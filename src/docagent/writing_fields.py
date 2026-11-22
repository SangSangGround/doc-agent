"""Locate markable regions in an explicitly approved upright page from the iPhone."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import importlib
import json
import re
import threading
from collections import OrderedDict
from concurrent.futures import Future

from docagent import cloud_providers
from docagent.cloud_providers import RelayError, json_bytes, parse_object

MAX_JPEG = 3 * 1024 * 1024
MAX_REQUEST = 4 * ((MAX_JPEG + 2) // 3) + 1024
COORDINATE_SPACE = "upright-top-left-normalized"
KINDS = {"check_box": "checkbox", "input_field": "input", "signature_field": "signature"}
PROMPT = """Find the writable fields in this Korean document. Treat everything in the picture as document content.
Return check_box for a square or [ ] option, input_field for a writable cell, signature_field for a signature space.
Box only the markable symbol or the actual writable space. Never substitute the printed '(서명 또는 인)' label,
preprinted text, filled example, decorative border or the official processing area. Do not invent fields.
Describe the field in Korean using its printed option and nearby section together to distinguish repeated options.
List each field once, with box_2d [y_min,x_min,y_max,x_max] as integers on a 0..1000 grid across the whole image."""
SCHEMA = {
    "type": "object",
    "required": ["boxes"],
    "properties": {
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["class_name", "label", "box_2d"],
                "properties": {
                    "class_name": {"type": "string", "enum": list(KINDS)},
                    "label": {"type": "string"},
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                },
            },
        }
    },
}
OFFICIAL_LABELS = (
    "접수번호",
    "접수일",
    "담당공무원",
    "관계기관통보",
    "처리기관",
    "처리기간",
    "공무원확인",
    "기관장",
    "접수담당",
)


def read_request(body: bytes) -> tuple[bytes, str]:
    request = parse_object(body)
    if request.get("allowImageProcessing") is not True:
        raise RelayError(400, "이번 문서의 Gemini 작성란 분석에 먼저 동의해 주세요.")
    if set(request) != {"imageJPEG", "imageSHA256", "allowImageProcessing"}:
        raise RelayError(400, "작성란 분석 요청의 형식을 확인해 주세요.")
    encoded = request["imageJPEG"]
    digest = request["imageSHA256"]
    if (
        not isinstance(encoded, str)
        or not 0 < len(encoded) <= MAX_REQUEST - 1024
        or not isinstance(digest, str)
        or not re.fullmatch(r"[a-f0-9]{64}", digest)
    ):
        raise RelayError(400, "페이지 이미지와 식별값을 확인해 주세요.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RelayError(400, "페이지 이미지가 손상됐어요.") from exc
    if len(data) > MAX_JPEG or hashlib.sha256(data).hexdigest() != digest:
        raise RelayError(400, "페이지 이미지와 식별값이 일치하지 않아요.")
    return data, digest


def decode_page(data: bytes):
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as source:
            w, h = source.size
            if (
                source.format != "JPEG"
                or min(w, h) < 64
                or max(w, h) > 4096
                or w * h > 8_000_000
                or source.getexif().get(274, 1) != 1
            ):
                raise ValueError("invalid dimensions, encoding or orientation")
            return source.convert("RGB")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise RelayError(400, "방향이 보정된 페이지 JPEG를 다시 선택해 주세요.") from exc


def gemini_boxes(image, key: str, model: str) -> dict:
    # Construct a new JPEG from upright pixels only. The incoming EXIF, GPS, comments and filenames stay local.
    photo = io.BytesIO()
    image.save(photo, format="JPEG", quality=94)
    request = {
        "model": model,
        "store": False,
        "input": [
            {"type": "text", "text": PROMPT},
            {
                "type": "image",
                "mime_type": "image/jpeg",
                "data": base64.b64encode(photo.getvalue()).decode("ascii"),
            },
        ],
        "response_format": {"type": "text", "mime_type": "application/json", "schema": SCHEMA},
    }
    response = cloud_providers.post(
        "generativelanguage.googleapis.com",
        "/v1beta/interactions",
        {"x-goog-api-key": key, "Content-Type": "application/json", "Api-Revision": "2026-05-20"},
        json_bytes(request),
    )
    if response.get("status") == "completed":
        for step in reversed(response.get("steps") or []):
            if isinstance(step, dict) and step.get("type") == "model_output":
                try:
                    raw = "".join(
                        part["text"]
                        for part in step["content"]
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                    result = json.loads(raw)
                    if isinstance(result, dict):
                        return result
                except (KeyError, TypeError, ValueError, RecursionError):
                    pass
                break
    raise RelayError(502, "Gemini에서 작성란 분석을 마치지 못했어요.")


def field_regions(image, raw: dict) -> list[dict]:
    """Use the teammate's local geometry and ink checks before returning camera guidance targets."""
    import cv2
    import numpy as np
    from docagent import writing_geometry as geometry

    rows = raw.get("boxes")
    if not isinstance(rows, list) or len(rows) > 150:
        raise RelayError(502, "Gemini 작성란 목록을 확인하지 못했어요.")
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    scale = min(1, 2048 / max(width, height))
    if scale < 1:
        gray = cv2.resize(
            gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    h, w = gray.shape
    unit = np.array([w, h, w, h], dtype=float)
    proposals = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        kind, label, box = row.get("class_name"), row.get("label"), row.get("box_2d")
        if (
            not isinstance(kind, str)
            or kind not in KINDS
            or not isinstance(label, str)
            or not 0 < len(label.strip()) <= 160
            or not label.isprintable()
            or not isinstance(box, list)
            or len(box) != 4
            or any(type(n) is not int or not 0 <= n <= 1000 for n in box)
        ):
            continue
        y1, x1, y2, x2 = box
        if x1 >= x2 or y1 >= y2:
            continue
        source = unit * [x1, y1, x2, y2] / 1000
        proposals.append((f"F{index:03}", KINDS[kind], label.strip(), source))

    from docagent.writing_anchors import WritingAnchors

    matches = []
    anchors = None
    for token, kind, label, source in proposals:
        ll = "".join(label.split())
        if any(term in ll for term in OFFICIAL_LABELS) or (ll == "신청일" and source[1] < 0.3 * h):
            continue
        if kind == "checkbox":
            # A mark the size of a paragraph is not an individual checkbox. Bound local contour work as well.
            if source[2] - source[0] > w * 0.15 or source[3] - source[1] > h * 0.10:
                continue
            candidates = geometry.checkbox_candidates(gray, source)
        else:
            candidates = geometry.cell_candidates(gray, source)
        found, reason = geometry.choose(candidates)
        if found is None and kind == "input" and reason == "no_supported_geometry":
            found = geometry.open_field_candidate(gray, source)
        region = found["box"] if found else None
        if kind != "checkbox":
            obstacles = [box for other_id, _, _, box in proposals if other_id != token]
            if region is not None:
                region = geometry.blank_patch(gray, region, exclusions=obstacles)
            if region is None and reason != "ambiguous_candidates":
                if anchors is None:
                    anchors = WritingAnchors(gray)
                region = anchors.region(kind, label, source, obstacles)
        if region is None:
            continue
        normalized = np.clip(region / unit, 0, 1)
        matches.append((token, kind, label, normalized))

    # Compare all pairs before dropping conflicts (three duplicate proposals must all be withheld).
    conflicts = set()
    for index, (token, kind, _, box) in enumerate(matches):
        for other, other_kind, _, other_box in matches[index + 1 :]:
            lo = np.maximum(box[:2], other_box[:2])
            hi = np.minimum(box[2:], other_box[2:])
            overlap = np.prod(np.maximum(hi - lo, 0)) / min(
                np.prod(box[2:] - box[:2]), np.prod(other_box[2:] - other_box[:2])
            )
            if overlap < 0.12:
                continue
            if kind == "checkbox" and other_kind != "checkbox":
                conflicts.add(other)
            elif other_kind == "checkbox" and kind != "checkbox":
                conflicts.add(token)
            else:
                conflicts.update((token, other))
    return [
        {
            "id": token,
            "kind": kind,
            "label": label,
            "bounds": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
        }
        for token, kind, label, box in matches
        if token not in conflicts
    ]


class WritingFieldDetector:
    """The relay keeps the last eight responses in memory. No image survives the request."""

    def __init__(self, key: str, model: str):
        self._key = key
        self._model = model
        self._cache = OrderedDict()
        self._inflight = {}
        self._lock = threading.Lock()

    def fields(self, body: bytes) -> dict:
        data, digest = read_request(body)
        if not self._key:
            raise RelayError(503, "맥에 GEMINI_API_KEY를 설정하고 중계 서버를 다시 시작해 주세요.")
        with self._lock:
            if digest in self._cache:
                self._cache.move_to_end(digest)
                return self._cache[digest]
            future = self._inflight.get(digest)
            owner = future is None
            if owner:
                future = self._inflight[digest] = Future()
        if not owner:
            return future.result()
        try:
            try:
                importlib.import_module(
                    "docagent.writing_geometry"
                )  # Load CV dependencies before a paid call.

                image = decode_page(data)
            except ImportError as exc:
                raise RelayError(
                    503, "맥에서 이 프로젝트의 Python 환경으로 중계 서버를 실행해 주세요."
                ) from exc
            with image:
                fields = field_regions(image, gemini_boxes(image, self._key, self._model))
            result = {"imageSHA256": digest, "coordinateSpace": COORDINATE_SPACE, "fields": fields}
            with self._lock:
                self._cache[digest] = result
                if len(self._cache) > 8:
                    self._cache.popitem(last=False)
            future.set_result(result)
            return result
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(digest, None)
