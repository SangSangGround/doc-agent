"""Roboflow Workflow(RF-DETR) 기반 기입란 탐지 어댑터(선택적 의존).

Roboflow 에서 학습한 RF-DETR 모델을 서버리스 Workflow 로 호출해
체크박스(``check_box``)와 서명란(``signature_field``)을 탐지한다.

``inference-sdk`` 는 **선택적 패키지**다. 이 모듈은 최상단에서 import 하지 않으므로
패키지가 없어도 ``import`` 는 항상 성공하고, 생성자에서 지연 import 하여
``ImportError`` 를 :class:`~docagent.errors.AdapterUnavailable` 로 감싼다.

출력 계약은 다른 탐지기와 같다. :meth:`RoboflowDetector.detect` 는 픽셀 결과를
A4 mm 좌표의 :class:`~docagent.contracts.Detection` 으로 변환해 돌려준다.

Workflow 응답 구조 (2026-10 실제 호출로 확인)
---------------------------------------------
::

    [
      {
        "predictions": {
          "image": {"width": 2067, "height": 2924},
          "predictions": [
            {"x": 717.5, "y": 2007.5, "width": 67.0, "height": 79.0,
             "confidence": 0.76, "class": "check_box", "class_id": 0, ...}
          ]
        },
        "inference_id": "...",
        "model_id": "..."
      }
    ]

* ``x``/``y`` 는 상자 **중심** 픽셀 좌표다.
* 탐지가 0건이면 ``image.width/height`` 가 ``null`` 로 온다(오류 아님).
* 클래스 이름은 ``check_box`` 다(``checkbox`` 아님). 둘 다 체크박스로 매핑한다.

API 키
------
``ROBOFLOW_API_KEY`` 환경변수(또는 생성자 인자)로만 받는다. 코드·저장소에 키를 두지 않는다.
CLI 는 ``python-dotenv`` 가 있으면 저장소 루트의 ``.env`` 를 읽는다.

CLI 사용 예::

    python -m docagent.detector.roboflow_detector images/test.jpg
    python -m docagent.detector.roboflow_detector images/test.jpg --conf 0.3 --output outputs/result.jpg
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from docagent.contracts import BoxPx, Detection, FieldType
from docagent.errors import AdapterUnavailable, VisionError
from docagent.vision.heuristic import coordinate_system_from_image

__all__ = [
    "SOURCE_ROBOFLOW",
    "ROBOFLOW_FEATURE",
    "ROBOFLOW_EXTRA",
    "API_KEY_ENV",
    "MODEL_ID_ENV",
    "DEFAULT_API_URL",
    "DEFAULT_WORKSPACE",
    "DEFAULT_WORKFLOW_ID",
    "DEFAULT_MODEL_ID",
    "DEFAULT_CONFIDENCE",
    "CLASS_NAME_TO_FIELD_TYPE",
    "RawPrediction",
    "parse_workflow_result",
    "RoboflowDetector",
    "main",
]

_LOG = logging.getLogger(__name__)

#: Roboflow 탐지 결과의 :attr:`Detection.source` 값.
SOURCE_ROBOFLOW: str = "roboflow_rfdetr"
#: :class:`AdapterUnavailable` 메시지에 쓰는 기능 설명.
ROBOFLOW_FEATURE: str = "Roboflow RF-DETR 기입란 탐지"
#: ``pip install "docagent[<extra>]"`` 안내에 쓰는 optional 그룹 이름.
ROBOFLOW_EXTRA: str = "roboflow"

#: API 키 환경변수 이름.
API_KEY_ENV: str = "ROBOFLOW_API_KEY"
#: 모델 id 덮어쓰기 환경변수 이름.
MODEL_ID_ENV: str = "ROBOFLOW_MODEL_ID"

DEFAULT_API_URL: str = "https://serverless.roboflow.com"
DEFAULT_WORKSPACE: str = "-0qtd3"
DEFAULT_WORKFLOW_ID: str = "my-first-project-lkpc5"
#: Workflow 의 기본 model_id(``...--9602f1``)는 실측 결과 탐지가 0건이라,
#: 학습된 RF-DETR 모델을 workflow 파라미터로 명시해서 넘긴다.
DEFAULT_MODEL_ID: str = "-0qtd3/my-first-project-lkpc5-3-rfdetr-nas-t1--f98b3b"
#: 기본 신뢰도 임계값.
DEFAULT_CONFIDENCE: float = 0.5

#: Roboflow 클래스 이름 → 도메인 :class:`FieldType`.
CLASS_NAME_TO_FIELD_TYPE: dict[str, FieldType] = {
    "check_box": FieldType.CHECKBOX,
    "checkbox": FieldType.CHECKBOX,
    "signature_field": FieldType.SIGNATURE,
}


# --------------------------------------------------------------------------
# 응답 파싱 (네트워크 무관, 순수 함수)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawPrediction:
    """Workflow 가 돌려준 탐지 1건(픽셀 좌표, Vision 내부 전용).

    :param class_name: Roboflow 클래스 이름(예: ``"check_box"``).
    :param confidence: 0.0~1.0 신뢰도.
    :param x: 상자 중심 x(px).
    :param y: 상자 중심 y(px).
    :param width: 상자 폭(px).
    :param height: 상자 높이(px).
    """

    class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    def to_box_px(self) -> BoxPx:
        """중심 좌표 상자를 좌상단 기준 :class:`BoxPx` 로 바꾼다.

        :returns: 정수 픽셀 상자.
        """
        return BoxPx(
            x=int(round(self.x - self.width / 2.0)),
            y=int(round(self.y - self.height / 2.0)),
            w=max(0, int(round(self.width))),
            h=max(0, int(round(self.height))),
        )


def parse_workflow_result(result: Any) -> list[RawPrediction]:
    """``client.run_workflow()`` 반환값에서 탐지 목록을 꺼낸다.

    :param result: Workflow 응답(모듈 독스트링의 구조).
    :returns: :class:`RawPrediction` 목록. 탐지가 없으면 빈 리스트.
    :raises docagent.errors.VisionError: 응답 구조가 예상과 다른 경우.
    """
    if not isinstance(result, list) or not result or not isinstance(result[0], dict):
        raise VisionError(
            f"예상하지 못한 Roboflow 응답입니다(list[dict] 가 아님): {str(result)[:300]}"
        )
    output = result[0].get("predictions")
    if not isinstance(output, dict) or not isinstance(output.get("predictions"), list):
        raise VisionError(
            "예상하지 못한 Roboflow 응답입니다('predictions.predictions' 리스트 없음): "
            f"{str(result[0])[:300]}"
        )

    predictions: list[RawPrediction] = []
    for item in output["predictions"]:
        try:
            predictions.append(
                RawPrediction(
                    class_name=str(item["class"]),
                    confidence=float(item["confidence"]),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    width=float(item["width"]),
                    height=float(item["height"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VisionError(f"예상하지 못한 Roboflow 탐지 형식입니다: {item}") from exc
    return predictions


# --------------------------------------------------------------------------
# 지연 import
# --------------------------------------------------------------------------


def _require_sdk() -> tuple[Any, Any]:
    """``inference_sdk`` 의 클라이언트·설정 클래스를 지연 import 한다.

    :returns: ``(InferenceHTTPClient, InferenceConfiguration)``.
    :raises docagent.errors.AdapterUnavailable: inference-sdk 가 설치되지 않은 경우.
    """
    try:
        from inference_sdk import (  # noqa: PLC0415 - 선택적 의존이라 지연 import 한다.
            InferenceConfiguration,
            InferenceHTTPClient,
        )
    except ImportError as exc:
        raise AdapterUnavailable(
            package="inference-sdk", feature=ROBOFLOW_FEATURE, extra=ROBOFLOW_EXTRA
        ) from exc
    return InferenceHTTPClient, InferenceConfiguration


# --------------------------------------------------------------------------
# 탐지기
# --------------------------------------------------------------------------


class RoboflowDetector:
    """Roboflow Workflow(RF-DETR)로 체크박스·서명란을 탐지하는 어댑터.

    :class:`docagent.interfaces.Detector` 프로토콜을 만족한다.

    :param api_key: Roboflow API 키. ``None`` 이면 ``ROBOFLOW_API_KEY`` 환경변수.
    :param conf: 결과에 남길 최소 신뢰도(0.0~1.0).
    :param model_id: Workflow 에 넘길 모델 id. ``None`` 이면 ``ROBOFLOW_MODEL_ID``
        환경변수, 그것도 없으면 :data:`DEFAULT_MODEL_ID`.
    :param workspace: Roboflow workspace 이름.
    :param workflow_id: Workflow id.
    :param api_url: 추론 서버 URL.
    :param client: 테스트용 주입 클라이언트(``run_workflow`` 메서드만 필요).
        주어지면 inference-sdk 를 import 하지 않는다.
    :raises ValueError: ``conf`` 가 범위를 벗어났거나 API 키가 없는 경우.
    :raises docagent.errors.AdapterUnavailable: inference-sdk 가 설치되지 않은 경우.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        conf: float = DEFAULT_CONFIDENCE,
        model_id: str | None = None,
        workspace: str = DEFAULT_WORKSPACE,
        workflow_id: str = DEFAULT_WORKFLOW_ID,
        api_url: str = DEFAULT_API_URL,
        client: Any | None = None,
    ) -> None:
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"conf 는 0.0~1.0 이어야 합니다: {conf}")
        self.conf = float(conf)
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.model_id = (
            model_id or os.environ.get(MODEL_ID_ENV, "").strip() or DEFAULT_MODEL_ID
        )

        if client is not None:
            self._client = client
            return

        key = (api_key or os.environ.get(API_KEY_ENV, "")).strip()
        if not key:
            raise ValueError(
                f"Roboflow API 키가 없습니다. 환경변수 {API_KEY_ENV} 또는 저장소 루트 "
                f".env 파일에 '{API_KEY_ENV}=<키>' 형태로 설정하십시오."
            )
        client_class, config_class = _require_sdk()
        self._client = client_class(api_url=api_url, api_key=key)
        self._client.configure(config_class(api_key_transport="header"))

    # -- 공개 API --------------------------------------------------------

    def predict(self, image: Any) -> list[RawPrediction]:
        """Workflow 를 호출해 픽셀 좌표 탐지 결과를 돌려준다(신뢰도 필터 적용).

        :param image: 이미지 파일 경로(``str``/:class:`~pathlib.Path`) 또는
            ``(H, W, 3)`` uint8 BGR 배열.
        :returns: 신뢰도 내림차순 :class:`RawPrediction` 목록.
        :raises docagent.errors.VisionError: API 호출 실패 또는 응답 구조 이상.
        """
        source = str(image) if isinstance(image, Path) else image
        try:
            result = self._client.run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow_id,
                images={"image": source},
                parameters={"model_id": self.model_id},
                use_cache=True,
            )
        except Exception as exc:  # noqa: BLE001 - SDK 의 HTTP·연결 예외를 도메인 예외로 감싼다.
            raise VisionError(
                "Roboflow API 호출에 실패했습니다. API 키·모델 설정과 네트워크를 확인하십시오."
            ) from exc

        predictions = [p for p in parse_workflow_result(result) if p.confidence >= self.conf]
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        _LOG.debug("Roboflow 탐지 %d건(conf>=%.2f).", len(predictions), self.conf)
        return predictions

    def detect(self, image: Any) -> list[Detection]:
        """이미지에서 기입란 후보를 탐지한다.

        :param image: A4 로 정립된 문서 이미지.
            ``(H, W, 3)`` uint8 BGR 또는 ``(H, W)`` uint8 그레이스케일.
        :returns: A4 mm 좌표의 :class:`~docagent.contracts.Detection` 목록.
            ``(type, y_mm, x_mm)`` 순으로 정렬된다. 후보가 없으면 빈 리스트.
        :raises docagent.errors.VisionError: 입력이 유효하지 않거나 추론에 실패한 경우.
        """
        if not isinstance(image, np.ndarray):
            raise VisionError(
                f"입력 이미지는 numpy 배열이어야 합니다: {type(image).__name__}"
            )
        if image.size == 0:
            raise VisionError("입력 이미지가 비어 있습니다.")

        coords = coordinate_system_from_image(image)
        detections: list[Detection] = []
        for prediction in self.predict(image):
            field_type = CLASS_NAME_TO_FIELD_TYPE.get(prediction.class_name)
            if field_type is None:
                _LOG.warning(
                    "도메인 유형에 매핑되지 않은 Roboflow 클래스라 제외합니다: %s",
                    prediction.class_name,
                )
                continue
            detections.append(
                Detection(
                    type=field_type,
                    box_mm=coords.to_mm(prediction.to_box_px(), clamp=True),
                    confidence=float(min(1.0, max(0.0, prediction.confidence))),
                    source=SOURCE_ROBOFLOW,
                )
            )
        detections.sort(
            key=lambda d: (d.type.value, round(d.box_mm.y_mm, 3), round(d.box_mm.x_mm, 3))
        )
        return detections


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_dotenv() -> None:
    """``python-dotenv`` 가 있으면 저장소 루트·현재 디렉터리의 ``.env`` 를 읽는다."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415 - 선택적 의존.
    except ImportError:
        _LOG.info("python-dotenv 가 없어 .env 를 읽지 않습니다(환경변수만 사용).")
        return
    repo_env = Path(__file__).resolve().parents[3] / ".env"
    for candidate in (Path.cwd() / ".env", repo_env):
        if candidate.is_file():
            load_dotenv(candidate)
            return


def format_results(predictions: Sequence[RawPrediction]) -> str:
    """탐지 결과를 사람이 읽는 텍스트로 만든다.

    :param predictions: 출력할 탐지 목록.
    :returns: 여러 줄 문자열.
    """
    lines = ["=== Detection Results ===", "(bbox x, y = 박스 중심 좌표, 단위 px)"]
    if not predictions:
        lines += ["", "탐지된 객체가 없습니다."]
    for index, p in enumerate(predictions, start=1):
        lines += [
            "",
            f"[{index}]",
            f"class      : {p.class_name}",
            f"confidence : {p.confidence * 100:.1f}%",
            f"bbox       : x={p.x:.1f}, y={p.y:.1f}, width={p.width:.1f}, height={p.height:.1f}",
        ]
    lines += ["", f"Total objects: {len(predictions)}"]
    return "\n".join(lines)


def draw_results(
    image_path: Path, predictions: Sequence[RawPrediction], output_path: Path
) -> None:
    """원본 이미지에 bbox 와 class/confidence 를 그려 저장한다.

    :param image_path: 원본 이미지 경로.
    :param predictions: 그릴 탐지 목록.
    :param output_path: 저장 경로(상위 디렉터리는 자동 생성).
    """
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415 - CLI 전용.

    colors = {FieldType.CHECKBOX: (230, 50, 50), FieldType.SIGNATURE: (40, 110, 230)}
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(max(image.size) / 800))
    try:
        font: Any = ImageFont.truetype("arial.ttf", max(14, round(max(image.size) / 100)))
    except OSError:
        font = ImageFont.load_default()

    for p in predictions:
        color = colors.get(CLASS_NAME_TO_FIELD_TYPE.get(p.class_name), (0, 170, 0))
        x1, y1 = p.x - p.width / 2.0, p.y - p.height / 2.0
        x2, y2 = p.x + p.width / 2.0, p.y + p.height / 2.0
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label = f"{p.class_name} {p.confidence * 100:.1f}%"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        label_y = y1 - text_h - 6 if y1 - text_h - 6 >= 0 else y2
        draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 6], fill=color)
        draw.text((x1 + 3, label_y + 3 - top), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 진입점.

    :param argv: 명령행 인자. ``None`` 이면 :data:`sys.argv`.
    :returns: 종료 코드(성공 0, 실패 1).
    """
    parser = argparse.ArgumentParser(
        description="Roboflow RF-DETR 문서 객체 탐지 (check_box / signature_field)"
    )
    parser.add_argument("image", help="입력 이미지 경로")
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONFIDENCE,
        help=f"출력할 최소 confidence (기본 {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--output", default="outputs/result.jpg",
        help="bbox 결과 이미지 저장 경로 (기본 outputs/result.jpg)",
    )
    parser.add_argument("--no-draw", action="store_true", help="결과 이미지 저장 안 함")
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[ERROR] 이미지 파일이 없습니다: {image_path}", file=sys.stderr)
        return 1

    _load_dotenv()
    try:
        detector = RoboflowDetector(conf=args.conf)
        predictions = detector.predict(image_path)
    except (ValueError, AdapterUnavailable, VisionError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(format_results(predictions))
    if predictions and not args.no_draw:
        output_path = Path(args.output)
        try:
            draw_results(image_path, predictions, output_path)
        except OSError as exc:
            print(f"[WARN] 결과 이미지 저장 실패(좌표 출력은 정상): {exc}", file=sys.stderr)
        else:
            print(f"\n결과 이미지 저장: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
