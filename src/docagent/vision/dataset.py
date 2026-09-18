"""라벨 데이터셋 유틸 — YOLO 라벨 입출력 · 변환 · 분할 · 검증(로드맵 Phase 2).

이 모듈은 학습 데이터 준비 전용이며 **선택적 패키지에 전혀 의존하지 않는다**
(ultralytics·torch 없이도 import·실행된다). 표준 라이브러리와 numpy 만 쓴다.

클래스 정의
-----------
탐지 클래스는 두 개로 고정한다. 이 매핑은
:mod:`docagent.vision.yolo_adapter` 와 ``scripts/train_yolo.py`` 가 공유한다.

===== ================== =============================================
 id    이름               대응 :class:`~docagent.contracts.FieldType`
===== ================== =============================================
  0    ``signature_field`` :attr:`FieldType.SIGNATURE`
  1    ``checkbox``        :attr:`FieldType.CHECKBOX`
===== ================== =============================================

YOLO 라벨 포맷
--------------
한 줄에 ``class_id cx cy w h`` 다섯 값을 공백으로 구분해 적는다.
``cx``/``cy``/``w``/``h`` 는 이미지 크기로 나눈 **정규화 값(0~1)** 이며,
``cx``/``cy`` 는 상자의 **중심**이다. 파일 인코딩은 항상 UTF-8 이다.

좌표 규약
---------
:func:`structure_to_yolo_labels` 는 정답 :class:`DocumentStructure` 의 mm 좌표를
페이지 크기로 나눠 정규화한다. 즉 **정규화 값은 해상도(dpi)와 무관**하므로,
같은 서식을 여러 dpi 로 렌더링해도 라벨 파일 하나를 그대로 쓸 수 있다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BoxMm,
    BoxPx,
    DocumentStructure,
    FieldType,
)

__all__ = [
    "CLASS_MAP",
    "CLASS_NAMES",
    "NAME_TO_CLASS_ID",
    "FIELD_TYPE_TO_CLASS_ID",
    "COORD_PRECISION",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_SPLIT_SEED",
    "YoloLabel",
    "LabelReport",
    "DatasetSplit",
    "read_yolo_labels",
    "write_yolo_labels",
    "structure_to_yolo_labels",
    "export_structure_labels",
    "coco_to_yolo",
    "split_dataset",
    "write_data_yaml",
    "validate_labels",
    "validate_label_file",
    "validate_label_dir",
]

_LOG = logging.getLogger(__name__)

#: 클래스 id → 이름.
CLASS_MAP: dict[int, str] = {0: "signature_field", 1: "checkbox"}
#: 클래스 id 순서대로 정렬된 이름 튜플(data.yaml 의 ``names`` 순서).
CLASS_NAMES: tuple[str, ...] = tuple(CLASS_MAP[key] for key in sorted(CLASS_MAP))
#: 이름 → 클래스 id.
NAME_TO_CLASS_ID: dict[str, int] = {name: key for key, name in CLASS_MAP.items()}
#: 도메인 :class:`FieldType` → 클래스 id.
FIELD_TYPE_TO_CLASS_ID: dict[FieldType, int] = {
    FieldType.SIGNATURE: 0,
    FieldType.CHECKBOX: 1,
}

#: 정규화 좌표를 파일에 적을 때 쓰는 소수점 자리수.
#: :class:`YoloLabel` 은 생성 시점에 이 자리수로 반올림하므로
#: ``YoloLabel.from_line(label.to_line()) == label`` 이 항상 성립한다.
COORD_PRECISION: int = 6

#: :func:`split_dataset` 의 기본 분할 비율 ``(train, val, test)``.
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)
#: :func:`split_dataset` 의 기본 난수 시드.
DEFAULT_SPLIT_SEED: int = 20260909


# --------------------------------------------------------------------------
# 라벨 1건
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class YoloLabel:
    """YOLO 라벨 한 줄 — 클래스 id 와 정규화된 중심·크기.

    생성 시점에 좌표를 :data:`COORD_PRECISION` 자리로 반올림하므로
    텍스트 왕복(:meth:`to_line` → :meth:`from_line`)이 **무손실**이다.

    :param class_id: :data:`CLASS_MAP` 에 정의된 클래스 id.
    :param cx: 상자 중심 x(0.0~1.0, 이미지 폭으로 정규화).
    :param cy: 상자 중심 y(0.0~1.0, 이미지 높이로 정규화).
    :param w: 상자 폭(0.0 초과 1.0 이하).
    :param h: 상자 높이(0.0 초과 1.0 이하).
    :raises ValueError: 클래스 id 가 미정의이거나 좌표가 범위를 벗어난 경우.
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.class_id not in CLASS_MAP:
            allowed = ", ".join(f"{key}={name}" for key, name in sorted(CLASS_MAP.items()))
            raise ValueError(
                f"정의되지 않은 클래스 id 입니다: {self.class_id}. 허용 값: {allowed}"
            )
        for name in ("cx", "cy", "w", "h"):
            object.__setattr__(
                self, name, round(float(getattr(self, name)), COORD_PRECISION)
            )
        for name in ("cx", "cy"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"YoloLabel.{name} 은 0.0~1.0 이어야 합니다: {value}"
                )
        for name in ("w", "h"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"YoloLabel.{name} 는 0.0 초과 1.0 이하여야 합니다: {value}"
                )

    @property
    def class_name(self) -> str:
        """클래스 이름(예: ``"checkbox"``)."""
        return CLASS_MAP[self.class_id]

    @property
    def field_type(self) -> FieldType:
        """대응하는 도메인 :class:`FieldType`."""
        for field_type, class_id in FIELD_TYPE_TO_CLASS_ID.items():
            if class_id == self.class_id:
                return field_type
        raise ValueError(f"클래스 id 에 대응하는 FieldType 이 없습니다: {self.class_id}")

    # -- 텍스트 왕복 -----------------------------------------------------

    def to_line(self) -> str:
        """YOLO 라벨 한 줄 문자열로 직렬화한다.

        :returns: ``"<class_id> <cx> <cy> <w> <h>"`` (좌표는 소수점 6자리).
        """
        values = " ".join(f"{value:.{COORD_PRECISION}f}" for value in (self.cx, self.cy, self.w, self.h))
        return f"{self.class_id} {values}"

    @classmethod
    def from_line(cls, line: str) -> "YoloLabel":
        """YOLO 라벨 한 줄을 파싱한다.

        :param line: ``"<class_id> <cx> <cy> <w> <h>"`` 형식 문자열.
        :returns: :class:`YoloLabel`.
        :raises ValueError: 토큰 수가 5개가 아니거나 숫자로 변환할 수 없는 경우.
        """
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(
                f"YOLO 라벨 한 줄은 5개 값이어야 합니다(class_id cx cy w h): {line!r}"
            )
        try:
            class_id = int(tokens[0])
            cx, cy, w, h = (float(token) for token in tokens[1:])
        except ValueError as exc:
            raise ValueError(f"YOLO 라벨을 숫자로 변환할 수 없습니다: {line!r}") from exc
        return cls(class_id=class_id, cx=cx, cy=cy, w=w, h=h)

    # -- 좌표 변환 -------------------------------------------------------

    def to_box_px(self, width_px: int, height_px: int) -> BoxPx:
        """정규화 좌표를 픽셀 사각형으로 되돌린다.

        :param width_px: 이미지 폭(px, 1 이상).
        :param height_px: 이미지 높이(px, 1 이상).
        :returns: :class:`~docagent.contracts.BoxPx`.
        :raises ValueError: 이미지 크기가 1 미만인 경우.
        """
        if width_px < 1 or height_px < 1:
            raise ValueError(
                f"이미지 크기는 1px 이상이어야 합니다: {width_px}x{height_px}"
            )
        w_px = self.w * width_px
        h_px = self.h * height_px
        return BoxPx(
            x=int(round(self.cx * width_px - w_px / 2.0)),
            y=int(round(self.cy * height_px - h_px / 2.0)),
            w=max(0, int(round(w_px))),
            h=max(0, int(round(h_px))),
        )

    @classmethod
    def from_box_px(
        cls, class_id: int, box_px: BoxPx, width_px: int, height_px: int
    ) -> "YoloLabel":
        """픽셀 사각형을 정규화 라벨로 바꾼다.

        :param class_id: 클래스 id.
        :param box_px: 픽셀 사각형.
        :param width_px: 이미지 폭(px, 1 이상).
        :param height_px: 이미지 높이(px, 1 이상).
        :returns: :class:`YoloLabel`.
        :raises ValueError: 이미지 크기가 1 미만이거나 좌표가 범위를 벗어난 경우.
        """
        if width_px < 1 or height_px < 1:
            raise ValueError(
                f"이미지 크기는 1px 이상이어야 합니다: {width_px}x{height_px}"
            )
        return cls(
            class_id=class_id,
            cx=(box_px.x + box_px.w / 2.0) / width_px,
            cy=(box_px.y + box_px.h / 2.0) / height_px,
            w=box_px.w / width_px,
            h=box_px.h / height_px,
        )

    def to_box_mm(
        self, page_size_mm: tuple[float, float] = (A4_WIDTH_MM, A4_HEIGHT_MM)
    ) -> BoxMm:
        """정규화 좌표를 mm 사각형으로 되돌린다.

        :param page_size_mm: 페이지 크기 ``(가로_mm, 세로_mm)``. 기본 A4.
        :returns: :class:`~docagent.contracts.BoxMm`.
        :raises ValueError: 페이지 크기가 0 이하인 경우.
        """
        page_w, page_h = _check_page_size(page_size_mm)
        w_mm = self.w * page_w
        h_mm = self.h * page_h
        return BoxMm(
            x_mm=self.cx * page_w - w_mm / 2.0,
            y_mm=self.cy * page_h - h_mm / 2.0,
            w_mm=w_mm,
            h_mm=h_mm,
        )

    @classmethod
    def from_box_mm(
        cls,
        class_id: int,
        box_mm: BoxMm,
        page_size_mm: tuple[float, float] = (A4_WIDTH_MM, A4_HEIGHT_MM),
    ) -> "YoloLabel":
        """mm 사각형을 정규화 라벨로 바꾼다.

        :param class_id: 클래스 id.
        :param box_mm: mm 사각형.
        :param page_size_mm: 페이지 크기 ``(가로_mm, 세로_mm)``. 기본 A4.
        :returns: :class:`YoloLabel`.
        :raises ValueError: 페이지 크기가 0 이하이거나 정규화 값이 범위를 벗어난 경우.
        """
        page_w, page_h = _check_page_size(page_size_mm)
        return cls(
            class_id=class_id,
            cx=box_mm.center().x_mm / page_w,
            cy=box_mm.center().y_mm / page_h,
            w=box_mm.w_mm / page_w,
            h=box_mm.h_mm / page_h,
        )


def _check_page_size(page_size_mm: Sequence[float]) -> tuple[float, float]:
    """페이지 크기 시퀀스를 검증해 ``(가로, 세로)`` 로 돌려준다.

    :param page_size_mm: 길이 2 시퀀스.
    :returns: ``(page_w_mm, page_h_mm)``.
    :raises ValueError: 길이가 2 가 아니거나 값이 0 이하인 경우.
    """
    values = list(page_size_mm)
    if len(values) != 2:
        raise ValueError(f"페이지 크기는 (가로_mm, 세로_mm) 두 값이어야 합니다: {page_size_mm!r}")
    page_w, page_h = float(values[0]), float(values[1])
    if page_w <= 0 or page_h <= 0:
        raise ValueError(f"페이지 크기는 0보다 커야 합니다: {page_w}x{page_h}")
    return (page_w, page_h)


# --------------------------------------------------------------------------
# 파일 입출력
# --------------------------------------------------------------------------


def read_yolo_labels(path: str | Path) -> list[YoloLabel]:
    """YOLO 라벨 txt 파일을 읽는다.

    빈 줄과 ``#`` 로 시작하는 주석 줄은 건너뛴다.

    :param path: 라벨 파일 경로.
    :returns: :class:`YoloLabel` 목록. 라벨이 없으면 빈 리스트.
    :raises FileNotFoundError: 파일이 없는 경우.
    :raises ValueError: 형식이 잘못된 줄이 있는 경우(줄 번호를 메시지에 담는다).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다: {file_path}")
    labels: list[YoloLabel] = []
    text = file_path.read_text(encoding="utf-8")
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            labels.append(YoloLabel.from_line(line))
        except ValueError as exc:
            raise ValueError(f"{file_path} {number}번째 줄: {exc}") from exc
    return labels


def write_yolo_labels(path: str | Path, labels: Iterable[YoloLabel]) -> Path:
    """YOLO 라벨 txt 파일을 쓴다(상위 디렉터리를 자동 생성한다).

    :param path: 저장할 라벨 파일 경로.
    :param labels: :class:`YoloLabel` 반복자.
    :returns: 저장된 :class:`pathlib.Path`.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [label.to_line() for label in labels]
    file_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return file_path


# --------------------------------------------------------------------------
# DocumentStructure → YOLO 라벨
# --------------------------------------------------------------------------


def structure_to_yolo_labels(structure: DocumentStructure) -> list[YoloLabel]:
    """정답 :class:`DocumentStructure` 를 YOLO 라벨 목록으로 바꾼다.

    합성 데이터(:mod:`docagent.testing.synthetic`)로 학습셋을 만들 수 있게 하는
    함수다. 다음 규칙으로 상자를 뽑는다.

    * :attr:`FieldType.SIGNATURE` 항목의 ``box_mm`` → 클래스 ``signature_field``
    * **선택지가 없는** :attr:`FieldType.CHECKBOX` 항목의 ``box_mm`` → 클래스 ``checkbox``
    * 모든 항목의 ``options[].box_mm`` → 클래스 ``checkbox``
      (:attr:`FieldType.CHOICE` 항목의 선택지 네모 칸이 여기 해당한다)

    **선택지를 가진 항목의 상자는 라벨로 내보내지 않는다.** 구조화 단계는 체크박스
    항목의 ``box_mm`` 을 제목 행까지 포함한 합집합으로 잡고 같은 물리적 칸을
    ``options[0].box_mm`` 으로도 담는다. 둘 다 내보내면 같은 칸이 두 번 라벨되고,
    그중 하나는 체크박스가 아니라 문구 블록 전체(예: 39×16mm)라서 학습셋에
    가짜 정답이 섞인다.

    좌표가 없는(``box_mm is None``) 항목은 조용히 건너뛰지 않고 경고 로그를 남긴다.

    :param structure: 정답 문서 구조.
    :returns: ``(order, y, x)`` 순으로 정렬된 :class:`YoloLabel` 목록.
    :raises ValueError: 좌표가 페이지 범위를 벗어나 정규화 값이 0~1 을 넘는 경우.
    """
    page_size = _check_page_size(structure.page_size_mm)
    labels: list[tuple[float, float, YoloLabel]] = []

    for item in sorted(structure.fields, key=lambda f: f.order):
        class_id = FIELD_TYPE_TO_CLASS_ID.get(item.type)
        if item.options:
            # 선택지 상자가 실제 칸이다. 항목 상자(제목 행 포함 합집합)를 함께
            # 내보내면 같은 칸이 중복 라벨되고 문구 블록이 체크박스로 학습된다.
            class_id = None
        if class_id is not None:
            if item.box_mm is None:
                _LOG.warning(
                    "좌표가 없어 라벨에서 제외합니다: 항목 id=%s, 유형=%s",
                    item.id,
                    item.type.value,
                )
            else:
                labels.append(
                    (
                        item.box_mm.y_mm,
                        item.box_mm.x_mm,
                        YoloLabel.from_box_mm(class_id, item.box_mm, page_size),
                    )
                )
        for option in item.options:
            labels.append(
                (
                    option.box_mm.y_mm,
                    option.box_mm.x_mm,
                    YoloLabel.from_box_mm(
                        FIELD_TYPE_TO_CLASS_ID[FieldType.CHECKBOX],
                        option.box_mm,
                        page_size,
                    ),
                )
            )

    labels.sort(key=lambda entry: (round(entry[0], 4), round(entry[1], 4)))
    return [label for _y, _x, label in labels]


def export_structure_labels(structure: DocumentStructure, path: str | Path) -> Path:
    """정답 구조를 YOLO 라벨 파일로 저장한다.

    :param structure: 정답 문서 구조.
    :param path: 저장할 라벨 파일 경로.
    :returns: 저장된 :class:`pathlib.Path`.
    """
    return write_yolo_labels(path, structure_to_yolo_labels(structure))


# --------------------------------------------------------------------------
# COCO → YOLO
# --------------------------------------------------------------------------


def coco_to_yolo(
    coco: str | Path | Mapping[str, Any],
    out_dir: str | Path,
    *,
    category_map: Mapping[str, int] | None = None,
    skip_unknown: bool = False,
) -> dict[str, Path]:
    """COCO JSON 어노테이션을 YOLO txt 라벨로 변환한다.

    이미지 한 장당 라벨 파일 하나(``<이미지 파일명 stem>.txt``)를 만든다.
    어노테이션이 하나도 없는 이미지도 **빈 라벨 파일**을 만들어, 배경 전용
    샘플이 학습에서 누락되지 않게 한다.

    :param coco: COCO JSON 파일 경로 또는 이미 파싱된 dict.
        ``images`` / ``annotations`` / ``categories`` 키를 가져야 한다.
    :param out_dir: 라벨 txt 를 저장할 디렉터리(없으면 만든다).
    :param category_map: COCO 카테고리 이름 → YOLO 클래스 id.
        ``None`` 이면 :data:`NAME_TO_CLASS_ID` 를 쓴다.
    :param skip_unknown: True 면 매핑에 없는 카테고리를 경고 후 건너뛴다.
        False(기본)면 :class:`ValueError` 를 던진다(조용한 실패 금지).
    :returns: ``{이미지 파일명: 저장된 라벨 경로}``.
    :raises FileNotFoundError: JSON 파일이 없는 경우.
    :raises ValueError: 필수 키 누락, 이미지 크기 이상, 미정의 카테고리인 경우.
    """
    payload = _load_coco(coco)
    mapping = dict(category_map) if category_map is not None else dict(NAME_TO_CLASS_ID)

    for key in ("images", "annotations", "categories"):
        if key not in payload:
            raise ValueError(f"COCO JSON 에 '{key}' 키가 없습니다.")

    category_names: dict[int, str] = {
        int(category["id"]): str(category["name"]) for category in payload["categories"]
    }
    images: dict[int, dict[str, Any]] = {
        int(image["id"]): image for image in payload["images"]
    }
    grouped: dict[int, list[YoloLabel]] = {image_id: [] for image_id in images}

    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in images:
            raise ValueError(
                f"COCO 어노테이션이 존재하지 않는 image_id 를 가리킵니다: {image_id}"
            )
        category_id = int(annotation["category_id"])
        name = category_names.get(category_id)
        if name is None:
            raise ValueError(f"COCO categories 에 없는 category_id 입니다: {category_id}")
        if name not in mapping:
            if skip_unknown:
                _LOG.warning("매핑에 없는 카테고리를 건너뜁니다: %s", name)
                continue
            allowed = ", ".join(sorted(mapping))
            raise ValueError(
                f"YOLO 클래스로 매핑되지 않은 카테고리입니다: {name!r}. 허용 값: {allowed}"
            )
        image = images[image_id]
        width_px, height_px = int(image["width"]), int(image["height"])
        bbox = [float(value) for value in annotation["bbox"]]
        if len(bbox) != 4:
            raise ValueError(f"COCO bbox 는 [x, y, w, h] 네 값이어야 합니다: {bbox!r}")
        box_px = BoxPx(
            x=int(round(bbox[0])),
            y=int(round(bbox[1])),
            w=max(0, int(round(bbox[2]))),
            h=max(0, int(round(bbox[3]))),
        )
        grouped[image_id].append(
            YoloLabel.from_box_px(mapping[name], box_px, width_px, height_px)
        )

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for image_id, image in images.items():
        file_name = str(image["file_name"])
        target = output / f"{Path(file_name).stem}.txt"
        written[file_name] = write_yolo_labels(target, grouped[image_id])
    return written


def _load_coco(coco: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    """COCO 입력을 dict 로 정규화한다.

    :param coco: JSON 파일 경로 또는 dict.
    :returns: 파싱된 매핑.
    :raises FileNotFoundError: 파일이 없는 경우.
    :raises ValueError: JSON 파싱에 실패했거나 최상위가 객체가 아닌 경우.
    """
    if isinstance(coco, Mapping):
        return coco
    path = Path(coco)
    if not path.is_file():
        raise FileNotFoundError(f"COCO JSON 파일을 찾을 수 없습니다: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"COCO JSON 파싱에 실패했습니다: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"COCO JSON 최상위는 객체여야 합니다: {path}")
    return payload


# --------------------------------------------------------------------------
# 분할
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSplit:
    """train / val / test 분할 결과.

    :param train: 학습 파일 목록.
    :param val: 검증 파일 목록.
    :param test: 시험 파일 목록.
    """

    train: tuple[str, ...] = ()
    val: tuple[str, ...] = ()
    test: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("train", "val", "test"):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))

    @property
    def counts(self) -> dict[str, int]:
        """분할별 개수 ``{"train": n, "val": n, "test": n}``."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    @property
    def total(self) -> int:
        """전체 파일 수."""
        return len(self.train) + len(self.val) + len(self.test)

    def to_dict(self) -> dict[str, list[str]]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}


def split_dataset(
    files: Sequence[str | Path],
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> DatasetSplit:
    """파일 목록을 train/val/test 로 **결정론적으로** 나눈다.

    입력 순서에 흔들리지 않도록 먼저 문자열 정렬한 뒤,
    ``numpy.random.default_rng(seed)`` 로 섞는다. 같은 입력 집합과 같은 시드는
    입력 순서가 달라도 항상 같은 결과를 낸다.

    개수는 누적 비율을 반올림해 배분하므로 합이 항상 입력 개수와 같다.

    :param files: 파일 경로(또는 식별자) 목록. 중복은 허용하지 않는다.
    :param ratios: ``(train, val, test)`` 비율. 합이 1.0 이어야 한다(오차 1e-6).
    :param seed: 난수 시드.
    :returns: :class:`DatasetSplit`.
    :raises ValueError: 비율 개수·합이 잘못되었거나 중복 파일이 있는 경우.
    """
    values = [float(ratio) for ratio in ratios]
    if len(values) != 3:
        raise ValueError(f"ratios 는 (train, val, test) 세 값이어야 합니다: {ratios!r}")
    if any(value < 0.0 for value in values):
        raise ValueError(f"ratios 는 음수일 수 없습니다: {ratios!r}")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"ratios 의 합은 1.0 이어야 합니다: {sum(values)}")

    items = [str(item) for item in files]
    duplicates = sorted({item for item in items if items.count(item) > 1})
    if duplicates:
        raise ValueError(f"파일 목록에 중복이 있습니다: {', '.join(duplicates[:5])}")
    if not items:
        return DatasetSplit()

    ordered = sorted(items)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))
    shuffled = [ordered[int(index)] for index in permutation]

    total = len(shuffled)
    bounds: list[int] = []
    cumulative = 0.0
    for value in values[:-1]:
        cumulative += value
        bounds.append(int(round(cumulative * total)))
    train_end = min(bounds[0], total)
    val_end = min(max(bounds[1], train_end), total)
    return DatasetSplit(
        train=tuple(shuffled[:train_end]),
        val=tuple(shuffled[train_end:val_end]),
        test=tuple(shuffled[val_end:]),
    )


# --------------------------------------------------------------------------
# data.yaml
# --------------------------------------------------------------------------


def write_data_yaml(
    path: str | Path,
    root: str | Path,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    train: str = "images/train",
    val: str = "images/val",
    test: str | None = "images/test",
) -> Path:
    """Ultralytics 학습용 ``data.yaml`` 을 만든다(PyYAML 없이 직접 기록).

    경로는 Windows 역슬래시가 아니라 POSIX 슬래시로 적는다(ultralytics 관례).

    :param path: 저장할 yaml 경로.
    :param root: 데이터셋 루트 디렉터리(``path:`` 값).
    :param class_names: 클래스 id 순서대로 정렬된 이름 목록.
    :param train: 루트 기준 학습 이미지 경로.
    :param val: 루트 기준 검증 이미지 경로.
    :param test: 루트 기준 시험 이미지 경로. ``None`` 이면 항목을 적지 않는다.
    :returns: 저장된 :class:`pathlib.Path`.
    :raises ValueError: 클래스 이름 목록이 비어 있거나 중복인 경우.
    """
    names = [str(name) for name in class_names]
    if not names:
        raise ValueError("class_names 가 비어 있습니다.")
    if len(set(names)) != len(names):
        raise ValueError(f"class_names 에 중복이 있습니다: {names}")

    yaml_path = Path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# docagent 기입란 탐지 데이터셋 정의 (자동 생성)",
        f"path: {Path(root).as_posix()}",
        f"train: {train}",
        f"val: {val}",
    ]
    if test is not None:
        lines.append(f"test: {test}")
    lines.append(f"nc: {len(names)}")
    lines.append("names:")
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


# --------------------------------------------------------------------------
# 검증
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelReport:
    """라벨 검증 결과 1건.

    :param source: 검사 대상 식별자(파일 경로 또는 ``"<메모리>"``).
    :param label_count: 정상 파싱된 라벨 개수.
    :param issues: 발견된 문제 설명(한국어) 목록. 비어 있으면 정상.
    """

    source: str
    label_count: int = 0
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(str(item) for item in self.issues))
        if self.label_count < 0:
            raise ValueError(
                f"LabelReport.label_count 는 0 이상이어야 합니다: {self.label_count}"
            )

    @property
    def ok(self) -> bool:
        """문제가 하나도 없으면 True."""
        return not self.issues

    def format_report(self) -> str:
        """사람이 읽는 한 줄 요약(문제가 있으면 줄바꿈으로 상세 추가).

        :returns: 한국어 요약 문자열.
        """
        head = f"[{'정상' if self.ok else '문제'}] {self.source} — 라벨 {self.label_count}건"
        if self.ok:
            return head
        detail = "\n".join(f"  - {issue}" for issue in self.issues)
        return f"{head}, 문제 {len(self.issues)}건\n{detail}"

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "source": self.source,
            "label_count": self.label_count,
            "issues": list(self.issues),
            "ok": self.ok,
        }


def validate_labels(
    lines: Iterable[str], *, source: str = "<메모리>", allow_empty: bool = False
) -> LabelReport:
    """라벨 텍스트 줄들을 검증한다(예외를 던지지 않고 문제를 모아 보고한다).

    검사 항목: 토큰 수, 숫자 변환, 클래스 id 정의 여부, 좌표 0~1 범위,
    폭·높이 양수 여부, 상자가 이미지 밖으로 나가는지(중심 ± 절반).

    :param lines: 라벨 텍스트 줄 반복자.
    :param source: 보고서에 표기할 대상 이름.
    :param allow_empty: True 면 라벨 0건을 문제로 보지 않는다(배경 전용 샘플).
    :returns: :class:`LabelReport`.
    """
    issues: list[str] = []
    count = 0
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) != 5:
            issues.append(f"{number}번째 줄: 값이 5개가 아닙니다({len(tokens)}개).")
            continue
        try:
            class_id = int(tokens[0])
        except ValueError:
            issues.append(f"{number}번째 줄: 클래스 id 가 정수가 아닙니다({tokens[0]!r}).")
            continue
        try:
            cx, cy, w, h = (float(token) for token in tokens[1:])
        except ValueError:
            issues.append(f"{number}번째 줄: 좌표를 숫자로 변환할 수 없습니다.")
            continue
        if class_id not in CLASS_MAP:
            issues.append(
                f"{number}번째 줄: 정의되지 않은 클래스 id 입니다({class_id}). "
                f"허용 값: {sorted(CLASS_MAP)}"
            )
            continue
        line_issues: list[str] = []
        for name, value in (("cx", cx), ("cy", cy)):
            if not 0.0 <= value <= 1.0:
                line_issues.append(f"{name}={value} 가 0~1 범위를 벗어납니다")
        for name, value in (("w", w), ("h", h)):
            if not 0.0 < value <= 1.0:
                line_issues.append(f"{name}={value} 가 0 초과 1 이하가 아닙니다")
        if not line_issues:
            if cx - w / 2.0 < -1e-6 or cx + w / 2.0 > 1.0 + 1e-6:
                line_issues.append("상자가 이미지 좌우 경계를 벗어납니다")
            if cy - h / 2.0 < -1e-6 or cy + h / 2.0 > 1.0 + 1e-6:
                line_issues.append("상자가 이미지 상하 경계를 벗어납니다")
        if line_issues:
            issues.append(f"{number}번째 줄: " + ", ".join(line_issues) + ".")
            continue
        count += 1

    if count == 0 and not allow_empty:
        issues.append("유효한 라벨이 하나도 없습니다(빈 라벨 파일).")
    return LabelReport(source=source, label_count=count, issues=tuple(issues))


def validate_label_file(path: str | Path, *, allow_empty: bool = False) -> LabelReport:
    """라벨 파일 하나를 검증한다.

    :param path: 라벨 파일 경로.
    :param allow_empty: 라벨 0건을 허용할지 여부.
    :returns: :class:`LabelReport`. 파일이 없으면 그 사실을 문제로 담는다.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return LabelReport(
            source=str(file_path),
            label_count=0,
            issues=(f"라벨 파일을 찾을 수 없습니다: {file_path}",),
        )
    text = file_path.read_text(encoding="utf-8")
    report = validate_labels(
        text.splitlines(), source=str(file_path), allow_empty=allow_empty
    )
    return report


def validate_label_dir(
    directory: str | Path, *, pattern: str = "*.txt", allow_empty: bool = False
) -> list[LabelReport]:
    """디렉터리 안의 라벨 파일을 모두 검증한다.

    :param directory: 라벨 디렉터리.
    :param pattern: 파일 glob 패턴.
    :param allow_empty: 라벨 0건을 허용할지 여부.
    :returns: 파일명 정렬 순서의 :class:`LabelReport` 목록.
    :raises FileNotFoundError: 디렉터리가 없는 경우.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"라벨 디렉터리를 찾을 수 없습니다: {root}")
    return [
        validate_label_file(path, allow_empty=allow_empty)
        for path in sorted(root.glob(pattern))
    ]
