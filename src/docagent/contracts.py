"""도메인 계약(Contract) 정의 — 모든 모듈이 공유하는 단일 진실 원천.

이 모듈은 **순수 표준 라이브러리만** 사용한다. numpy·opencv·Pillow 를 포함한
어떤 외부 패키지도 import 하지 않는다. Vision / PII / Agent / IO 계층이 모두
이 모듈에 의존하므로, 여기에 외부 의존이 들어오면 계약 자체가 오염된다.

좌표계 규약
-----------
* 모든 **도메인 좌표는 A4 밀리미터(mm)** 기준이다.
* 원점은 페이지의 **좌상단**, x 는 오른쪽(+), y 는 **아래쪽(+)** 방향이다.
* 픽셀 좌표(:class:`BoxPx`)는 Vision 모듈 **내부에서만** 존재하며,
  모듈 경계를 넘어갈 때는 반드시 mm 로 변환된 :class:`BoxMm` 이어야 한다.

직렬화 규약
-----------
모든 dataclass 는 ``to_dict()`` / ``from_dict()`` 를 제공하며,
``from_dict(x.to_dict()) == x`` 가 항상 성립한다(JSON 왕복 무손실).
이는 "세션 복원율 100%" KPI 의 기술적 근거다.
Enum 은 문자열 ``value`` 로 직렬화되고, tuple 은 list 로 직렬화된 뒤
``from_dict`` 에서 tuple 로 복원된다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Mapping, Sequence

__all__ = [
    # 상수
    "A4_WIDTH_MM",
    "A4_HEIGHT_MM",
    "A4_PAGE_SIZE_MM",
    "EXPLAIN_THRESHOLD",
    "PARTIAL_THRESHOLD",
    "VISION_TRUST_THRESHOLD",
    "MOTION_TOLERANCE_MM",
    # 기하
    "Point",
    "BoxMm",
    "BoxPx",
    # 열거형
    "FieldType",
    "FieldRole",
    "Sensitivity",
    # Vision
    "Detection",
    "OcrWord",
    "Option",
    "Field",
    "DocumentStructure",
    "VerificationResult",
    # PII
    "PiiSpan",
    "SanitizedText",
    # 검색·Agent
    "RetrievedChunk",
    "ToolCall",
    "ToolResult",
    "AgentTurn",
]


# --------------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------------

#: A4 용지 가로 길이(mm).
A4_WIDTH_MM: float = 210.0
#: A4 용지 세로 길이(mm).
A4_HEIGHT_MM: float = 297.0
#: A4 페이지 크기 ``(가로_mm, 세로_mm)``.
A4_PAGE_SIZE_MM: tuple[float, float] = (A4_WIDTH_MM, A4_HEIGHT_MM)

#: 이 값 이상이면 에이전트가 항목을 **단정적으로 설명**한다.
EXPLAIN_THRESHOLD: float = 0.85
#: 이 값 이상 ~ :data:`EXPLAIN_THRESHOLD` 미만이면 **부분 확신**으로 재확인 질문을 붙인다.
#: 이 값 미만이면 사람 지원(:class:`docagent.errors.HandoffRequired`)으로 넘긴다.
PARTIAL_THRESHOLD: float = 0.70
#: Vision 탐지 결과를 추가 확인 없이 신뢰할 수 있는 최소 신뢰도.
VISION_TRUST_THRESHOLD: float = 0.85
#: 펜이 목표 지점에 도달했다고 볼 수 있는 허용 오차(mm).
#:
#: 임계값은 한 곳에서만 정의한다. :mod:`docagent.agent.tools` 와
#: :class:`docagent.config.DocAgentConfig` 는 이 값을 참조하기만 한다.
MOTION_TOLERANCE_MM: float = 1.0


# --------------------------------------------------------------------------
# 내부 직렬화 헬퍼
# --------------------------------------------------------------------------


def _as_enum(enum_cls: type[Enum], value: Any) -> Any:
    """문자열 또는 Enum 인스턴스를 ``enum_cls`` 로 변환한다.

    :param enum_cls: 대상 Enum 클래스.
    :param value: Enum 인스턴스이거나 해당 Enum 의 ``value`` 문자열.
    :returns: ``enum_cls`` 인스턴스.
    :raises ValueError: 정의되지 않은 값인 경우(조용한 실패 금지).
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:  # 도메인 메시지로 감싸 올린다.
        allowed = ", ".join(repr(member.value) for member in enum_cls)
        raise ValueError(
            f"{enum_cls.__name__} 에 정의되지 않은 값입니다: {value!r}. "
            f"허용 값: {allowed}"
        ) from exc


def _as_float_pair(value: Any) -> tuple[float, float]:
    """길이 2 의 시퀀스를 ``(float, float)`` 튜플로 변환한다.

    :param value: 길이 2 의 시퀀스(list 또는 tuple).
    :returns: ``(float, float)``.
    :raises ValueError: 길이가 2 가 아닌 경우.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"길이 2 의 시퀀스가 필요합니다: {value!r}")
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"길이 2 의 시퀀스가 필요합니다: {value!r}")
    return (float(items[0]), float(items[1]))


def _require_keys(data: Mapping[str, Any], keys: Sequence[str], owner: str) -> None:
    """``data`` 에 필수 키가 모두 존재하는지 검사한다.

    :param data: 검사 대상 매핑.
    :param keys: 필수 키 목록.
    :param owner: 오류 메시지에 표기할 dataclass 이름.
    :raises ValueError: 누락된 키가 있는 경우.
    """
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(
            f"{owner}.from_dict 에 필수 키가 없습니다: {', '.join(missing)}"
        )


# --------------------------------------------------------------------------
# 기하 타입
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """A4 밀리미터 좌표계 위의 한 점.

    :param x_mm: 좌측 여백 기준 가로 위치(mm, 오른쪽이 +).
    :param y_mm: 상단 여백 기준 세로 위치(mm, 아래쪽이 +).
    """

    x_mm: float
    y_mm: float

    def to_tuple(self) -> tuple[float, float]:
        """``(x_mm, y_mm)`` 튜플을 반환한다."""
        return (self.x_mm, self.y_mm)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {"x_mm": self.x_mm, "y_mm": self.y_mm}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Point":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("x_mm", "y_mm"), cls.__name__)
        return cls(x_mm=float(data["x_mm"]), y_mm=float(data["y_mm"]))


@dataclass(frozen=True)
class BoxMm:
    """A4 밀리미터 좌표계의 축 정렬 사각형.

    :param x_mm: 좌상단 x(mm).
    :param y_mm: 좌상단 y(mm).
    :param w_mm: 가로 길이(mm, 0 이상).
    :param h_mm: 세로 길이(mm, 0 이상).
    :raises ValueError: 폭 또는 높이가 음수인 경우.
    """

    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float

    def __post_init__(self) -> None:
        if self.w_mm < 0 or self.h_mm < 0:
            raise ValueError(
                f"BoxMm 의 폭·높이는 0 이상이어야 합니다: w_mm={self.w_mm}, h_mm={self.h_mm}"
            )

    @property
    def right_mm(self) -> float:
        """사각형 오른쪽 경계 x(mm)."""
        return self.x_mm + self.w_mm

    @property
    def bottom_mm(self) -> float:
        """사각형 아래쪽 경계 y(mm)."""
        return self.y_mm + self.h_mm

    @property
    def area_mm2(self) -> float:
        """사각형 넓이(mm^2)."""
        return self.w_mm * self.h_mm

    def center(self) -> Point:
        """사각형 중심점을 :class:`Point` 로 반환한다.

        하드웨어(펜 액추에이터)의 목표 좌표로 사용된다.
        """
        return Point(self.x_mm + self.w_mm / 2.0, self.y_mm + self.h_mm / 2.0)

    def contains(self, point: Point) -> bool:
        """``point`` 가 사각형 내부(경계 포함)에 있으면 True.

        :param point: 검사할 점(mm 좌표).
        :returns: 포함 여부.
        """
        return (
            self.x_mm <= point.x_mm <= self.right_mm
            and self.y_mm <= point.y_mm <= self.bottom_mm
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        """``(x_mm, y_mm, w_mm, h_mm)`` 튜플을 반환한다."""
        return (self.x_mm, self.y_mm, self.w_mm, self.h_mm)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "w_mm": self.w_mm,
            "h_mm": self.h_mm,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoxMm":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("x_mm", "y_mm", "w_mm", "h_mm"), cls.__name__)
        return cls(
            x_mm=float(data["x_mm"]),
            y_mm=float(data["y_mm"]),
            w_mm=float(data["w_mm"]),
            h_mm=float(data["h_mm"]),
        )


@dataclass(frozen=True)
class BoxPx:
    """이미지 픽셀 좌표계의 축 정렬 사각형(Vision 내부 전용).

    이 타입은 **모듈 경계를 넘지 않는다.** Vision 이 외부로 내보내는 좌표는
    항상 :class:`BoxMm` 이어야 한다.

    :param x: 좌상단 x(px).
    :param y: 좌상단 y(px).
    :param w: 가로 길이(px, 0 이상).
    :param h: 세로 길이(px, 0 이상).
    :raises ValueError: 폭 또는 높이가 음수인 경우.
    """

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w < 0 or self.h < 0:
            raise ValueError(
                f"BoxPx 의 폭·높이는 0 이상이어야 합니다: w={self.w}, h={self.h}"
            )

    def to_tuple(self) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` 튜플을 반환한다."""
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoxPx":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("x", "y", "w", "h"), cls.__name__)
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
        )


# --------------------------------------------------------------------------
# 열거형
# --------------------------------------------------------------------------


class FieldType(Enum):
    """문서 기입란의 유형."""

    SIGNATURE = "signature"
    CHECKBOX = "checkbox"
    CHOICE = "choice"
    TEXT_INPUT = "text_input"
    DATE = "date"
    UNKNOWN = "unknown"


class FieldRole(Enum):
    """기입란을 채워야 하는 주체."""

    APPLICANT = "applicant"
    REPRESENTATIVE = "representative"
    OFFICIAL = "official"
    UNKNOWN = "unknown"


class Sensitivity(Enum):
    """정보 민감도 구분.

    * :attr:`PUBLIC` — 공개 정보 영역. 약관·안내문·항목 라벨 등 문서 자체의 내용.
      외부 LLM 에 전송 가능하다.
    * :attr:`PRIVATE` — 개인정보 영역. 이름·주민등록번호·계좌번호 등 이용자가
      기입하는 값. **원문 그대로 외부로 나가서는 안 된다.**
    """

    PUBLIC = "public"
    PRIVATE = "private"


# --------------------------------------------------------------------------
# Vision 산출물
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """탐지기(:class:`docagent.interfaces.Detector`) 가 내놓는 원시 탐지 1건.

    :param type: 추정된 기입란 유형.
    :param box_mm: mm 좌표 경계 상자.
    :param confidence: 0.0~1.0 신뢰도.
    :param source: 탐지 출처 식별자(예: ``"opencv_contour"``, ``"yolo"``).
    :raises ValueError: ``confidence`` 가 0.0~1.0 범위를 벗어난 경우.
    """

    type: FieldType
    box_mm: BoxMm
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Detection.confidence 는 0.0~1.0 이어야 합니다: {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "type": self.type.value,
            "box_mm": self.box_mm.to_dict(),
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Detection":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("type", "box_mm", "confidence", "source"), cls.__name__)
        return cls(
            type=_as_enum(FieldType, data["type"]),
            box_mm=BoxMm.from_dict(data["box_mm"]),
            confidence=float(data["confidence"]),
            source=str(data["source"]),
        )


@dataclass(frozen=True)
class OcrWord:
    """OCR 엔진이 인식한 단어 1개.

    :param text: 인식된 문자열.
    :param box_mm: mm 좌표 경계 상자.
    :param confidence: 0.0~1.0 인식 신뢰도.
    """

    text: str
    box_mm: BoxMm
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "text": self.text,
            "box_mm": self.box_mm.to_dict(),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OcrWord":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("text", "box_mm", "confidence"), cls.__name__)
        return cls(
            text=str(data["text"]),
            box_mm=BoxMm.from_dict(data["box_mm"]),
            confidence=float(data["confidence"]),
        )


@dataclass(frozen=True)
class Option:
    """체크박스·선택형 항목의 선택지 하나.

    :param label: 선택지 라벨(예: ``"동의함"``).
    :param box_mm: 체크 표시를 넣을 네모 칸의 mm 좌표.
    :param checked: 체크 여부. ``None`` 은 **아직 확인되지 않음**을 뜻하며,
        ``False``(명시적 미체크)와 구분된다.

    .. note::
       현재 파이프라인에서 Vision 구조화 산출물
       (:func:`docagent.vision.structuring.build_structure`)의 ``checked`` 는
       **항상 ``None``** 이다. 기입 여부는 그때그때
       :func:`docagent.vision.verify.verify_options` 로 판정하며 그 결과를
       구조에 되먹이지 않는다. ``True``/``False`` 가 실제로 들어가는 곳은 정답
       데이터(:mod:`docagent.testing.synthetic`)와 그 정답을 직렬화·복원하는
       경로다. 구조를 보고 "지금 어디에 표시돼 있는지" 판단해서는 안 된다 —
       판정은 반드시 검증 함수를 호출해서 얻는다.
    """

    label: str
    box_mm: BoxMm
    checked: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "label": self.label,
            "box_mm": self.box_mm.to_dict(),
            "checked": self.checked,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Option":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("label", "box_mm"), cls.__name__)
        checked = data.get("checked")
        return cls(
            label=str(data["label"]),
            box_mm=BoxMm.from_dict(data["box_mm"]),
            checked=None if checked is None else bool(checked),
        )


@dataclass(frozen=True)
class Field:
    """문서의 기입란 1개 — Vision → Agent 인터페이스의 최소 단위.

    :param id: 문서 내 고유 식별자(예: ``"consent_01"``, ``"signature_01"``).
    :param type: 기입란 유형.
    :param title: 사람이 읽는 항목명(예: ``"개인정보 수집·이용 동의"``).
    :param role: 이 항목을 채워야 하는 주체.
    :param options: 선택지 목록. 선택형이 아니면 빈 튜플.
    :param required: 필수 기입 여부.
    :param sensitivity: 공개/개인정보 영역 구분. 기본값은 안전한 :attr:`Sensitivity.PRIVATE`.
    :param box_mm: 기입 영역의 mm 좌표. 좌표 미확정이면 ``None``.
    :param clause_text: 항목에 딸린 약관·안내 문구 원문(공개 정보).
    :param order: 낭독·기입 순서(0 부터). 정렬 키.
    :param confidence: 이 항목 해석에 대한 종합 신뢰도(0.0~1.0).
    :raises ValueError: ``confidence`` 범위 위반 또는 ``id`` 가 빈 문자열인 경우.
    """

    id: str
    type: FieldType = FieldType.UNKNOWN
    title: str = ""
    role: FieldRole = FieldRole.UNKNOWN
    options: tuple[Option, ...] = ()
    required: bool = False
    sensitivity: Sensitivity = Sensitivity.PRIVATE
    box_mm: BoxMm | None = None
    clause_text: str = ""
    order: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Field.id 는 빈 문자열일 수 없습니다.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Field.confidence 는 0.0~1.0 이어야 합니다: {self.confidence}"
            )
        # list 로 들어와도 tuple 로 정규화하여 해시 가능·불변 상태를 유지한다.
        object.__setattr__(self, "options", tuple(self.options))

    def is_public(self) -> bool:
        """공개 정보 영역이면 True(외부 LLM 전송 허용 대상)."""
        return self.sensitivity is Sensitivity.PUBLIC

    def target_point(self) -> Point | None:
        """액추에이터가 이동할 목표 좌표. ``box_mm`` 이 없으면 ``None``."""
        return None if self.box_mm is None else self.box_mm.center()

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "role": self.role.value,
            "options": [option.to_dict() for option in self.options],
            "required": self.required,
            "sensitivity": self.sensitivity.value,
            "box_mm": None if self.box_mm is None else self.box_mm.to_dict(),
            "clause_text": self.clause_text,
            "order": self.order,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Field":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("id",), cls.__name__)
        box = data.get("box_mm")
        return cls(
            id=str(data["id"]),
            type=_as_enum(FieldType, data.get("type", FieldType.UNKNOWN.value)),
            title=str(data.get("title", "")),
            role=_as_enum(FieldRole, data.get("role", FieldRole.UNKNOWN.value)),
            options=tuple(Option.from_dict(item) for item in data.get("options", ())),
            required=bool(data.get("required", False)),
            sensitivity=_as_enum(
                Sensitivity, data.get("sensitivity", Sensitivity.PRIVATE.value)
            ),
            box_mm=None if box is None else BoxMm.from_dict(box),
            clause_text=str(data.get("clause_text", "")),
            order=int(data.get("order", 0)),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class DocumentStructure:
    """한 장의 문서를 해석한 결과 전체 — Vision 모듈의 최종 산출물.

    이 객체의 ``to_dict()`` 결과가 로드맵에서 말하는 **fields JSON** 이며,
    Vision 과 Agent 사이의 유일한 공식 인터페이스다.

    :param document_id: 문서 인스턴스 식별자(세션 복원 키).
    :param doc_title: 문서 제목(예: ``"예금계좌 개설 신청서"``).
    :param fields: 기입란 목록. ``order`` 순 정렬을 권장한다.
    :param page_size_mm: 페이지 크기 ``(가로_mm, 세로_mm)``. 기본 A4.
    :param source_image: 원본 이미지 경로 또는 식별자. 없으면 ``None``.
    :param warnings: 해석 중 발생한 경고 메시지(한국어) 목록.
    """

    document_id: str
    doc_title: str = ""
    fields: tuple[Field, ...] = ()
    page_size_mm: tuple[float, float] = A4_PAGE_SIZE_MM
    source_image: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("DocumentStructure.document_id 는 빈 문자열일 수 없습니다.")
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "page_size_mm", _as_float_pair(self.page_size_mm))
        ids = [f.id for f in self.fields]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(
                f"DocumentStructure.fields 의 id 가 중복되었습니다: {', '.join(duplicates)}"
            )

    def required_fields(self) -> tuple[Field, ...]:
        """필수 기입 항목만 ``order`` 순으로 반환한다."""
        return tuple(
            sorted((f for f in self.fields if f.required), key=lambda f: f.order)
        )

    def field_by_id(self, field_id: str) -> Field | None:
        """``field_id`` 에 해당하는 항목을 반환한다. 없으면 ``None``.

        :param field_id: 찾을 항목 id.
        :returns: :class:`Field` 또는 ``None``.
        """
        for item in self.fields:
            if item.id == field_id:
                return item
        return None

    def public_payload(self) -> dict[str, Any]:
        """외부 LLM 에 전송 가능한 **공개 정보 영역만** 담은 dict 를 반환한다.

        분리 원칙:

        * :attr:`Sensitivity.PUBLIC` 항목은 ``title`` · ``clause_text`` · ``options``
          라벨까지 포함하되 **좌표(``box_mm``)와 인식 신뢰도(``confidence``)는
          제외한다.** 좌표가 필요한 시점(Act 단계)에는 LLM 이 아니라 로컬 코드가
          원본 :class:`DocumentStructure` 에서 직접 읽는다(§6). 좌표를 프롬프트에
          실으면 모델이 만들어 낸 숫자가 펜 좌표로 되돌아오는 경로가 열린다.
        * :attr:`Sensitivity.PRIVATE` 항목은 **구조 정보만** 남긴 축약형
          (``id`` / ``type`` / ``role`` / ``required`` / ``order`` / ``redacted=True``)
          으로 내려간다. ``title`` · ``clause_text`` · ``options`` · ``box_mm`` 은 제외된다.
          에이전트는 "몇 번째에 어떤 유형의 개인정보 항목이 있다"는 사실은 알 수 있지만
          그 내용은 알 수 없다.

        .. note::
           내용 기반 재판정(탐지기로 값 유입을 다시 확인)까지 필요한 운영 경로는
           :func:`docagent.pii.policy.build_public_payload` 를 쓴다. 이 메서드는
           같은 키 집합을 만들되 탐지기에 의존하지 않는 계약 기본 구현이다.

        :returns: JSON 직렬화 가능한 dict.
            키는 ``document_id`` / ``doc_title`` / ``page_size_mm`` /
            ``fields`` / ``redacted_field_ids`` / ``warnings``.
        """
        public_fields: list[dict[str, Any]] = []
        redacted_ids: list[str] = []
        for item in sorted(self.fields, key=lambda f: f.order):
            if item.is_public():
                public_fields.append(
                    {
                        "id": item.id,
                        "type": item.type.value,
                        "role": item.role.value,
                        "required": item.required,
                        "order": item.order,
                        "sensitivity": item.sensitivity.value,
                        "title": item.title,
                        "clause_text": item.clause_text,
                        "options": [
                            {"label": option.label} for option in item.options
                        ],
                    }
                )
            else:
                redacted_ids.append(item.id)
                public_fields.append(
                    {
                        "id": item.id,
                        "type": item.type.value,
                        "role": item.role.value,
                        "required": item.required,
                        "order": item.order,
                        "sensitivity": item.sensitivity.value,
                        "redacted": True,
                    }
                )
        return {
            "document_id": self.document_id,
            "doc_title": self.doc_title,
            "page_size_mm": list(self.page_size_mm),
            "fields": public_fields,
            "redacted_field_ids": redacted_ids,
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict(= fields JSON)를 반환한다."""
        return {
            "document_id": self.document_id,
            "doc_title": self.doc_title,
            "fields": [item.to_dict() for item in self.fields],
            "page_size_mm": list(self.page_size_mm),
            "source_image": self.source_image,
            "warnings": list(self.warnings),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """fields JSON 문자열을 반환한다(한글은 이스케이프하지 않는다).

        :param indent: JSON 들여쓰기. ``None`` 이면 한 줄로 직렬화한다.
        :returns: UTF-8 그대로 읽히는 JSON 문자열.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DocumentStructure":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("document_id",), cls.__name__)
        source_image = data.get("source_image")
        return cls(
            document_id=str(data["document_id"]),
            doc_title=str(data.get("doc_title", "")),
            fields=tuple(Field.from_dict(item) for item in data.get("fields", ())),
            page_size_mm=_as_float_pair(data.get("page_size_mm", A4_PAGE_SIZE_MM)),
            source_image=None if source_image is None else str(source_image),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
        )

    @classmethod
    def from_json(cls, text: str) -> "DocumentStructure":
        """fields JSON 문자열로부터 인스턴스를 복원한다.

        :param text: :meth:`to_json` 이 만든 JSON 문자열.
        :returns: :class:`DocumentStructure`.
        :raises ValueError: JSON 파싱 실패 또는 스키마 불일치.
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"fields JSON 파싱에 실패했습니다: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("fields JSON 최상위는 객체여야 합니다.")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class VerificationResult:
    """Verify 단계 결과 — 기입 전/후 잉크 비율 비교로 실제 기입을 확인한다.

    :param field_id: 검증 대상 항목 id.
    :param written: 기입이 실제로 이루어졌다고 판정했는지 여부.
    :param ink_ratio_before: 기입 전 해당 영역의 잉크 픽셀 비율(0.0~1.0).
    :param ink_ratio_after: 기입 후 잉크 픽셀 비율(0.0~1.0).
    :param confidence: 판정 신뢰도(0.0~1.0).
    :param reason: 판정 근거(한국어 한 문장).
    """

    field_id: str
    written: bool
    ink_ratio_before: float
    ink_ratio_after: float
    confidence: float
    reason: str = ""

    @property
    def ink_delta(self) -> float:
        """기입 전후 잉크 비율 증가량(음수 가능)."""
        return self.ink_ratio_after - self.ink_ratio_before

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "field_id": self.field_id,
            "written": self.written,
            "ink_ratio_before": self.ink_ratio_before,
            "ink_ratio_after": self.ink_ratio_after,
            "confidence": self.confidence,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationResult":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(
            data,
            ("field_id", "written", "ink_ratio_before", "ink_ratio_after", "confidence"),
            cls.__name__,
        )
        return cls(
            field_id=str(data["field_id"]),
            written=bool(data["written"]),
            ink_ratio_before=float(data["ink_ratio_before"]),
            ink_ratio_after=float(data["ink_ratio_after"]),
            confidence=float(data["confidence"]),
            reason=str(data.get("reason", "")),
        )


# --------------------------------------------------------------------------
# PII
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PiiSpan:
    """텍스트에서 탐지된 개인정보 구간의 **메타데이터**.

    .. warning::
       이 객체는 **개인정보 원문 값을 절대 담지 않는다.** 위치(``start``/``end``),
       유형(``pii_type``), 원문 길이(``raw_len``)만 보관한다. 로그·telemetry·
       세션 저장소에 그대로 남겨도 원문을 복원할 수 없어야 하며, 구현체가
       원문 값을 담는 필드를 추가하는 것은 계약 위반이다.

    :param start: 원문 텍스트에서의 시작 인덱스(포함).
    :param end: 끝 인덱스(제외). ``start`` 보다 커야 한다.
    :param pii_type: 개인정보 유형 식별자(예: ``"rrn"``, ``"phone"``, ``"account"``, ``"name"``).
    :param raw_len: 마스킹 전 원문 길이(문자 수).
    :param confidence: 탐지 신뢰도(0.0~1.0).
    :raises ValueError: ``start`` 가 음수이거나 ``end <= start`` 인 경우.
    """

    start: int
    end: int
    pii_type: str
    raw_len: int
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"PiiSpan.start 는 0 이상이어야 합니다: {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"PiiSpan.end 는 start 보다 커야 합니다: start={self.start}, end={self.end}"
            )
        if self.raw_len < 0:
            raise ValueError(f"PiiSpan.raw_len 은 0 이상이어야 합니다: {self.raw_len}")

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다(원문 값 미포함)."""
        return {
            "start": self.start,
            "end": self.end,
            "pii_type": self.pii_type,
            "raw_len": self.raw_len,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PiiSpan":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("start", "end", "pii_type", "raw_len"), cls.__name__)
        return cls(
            start=int(data["start"]),
            end=int(data["end"]),
            pii_type=str(data["pii_type"]),
            raw_len=int(data["raw_len"]),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class SanitizedText:
    """마스킹이 끝난 텍스트와 그 근거.

    :param text: 마스킹 완료 텍스트. 이 값만 외부(LLM·로그)로 나갈 수 있다.
    :param spans: 탐지된 개인정보 구간 메타데이터. 인덱스는 **마스킹 전 원문 기준**이다.
    :param blocked: 마스킹으로도 안전을 보장할 수 없어 **전송 자체를 차단**해야 하면 True.
        이 경우 호출자는 :class:`docagent.errors.PiiEgressBlocked` 를 던져야 한다.
    """

    text: str
    spans: tuple[PiiSpan, ...] = ()
    blocked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "spans", tuple(self.spans))

    @property
    def has_pii(self) -> bool:
        """개인정보가 하나라도 탐지되었으면 True."""
        return len(self.spans) > 0

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "text": self.text,
            "spans": [span.to_dict() for span in self.spans],
            "blocked": self.blocked,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SanitizedText":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("text",), cls.__name__)
        return cls(
            text=str(data["text"]),
            spans=tuple(PiiSpan.from_dict(item) for item in data.get("spans", ())),
            blocked=bool(data.get("blocked", False)),
        )


# --------------------------------------------------------------------------
# 검색 · Agent
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """검색기(:class:`docagent.interfaces.Retriever`) 가 반환하는 근거 조각.

    :param chunk_id: 조각 식별자.
    :param text: 조각 본문(공개 정보 영역이어야 한다).
    :param source: 출처 표기(문서명·조문 등). 사용자에게 낭독할 수 있어야 한다.
    :param score: 질의 적합도 점수(클수록 적합).
    """

    chunk_id: str
    text: str
    source: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievedChunk":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("chunk_id", "text"), cls.__name__)
        return cls(
            chunk_id=str(data["chunk_id"]),
            text=str(data["text"]),
            source=str(data.get("source", "")),
            score=float(data.get("score", 0.0)),
        )


@dataclass(frozen=True)
class ToolCall:
    """에이전트가 요청한 도구 호출 1건.

    :param name: 도구 이름(예: ``"read_field"``, ``"write_signature"``).
    :param arguments: 도구 인자. JSON 직렬화 가능한 값만 담는다.
    """

    name: str
    arguments: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {"name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolCall":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("name",), cls.__name__)
        return cls(name=str(data["name"]), arguments=dict(data.get("arguments", {})))


@dataclass(frozen=True)
class ToolResult:
    """도구 실행 결과.

    :param ok: 성공 여부.
    :param speech: 사용자에게 음성으로 읽어 줄 한국어 문장. 없으면 빈 문자열.
    :param state_patch: 대화 상태에 병합할 부분 갱신 dict.
    :param error: 실패 사유(한국어). 성공 시 ``None``.
    :param handoff: 사람 지원으로 넘겨야 하면 True.
    """

    ok: bool
    speech: str = ""
    state_patch: dict[str, Any] = dc_field(default_factory=dict)
    error: str | None = None
    handoff: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "ok": self.ok,
            "speech": self.speech,
            "state_patch": dict(self.state_patch),
            "error": self.error,
            "handoff": self.handoff,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolResult":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        _require_keys(data, ("ok",), cls.__name__)
        error = data.get("error")
        return cls(
            ok=bool(data["ok"]),
            speech=str(data.get("speech", "")),
            state_patch=dict(data.get("state_patch", {})),
            error=None if error is None else str(error),
            handoff=bool(data.get("handoff", False)),
        )


@dataclass(frozen=True)
class AgentTurn:
    """에이전트 대화 1턴의 입출력 기록(호출자·KPI 용 반환값).

    .. note::
       **세션 복원 단위가 아니다.** 세션을 되살리는 정본은
       :class:`docagent.agent.state.SessionState` 이며, 감사 추적은 그 안의
       ``history``(턴 번호·단계·항목·행위·설명)가 담당한다. 이 객체는
       ``user_text`` 에 이용자의 **원문 발화**를 그대로 담으므로 세션 JSON 으로
       저장하지 않는다 — 저장하면 개인정보가 그대로 디스크에 남는다.
       (저장이 필요하면 :class:`docagent.interfaces.PiiGate` 를 먼저 통과시켜야
       한다.)

    :param user_text: 사용자 발화(STT 결과). 시스템 발화 시작 턴이면 빈 문자열.
    :param intent: 분류된 의도 식별자(예: ``"explain_field"``, ``"confirm_write"``).
    :param tool_calls: 이 턴에서 요청한 도구 호출 목록.
    :param speech: 이 턴에서 사용자에게 낭독한 한국어 문장.
    :param confidence: 의도 분류·응답 신뢰도(0.0~1.0).
        :data:`EXPLAIN_THRESHOLD` / :data:`PARTIAL_THRESHOLD` 와 비교해 분기한다.
    :param handoff_reason: 사람 지원으로 넘긴 사유(한국어). 넘기지 않았으면 ``None``.
    """

    user_text: str = ""
    intent: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    speech: str = ""
    confidence: float = 0.0
    handoff_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"AgentTurn.confidence 는 0.0~1.0 이어야 합니다: {self.confidence}"
            )

    def needs_handoff(self) -> bool:
        """사람 지원으로 넘어간 턴이면 True.

        판정 근거는 ``handoff_reason`` **하나뿐**이다. 이 값을 채우는 주체는
        오케스트레이터이므로, 이 술어는 언제나 실제 트리거 집합과 일치한다.

        신뢰도는 여기서 보지 않는다. 낮은 신뢰도가 곧 직원 연결은 아니기 때문이다.
        예를 들어 발화를 알아듣지 못한 **첫** 턴은 ``confidence=0.0`` 이지만
        핸드오프가 아니라 재질문이며, 연속 오인식이 누적되어야 비로소 넘어간다.
        신뢰도 밴드 판정이 필요하면 :func:`docagent.agent.confidence.band_for` 를 쓴다.
        """
        return self.handoff_reason is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "user_text": self.user_text,
            "intent": self.intent,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "speech": self.speech,
            "confidence": self.confidence,
            "handoff_reason": self.handoff_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentTurn":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다."""
        reason = data.get("handoff_reason")
        return cls(
            user_text=str(data.get("user_text", "")),
            intent=str(data.get("intent", "")),
            tool_calls=tuple(
                ToolCall.from_dict(item) for item in data.get("tool_calls", ())
            ),
            speech=str(data.get("speech", "")),
            confidence=float(data.get("confidence", 0.0)),
            handoff_reason=None if reason is None else str(reason),
        )
