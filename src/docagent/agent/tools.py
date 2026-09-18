"""Tool Calling 파이프라인 — 에이전트가 세상에 영향을 주는 유일한 통로.

핵심 안전장치
-------------
**LLM 은 좌표를 만들지 못한다.** 모든 도구의 파라미터 스키마에는 좌표 인자가
아예 존재하지 않는다. :func:`ToolRegistry.move_to_field` 는 ``field_id`` 만 받고,
실제 mm 좌표는 로컬 코드가 :class:`~docagent.contracts.DocumentStructure` 에서
직접 읽는다. 모델이 환각으로 만들어 낸 숫자가 펜 좌표가 되는 경로 자체를 없앤 것이다.

두 번째 안전장치는 :meth:`ToolRegistry.describe` 다. LLM function-calling 용
스키마를 내보내기 전에 좌표·개인정보 토큰이 섞였는지 자체 검사하고,
걸리면 :class:`~docagent.errors.ToolExecutionError` 로 실패한다.

예외 규약
---------
모든 handler 는 :class:`~docagent.contracts.ToolResult` 를 반환하고 **예외를 밖으로
던지지 않는다.** :class:`~docagent.errors.HandoffRequired` 는 ``handoff=True`` 로,
그 밖의 오류는 ``ok=False`` + ``error`` 한국어 메시지로 변환된다. 다만 조용히
삼키지는 않는다 — 실패 사유는 항상 ``error`` 에 남고 세션 이력에도 기록된다.

의존성 주입
-----------
``explainer`` / ``verifier`` / ``motion`` / ``speech`` 는 전부 선택 인자다.
``None`` 이면 아래의 안전한 내장 기본값이 쓰여 단독 테스트가 가능하다.

============ ================================= ==========================================
인자          기대 인터페이스                     ``None`` 일 때의 기본값
============ ================================= ==========================================
``explainer`` ``explain(field, question)``      :class:`VerbatimExplainer` (원문 낭독만)
``verifier``  ``verify(field_id)``              :class:`AssumeWrittenVerifier` (데모용)
``motion``    :class:`~docagent.interfaces.MotionController` :class:`MemoryMotionController`
``speech``    :class:`~docagent.interfaces.SpeechIO`         :class:`RecordingSpeech`
============ ================================= ==========================================

``explainer.explain`` 의 반환값(``text`` / ``confidence`` / ``sources`` /
``needs_handoff`` / ``disclaimer``)은 다른 에이전트가 소유한 타입이므로
:func:`getattr` 로 덕 타이핑 접근한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Mapping, Sequence

from docagent.contracts import (
    A4_PAGE_SIZE_MM,
    MOTION_TOLERANCE_MM,
    PARTIAL_THRESHOLD,
    VISION_TRUST_THRESHOLD,
    BoxMm,
    DocumentStructure,
    Field,
    FieldType,
    Point,
    ToolResult,
    VerificationResult,
)
from docagent.errors import DocAgentError, HandoffRequired, ToolExecutionError
from docagent.agent.guardrails import DISCLAIMER
from docagent.agent.state import SessionPhase, SessionState

__all__ = [
    "Tool",
    "ToolRegistry",
    "VerbatimExplainer",
    "AssumeWrittenVerifier",
    "MemoryMotionController",
    "RecordingSpeech",
    "ORIGINAL_NOTICE",
    "MOTION_TOLERANCE_MM",
    "MAX_FIELD_FAILURES",
]


#: 쉬운 설명 뒤에 항상 붙이는 고지. 설명은 원문의 대체물이 아니다.
#:
#: 정의는 :data:`docagent.agent.guardrails.DISCLAIMER` 한 곳뿐이며 여기서는
#: 이름만 다시 공개한다. 문구를 복사해 두면 설명기가 붙인 고지와 도구가 붙인
#: 고지가 한 발화에 함께 실려 같은 안내가 두 번 낭독된다.
ORIGINAL_NOTICE: str = DISCLAIMER

# 펜 도달 판정 허용 오차(mm)인 MOTION_TOLERANCE_MM 은 계약
# (:data:`docagent.contracts.MOTION_TOLERANCE_MM`)에서 가져와 이름만 다시 공개한다.
# 임계값은 한 곳에서만 정의한다 — 여기서 리터럴을 복사하면
# docagent.config 의 기본값과 어긋난 채 각자 굳는다.

#: 같은 항목에서 이 횟수만큼 연속 실패하면 사람 지원으로 넘긴다.
MAX_FIELD_FAILURES: int = 3

#: 도구 스키마에 절대 나타나서는 안 되는 토큰(좌표 유출 방지).
_BANNED_SCHEMA_TOKENS: tuple[str, ...] = (
    "x_mm",
    "y_mm",
    "w_mm",
    "h_mm",
    "box",
    "coord",
    "point",
    "pixel",
    "_px",
    "좌표",
)

#: 도구 설명·스키마에서 개인정보 유출을 잡아내는 패턴(숫자 6자리 이상, 주민번호 형태).
_PII_LIKE: re.Pattern[str] = re.compile(r"\d{6,}|\d{6}\s*-\s*\d{7}|\d{2,4}-\d{3,4}-\d{4}")


# --------------------------------------------------------------------------
# 내장 기본 구현
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Explanation:
    """내장 기본 설명 결과. 외부 ``ExplanationResult`` 와 같은 속성 이름을 쓴다.

    :param text: 사용자에게 읽어 줄 본문.
    :param confidence: 0.0~1.0 신뢰도.
    :param sources: 근거 출처 표기 목록.
    :param needs_handoff: 사람 지원이 필요하면 True.
    :param disclaimer: 본문 뒤에 붙일 고지 문구.
    """

    text: str
    confidence: float = 1.0
    sources: tuple[str, ...] = ()
    needs_handoff: bool = False
    disclaimer: str = ""


class VerbatimExplainer:
    """설명 모듈이 연결되지 않았을 때 쓰는 기본 설명기.

    **해석하지 않는다.** 항목의 원문(약관 문구)이나 제목을 그대로 읽어 주고,
    쉬운 설명 기능이 아직 연결되지 않았음을 고지한다. 근거 없는 요약으로
    사용자를 오도하지 않기 위한 보수적 기본값이다.
    """

    def explain(self, field: Field, question: str = "") -> _Explanation:
        """항목 원문을 그대로 담은 설명 결과를 만든다.

        :param field: 대상 항목.
        :param question: 사용자의 질문(이 기본 구현은 사용하지 않는다).
        :returns: :class:`_Explanation`.
        """
        body = field.clause_text.strip() or field.title.strip()
        if not body:
            body = "이 항목에는 읽어 드릴 안내 문구가 없습니다."
        return _Explanation(
            text=body,
            confidence=1.0,
            sources=("문서 원문",),
            needs_handoff=False,
            disclaimer="쉬운 설명 기능이 연결되지 않아 원문을 그대로 읽어 드렸습니다.",
        )


class AssumeWrittenVerifier:
    """검증 모듈이 연결되지 않았을 때 쓰는 기본 검증기(데모·테스트 전용).

    실제 이미지 비교를 하지 않고 기입이 이루어졌다고 간주한다. 그 사실을
    :attr:`~docagent.contracts.VerificationResult.reason` 에 명시하므로
    결과를 보는 쪽이 "검증되지 않았음"을 알 수 있다. 운영 환경에서는 반드시
    실제 Verify 구현을 주입해야 한다.
    """

    def verify(self, field_id: str) -> VerificationResult:
        """항목이 기입되었다고 간주하는 결과를 만든다.

        :param field_id: 대상 항목 id.
        :returns: :class:`~docagent.contracts.VerificationResult`.
        """
        return VerificationResult(
            field_id=field_id,
            written=True,
            ink_ratio_before=0.0,
            ink_ratio_after=0.0,
            confidence=1.0,
            reason="검증 모듈이 연결되지 않아 기입을 확인된 것으로 간주했습니다.",
        )


class MemoryMotionController:
    """메모리 상에서만 동작하는 기본 펜 제어기.

    실제 하드웨어 없이 이동 이력을 기록한다. 하드웨어는 에이전트의 액추에이터일
    뿐이므로, 이 대체 구현으로도 에이전트 로직 전체를 검증할 수 있다.

    :param page_size_mm: 가동 범위 ``(가로_mm, 세로_mm)``. 기본 A4.
    """

    def __init__(self, page_size_mm: tuple[float, float] = A4_PAGE_SIZE_MM) -> None:
        self.page_size_mm = (float(page_size_mm[0]), float(page_size_mm[1]))
        self._position = Point(0.0, 0.0)
        #: 이동 이력. 테스트에서 호출 여부를 확인하는 데 쓴다.
        self.moves: list[Point] = []

    def move_to(self, x_mm: float, y_mm: float) -> bool:
        """지정한 mm 좌표로 이동한다.

        :param x_mm: 목표 x(mm).
        :param y_mm: 목표 y(mm).
        :returns: 가동 범위 안이면 True, 밖이면 False(이동하지 않는다).
        """
        if not (0.0 <= x_mm <= self.page_size_mm[0]):
            return False
        if not (0.0 <= y_mm <= self.page_size_mm[1]):
            return False
        self._position = Point(float(x_mm), float(y_mm))
        self.moves.append(self._position)
        return True

    def home(self) -> bool:
        """원점으로 복귀한다.

        :returns: 항상 True.
        """
        self._position = Point(0.0, 0.0)
        self.moves.append(self._position)
        return True

    def position(self) -> Point:
        """현재 위치를 반환한다.

        :returns: :class:`~docagent.contracts.Point`.
        """
        return self._position


class RecordingSpeech:
    """발화를 리스트에 쌓아 두는 기본 음성 입출력.

    실제 TTS/STT 는 이번 범위 밖이다. 이 구현은 낭독 내용을 검증 가능한 형태로
    남기는 역할만 한다.
    """

    def __init__(self) -> None:
        #: 낭독한 문장 목록.
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        """문장을 기록한다.

        :param text: 낭독할 한국어 문장.
        :returns: ``None``.
        """
        self.spoken.append(text)

    def listen(self, timeout_s: float = 10.0) -> str:
        """입력을 받지 않는다(항상 빈 문자열).

        :param timeout_s: 최대 대기 시간(사용하지 않는다).
        :returns: 빈 문자열.
        """
        return ""


# --------------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """등록 가능한 도구 1개.

    :param name: 도구 이름(LLM function-calling 의 함수명).
    :param description: 한국어 설명. LLM 이 언제 이 도구를 쓸지 판단하는 근거.
    :param json_schema: 파라미터 JSON Schema. **좌표 인자를 담을 수 없다.**
    :param handler: 인자 매핑을 받아 :class:`~docagent.contracts.ToolResult` 를
        돌려주는 실행 함수.
    :raises ValueError: 이름이 비었거나 스키마가 객체 타입이 아닌 경우.
    """

    name: str
    description: str
    json_schema: dict[str, Any]
    handler: Callable[[Mapping[str, Any]], ToolResult]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool.name 은 빈 문자열일 수 없습니다.")
        if self.json_schema.get("type") != "object":
            raise ValueError(
                f"Tool.json_schema 는 type='object' 여야 합니다: {self.name}"
            )

    def schema_dict(self) -> dict[str, Any]:
        """LLM function-calling 규격의 도구 서술 dict 를 반환한다.

        :returns: ``{"name", "description", "parameters"}``.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": json.loads(json.dumps(self.json_schema, ensure_ascii=False)),
        }


def _schema(
    properties: dict[str, Any] | None = None, required: Sequence[str] = ()
) -> dict[str, Any]:
    """파라미터 JSON Schema 를 만든다.

    :param properties: 속성 정의. ``None`` 이면 인자 없는 도구.
    :param required: 필수 속성 이름 목록.
    :returns: JSON Schema dict.
    """
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required),
        "additionalProperties": False,
    }


_FIELD_ID_PROPERTY: dict[str, Any] = {
    "type": "string",
    "description": "대상 항목의 식별자. 생략하면 지금 안내 중인 항목을 사용합니다.",
}


# --------------------------------------------------------------------------
# ToolRegistry
# --------------------------------------------------------------------------


class ToolRegistry:
    """도구 등록·서술·실행을 담당하는 레지스트리.

    생성 시 12개 기본 도구를 등록한다: ``explain`` / ``read_original`` /
    ``next_field`` / ``previous_field`` / ``go_to_field`` / ``select_option`` /
    ``move_to_field`` / ``verify_field`` / ``repeat`` / ``revise_field`` /
    ``request_human`` / ``summarize_progress``.

    :param structure: 문서 구조(좌표의 유일한 출처).
    :param state: 세션 상태. handler 들이 직접 갱신한다.
    :param motion: 펜 제어기. ``None`` 이면 :class:`MemoryMotionController`.
    :param explainer: 설명기. ``None`` 이면 :class:`VerbatimExplainer`.
    :param verifier: 검증기. ``None`` 이면 :class:`AssumeWrittenVerifier`.
    :param speech: 음성 입출력. ``None`` 이면 :class:`RecordingSpeech`.
    :param motion_tolerance_mm: 펜 도달 판정 허용 오차(mm).

    .. note::
       handler 는 ``state`` 를 **직접 변경**한다.
       :attr:`~docagent.contracts.ToolResult.state_patch` 는 그 결과를 호출자에게
       알리는 요약이며, 다시 적용할 필요가 없다(중복 적용 금지).
    """

    def __init__(
        self,
        structure: DocumentStructure,
        state: SessionState,
        *,
        motion: Any | None = None,
        explainer: Any | None = None,
        verifier: Any | None = None,
        speech: Any | None = None,
        motion_tolerance_mm: float = MOTION_TOLERANCE_MM,
    ) -> None:
        self.structure = structure
        self.state = state
        self.motion = motion if motion is not None else MemoryMotionController(
            (float(structure.page_size_mm[0]), float(structure.page_size_mm[1]))
        )
        self.explainer = explainer if explainer is not None else VerbatimExplainer()
        self.verifier = verifier if verifier is not None else AssumeWrittenVerifier()
        self.speech = speech if speech is not None else RecordingSpeech()
        self.motion_tolerance_mm = float(motion_tolerance_mm)
        self._tools: dict[str, Tool] = {}
        for tool in self._build_default_tools():
            self.register(tool)

    # ------------------------------------------------------------------
    # 등록 · 조회
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """도구를 등록한다.

        :param tool: 등록할 :class:`Tool`.
        :returns: ``None``.
        :raises ToolExecutionError: 같은 이름이 이미 등록된 경우.
        """
        if tool.name in self._tools:
            raise ToolExecutionError(
                "이미 등록된 도구 이름입니다.", tool_name=tool.name
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """이름으로 도구를 찾는다.

        :param name: 도구 이름.
        :returns: :class:`Tool`.
        :raises ToolExecutionError: 등록되지 않은 이름인 경우.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolExecutionError("등록되지 않은 도구입니다.", tool_name=name)
        return tool

    def names(self) -> tuple[str, ...]:
        """등록된 도구 이름을 등록 순서대로 반환한다.

        :returns: 이름 튜플.
        """
        return tuple(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        """LLM function-calling 용 도구 스키마 목록을 만든다.

        내보내기 전에 스키마에 좌표·개인정보로 보이는 내용이 섞였는지 검사한다.
        이 검사는 "LLM 이 좌표를 직접 만들지 못하게 한다"는 설계 불변식을
        런타임에서 강제하는 마지막 방어선이다.

        :returns: ``{"name", "description", "parameters"}`` dict 목록.
        :raises ToolExecutionError: 스키마에 좌표 토큰이나 개인정보로 보이는
            문자열이 포함된 경우.
        """
        described: list[dict[str, Any]] = []
        for tool in self._tools.values():
            item = tool.schema_dict()
            blob = json.dumps(item["parameters"], ensure_ascii=False).lower()
            for token in _BANNED_SCHEMA_TOKENS:
                if token.lower() in blob:
                    raise ToolExecutionError(
                        f"도구 스키마에 좌표 관련 토큰 '{token}' 이 포함되어 있습니다. "
                        "좌표는 LLM 이 아니라 로컬 코드가 문서 구조에서 직접 읽어야 합니다.",
                        tool_name=tool.name,
                    )
            whole = json.dumps(item, ensure_ascii=False)
            if _PII_LIKE.search(whole) is not None:
                raise ToolExecutionError(
                    "도구 스키마에 개인정보로 보이는 문자열이 포함되어 있습니다.",
                    tool_name=tool.name,
                )
            described.append(item)
        return described

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        """도구를 실행한다. **예외를 밖으로 던지지 않는다.**

        :param name: 도구 이름.
        :param arguments: 인자 매핑. ``None`` 이면 빈 dict.
        :returns: :class:`~docagent.contracts.ToolResult`.
            :class:`~docagent.errors.HandoffRequired` 는 ``handoff=True`` 로,
            그 밖의 오류는 ``ok=False`` + ``error`` 로 변환된다.
            세션 이력(``state.history``)에는 예외 **유형 이름만** 남긴다.
            이력은 세션 복원 파일로 직렬화되므로 예외 문자열을 그대로 넣으면
            검사 대상 값이 영구 저장될 수 있다.
        """
        args = dict(arguments or {})
        try:
            tool = self.get(name)
            return tool.handler(args)
        except HandoffRequired as exc:
            self.state.log(
                "tool_handoff", note=f"[{name}] {exc.reason}", field_id=exc.field_id
            )
            return ToolResult(
                ok=False,
                speech="",
                state_patch={"tool": name},
                error=exc.reason,
                handoff=True,
            )
        except DocAgentError as exc:
            # 예외 **문자열**은 세션 이력에 남기지 않는다. 오류 메시지에는 검사 대상
            # 값이나 키 이름이 섞일 수 있고, state.history 는 to_json() 으로
            # 세션 복원 파일에 영구 저장되기 때문이다. 유형만 남긴다.
            self.state.log("tool_error", note=f"[{name}] {type(exc).__name__}")
            return ToolResult(
                ok=False, speech="", state_patch={"tool": name}, error=str(exc)
            )
        except Exception as exc:  # 예상 못한 오류도 도메인 메시지로 감싸 보고한다.
            wrapped = ToolExecutionError(
                f"도구 실행 중 예상치 못한 오류가 발생했습니다: {type(exc).__name__}",
                tool_name=name,
            )
            self.state.log("tool_error", note=f"[{name}] {type(exc).__name__}")
            return ToolResult(
                ok=False, speech="", state_patch={"tool": name}, error=str(wrapped)
            )

    # ------------------------------------------------------------------
    # 공통 헬퍼
    # ------------------------------------------------------------------

    def _resolve_field(self, arguments: Mapping[str, Any], tool_name: str) -> Field:
        """인자 또는 현재 상태에서 대상 항목을 확정한다.

        :param arguments: 도구 인자.
        :param tool_name: 오류 메시지에 표기할 도구 이름.
        :returns: :class:`~docagent.contracts.Field`.
        :raises ToolExecutionError: 대상을 정할 수 없거나 문서에 없는 항목인 경우.
        """
        raw = arguments.get("field_id") or self.state.current_field_id
        if not raw:
            raise ToolExecutionError(
                "지금 안내 중인 항목이 없어 대상을 정할 수 없습니다.",
                tool_name=tool_name,
            )
        field = self.structure.field_by_id(str(raw))
        if field is None:
            raise ToolExecutionError(
                f"문서에 존재하지 않는 항목입니다: {raw}", tool_name=tool_name
            )
        return field

    def _patch(self) -> dict[str, Any]:
        """현재 상태 요약을 만든다(``state_patch`` 용).

        :returns: ``phase`` / ``current_field_id`` / ``completed`` / ``pending`` 요약.
        """
        return {
            "phase": self.state.phase.value,
            "current_field_id": self.state.current_field_id,
            "completed": len(self.state.completed_fields),
            "pending": len(self.state.pending_fields),
        }

    def _target_box(self, field: Field, tool_name: str) -> BoxMm:
        """펜이 이동할 실제 사각형을 문서 구조에서 직접 읽는다.

        선택형 항목이면 **사용자가 고른 선택지**의 네모 칸을, 그 외에는 항목
        영역을 쓴다. 어떤 경우에도 인자로 받은 좌표를 쓰지 않는다.

        :param field: 대상 항목.
        :param tool_name: 오류 메시지에 표기할 도구 이름.
        :returns: :class:`~docagent.contracts.BoxMm`.
        :raises ToolExecutionError: 선택이 아직 이루어지지 않았거나 라벨이 문서에 없는 경우.
        :raises HandoffRequired: 좌표가 확정되지 않은 경우.
        """
        if field.options:
            label = self.state.selected_options.get(field.id)
            if label is None:
                raise ToolExecutionError(
                    f"'{field.title}' 항목은 선택지를 먼저 고르셔야 이동할 수 있습니다.",
                    tool_name=tool_name,
                )
            for option in field.options:
                if option.label == label:
                    return option.box_mm
            raise ToolExecutionError(
                f"문서에 없는 선택지입니다: {label}", tool_name=tool_name
            )
        if field.box_mm is None:
            raise HandoffRequired(
                f"'{field.title}' 항목의 위치를 확정하지 못했습니다. "
                "담당 직원의 도움이 필요합니다.",
                field_id=field.id,
            )
        return field.box_mm

    # ------------------------------------------------------------------
    # 기본 도구 정의
    # ------------------------------------------------------------------

    def _build_default_tools(self) -> tuple[Tool, ...]:
        """기본 도구 12종을 만든다.

        :returns: :class:`Tool` 튜플(등록 순서).
        """
        return (
            Tool(
                name="explain",
                description=(
                    "지금 항목의 내용을 쉬운 말로 설명합니다. 설명 뒤에는 항상 "
                    "원문을 그대로 들을 수 있다는 안내를 덧붙입니다."
                ),
                json_schema=_schema(
                    {
                        "field_id": _FIELD_ID_PROPERTY,
                        "question": {
                            "type": "string",
                            "description": "사용자가 물어본 내용. 없으면 생략합니다.",
                        },
                    }
                ),
                handler=self._explain,
            ),
            Tool(
                name="read_original",
                description="항목의 원문 문구를 한 글자도 바꾸지 않고 그대로 읽어 줍니다.",
                json_schema=_schema({"field_id": _FIELD_ID_PROPERTY}),
                handler=self._read_original,
            ),
            Tool(
                name="next_field",
                description="지금 항목을 건너뛰고 다음 항목으로 넘어갑니다.",
                json_schema=_schema(),
                handler=self._next_field,
            ),
            Tool(
                name="previous_field",
                description="문서 순서상 바로 앞 항목으로 돌아갑니다.",
                json_schema=_schema(),
                handler=self._previous_field,
            ),
            Tool(
                name="go_to_field",
                description="지정한 항목으로 안내 위치를 옮깁니다. 기입 기록은 바꾸지 않습니다.",
                json_schema=_schema(
                    {
                        "field_id": {
                            "type": "string",
                            "description": "이동할 항목의 식별자.",
                        }
                    },
                    required=("field_id",),
                ),
                handler=self._go_to_field,
            ),
            Tool(
                name="select_option",
                description=(
                    "선택형 항목에서 사용자가 고른 선택지를 기록합니다. "
                    "선택지 이름은 반드시 문서에 실제로 있는 것이어야 합니다."
                ),
                json_schema=_schema(
                    {
                        "field_id": _FIELD_ID_PROPERTY,
                        "option_label": {
                            "type": "string",
                            "description": "사용자가 고른 선택지의 이름.",
                        },
                    },
                    required=("option_label",),
                ),
                handler=self._select_option,
            ),
            Tool(
                name="move_to_field",
                description=(
                    "펜을 해당 항목의 기입 위치로 옮깁니다. 위치 값은 인자로 받지 않으며 "
                    "문서 인식 결과에서 프로그램이 직접 읽습니다."
                ),
                json_schema=_schema({"field_id": _FIELD_ID_PROPERTY}),
                handler=self._move_to_field,
            ),
            Tool(
                name="verify_field",
                description="항목이 실제로 기입되었는지 확인하고 완료 처리합니다.",
                json_schema=_schema({"field_id": _FIELD_ID_PROPERTY}),
                handler=self._verify_field,
            ),
            Tool(
                name="repeat",
                description="직전 안내를 다시 들려 달라는 요청을 처리합니다.",
                json_schema=_schema(),
                handler=self._repeat,
            ),
            Tool(
                name="revise_field",
                description=(
                    "이미 마친 항목으로 되돌아가 선택을 취소하고 다시 하게 합니다. "
                    "되돌아간 항목 이후의 진행 기록도 함께 취소됩니다."
                ),
                json_schema=_schema(
                    {
                        "field_id": {
                            "type": "string",
                            "description": "되돌아갈 항목의 식별자.",
                        }
                    },
                    required=("field_id",),
                ),
                handler=self._revise_field,
            ),
            Tool(
                name="request_human",
                description="담당 직원의 도움을 요청하고 세션을 사람에게 넘깁니다.",
                json_schema=_schema(
                    {
                        "reason": {
                            "type": "string",
                            "description": "직원에게 넘기는 사유를 한 문장으로 적습니다.",
                        }
                    }
                ),
                handler=self._request_human,
            ),
            Tool(
                name="summarize_progress",
                description="지금까지 몇 개를 마쳤고 무엇이 남았는지 알려 줍니다.",
                json_schema=_schema(),
                handler=self._summarize_progress,
            ),
        )

    # ------------------------------------------------------------------
    # handler 구현
    # ------------------------------------------------------------------

    def _explain(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``explain`` 도구 본체.

        :param arguments: ``field_id``(선택), ``question``(선택).
        :returns: 설명 발화가 담긴 :class:`~docagent.contracts.ToolResult`.
        :raises HandoffRequired: 설명 신뢰도가 :data:`PARTIAL_THRESHOLD` 미만이거나
            설명기가 사람 지원을 요구한 경우.
        """
        field = self._resolve_field(arguments, "explain")
        question = str(arguments.get("question", ""))
        result = self.explainer.explain(field, question)
        text = str(getattr(result, "text", "") or "").strip()
        confidence = float(getattr(result, "confidence", 0.0))
        needs_handoff = bool(getattr(result, "needs_handoff", False))
        disclaimer = str(getattr(result, "disclaimer", "") or "").strip()
        sources = tuple(getattr(result, "sources", ()) or ())

        if needs_handoff or confidence < PARTIAL_THRESHOLD or not text:
            failures = self.state.bump(f"fail:{field.id}")
            reason = (
                f"'{field.title}' 항목을 정확히 설명드릴 자신이 없습니다. "
                "담당 직원의 도움을 받으시는 것이 안전합니다."
            )
            if failures >= MAX_FIELD_FAILURES:
                reason = (
                    f"'{field.title}' 항목에서 {failures}번 연속으로 진행하지 못했습니다. "
                    "담당 직원의 도움이 필요합니다."
                )
            raise HandoffRequired(reason, field_id=field.id)

        self.state.reset_counter(f"fail:{field.id}")
        parts = [text]
        # 설명기가 이미 본문에 고지를 넣어 두었으면 다시 붙이지 않는다.
        # (통합 시 발견: Explainer 는 text 안에 DISCLAIMER 를 포함시키면서 같은
        #  문장을 disclaimer 필드로도 돌려주어, 낭독이 두 번 반복되었다.)
        if disclaimer and disclaimer not in text:
            parts.append(disclaimer)
        if sources:
            parts.append("근거는 " + ", ".join(str(s) for s in sources) + " 입니다.")
        # 원문 고지는 발화당 **정확히 한 번**만 낭독한다.
        if ORIGINAL_NOTICE not in " ".join(parts):
            parts.append(ORIGINAL_NOTICE)
        self.state.log("explain", note="쉬운 설명을 제공했습니다.", field_id=field.id)
        return ToolResult(
            ok=True,
            speech=" ".join(parts),
            state_patch=self._patch(),
        )

    def _read_original(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``read_original`` 도구 본체. 원문을 가공 없이 읽는다.

        :param arguments: ``field_id``(선택).
        :returns: 원문 발화가 담긴 :class:`~docagent.contracts.ToolResult`.
        """
        field = self._resolve_field(arguments, "read_original")
        parts: list[str] = ["원문을 그대로 읽어 드리겠습니다."]
        body = field.clause_text.strip()
        if body:
            parts.append(body)
        elif field.title.strip():
            parts.append(f"항목 이름은 '{field.title}' 입니다.")
        else:
            parts.append("이 항목에는 읽어 드릴 문구가 없습니다.")
        if field.options:
            labels = ", ".join(option.label for option in field.options)
            parts.append(f"선택지는 {labels} 입니다.")
        self.state.log("read_original", note="원문을 낭독했습니다.", field_id=field.id)
        return ToolResult(ok=True, speech=" ".join(parts), state_patch=self._patch())

    def _next_field(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``next_field`` 도구 본체. 현재 항목을 건너뛰고 다음으로 넘어간다.

        :param arguments: 사용하지 않는다.
        :returns: :class:`~docagent.contracts.ToolResult`.
        """
        current = self.state.current_field_id
        following = self.state.next_pending(after=current)
        if current is not None and current not in self.state.completed_fields:
            self.state.skip_field(current)
            following = self.state.next_pending(after=current)
        if following is None:
            self.state.transition_to(
                SessionPhase.COMPLETED,
                action="all_fields_done",
                note="더 진행할 항목이 없습니다.",
            )
            return ToolResult(
                ok=True, speech="", state_patch=self._patch()
            )
        self.state.set_current(following)
        self.state.transition_to(
            SessionPhase.ANNOUNCE_FIELD, action="next_field", note="다음 항목으로 넘어갑니다."
        )
        return ToolResult(ok=True, speech="", state_patch=self._patch())

    def _previous_field(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``previous_field`` 도구 본체.

        :param arguments: 사용하지 않는다.
        :returns: :class:`~docagent.contracts.ToolResult`. 첫 항목이면 ``ok=False``.
        """
        previous = self.state.previous_field()
        if previous is None:
            return ToolResult(
                ok=False,
                speech="지금이 첫 번째 항목입니다.",
                state_patch=self._patch(),
                error="이전 항목이 없습니다.",
            )
        self.state.set_current(previous)
        self.state.transition_to(
            SessionPhase.ANNOUNCE_FIELD,
            action="previous_field",
            note="이전 항목으로 돌아갑니다.",
        )
        return ToolResult(ok=True, speech="", state_patch=self._patch())

    def _go_to_field(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``go_to_field`` 도구 본체.

        :param arguments: ``field_id``(필수).
        :returns: :class:`~docagent.contracts.ToolResult`.
        """
        field = self._resolve_field(arguments, "go_to_field")
        self.state.set_current(field.id)
        self.state.transition_to(
            SessionPhase.ANNOUNCE_FIELD,
            action="go_to_field",
            note=f"'{field.title}' 항목으로 이동했습니다.",
        )
        return ToolResult(ok=True, speech="", state_patch=self._patch())

    def _select_option(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``select_option`` 도구 본체. 문서에 실재하는 라벨만 받아들인다.

        :param arguments: ``field_id``(선택), ``option_label``(필수).
        :returns: :class:`~docagent.contracts.ToolResult`.
        :raises ToolExecutionError: 선택형이 아니거나 문서에 없는 라벨인 경우.
        """
        field = self._resolve_field(arguments, "select_option")
        label = str(arguments.get("option_label", "")).strip()
        if not label:
            raise ToolExecutionError(
                "선택할 항목 이름이 비어 있습니다.", tool_name="select_option"
            )
        if not field.options:
            raise ToolExecutionError(
                f"'{field.title}' 항목에는 고를 선택지가 없습니다.",
                tool_name="select_option",
            )
        labels = [option.label for option in field.options]
        if label not in labels:
            raise ToolExecutionError(
                f"'{label}' 은(는) 이 항목의 선택지가 아닙니다. "
                f"선택지는 {', '.join(labels)} 입니다.",
                tool_name="select_option",
            )
        self.state.select_option(field.id, label)
        self.state.transition_to(
            SessionPhase.GUIDING,
            action="option_selected",
            note=f"'{label}' 선택을 확정했습니다.",
        )
        return ToolResult(
            ok=True,
            speech=f"'{label}'(으)로 확인했습니다.",
            state_patch={**self._patch(), "selected_option": label},
        )

    def _move_to_field(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``move_to_field`` 도구 본체. **좌표 인자를 받지 않는다.**

        절차: 항목 조회 → 인식 신뢰도 확인 → 문서에서 위치 직접 읽기 →
        가동 범위 확인 → 이동 → 도착 확인.

        :param arguments: ``field_id``(선택).
        :returns: :class:`~docagent.contracts.ToolResult`.
        :raises HandoffRequired: 인식 신뢰도 미달, 위치 미확정, 도착 실패.
        """
        field = self._resolve_field(arguments, "move_to_field")
        if field.confidence < VISION_TRUST_THRESHOLD:
            raise HandoffRequired(
                f"'{field.title}' 항목의 인식 신뢰도가 기준에 미치지 못해 "
                "펜을 옮기지 않았습니다. 담당 직원의 확인이 필요합니다.",
                field_id=field.id,
            )
        box = self._target_box(field, "move_to_field")
        target = box.center()
        page_w, page_h = self.structure.page_size_mm
        if not (0.0 <= target.x_mm <= page_w and 0.0 <= target.y_mm <= page_h):
            return ToolResult(
                ok=False,
                speech="기입 위치가 용지 범위를 벗어나 펜을 옮기지 않았습니다.",
                state_patch=self._patch(),
                error="목표 위치가 가동 범위를 벗어났습니다.",
            )
        moved = bool(self.motion.move_to(target.x_mm, target.y_mm))
        if not moved:
            failures = self.state.bump(f"fail:{field.id}")
            if failures >= MAX_FIELD_FAILURES:
                raise HandoffRequired(
                    f"'{field.title}' 항목에서 {failures}번 연속으로 펜을 옮기지 "
                    "못했습니다. 담당 직원의 도움이 필요합니다.",
                    field_id=field.id,
                )
            return ToolResult(
                ok=False,
                speech="펜을 옮기지 못했습니다. 다시 시도해 보겠습니다.",
                state_patch=self._patch(),
                error="펜 이동에 실패했습니다(가동 범위 밖이거나 장치가 응답하지 않습니다).",
            )
        actual = self.motion.position()
        drift_x = abs(float(actual.x_mm) - target.x_mm)
        drift_y = abs(float(actual.y_mm) - target.y_mm)
        if drift_x > self.motion_tolerance_mm or drift_y > self.motion_tolerance_mm:
            raise HandoffRequired(
                f"'{field.title}' 항목에서 펜이 목표 위치에 도달하지 못했습니다. "
                "잘못된 곳에 표시되지 않도록 담당 직원의 확인을 요청합니다.",
                field_id=field.id,
            )
        self.state.reset_counter(f"fail:{field.id}")
        self.state.transition_to(
            SessionPhase.AWAIT_WRITE,
            action="move_to_field",
            note=f"'{field.title}' 기입 위치로 펜을 옮겼습니다.",
        )
        if field.type is FieldType.SIGNATURE:
            guide = "펜 끝에 손을 대시고 서명해 주세요."
        elif field.options:
            guide = "펜 끝에 손을 대시고 네모 칸 안에 표시해 주세요."
        else:
            guide = "펜 끝에 손을 대시고 내용을 적어 주세요."
        return ToolResult(
            ok=True,
            speech=f"펜을 기입 위치로 옮겼습니다. {guide} 다 되시면 '확인'이라고 말씀해 주세요.",
            state_patch=self._patch(),
        )

    def _verify_field(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``verify_field`` 도구 본체.

        :param arguments: ``field_id``(선택).
        :returns: :class:`~docagent.contracts.ToolResult`.
        :raises HandoffRequired: 같은 항목에서 :data:`MAX_FIELD_FAILURES` 회 연속 실패.
        """
        field = self._resolve_field(arguments, "verify_field")
        self.state.transition_to(
            SessionPhase.VERIFYING, action="verify_start", note="기입 여부를 확인합니다."
        )
        result = self.verifier.verify(field.id)
        written = bool(getattr(result, "written", False))
        confidence = float(getattr(result, "confidence", 0.0))
        reason = str(getattr(result, "reason", "") or "")

        if written and confidence >= PARTIAL_THRESHOLD:
            # 기입 사실만으로는 부족하다. 선택형 항목은 **말로 고른 선택지**와
            # 실제로 표시된 선택지가 같은지까지 확인해야 한다. 다른 칸에 표시된
            # 것을 "확인했습니다"로 넘기면 이용자는 자기가 고르지 않은 내용에
            # 동의한 서류를 제출하게 된다.
            marked = self._marked_option_label(field)
            selected = self.state.selected_options.get(field.id)
            if marked is not None and selected is not None and marked != selected:
                return self._option_mismatch_result(field, selected, marked)

            self.state.reset_counter(f"fail:{field.id}")
            self.state.mark_verified(field.id)
            self.state.complete_field(field.id)
            self.state.transition_to(
                SessionPhase.FIELD_DONE,
                action="verify_ok",
                note=reason or "기입이 확인되었습니다.",
            )
            return ToolResult(
                ok=True,
                speech=f"'{field.title}' 항목 기입을 확인했습니다.",
                state_patch={**self._patch(), "verified": field.id},
            )

        failures = self.state.bump(f"fail:{field.id}")
        if failures >= MAX_FIELD_FAILURES:
            raise HandoffRequired(
                f"'{field.title}' 항목에서 {failures}번 연속으로 기입을 확인하지 "
                "못했습니다. 담당 직원의 도움이 필요합니다.",
                field_id=field.id,
            )
        self.state.transition_to(
            SessionPhase.AWAIT_WRITE,
            action="verify_failed",
            note=reason or "기입을 확인하지 못했습니다.",
        )
        return ToolResult(
            ok=False,
            speech=(
                "아직 기입이 확인되지 않았습니다. 펜 위치는 그대로 두었으니 "
                "다시 표시하신 뒤 '확인'이라고 말씀해 주세요."
            ),
            state_patch={**self._patch(), "failures": failures},
            error=reason or "기입을 확인하지 못했습니다.",
        )

    def _marked_option_label(self, field: Field) -> str | None:
        """실제로 표시된 선택지 라벨을 돌려준다(판정 불가면 ``None``).

        검증기가 선택지별 판정을 제공할 때만 대조가 가능하다.
        ``verify_options(field_id)`` 를 가진 검증기(예:
        :class:`docagent.pipeline.ImageVerifier`)면 그 결과에서 ``written`` 인
        선택지 가운데 잉크 증가량이 가장 큰 것을 표시된 칸으로 본다.

        :param field: 대상 항목.
        :returns: 표시된 선택지 라벨. 선택형이 아니거나 검증기가 선택지별
            판정을 제공하지 않거나 표시된 칸이 없으면 ``None``.
        """
        if not field.options:
            return None
        verify_options = getattr(self.verifier, "verify_options", None)
        if not callable(verify_options):
            return None
        try:
            results = tuple(verify_options(field.id))
        except DocAgentError:
            raise
        except Exception as exc:  # noqa: BLE001 — 도메인 예외로 감싸 올린다.
            raise ToolExecutionError(
                f"선택지별 기입 확인에 실패했습니다: {exc}", tool_name="verify_field"
            ) from exc
        marked = [item for item in results if bool(getattr(item, "written", False))]
        if not marked:
            return None
        best = max(marked, key=lambda item: float(getattr(item, "ink_delta", 0.0)))
        return str(getattr(best, "field_id", "")).split(":", 1)[-1]

    def _option_mismatch_result(
        self, field: Field, selected: str, marked: str
    ) -> ToolResult:
        """말한 선택지와 표시된 선택지가 다를 때의 결과를 만든다.

        :param field: 대상 항목.
        :param selected: 이용자가 말로 고른 선택지 라벨.
        :param marked: 실제로 표시가 확인된 선택지 라벨.
        :returns: ``ok=False`` 인 :class:`~docagent.contracts.ToolResult`.
        :raises HandoffRequired: 같은 항목에서 :data:`MAX_FIELD_FAILURES` 회
            연속 실패한 경우.
        """
        detail = (
            f"'{selected}'(으)로 말씀하셨는데 '{marked}' 칸에 표시가 확인되었습니다."
        )
        failures = self.state.bump(f"fail:{field.id}")
        if failures >= MAX_FIELD_FAILURES:
            raise HandoffRequired(
                f"'{field.title}' 항목에서 {detail} "
                f"{failures}번 연속으로 바로잡지 못했으므로 담당 직원의 도움이 필요합니다.",
                field_id=field.id,
            )
        self.state.transition_to(
            SessionPhase.AWAIT_WRITE,
            action="verify_option_mismatch",
            note=detail,
        )
        return ToolResult(
            ok=False,
            speech=(
                f"말씀하신 곳과 다른 칸에 표시되었습니다. {detail} "
                "표시를 지우고 다시 해 주시거나, '동의함'처럼 원하시는 선택지를 "
                "다시 말씀해 주세요."
            ),
            state_patch={**self._patch(), "failures": failures},
            error=f"선택지 불일치: {detail}",
        )

    def _repeat(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``repeat`` 도구 본체. 상태를 바꾸지 않고 재안내 요청만 기록한다.

        :param arguments: 사용하지 않는다.
        :returns: :class:`~docagent.contracts.ToolResult`.
        """
        self.state.log("repeat", note="직전 안내를 다시 요청했습니다.")
        return ToolResult(ok=True, speech="", state_patch=self._patch())

    def _revise_field(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``revise_field`` 도구 본체.

        :param arguments: ``field_id``(필수).
        :returns: :class:`~docagent.contracts.ToolResult`.
        :raises ToolExecutionError: 문서에 없는 항목이거나 되돌릴 수 없는 단계인 경우.
        """
        field = self._resolve_field(arguments, "revise_field")
        try:
            self.state.revert_to_field(field.id)
        except (ValueError, DocAgentError) as exc:
            raise ToolExecutionError(
                f"'{field.title}' 항목으로 되돌릴 수 없습니다: {exc}",
                tool_name="revise_field",
            ) from exc
        return ToolResult(
            ok=True,
            speech=(
                f"'{field.title}' 항목으로 되돌아갑니다. 이전 선택은 취소했습니다."
            ),
            state_patch=self._patch(),
        )

    def _request_human(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``request_human`` 도구 본체.

        :param arguments: ``reason``(선택).
        :returns: ``handoff=True`` 인 :class:`~docagent.contracts.ToolResult`.
        """
        reason = str(
            arguments.get("reason") or "사용자가 담당 직원의 도움을 요청했습니다."
        )
        self.state.enter_handoff(reason)
        return ToolResult(
            ok=True,
            speech="담당 직원을 호출했습니다. 잠시만 기다려 주세요.",
            state_patch=self._patch(),
            error=None,
            handoff=True,
        )

    def _summarize_progress(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``summarize_progress`` 도구 본체.

        :param arguments: 사용하지 않는다.
        :returns: 진행 요약 발화가 담긴 :class:`~docagent.contracts.ToolResult`.
        """
        done, total, remaining = self.state.progress()
        parts = [f"전체 {total}개 항목 중 {done}개를 마쳤습니다."]
        if remaining:
            titles: list[str] = []
            for fid in remaining:
                item = self.structure.field_by_id(fid)
                titles.append(item.title if item is not None and item.title else fid)
            parts.append("남은 필수 항목은 " + ", ".join(titles) + " 입니다.")
        else:
            parts.append("남은 필수 항목은 없습니다.")
        return ToolResult(
            ok=True,
            speech=" ".join(parts),
            state_patch={**self._patch(), "remaining_required": list(remaining)},
        )
