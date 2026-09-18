"""doc-agent — 시각장애인용 AI 음성·촉각 문서작성 에이전트.

시스템 골격은 **See → Understand → Explain → Ask → Act → Verify** 이며,
하드웨어(펜 액추에이터)는 에이전트의 Actuator 일 뿐 주인공은 AI 에이전트다.

이 패키지의 최상위에서는 계약 계층만 재export 한다. 실제 구현은
:mod:`docagent.vision` / :mod:`docagent.pii` / :mod:`docagent.agent` 하위에 있다.
순환 import 를 막기 위해 하위 패키지의 ``__init__`` 은 비워 둔다.
"""

from __future__ import annotations

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_PAGE_SIZE_MM,
    A4_WIDTH_MM,
    EXPLAIN_THRESHOLD,
    MOTION_TOLERANCE_MM,
    PARTIAL_THRESHOLD,
    VISION_TRUST_THRESHOLD,
    AgentTurn,
    BoxMm,
    BoxPx,
    Detection,
    DocumentStructure,
    Field,
    FieldRole,
    FieldType,
    OcrWord,
    Option,
    PiiSpan,
    Point,
    RetrievedChunk,
    SanitizedText,
    Sensitivity,
    ToolCall,
    ToolResult,
    VerificationResult,
)
from docagent.errors import (
    AdapterUnavailable,
    DocAgentError,
    DocumentNotFoundError,
    HandoffRequired,
    InvalidTransition,
    LowConfidenceError,
    PiiEgressBlocked,
    ToolExecutionError,
    VisionError,
)
from docagent.interfaces import (
    Clock,
    Detector,
    LlmClient,
    MotionController,
    OcrEngine,
    PiiGate,
    Retriever,
    SpeechIO,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 상수
    "A4_WIDTH_MM",
    "A4_HEIGHT_MM",
    "A4_PAGE_SIZE_MM",
    "EXPLAIN_THRESHOLD",
    "PARTIAL_THRESHOLD",
    "VISION_TRUST_THRESHOLD",
    "MOTION_TOLERANCE_MM",
    # 계약 dataclass · Enum
    "Point",
    "BoxMm",
    "BoxPx",
    "FieldType",
    "FieldRole",
    "Sensitivity",
    "Detection",
    "OcrWord",
    "Option",
    "Field",
    "DocumentStructure",
    "VerificationResult",
    "PiiSpan",
    "SanitizedText",
    "RetrievedChunk",
    "ToolCall",
    "ToolResult",
    "AgentTurn",
    # 예외
    "DocAgentError",
    "VisionError",
    "DocumentNotFoundError",
    "LowConfidenceError",
    "PiiEgressBlocked",
    "HandoffRequired",
    "InvalidTransition",
    "ToolExecutionError",
    "AdapterUnavailable",
    # Protocol
    "Detector",
    "OcrEngine",
    "PiiGate",
    "Retriever",
    "LlmClient",
    "MotionController",
    "SpeechIO",
    "Clock",
]
