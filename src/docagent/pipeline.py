"""단일 진입점 — See → Understand → Explain → Ask → Act → Verify 를 한 번에 조립한다.

로드맵 3장의 아키텍처를 그대로 코드로 옮긴 모듈이다. 호출자는
:func:`build_session` 하나만 부르면 Vision · PII · Agent 세 모듈이 계약대로
연결된 :class:`DocumentSession` 을 얻는다.

조립 순서
---------
1. **정합** — :func:`docagent.vision.normalize.normalize_document`
2. **탐지** — :func:`docagent.vision.detector_factory.build_detector`
3. **OCR** — :class:`docagent.interfaces.OcrEngine` (기본값은 오프라인 :class:`StubOcr`)
4. **구조화** — :func:`docagent.vision.structuring.build_structure` → fields JSON
5. **PII 정책 분류** — :func:`docagent.pii.policy.classify_field` 로 민감도를
   **fail-closed 재판정**한다. Vision 이 PUBLIC 이라고 본 항목이라도 내용 검사에서
   개인정보가 나오면 PRIVATE 로 강등한다. 반대 방향(PRIVATE → PUBLIC)은 없다.
6. **공개 payload** — :func:`docagent.pii.policy.build_public_payload` 로 만들고,
   :meth:`docagent.pii.gate.LlmEgressGate.guard` 로 한 번 더 통과시킨다.
   좌표(``box_mm``)는 payload 에 들어가지 않는다.
7. **RAG 색인** — :func:`docagent.agent.rag.build_index`
8. **Explainer / Guard / Gate** 구성
9. **ToolRegistry** 구성 — 좌표는 도구가 문서 구조에서 **직접** 읽는다.
10. **DocumentAgent** 생성

게이트 우회 금지
----------------
:func:`build_session` 은 어떤 경로로도 게이트 없는 LLM 을 에이전트에 넘기지
않는다. ``llm`` 인자로 날 것의 :class:`~docagent.interfaces.LlmClient` 를 주면
:class:`~docagent.pii.gate.GatedLlmClient` 로 **강제로 감싼다**. 이미 감싸인
클라이언트를 주면 그 게이트가 이 세션의 게이트와 같은지 확인하고, 다르면
감사 로그가 갈라지지 않도록 세션 게이트로 다시 감싼다.

신뢰도 취급 방침(통합 시 확정)
------------------------------
정합·탐지·구조화 각 단계의 신뢰도는 :class:`StageReport` 로 모아 두되,
**Vision 이 매긴 항목별 신뢰도를 파이프라인이 임의로 깎지 않는다.** 항목별
신뢰도는 실측값이고, 여기에 문서 단위 계수를 곱하면 측정값의 의미가 사라지기
때문이다. 사람 지원 트리거는 지금처럼 항목별 신뢰도가 담당하고, 문서 단위
저신뢰는 ``warnings`` 와 세션 시작 안내로 사용자에게 전달한다.

문서 경계를 확인하지 못한 대체 경로(margin 이 없어 종이 윤곽이 안 보이는 스캔)는
좌표 오차가 오히려 작으므로(실측 0.00mm) 직원 연결 사유로 쓰지 않는다.
더 엄격한 정책이 필요한 배치에서는 ``propagate_document_confidence=True`` 로
문서 신뢰도를 항목 신뢰도에 곱해 전파할 수 있다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from docagent.config import DocAgentConfig
from docagent.contracts import (
    VISION_TRUST_THRESHOLD,
    AgentTurn,
    Detection,
    DocumentStructure,
    Field,
    OcrWord,
    Sensitivity,
    VerificationResult,
)
from docagent.errors import DocumentNotFoundError, VisionError
from docagent.agent.explainer import Explainer
from docagent.agent.guardrails import Guard
from docagent.agent.llm import ClaudeLlm, OfflineTemplateLlm
from docagent.agent.orchestrator import DocumentAgent
from docagent.agent.rag import LocalTfidfRetriever, build_index
from docagent.agent.state import SessionState
from docagent.agent.tools import ToolRegistry
from docagent.io.motion import MockMotionController
from docagent.io.speech import ScriptedSpeechIO
from docagent.pii.audit import AuditLog
from docagent.pii.gate import GatedLlmClient, LlmEgressGate
from docagent.pii.policy import build_public_payload, classify_field
from docagent.vision.detector_factory import build_detector
from docagent.vision.normalize import NormalizedDocument, normalize_document
from docagent.vision.ocr import StubOcr
from docagent.vision.structuring import build_structure
from docagent.vision.verify import verify_field as verify_field_images
from docagent.vision.verify import verify_options as verify_options_images

__all__ = [
    "DEFAULT_DOCUMENT_ID",
    "NO_OCR_WARNING",
    "SNAPSHOT_VERSION",
    "StageReport",
    "ImageVerifier",
    "DocumentSession",
    "align_to_page",
    "apply_pii_policy",
    "build_session",
]

_LOG = logging.getLogger(__name__)

#: 문서 식별자를 주지 않았을 때 쓰는 기본값.
DEFAULT_DOCUMENT_ID: str = "session_0001"

#: 세션 스냅숏 봉투 형식 버전(:meth:`DocumentSession.snapshot_json`).
SNAPSHOT_VERSION: int = 1

#: OCR 엔진이 없어 텍스트 없이 진행할 때 남기는 경고.
NO_OCR_WARNING: str = (
    "OCR 엔진이 연결되지 않아 문서의 글자를 읽지 못했습니다. "
    "항목명과 약관 문구 없이 좌표만으로 진행하므로 안내가 불완전할 수 있습니다."
)


# --------------------------------------------------------------------------
# 단계별 신뢰도 보고
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageReport:
    """파이프라인 한 단계의 신뢰도 보고.

    :param stage: 단계 이름(``normalize`` / ``detect`` / ``structure`` / ``pii``).
    :param confidence: 그 단계의 대표 신뢰도(0.0~1.0).
    :param ok: 임계값을 만족했는지 여부.
    :param detail: 사람이 읽는 한국어 설명.
    """

    stage: str
    confidence: float
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 돌려준다."""
        return {
            "stage": self.stage,
            "confidence": self.confidence,
            "ok": self.ok,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# 이미지 정렬 · Verify 어댑터
# --------------------------------------------------------------------------


def align_to_page(normalized: NormalizedDocument, image: Any) -> np.ndarray:
    """새 촬영 이미지를 **기존 정합과 같은 좌표계**로 옮긴다.

    기입 전/후 비교는 두 이미지가 픽셀 단위로 같은 프레임에 있을 때만 성립한다.
    새 이미지를 따로 정규화하면 사각형 검출이 미세하게 달라져 프레임이 어긋날 수
    있으므로, 여기서는 **최초 정합에서 구한 호모그래피를 그대로 재사용**한다.

    :param normalized: 최초 정합 결과.
    :param image: 같은 자세에서 다시 찍은 원본 이미지(``(H, W)`` 또는 ``(H, W, 3)``).
    :returns: 최초 정합 이미지와 같은 크기·좌표계의 배열.
    :raises docagent.errors.VisionError: 이미지가 비었거나 투영 변환이 실패한 경우.
    """
    array = np.asarray(image)
    if array.size == 0:
        raise VisionError("정렬할 이미지가 비어 있습니다.")
    width_px, height_px = normalized.coords.size_px
    try:
        warped = cv2.warpPerspective(
            array,
            normalized.homography,
            (width_px, height_px),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except cv2.error as exc:  # 조용한 실패 금지
        raise VisionError(f"기입 후 이미지 정렬에 실패했습니다: {exc}") from exc
    if warped.ndim != normalized.image.ndim:
        if warped.ndim == 3 and normalized.image.ndim == 2:
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        elif warped.ndim == 2 and normalized.image.ndim == 3:
            warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    return warped


class ImageVerifier:
    """이미지 비교 기반 Verify 어댑터.

    :class:`~docagent.agent.tools.ToolRegistry` 가 요구하는
    ``verify(field_id) -> VerificationResult`` 계약을, Vision 의
    :func:`docagent.vision.verify.verify_field` 로 구현한다.

    "기입 전" 이미지는 세션 시작 시점의 정합 이미지로 고정되고, "기입 후"
    이미지는 :meth:`update` 로 갱신한다. 갱신 전에는 두 이미지가 같으므로
    잉크 증가량이 0 이 되어 ``written=False`` 가 나온다 — 즉 **아무것도 쓰지
    않았는데 확인되었다고 말하는 일이 구조적으로 불가능하다.**

    :param structure: 문서 구조(좌표의 출처).
    :param before: 기입 전 정합 이미지.
    :param page_size_mm: 페이지 크기 ``(가로_mm, 세로_mm)``.
    """

    def __init__(
        self,
        structure: DocumentStructure,
        before: Any,
        *,
        page_size_mm: tuple[float, float] | None = None,
    ) -> None:
        self._structure = structure
        self._before = np.asarray(before)
        self._after = self._before
        self._page = (
            tuple(float(v) for v in page_size_mm)
            if page_size_mm is not None
            else tuple(float(v) for v in structure.page_size_mm)
        )
        #: 지금까지 수행한 검증 결과 이력(감사·KPI 용).
        self.results: list[VerificationResult] = []

    @property
    def before(self) -> np.ndarray:
        """기입 전 정합 이미지."""
        return self._before

    @property
    def after(self) -> np.ndarray:
        """현재 "기입 후" 이미지. 갱신 전에는 기입 전 이미지와 같다."""
        return self._after

    def update(self, image: Any) -> None:
        """기입 후 이미지를 갱신한다.

        :param image: 기입 전 이미지와 **같은 크기·좌표계**의 배열.
        :returns: ``None``.
        :raises docagent.errors.VisionError: 크기가 다른 경우
            (좌표계가 어긋난 채 판정하면 결과를 신뢰할 수 없다).
        """
        array = np.asarray(image)
        if array.shape[:2] != self._before.shape[:2]:
            raise VisionError(
                "기입 후 이미지의 크기가 기입 전과 다릅니다: "
                f"{array.shape[:2]} != {self._before.shape[:2]}. "
                "같은 정합 좌표계의 이미지를 주십시오(align_to_page 참조)."
            )
        self._after = array

    def rebind(self, structure: DocumentStructure) -> None:
        """검증에 쓸 문서 구조를 교체한다(민감도 재분류 후 재바인딩용).

        :param structure: 새 문서 구조.
        :returns: ``None``.
        """
        self._structure = structure

    def verify(self, field_id: str) -> VerificationResult:
        """항목 하나의 기입 여부를 판정한다.

        :param field_id: 검증할 항목 id.
        :returns: :class:`~docagent.contracts.VerificationResult`.
        :raises docagent.errors.VisionError: 항목이 없거나 좌표가 없는 경우.
            (조용히 ``written=True`` 로 넘기지 않는다.)
        """
        try:
            result = verify_field_images(
                self._structure, field_id, self._before, self._after, self._page
            )
        except ValueError as exc:
            raise VisionError(f"기입 검증 대상을 찾지 못했습니다: {exc}") from exc
        self.results.append(result)
        return result

    def verify_options(self, field_id: str) -> tuple[VerificationResult, ...]:
        """선택형 항목의 **선택지별** 기입 여부를 각각 판정한다.

        :class:`~docagent.agent.tools.ToolRegistry` 가 "말로 고른 선택지"와
        "실제로 표시된 선택지"를 대조하는 데 쓴다. :meth:`verify` 는 대표 결과
        하나만 돌려주므로 어느 칸에 표시되었는지 구조적으로 알 수 없다.

        :param field_id: 선택형 항목 id.
        :returns: 선택지 순서대로의 :class:`~docagent.contracts.VerificationResult`
            튜플. 각 결과의 ``field_id`` 는 ``"<항목id>:<선택지라벨>"`` 형식이다.
        :raises docagent.errors.VisionError: 항목이 없거나 선택지가 없는 경우.
        """
        try:
            return verify_options_images(
                self._structure, field_id, self._before, self._after, self._page
            )
        except ValueError as exc:
            raise VisionError(f"선택지 검증 대상을 찾지 못했습니다: {exc}") from exc

    def __repr__(self) -> str:
        """검증 횟수와 이미지 크기만 노출한다."""
        return (
            f"<ImageVerifier 검증 {len(self.results)}건, "
            f"프레임={self._before.shape[:2]}>"
        )


# --------------------------------------------------------------------------
# PII 정책 적용
# --------------------------------------------------------------------------


def apply_pii_policy(
    structure: DocumentStructure, *, detector: Any | None = None
) -> tuple[DocumentStructure, tuple[str, ...]]:
    """문서 구조의 민감도를 PII 정책으로 **재판정**한다(fail-closed).

    Vision 의 민감도 판정은 항목 유형·라벨 키워드 기반이고, PII 정책은 실제
    내용에 탐지기를 돌린다. 둘 중 더 보수적인 쪽을 택한다: 정책이 PRIVATE 이라고
    하면 무조건 PRIVATE 이며, 강등된 항목의 ``clause_text`` 는 비워 둔다
    (공개 payload 강등 이전 단계에서부터 유출 경로를 없앤다).

    :param structure: 구조화 결과.
    :param detector: 내용 검사에 쓸 탐지기. ``None`` 이면 정책 기본 탐지기.
    :returns: ``(재판정된 구조, 강등된 항목 id 튜플)``.
    """
    updated: list[Field] = []
    downgraded: list[str] = []
    for item in structure.fields:
        decided = classify_field(item, detector=detector)
        if decided is item.sensitivity:
            updated.append(item)
            continue
        if item.sensitivity is Sensitivity.PUBLIC and decided is Sensitivity.PRIVATE:
            downgraded.append(item.id)
        updated.append(
            Field(
                id=item.id,
                type=item.type,
                title=item.title,
                role=item.role,
                options=item.options,
                required=item.required,
                sensitivity=decided,
                box_mm=item.box_mm,
                clause_text=item.clause_text if decided is Sensitivity.PUBLIC else "",
                order=item.order,
                confidence=item.confidence,
            )
        )
    warnings = list(structure.warnings)
    if downgraded:
        warnings.append(
            "개인정보 정책 검사에서 다음 항목을 개인정보 영역으로 내렸습니다: "
            + ", ".join(downgraded)
            + ". 이 항목의 문구는 외부로 전송되지 않습니다."
        )
    return (
        DocumentStructure(
            document_id=structure.document_id,
            doc_title=structure.doc_title,
            fields=tuple(updated),
            page_size_mm=structure.page_size_mm,
            source_image=structure.source_image,
            warnings=tuple(warnings),
        ),
        tuple(downgraded),
    )


def _scale_confidence(structure: DocumentStructure, factor: float) -> DocumentStructure:
    """모든 항목 신뢰도에 문서 단위 계수를 곱한다(옵트인 정책).

    :param structure: 대상 구조.
    :param factor: 0.0~1.0 계수.
    :returns: 신뢰도가 조정된 새 구조.
    :raises ValueError: ``factor`` 가 0.0~1.0 을 벗어난 경우.
    """
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"신뢰도 계수는 0.0~1.0 이어야 합니다: {factor}")
    fields = tuple(
        Field(
            id=item.id,
            type=item.type,
            title=item.title,
            role=item.role,
            options=item.options,
            required=item.required,
            sensitivity=item.sensitivity,
            box_mm=item.box_mm,
            clause_text=item.clause_text,
            order=item.order,
            confidence=max(0.0, min(1.0, item.confidence * factor)),
        )
        for item in structure.fields
    )
    return DocumentStructure(
        document_id=structure.document_id,
        doc_title=structure.doc_title,
        fields=fields,
        page_size_mm=structure.page_size_mm,
        source_image=structure.source_image,
        warnings=structure.warnings,
    )


# --------------------------------------------------------------------------
# 세션
# --------------------------------------------------------------------------


@dataclass
class DocumentSession:
    """조립이 끝난 한 장짜리 문서 작성 세션.

    :param structure: fields JSON 의 원본(:class:`DocumentStructure`).
        좌표가 필요한 Act 단계는 payload 가 아니라 **이 객체**를 읽는다.
    :param agent: 대화를 이끄는 :class:`~docagent.agent.orchestrator.DocumentAgent`.
    :param normalized: 정합 결과. 좌표계·호모그래피의 출처.
    :param audit: 외부 전송 감사 로그. ``counters()["pii_leaked"]`` 가 안전 KPI.
    :param config: 이 세션을 만든 설정.
    :param gate: 개인정보 유출 차단 게이트.
    :param public_payload: LLM 에 나갈 수 있는 공개 영역 payload(좌표 없음).
    :param stages: 단계별 신뢰도 보고.
    :param detections: 탐지 원본(디버깅·KPI 용).
    :param words: OCR 단어 원본(디버깅용).
    :param verifier: Verify 어댑터. 이미지 기반이면 :class:`ImageVerifier`.
    :param motion: 펜 제어기.
    :param speech: 음성 입출력.
    :param retriever: 근거 검색기.
    :param llm: 게이트로 감싸인 LLM 클라이언트.
    """

    structure: DocumentStructure
    agent: DocumentAgent
    normalized: NormalizedDocument
    audit: AuditLog
    config: DocAgentConfig
    gate: LlmEgressGate
    public_payload: dict[str, Any]
    stages: tuple[StageReport, ...] = ()
    detections: tuple[Detection, ...] = ()
    words: tuple[OcrWord, ...] = ()
    verifier: Any | None = None
    motion: Any | None = None
    speech: Any | None = None
    retriever: LocalTfidfRetriever | None = None
    llm: Any | None = None
    turns: list[AgentTurn] = dc_field(default_factory=list)

    # ------------------------------------------------------------------
    # 대화
    # ------------------------------------------------------------------

    def start(self) -> AgentTurn:
        """세션을 시작하고 첫 안내를 낸다.

        :returns: :class:`~docagent.contracts.AgentTurn`.
        """
        turn = self.agent.start()
        self.turns.append(turn)
        return turn

    def handle(self, user_text: str) -> AgentTurn:
        """사용자 발화 1건을 처리한다.

        :param user_text: 사용자 발화(STT 결과).
        :returns: :class:`~docagent.contracts.AgentTurn`.
        """
        turn = self.agent.handle(user_text)
        self.turns.append(turn)
        return turn

    def run(self, script: Sequence[str]) -> list[AgentTurn]:
        """발화 목록을 순서대로 처리한다.

        :param script: 사용자 발화 목록.
        :returns: 처리 결과 턴 목록.
        """
        return [self.handle(line) for line in script]

    # ------------------------------------------------------------------
    # Verify 연동
    # ------------------------------------------------------------------

    def update_written_image(self, raw_image: Any, *, aligned: bool = False) -> None:
        """기입 후 이미지를 세션에 반영한다.

        :param raw_image: 다시 촬영한 원본 이미지, 또는 이미 정합된 이미지.
        :param aligned: True 면 ``raw_image`` 가 이미 정합 좌표계라고 보고
            정렬 단계를 건너뛴다.
        :returns: ``None``.
        :raises docagent.errors.VisionError: 정렬에 실패했거나 크기가 다른 경우.
        :raises TypeError: 이미지 기반 검증기가 연결되어 있지 않은 경우.
        """
        if not isinstance(self.verifier, ImageVerifier):
            raise TypeError(
                "이미지 기반 검증기(ImageVerifier)가 연결되어 있지 않아 "
                "기입 후 이미지를 반영할 수 없습니다."
            )
        image = raw_image if aligned else align_to_page(self.normalized, raw_image)
        self.verifier.update(image)

    # ------------------------------------------------------------------
    # 요약
    # ------------------------------------------------------------------

    @property
    def document_confidence(self) -> float:
        """단계 신뢰도 중 가장 낮은 값(문서 단위 신뢰도).

        :returns: 0.0~1.0. 단계 보고가 없으면 0.0.
        """
        if not self.stages:
            return 0.0
        return min(report.confidence for report in self.stages)

    def low_confidence_stages(self) -> tuple[StageReport, ...]:
        """임계값을 만족하지 못한 단계 보고만 돌려준다.

        :returns: :class:`StageReport` 튜플.
        """
        return tuple(report for report in self.stages if not report.ok)

    def notices(self) -> tuple[str, ...]:
        """세션 시작 시 사용자에게 알려야 할 한국어 안내 문장.

        문서 구조의 ``warnings`` 만 돌려준다(정합·탐지 경고는 조립 시점에
        여기로 합쳐진다). 단계별 신뢰도 설명은 :meth:`low_confidence_stages`
        가 따로 제공하므로 여기에 섞지 않는다 — 같은 문장이 두 번 낭독되면
        음성 인터페이스에서 특히 거슬린다.

        :returns: 안내 문장 튜플(중복 제거). 알릴 것이 없으면 빈 튜플.
        """
        seen: set[str] = set()
        unique: list[str] = []
        for message in self.structure.warnings:
            if message and message not in seen:
                seen.add(message)
                unique.append(message)
        return tuple(unique)

    def summary(self) -> dict[str, Any]:
        """세션 상태를 기계 판독 가능한 dict 로 돌려준다.

        **개인정보를 담지 않는다.** 항목 제목·약관 문구·기입 값은 들어가지 않고
        식별자·유형·진행 상태·신뢰도만 남는다.

        :returns: JSON 직렬화 가능한 dict.
        """
        done, total, remaining = self.agent.state.progress()
        return {
            "document_id": self.structure.document_id,
            "phase": self.agent.state.phase.value,
            "fields_total": total,
            "fields_completed": done,
            "required_remaining": list(remaining),
            "turns": len(self.turns),
            "document_confidence": round(self.document_confidence, 4),
            "stages": [report.to_dict() for report in self.stages],
            "audit": self.audit.counters(),
            "handoff_reason": self.agent.state.handoff_reason,
            "moves": [
                point.to_dict() for point in getattr(self.motion, "moves", ())
            ],
        }

    def session_json(self, *, indent: int | None = 2) -> str:
        """세션 상태를 JSON 문자열로 내보낸다(복원용).

        :param indent: JSON 들여쓰기.
        :returns: 세션 상태 JSON.
        """
        return self.agent.session_json(indent=indent)

    def snapshot_json(self, *, indent: int | None = 2) -> str:
        """문서 구조와 세션 상태를 **한 봉투**에 담아 내보낸다(복원용).

        세션 상태만 저장하면 복원에 필수인 :class:`~docagent.contracts.DocumentStructure`
        가 없어서 되살릴 수 없다. 저장·복원을 대칭으로 만들기 위해 둘을 함께 담는다.

        :param indent: JSON 들여쓰기.
        :returns: ``{"version", "structure", "state"}`` 를 담은 JSON 문자열.
        """
        envelope = {
            "version": SNAPSHOT_VERSION,
            "structure": json.loads(self.structure.to_json(indent=None)),
            "state": json.loads(self.session_json(indent=None)),
        }
        return json.dumps(envelope, ensure_ascii=False, indent=indent)

    @staticmethod
    def restore_agent(snapshot_json: str, **deps: Any) -> DocumentAgent:
        """:meth:`snapshot_json` 이 만든 봉투로부터 에이전트를 복원한다.

        :param snapshot_json: :meth:`snapshot_json` 결과 문자열.
        :param deps: ``explainer`` / ``guard`` / ``speech`` / ``motion`` /
            ``verifier`` / ``clock`` / ``classifier`` 주입 인자.
        :returns: 복원된 :class:`~docagent.agent.orchestrator.DocumentAgent`.
            대화 턴 기록(``turns``)은 복원되지 않는다. 감사 재료의 정본은
            ``state.history`` 이며 그쪽은 그대로 복원된다.
        :raises ValueError: 봉투가 깨졌거나 필수 키가 없는 경우(조용한 실패 금지).
        """
        try:
            envelope = json.loads(snapshot_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"세션 스냅숏을 JSON 으로 읽을 수 없습니다: {exc}") from exc
        if not isinstance(envelope, dict):
            raise ValueError("세션 스냅숏은 JSON 객체여야 합니다.")
        missing = [key for key in ("structure", "state") if key not in envelope]
        if missing:
            raise ValueError(
                f"세션 스냅숏에 필수 키가 없습니다: {', '.join(missing)}"
            )
        structure = DocumentStructure.from_json(
            json.dumps(envelope["structure"], ensure_ascii=False)
        )
        return DocumentAgent.restore(
            structure,
            json.dumps(envelope["state"], ensure_ascii=False),
            **deps,
        )


# --------------------------------------------------------------------------
# 조립
# --------------------------------------------------------------------------


def _resolve_ocr(ocr: Any | None, warnings: list[str]) -> Any:
    """OCR 엔진을 확정한다.

    ``pytesseract`` 는 설치되어 있지 않다. 그래서 기본값은 **글자를 하나도
    읽지 못하는 빈 :class:`StubOcr`** 이며, 그 사실을 경고로 남긴다.
    조용히 "OCR 결과가 없다"로 넘어가면 항목명이 비어 있는 이유를 알 수 없다.

    :param ocr: 주입된 OCR 엔진. ``None`` 이면 기본값을 만든다.
    :param warnings: 경고를 덧붙일 리스트(제자리 수정).
    :returns: :class:`~docagent.interfaces.OcrEngine` 구현.
    """
    if ocr is not None:
        return ocr
    warnings.append(NO_OCR_WARNING)
    _LOG.info("OCR 엔진이 주입되지 않아 빈 StubOcr 로 진행합니다.")
    return StubOcr(())


def _ensure_gated(llm: Any, gate: LlmEgressGate) -> GatedLlmClient:
    """LLM 클라이언트가 반드시 이 세션의 게이트를 지나가게 만든다.

    래퍼는 :class:`docagent.pii.gate.GatedLlmClient` 하나뿐이며 마스킹 +
    **마스킹 후 재검사** + 감사 기록을 모두 수행한다. 파이프라인은 안전 KPI
    ("개인정보 LLM 전송 0건")를 감사 로그로 증명해야 하므로, 다른 게이트로
    감싸인 클라이언트가 들어오면 세션 게이트로 다시 감싼다(이중 마스킹은
    안전 쪽으로만 작용한다).

    :param llm: 날 것의 클라이언트 또는 이미 감싸인 클라이언트.
    :param gate: 이 세션의 게이트.
    :returns: :class:`~docagent.pii.gate.GatedLlmClient`.
    :raises TypeError: ``complete`` 를 갖지 않은 객체를 준 경우.
    """
    if isinstance(llm, GatedLlmClient) and llm.gate is gate:
        return llm
    if isinstance(llm, GatedLlmClient):
        _LOG.info("다른 게이트로 감싸인 LLM 이 들어와 세션 게이트로 다시 감쌉니다.")
    return GatedLlmClient(inner=llm, gate=gate)


def _build_inner_llm(cfg: DocAgentConfig) -> Any:
    """설정에 맞는 **게이트로 감싸기 전** LLM 구현을 만든다.

    :param cfg: 조립 설정.
    :returns: :class:`~docagent.interfaces.LlmClient` 구현.
    :raises docagent.errors.AdapterUnavailable: ``llm_kind="claude"`` 인데
        ``anthropic`` 이 설치되어 있지 않은 경우.
    """
    if cfg.llm_kind == "claude":
        return ClaudeLlm(model=cfg.llm_model)
    return OfflineTemplateLlm()


def build_session(
    image: Any,
    *,
    detector: Any | None = None,
    ocr: Any | None = None,
    dpi: int | None = None,
    corpus_dir: Path | str | None = None,
    llm: Any | None = None,
    motion: Any | None = None,
    speech: Any | None = None,
    config: DocAgentConfig | None = None,
    document_id: str = DEFAULT_DOCUMENT_ID,
    doc_title: str = "",
    source_image: str | None = None,
    gate: LlmEgressGate | None = None,
    audit: AuditLog | None = None,
    retriever: LocalTfidfRetriever | None = None,
    verifier: Any | None = None,
    clock: Any | None = None,
    classifier: Any | None = None,
    propagate_document_confidence: bool = False,
) -> DocumentSession:
    """문서 이미지 한 장으로 대화 가능한 세션을 만든다.

    :param image: 촬영·스캔 원본 이미지. ``(H, W)`` 또는 ``(H, W, 3)`` uint8.
    :param detector: 기입란 탐지기. ``None`` 이면
        :func:`~docagent.vision.detector_factory.build_detector` 로 만든다.
    :param ocr: OCR 엔진. ``None`` 이면 빈 :class:`StubOcr`(경고 기록).
    :param dpi: 정합 결과 해상도. ``None`` 이면 ``config.dpi``.
    :param corpus_dir: 근거 코퍼스 디렉터리. ``None`` 이면 ``config`` 값.
    :param llm: LLM 클라이언트. ``None`` 이면 오프라인 구현. 어느 쪽이든
        **반드시 게이트로 감싸서** 에이전트에 넘긴다.
    :param motion: 펜 제어기. ``None`` 이면 :class:`~docagent.io.motion.MockMotionController`.
    :param speech: 음성 입출력. ``None`` 이면 :class:`~docagent.io.speech.ScriptedSpeechIO`.
    :param config: 조립 설정. ``None`` 이면 기본 :class:`~docagent.config.DocAgentConfig`.
    :param document_id: 문서 인스턴스 식별자(세션 복원 키).
    :param doc_title: 문서 제목. 빈 문자열이면 OCR 최상단 행에서 유추한다.
    :param source_image: 원본 이미지 경로·식별자(감사 표기용).
    :param gate: 개인정보 게이트. ``None`` 이면 새로 만든다.
    :param audit: 감사 로그. ``None`` 이면 ``config.audit_path`` 로 만든다.
    :param retriever: 근거 검색기. ``None`` 이면 코퍼스로 색인을 만든다.
    :param verifier: Verify 구현. ``None`` 이면 :class:`ImageVerifier`.
    :param clock: :class:`~docagent.interfaces.Clock`. ``None`` 이면 에이전트 기본 시계.
    :param classifier: 의도 분류기. ``None`` 이면 규칙 기반 분류기.
    :param propagate_document_confidence: True 면 문서 단위 신뢰도를 항목별
        신뢰도에 곱해 전파한다(더 엄격한 배치용, 기본 False).
    :returns: :class:`DocumentSession`.
    :raises docagent.errors.DocumentNotFoundError: 이미지에서 문서를 찾지 못했거나
        탐지·OCR 결과가 모두 비어 구조를 만들 수 없는 경우.
    :raises docagent.errors.VisionError: 정합·탐지 단계가 실패한 경우.
    :raises docagent.agent.rag.CorpusError: 근거 코퍼스를 읽지 못한 경우.
    """
    cfg = config if config is not None else DocAgentConfig()
    resolved_dpi = int(dpi) if dpi is not None else cfg.dpi
    page_size_mm = cfg.page_size_mm

    stage_warnings: list[str] = []
    stages: list[StageReport] = []

    # --- 1. 정합 ------------------------------------------------------
    normalized = normalize_document(image, dpi=resolved_dpi)
    stage_warnings.extend(normalized.warnings)
    stages.append(
        StageReport(
            stage="normalize",
            confidence=float(normalized.confidence),
            ok=not normalized.low_confidence,
            detail=(
                f"문서 정합 신뢰도 {normalized.confidence:.2f}, "
                f"기대 좌표 오차 {normalized.expected_error_mm:.1f}mm, "
                f"기울기 {normalized.skew_deg:+.1f}도."
            ),
        )
    )

    # --- 2. 탐지 ------------------------------------------------------
    engine = (
        detector
        if detector is not None
        else build_detector(cfg.detector_weights, cfg.detector_prefer)
    )
    detections = tuple(engine.detect(normalized.image))
    detect_confidence = (
        min(item.confidence for item in detections) if detections else 0.0
    )
    stages.append(
        StageReport(
            stage="detect",
            confidence=float(detect_confidence),
            ok=bool(detections) and detect_confidence >= VISION_TRUST_THRESHOLD,
            detail=(
                f"기입란 후보 {len(detections)}건을 찾았습니다"
                f"(최저 신뢰도 {detect_confidence:.2f})."
                if detections
                else "기입란 후보를 하나도 찾지 못했습니다."
            ),
        )
    )

    # --- 3. OCR -------------------------------------------------------
    reader = _resolve_ocr(ocr, stage_warnings)
    words = tuple(reader.read(normalized.image))

    # --- 4. 구조화 ----------------------------------------------------
    if not detections and not words:
        raise DocumentNotFoundError(
            "탐지 결과와 OCR 결과가 모두 비어 문서 구조를 만들 수 없습니다.",
            source=source_image,
        )
    structure = build_structure(
        detections,
        words,
        page_size_mm,
        document_id,
        doc_title=doc_title,
        source_image=source_image,
    )

    # --- 5. PII 정책 재판정 -------------------------------------------
    structure, downgraded = apply_pii_policy(structure)
    field_confidence = (
        min(item.confidence for item in structure.fields) if structure.fields else 0.0
    )
    stages.append(
        StageReport(
            stage="structure",
            confidence=float(field_confidence),
            ok=bool(structure.fields) and field_confidence >= VISION_TRUST_THRESHOLD,
            detail=(
                f"작성 항목 {len(structure.fields)}개를 확정했습니다"
                f"(최저 해석 신뢰도 {field_confidence:.2f})."
                if structure.fields
                else "작성 항목을 확정하지 못했습니다."
            ),
        )
    )

    # 정합·탐지 경고를 문서 구조에 합쳐 사용자 안내에 쓴다.
    merged_warnings = tuple(dict.fromkeys((*structure.warnings, *stage_warnings)))
    structure = DocumentStructure(
        document_id=structure.document_id,
        doc_title=structure.doc_title,
        fields=structure.fields,
        page_size_mm=structure.page_size_mm,
        source_image=structure.source_image,
        warnings=merged_warnings,
    )

    document_confidence = min(report.confidence for report in stages)
    if propagate_document_confidence:
        structure = _scale_confidence(structure, document_confidence)

    # --- 6. 게이트 · 공개 payload -------------------------------------
    resolved_audit = (
        audit
        if audit is not None
        else AuditLog(cfg.audit_path) if cfg.audit_path is not None else AuditLog()
    )
    resolved_gate = (
        gate
        if gate is not None
        else LlmEgressGate(strict=cfg.strict_pii, audit=resolved_audit)
    )
    payload = build_public_payload(structure)
    # 게이트를 한 번 더 통과시켜 "공개 영역이라고 믿었는데 아니었던" 경우를 잡는다.
    guarded_payload = resolved_gate.guard(payload, caller="pipeline.public_payload")
    stages.append(
        StageReport(
            stage="pii",
            confidence=1.0,
            ok=True,
            detail=(
                f"공개 영역 {len(payload['fields']) - len(payload['redacted_field_ids'])}개, "
                f"개인정보 강등 {len(payload['redacted_field_ids'])}개"
                + (f" (정책 재판정 {len(downgraded)}개)" if downgraded else "")
                + "."
            ),
        )
    )

    # --- 7. RAG ------------------------------------------------------
    resolved_corpus = (
        Path(corpus_dir) if corpus_dir is not None else cfg.resolved_corpus_dir()
    )
    resolved_retriever = (
        retriever
        if retriever is not None
        else build_index(resolved_corpus, cache_path=cfg.index_cache_path)
    )

    # --- 8. Explainer / Guard / LLM ----------------------------------
    inner_llm = llm if llm is not None else _build_inner_llm(cfg)
    gated_llm: Any = _ensure_gated(inner_llm, resolved_gate)
    guard = Guard(resolved_gate)
    explainer = Explainer(
        resolved_retriever, gated_llm, guard, clock, top_k=cfg.explain_top_k
    )

    # --- 9. 액추에이터 · 음성 · 검증 ----------------------------------
    resolved_motion = (
        motion if motion is not None else MockMotionController(page_size_mm)
    )
    resolved_speech = speech if speech is not None else ScriptedSpeechIO()
    resolved_verifier = (
        verifier
        if verifier is not None
        else ImageVerifier(structure, normalized.image, page_size_mm=page_size_mm)
    )
    if isinstance(resolved_verifier, ImageVerifier):
        resolved_verifier.rebind(structure)

    # --- 10. 도구 · 에이전트 ------------------------------------------
    state = SessionState.from_structure(structure)
    tools = ToolRegistry(
        structure,
        state,
        motion=resolved_motion,
        explainer=explainer,
        verifier=resolved_verifier,
        speech=resolved_speech,
        motion_tolerance_mm=cfg.motion_tolerance_mm,
    )
    # 주입된 의도 분류기도 세션 게이트를 지나게 만든다. LLM 분류기는 사용자
    # 발화 원문을 프롬프트로 보내므로, llm 인자와 동일한 규율을 적용한다.
    if classifier is not None and hasattr(classifier, "bind_gate"):
        classifier.bind_gate(resolved_gate)

    agent = DocumentAgent(
        structure,
        state=state,
        tools=tools,
        explainer=explainer,
        guard=guard,
        speech=resolved_speech,
        motion=resolved_motion,
        verifier=resolved_verifier,
        clock=clock,
        classifier=classifier,
    )

    return DocumentSession(
        structure=structure,
        agent=agent,
        normalized=normalized,
        audit=resolved_audit,
        config=cfg,
        gate=resolved_gate,
        public_payload=guarded_payload,
        stages=tuple(stages),
        detections=detections,
        words=words,
        verifier=resolved_verifier,
        motion=resolved_motion,
        speech=resolved_speech,
        retriever=resolved_retriever,
        llm=gated_llm,
    )
