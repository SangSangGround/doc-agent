"""통합 설정 — 6개 모듈을 한 벌로 조립할 때 쓰는 단일 설정 객체.

각 모듈은 자기 임계값을 자기 모듈 상수로 갖는다. 이 모듈은 그것들을 **다시
정의하지 않는다.** 여기 있는 것은 "조립 시점에 정해야 하는 값"(해상도·경로·
어댑터 선택·재시도 횟수)뿐이며, 판정 임계값은 전부
:mod:`docagent.contracts` 와 각 모듈 상수를 그대로 참조한다.

설계 원칙
---------
* **결정론** — 기본값에 난수·현재 시각이 들어가지 않는다. 같은 설정이면
  같은 세션이 만들어진다.
* **경로는 pathlib** — POSIX 경로를 하드코딩하지 않는다. 저장소 루트는 이
  파일 위치에서 역산한다(``src/docagent/config.py`` → 두 단계 위).
* **선택적 패키지 없음** — 이 모듈은 표준 라이브러리만 import 한다.

환경변수(:meth:`DocAgentConfig.from_env`)로 덮어쓸 수 있는 값은 접두사
``DOCAGENT_`` 를 쓴다. 환경변수를 읽지 않는 것이 기본값이므로, 테스트는
:class:`DocAgentConfig` 를 직접 만들어 쓴다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from docagent.contracts import (
    A4_PAGE_SIZE_MM,
    EXPLAIN_THRESHOLD,
    MOTION_TOLERANCE_MM,
    PARTIAL_THRESHOLD,
    VISION_TRUST_THRESHOLD,
)

__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_LLM_KIND",
    "DEFAULT_DETECTOR_PREFERENCE",
    "ENV_PREFIX",
    "DocAgentConfig",
    "repo_root",
    "package_root",
    "default_corpus_dir",
    "default_audit_path",
    "venv_python_path",
    "thresholds",
]


#: 정합(normalize) 기본 해상도. 300dpi 면 1mm 가 약 11.8px 이라
#: 2.5mm 짜리 체크칸도 30px 가까이 되어 윤곽선 근사가 안정적이다.
DEFAULT_DPI: int = 300

#: 기본 LLM 종류. 네트워크·API 키 없이 100% 동작해야 하므로 오프라인이 기본이다.
DEFAULT_LLM_KIND: str = "offline"

#: 기본 탐지기 선택 전략(:func:`docagent.vision.detector_factory.build_detector`).
DEFAULT_DETECTOR_PREFERENCE: str = "auto"

#: 환경변수 접두사.
ENV_PREFIX: str = "DOCAGENT_"


def package_root() -> Path:
    """``docagent`` 패키지 디렉터리를 돌려준다.

    :returns: ``<repo>/src/docagent`` 경로.
    """
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """저장소 루트를 돌려준다.

    ``src/docagent/config.py`` 기준으로 두 단계 위가 저장소 루트다.

    :returns: 저장소 루트 :class:`~pathlib.Path`.
    """
    return package_root().parent.parent


def default_corpus_dir() -> Path:
    """근거 코퍼스 기본 디렉터리를 돌려준다.

    :returns: ``<repo>/data/corpus`` 경로.
    """
    return repo_root() / "data" / "corpus"


def default_audit_path() -> Path:
    """감사 로그 JSONL 기본 경로를 돌려준다.

    :returns: ``<repo>/data/audit/egress.jsonl`` 경로.
    """
    return repo_root() / "data" / "audit" / "egress.jsonl"


def venv_python_path() -> Path:
    """저장소 venv 인터프리터 경로를 돌려준다(오류 안내 문구용).

    :returns: ``<repo>/.venv/Scripts/python.exe`` 경로(Windows 기준).
    """
    return repo_root() / ".venv" / "Scripts" / "python.exe"


def thresholds() -> dict[str, float]:
    """판정 임계값 모음을 돌려준다(계약에서 그대로 읽어 온다).

    이 함수는 값을 **새로 정의하지 않는다.** 리포트·로그에 임계값을 함께
    남길 때 한 곳에서 꺼내 쓰기 위한 편의 함수다.

    :returns: ``vision_trust`` / ``explain`` / ``partial`` 키를 갖는 dict.
    """
    return {
        "vision_trust": VISION_TRUST_THRESHOLD,
        "explain": EXPLAIN_THRESHOLD,
        "partial": PARTIAL_THRESHOLD,
    }


def _as_bool(value: str) -> bool:
    """환경변수 문자열을 불리언으로 바꾼다.

    :param value: 환경변수 값.
    :returns: ``1/true/yes/on`` (대소문자 무시)이면 True.
    """
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class DocAgentConfig:
    """파이프라인 조립 설정.

    :param dpi: 정합 결과 이미지의 해상도(dpi, 72 이상).
    :param page_size_mm: 문서 페이지 크기 ``(가로_mm, 세로_mm)``.
        **이번 범위는 A4 전용**이며 다른 값은 :class:`ValueError` 로 거부한다
        (탐지 좌표 환산이 A4 로 고정되어 있어 좌표계가 갈라지기 때문이다).
    :param corpus_dir: 근거 코퍼스 디렉터리. ``None`` 이면
        :func:`default_corpus_dir`.
    :param audit_path: 감사 로그 JSONL 경로. ``None`` 이면 메모리에만 남긴다
        (데모·테스트 기본값. 파일을 남기지 않아도 카운터는 그대로 집계된다).
    :param index_cache_path: RAG 색인 캐시 JSON 경로. ``None`` 이면 캐시 없음.
    :param llm_kind: ``"offline"`` 또는 ``"claude"``.
    :param llm_model: ``llm_kind="claude"`` 일 때 쓸 모델 id.
    :param detector_weights: YOLO 가중치 경로. ``None`` 이면 규칙 기반.
    :param detector_prefer: ``"auto"`` / ``"yolo"`` / ``"heuristic"``.
    :param motion_tolerance_mm: 펜 도달 판정 허용 오차(mm, 0 초과).
    :param serial_port: 시리얼 포트 이름(예: ``"COM3"``). ``None`` 이면 하드웨어 미사용.
    :param serial_baudrate: 시리얼 통신 속도.
    :param serial_timeout_s: 시리얼 응답 대기 시간(초).
    :param serial_retries: 이동 명령 재시도 횟수(0 이상).
    :param explain_top_k: 설명 1건당 검색할 근거 청크 수(1 이상).
    :param strict_pii: PII 게이트를 fail-closed(strict)로 둘지 여부.
        운영에서 False 로 두는 것은 권장하지 않는다.
    :raises ValueError: 수치 설정이 허용 범위를 벗어난 경우.
    """

    dpi: int = DEFAULT_DPI
    page_size_mm: tuple[float, float] = A4_PAGE_SIZE_MM
    corpus_dir: Path | None = None
    audit_path: Path | None = None
    index_cache_path: Path | None = None
    llm_kind: str = DEFAULT_LLM_KIND
    llm_model: str = "claude-sonnet-5"
    detector_weights: Path | None = None
    detector_prefer: str = DEFAULT_DETECTOR_PREFERENCE
    motion_tolerance_mm: float = MOTION_TOLERANCE_MM
    serial_port: str | None = None
    serial_baudrate: int = 115200
    serial_timeout_s: float = 5.0
    serial_retries: int = 2
    explain_top_k: int = 4
    strict_pii: bool = True

    def __post_init__(self) -> None:
        if self.dpi < 72:
            raise ValueError(f"dpi 는 72 이상이어야 합니다: {self.dpi}")
        page = tuple(float(v) for v in self.page_size_mm)
        if len(page) != 2 or page[0] <= 0 or page[1] <= 0:
            raise ValueError(
                f"page_size_mm 은 0 보다 큰 (가로_mm, 세로_mm) 이어야 합니다: {self.page_size_mm!r}"
            )
        # 이번 범위는 **A4 전용**이다. 탐지 경로의 좌표 환산
        # (:func:`docagent.vision.geometry.coordinate_system_from_image`)이 A4 로
        # 고정되어 있어, 다른 용지 크기를 설정하면 탐지 좌표(A4 기준)와 구조·검증·
        # 펜 이동 좌표(설정값 기준)가 어긋난다. 오차는 위치에 비례해 커지고
        # 허용 오차(MOTION_TOLERANCE_MM)를 쉽게 넘기므로 조용히 진행하지 않는다.
        if page != tuple(float(v) for v in A4_PAGE_SIZE_MM):
            raise ValueError(
                "이번 범위는 A4 문서만 지원합니다. page_size_mm 은 "
                f"{A4_PAGE_SIZE_MM} 이어야 합니다: {self.page_size_mm!r}. "
                "다른 용지를 지원하려면 Vision 의 좌표 환산부터 페이지 크기를 "
                "인자로 받도록 일반화해야 합니다."
            )
        object.__setattr__(self, "page_size_mm", page)
        if self.llm_kind not in ("offline", "claude"):
            raise ValueError(
                f"llm_kind 는 'offline' 또는 'claude' 여야 합니다: {self.llm_kind!r}"
            )
        if self.detector_prefer not in ("auto", "yolo", "heuristic"):
            raise ValueError(
                "detector_prefer 는 auto, yolo, heuristic 중 하나여야 합니다: "
                f"{self.detector_prefer!r}"
            )
        if self.motion_tolerance_mm <= 0:
            raise ValueError(
                f"motion_tolerance_mm 은 0 보다 커야 합니다: {self.motion_tolerance_mm}"
            )
        if self.serial_baudrate <= 0:
            raise ValueError(f"serial_baudrate 는 0 보다 커야 합니다: {self.serial_baudrate}")
        if self.serial_timeout_s <= 0:
            raise ValueError(f"serial_timeout_s 는 0 보다 커야 합니다: {self.serial_timeout_s}")
        if self.serial_retries < 0:
            raise ValueError(f"serial_retries 는 0 이상이어야 합니다: {self.serial_retries}")
        if self.explain_top_k < 1:
            raise ValueError(f"explain_top_k 는 1 이상이어야 합니다: {self.explain_top_k}")
        for name in ("corpus_dir", "audit_path", "index_cache_path", "detector_weights"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

    # ------------------------------------------------------------------

    def resolved_corpus_dir(self) -> Path:
        """실제로 사용할 코퍼스 디렉터리를 돌려준다.

        :returns: :attr:`corpus_dir` 또는 :func:`default_corpus_dir`.
        """
        return self.corpus_dir if self.corpus_dir is not None else default_corpus_dir()

    def with_(self, **changes: Any) -> "DocAgentConfig":
        """일부 값만 바꾼 새 설정을 만든다(frozen dataclass 이므로 사본).

        :param changes: 바꿀 필드.
        :returns: 새 :class:`DocAgentConfig`.
        :raises TypeError: 정의되지 않은 필드 이름을 준 경우.
        """
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 돌려준다(경로는 문자열).

        :returns: 설정 dict.
        """
        return {
            "dpi": self.dpi,
            "page_size_mm": list(self.page_size_mm),
            "corpus_dir": None if self.corpus_dir is None else str(self.corpus_dir),
            "audit_path": None if self.audit_path is None else str(self.audit_path),
            "index_cache_path": (
                None if self.index_cache_path is None else str(self.index_cache_path)
            ),
            "llm_kind": self.llm_kind,
            "llm_model": self.llm_model,
            "detector_weights": (
                None if self.detector_weights is None else str(self.detector_weights)
            ),
            "detector_prefer": self.detector_prefer,
            "motion_tolerance_mm": self.motion_tolerance_mm,
            "serial_port": self.serial_port,
            "serial_baudrate": self.serial_baudrate,
            "serial_timeout_s": self.serial_timeout_s,
            "serial_retries": self.serial_retries,
            "explain_top_k": self.explain_top_k,
            "strict_pii": self.strict_pii,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DocAgentConfig":
        """:meth:`to_dict` 결과로부터 설정을 복원한다.

        :param data: 설정 dict.
        :returns: :class:`DocAgentConfig`.
        """
        defaults = cls()
        page = data.get("page_size_mm", defaults.page_size_mm)
        return cls(
            dpi=int(data.get("dpi", defaults.dpi)),
            page_size_mm=(float(page[0]), float(page[1])),
            corpus_dir=_opt_path(data.get("corpus_dir")),
            audit_path=_opt_path(data.get("audit_path")),
            index_cache_path=_opt_path(data.get("index_cache_path")),
            llm_kind=str(data.get("llm_kind", defaults.llm_kind)),
            llm_model=str(data.get("llm_model", defaults.llm_model)),
            detector_weights=_opt_path(data.get("detector_weights")),
            detector_prefer=str(data.get("detector_prefer", defaults.detector_prefer)),
            motion_tolerance_mm=float(
                data.get("motion_tolerance_mm", defaults.motion_tolerance_mm)
            ),
            serial_port=_opt_str(data.get("serial_port")),
            serial_baudrate=int(data.get("serial_baudrate", defaults.serial_baudrate)),
            serial_timeout_s=float(data.get("serial_timeout_s", defaults.serial_timeout_s)),
            serial_retries=int(data.get("serial_retries", defaults.serial_retries)),
            explain_top_k=int(data.get("explain_top_k", defaults.explain_top_k)),
            strict_pii=bool(data.get("strict_pii", defaults.strict_pii)),
        )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, base: "DocAgentConfig | None" = None
    ) -> "DocAgentConfig":
        """``DOCAGENT_*`` 환경변수로 기본 설정을 덮어쓴다.

        읽는 변수: ``DOCAGENT_DPI`` / ``DOCAGENT_CORPUS_DIR`` /
        ``DOCAGENT_AUDIT_PATH`` / ``DOCAGENT_INDEX_CACHE`` / ``DOCAGENT_LLM_KIND`` /
        ``DOCAGENT_LLM_MODEL`` / ``DOCAGENT_DETECTOR_WEIGHTS`` /
        ``DOCAGENT_DETECTOR_PREFER`` / ``DOCAGENT_SERIAL_PORT`` /
        ``DOCAGENT_SERIAL_BAUDRATE`` / ``DOCAGENT_STRICT_PII``.

        :param environ: 환경변수 매핑. ``None`` 이면 :data:`os.environ`
            (테스트는 dict 를 주입해 결정론을 확보한다).
        :param base: 덮어쓸 기준 설정. ``None`` 이면 기본 설정.
        :returns: :class:`DocAgentConfig`.
        :raises ValueError: 숫자 변수의 값이 정수·실수로 해석되지 않는 경우.
        """
        env = os.environ if environ is None else environ
        config = base if base is not None else cls()
        changes: dict[str, Any] = {}

        raw_dpi = env.get(ENV_PREFIX + "DPI")
        if raw_dpi:
            try:
                changes["dpi"] = int(raw_dpi)
            except ValueError as exc:
                raise ValueError(
                    f"{ENV_PREFIX}DPI 는 정수여야 합니다: {raw_dpi!r}"
                ) from exc
        for env_name, field_name in (
            ("CORPUS_DIR", "corpus_dir"),
            ("AUDIT_PATH", "audit_path"),
            ("INDEX_CACHE", "index_cache_path"),
            ("DETECTOR_WEIGHTS", "detector_weights"),
        ):
            raw = env.get(ENV_PREFIX + env_name)
            if raw:
                changes[field_name] = Path(raw)
        for env_name, field_name in (
            ("LLM_KIND", "llm_kind"),
            ("LLM_MODEL", "llm_model"),
            ("DETECTOR_PREFER", "detector_prefer"),
            ("SERIAL_PORT", "serial_port"),
        ):
            raw = env.get(ENV_PREFIX + env_name)
            if raw:
                changes[field_name] = raw
        raw_baud = env.get(ENV_PREFIX + "SERIAL_BAUDRATE")
        if raw_baud:
            try:
                changes["serial_baudrate"] = int(raw_baud)
            except ValueError as exc:
                raise ValueError(
                    f"{ENV_PREFIX}SERIAL_BAUDRATE 는 정수여야 합니다: {raw_baud!r}"
                ) from exc
        raw_strict = env.get(ENV_PREFIX + "STRICT_PII")
        if raw_strict:
            changes["strict_pii"] = _as_bool(raw_strict)
        return config.with_(**changes)


def _opt_path(value: Any) -> Path | None:
    """``None`` 을 허용하는 경로 변환 헬퍼.

    :param value: 경로 문자열 또는 ``None``.
    :returns: :class:`~pathlib.Path` 또는 ``None``.
    """
    return None if value is None else Path(str(value))


def _opt_str(value: Any) -> str | None:
    """``None`` 을 허용하는 문자열 변환 헬퍼.

    :param value: 문자열 또는 ``None``.
    :returns: 문자열 또는 ``None``.
    """
    return None if value is None else str(value)
