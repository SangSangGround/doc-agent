"""한글 폰트 탐색·로딩 유틸리티 — 합성 문서 렌더링 전용.

합성 신청서 이미지는 한글 텍스트를 그려야 하므로 시스템에 설치된
한글 트루타입 폰트를 찾아 쓴다. 다만 **폰트를 찾지 못해도 테스트가 실패해서는
안 된다.** 이 경우 Pillow 기본 비트맵 폰트로 폴백하고 한국어 경고를 한 번만
발생시킨다. 도형(표 괘선·체크박스 테두리·서명 밑줄) 기반 검증은 폰트와
무관하게 항상 유효하므로, Vision 파이프라인 테스트는 폰트 없이도 성립한다.

탐색 순서
---------
1. 환경변수 ``DOCAGENT_FONT_PATH`` (테스트에서 특정 폰트를 강제할 때 사용)
2. Windows 기본 한글 폰트(맑은 고딕 → 굴림 → 바탕)
3. 나눔고딕 등 리눅스/맥 배포 폰트
4. 전부 실패하면 :func:`PIL.ImageFont.load_default`

이 모듈은 표준 라이브러리와 Pillow 만 사용한다.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import ImageFont

__all__ = [
    "FONT_PATH_ENV",
    "REGULAR_FONT_CANDIDATES",
    "BOLD_FONT_CANDIDATES",
    "MissingKoreanFontWarning",
    "find_font_path",
    "load_font",
    "korean_font_available",
    "font_diagnostics",
]

#: 폰트 경로를 강제 지정하는 환경변수 이름.
FONT_PATH_ENV: str = "DOCAGENT_FONT_PATH"

#: 일반(regular) 굵기 후보 경로. 위에서부터 순서대로 탐색한다.
REGULAR_FONT_CANDIDATES: tuple[Path, ...] = (
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/gulim.ttc"),
    Path("C:/Windows/Fonts/batang.ttc"),
    Path("C:/Windows/Fonts/NanumGothic.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
)

#: 굵은(bold) 굵기 후보 경로. 실패 시 :data:`REGULAR_FONT_CANDIDATES` 로 내려간다.
BOLD_FONT_CANDIDATES: tuple[Path, ...] = (
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("C:/Windows/Fonts/NanumGothicBold.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
)


class MissingKoreanFontWarning(UserWarning):
    """한글 폰트를 찾지 못해 Pillow 기본 폰트로 폴백했을 때 발생하는 경고."""


#: 경고를 한 번만 내보내기 위한 모듈 플래그.
_fallback_warned: bool = False


def _env_font_path() -> Path | None:
    """환경변수로 지정된 폰트 경로를 반환한다. 미지정·미존재면 ``None``.

    :returns: 존재하는 폰트 파일 경로 또는 ``None``.
    """
    raw = os.environ.get(FONT_PATH_ENV, "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    return candidate if candidate.is_file() else None


@lru_cache(maxsize=8)
def _find_font_path_cached(bold: bool, env_hint: str) -> Path | None:
    """폰트 경로 탐색 본체(캐시 대상).

    :param bold: 굵은 글꼴을 우선 탐색할지 여부.
    :param env_hint: 환경변수 값. 값이 바뀌면 캐시가 무효화되도록 키에 포함한다.
    :returns: 폰트 파일 경로 또는 ``None``.
    """
    if env_hint:
        env_path = Path(env_hint)
        if env_path.is_file():
            return env_path
    candidates: tuple[Path, ...] = (
        BOLD_FONT_CANDIDATES + REGULAR_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            # 접근 불가 경로는 후보에서 조용히 제외한다(다음 후보로 계속 탐색).
            continue
    return None


def find_font_path(*, bold: bool = False) -> Path | None:
    """사용 가능한 한글 폰트 파일 경로를 찾는다.

    :param bold: True 이면 굵은 글꼴을 먼저 찾고, 없으면 일반 글꼴로 내려간다.
    :returns: 존재하는 폰트 파일 :class:`pathlib.Path`. 하나도 없으면 ``None``.
    """
    env_value = os.environ.get(FONT_PATH_ENV, "").strip()
    return _find_font_path_cached(bold, env_value)


def korean_font_available() -> bool:
    """한글 트루타입 폰트를 하나라도 찾았으면 True.

    :returns: 폰트 발견 여부. False 이면 :func:`load_font` 가 기본 폰트를 돌려준다.
    """
    return find_font_path() is not None


def _load_default_font(size_px: int) -> Any:
    """Pillow 기본 비트맵 폰트를 반환한다(폴백 경로).

    :param size_px: 요청 크기(px). Pillow 10.1 미만에서는 무시된다.
    :returns: Pillow 폰트 객체.
    """
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:
        # Pillow < 10.1 에는 size 인자가 없다.
        return ImageFont.load_default()


@lru_cache(maxsize=64)
def _load_font_cached(size_px: int, bold: bool, path_key: str) -> Any:
    """폰트 로딩 본체(캐시 대상).

    :param size_px: 글자 크기(px, 1 이상).
    :param bold: 굵은 글꼴 여부.
    :param path_key: 탐색된 폰트 경로 문자열. 빈 문자열이면 폴백.
    :returns: Pillow 폰트 객체.
    """
    if not path_key:
        return _load_default_font(size_px)
    try:
        return ImageFont.truetype(path_key, size_px)
    except OSError:
        # 파일은 있으나 Pillow 가 읽지 못하는 경우도 폴백한다(테스트 실패 금지).
        warnings.warn(
            f"폰트 파일을 읽지 못해 Pillow 기본 폰트로 대체합니다: {path_key}",
            MissingKoreanFontWarning,
            stacklevel=2,
        )
        return _load_default_font(size_px)


def load_font(size_px: int, *, bold: bool = False) -> Any:
    """지정한 픽셀 크기의 한글 폰트를 반환한다.

    폰트를 찾지 못하면 Pillow 기본 비트맵 폰트로 폴백하고
    :class:`MissingKoreanFontWarning` 을 **프로세스당 한 번만** 발생시킨다.
    폴백 상태에서도 예외를 던지지 않는다.

    :param size_px: 글자 크기(px). 1 이상이어야 한다.
    :returns: Pillow 폰트 객체(``FreeTypeFont`` 또는 기본 ``ImageFont``).
    :raises ValueError: ``size_px`` 가 1 미만인 경우.
    """
    if size_px < 1:
        raise ValueError(f"폰트 크기는 1px 이상이어야 합니다: {size_px}")
    path = find_font_path(bold=bold)
    if path is None:
        global _fallback_warned
        if not _fallback_warned:
            _fallback_warned = True
            warnings.warn(
                "한글 트루타입 폰트를 찾지 못해 Pillow 기본 폰트로 렌더링합니다. "
                "한글 글자는 제대로 보이지 않을 수 있으나 도형·선 기반 검증은 그대로 유효합니다. "
                f"특정 폰트를 쓰려면 환경변수 {FONT_PATH_ENV} 에 .ttf 경로를 지정하십시오.",
                MissingKoreanFontWarning,
                stacklevel=2,
            )
    return _load_font_cached(int(size_px), bold, "" if path is None else str(path))


def font_diagnostics() -> dict[str, Any]:
    """폰트 탐색 상태를 담은 진단 dict 를 반환한다(디버깅·CLI 출력용).

    :returns: ``{"available": bool, "regular": str|None, "bold": str|None,
        "env_var": str, "env_value": str|None}``.
    """
    regular = find_font_path()
    bold = find_font_path(bold=True)
    env_path = _env_font_path()
    return {
        "available": regular is not None,
        "regular": None if regular is None else str(regular),
        "bold": None if bold is None else str(bold),
        "env_var": FONT_PATH_ENV,
        "env_value": None if env_path is None else str(env_path),
    }
