"""전 테스트 공용 pytest fixture.

Vision·PII·Agent 에이전트가 그대로 가져다 쓰는 결정론적 픽스처 모음이다.
합성 신청서 생성기(:mod:`docagent.testing.synthetic`)에만 의존하므로
실제 관공서 서식 데이터셋 없이도 파이프라인 전체를 검증할 수 있다.

렌더링 비용을 줄이기 위해 이미지 픽스처는 **module scope 캐시**로 둔다.
반환되는 :class:`~docagent.testing.synthetic.SyntheticForm` 은 frozen 이지만
내부 ndarray 는 가변이므로, **테스트에서 이미지를 직접 수정하지 말 것.**
수정이 필요하면 ``form.image.copy()`` 를 쓴다.

픽스처 목록
-----------
============================ ==================================================
``clean_form``               변형 없음. mm ↔ px 환산이 그대로 성립한다.
``skewed_form``              15도 회전 + 여백. 기울기 보정 테스트용.
``lowres_form``              150dpi 저해상도 + JPEG 열화.
``perspective_form``         원근 왜곡 + 여백 + 조명 불균일.
``written_form``             ``(form, written_image)`` 튜플. Verify 전/후 비교용.
``sample_structure``         이미지 없는 정답 구조. Agent·PII 테스트용(즉시 생성).
``sample_form_spec``         ``clean_form`` 이 쓰는 사양.
``synthetic_module``         생성기 모듈 자체(헬퍼 함수 접근용).
============================ ==================================================

``sample_structure`` 는 다음을 반드시 포함한다.

* 필수 CHOICE 항목 1개(선택지 ``동의함`` / ``동의하지 않음``), ``clause_text`` 채워짐
* ``FieldRole.APPLICANT`` SIGNATURE 항목 1개
* ``Sensitivity.PRIVATE`` TEXT_INPUT 항목 2개 이상(성명 / 주민등록번호)
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from docagent.contracts import DocumentStructure
from docagent.testing import synthetic as _synthetic
from docagent.testing.synthetic import (
    AGREE_LABEL,
    DEFAULT_SEED,
    FormSpec,
    SyntheticForm,
    build_truth,
    make_application_form,
    render_written,
)

__all__ = [
    "synthetic_module",
    "sample_form_spec",
    "clean_form",
    "skewed_form",
    "lowres_form",
    "perspective_form",
    "written_form",
    "sample_structure",
]


@pytest.fixture(scope="session")
def synthetic_module() -> ModuleType:
    """합성 문서 생성기 모듈 자체를 돌려준다(상수·헬퍼 접근용).

    :returns: :mod:`docagent.testing.synthetic` 모듈.
    """
    return _synthetic


@pytest.fixture(scope="module")
def sample_form_spec() -> FormSpec:
    """``clean_form`` 이 사용하는 변형 없는 기준 사양.

    :returns: :class:`FormSpec`.
    """
    return FormSpec(
        document_id="fixture_clean",
        dpi=200,
        include_representative=True,
        seed=DEFAULT_SEED,
    )


@pytest.fixture(scope="module")
def clean_form(sample_form_spec: FormSpec) -> SyntheticForm:
    """변형이 전혀 없는 합성 신청서.

    ``form.flat_box_px(box_mm)`` 로 얻은 픽셀 좌표가 ``form.image`` 위의
    좌표와 그대로 일치하므로, 좌표 정확도 검증의 기준으로 쓴다.

    :returns: :class:`SyntheticForm`.
    """
    return make_application_form(sample_form_spec)


@pytest.fixture(scope="module")
def skewed_form() -> SyntheticForm:
    """15도 기울어진 합성 신청서(여백 포함).

    문서 윤곽선 검출과 기울기 보정(deskew) 테스트용이다.

    :returns: :class:`SyntheticForm`.
    """
    return make_application_form(
        FormSpec(
            document_id="fixture_skewed",
            dpi=200,
            rotation_deg=15.0,
            margin_px=60,
            background_gray=185,
            seed=DEFAULT_SEED + 1,
        )
    )


@pytest.fixture(scope="module")
def lowres_form() -> SyntheticForm:
    """150dpi 저해상도 + JPEG 열화가 걸린 합성 신청서.

    :returns: :class:`SyntheticForm`.
    """
    return make_application_form(
        FormSpec(
            document_id="fixture_lowres",
            dpi=150,
            jpeg_quality=70,
            noise_sigma=2.5,
            seed=DEFAULT_SEED + 2,
        )
    )


@pytest.fixture(scope="module")
def perspective_form() -> SyntheticForm:
    """원근 왜곡·여백·조명 불균일이 걸린 합성 신청서.

    카메라로 비스듬히 촬영한 상황을 흉내낸다.

    :returns: :class:`SyntheticForm`.
    """
    return make_application_form(
        FormSpec(
            document_id="fixture_perspective",
            dpi=200,
            perspective_strength=0.25,
            margin_px=50,
            background_gray=195,
            illumination_gradient=0.3,
            seed=DEFAULT_SEED + 3,
        )
    )


@pytest.fixture(scope="module")
def written_form(clean_form: SyntheticForm) -> tuple[SyntheticForm, np.ndarray]:
    """``(원본 form, 작성 후 이미지)`` 튜플.

    작성 후 이미지는 ``동의함`` 체크 + 신청인 서명이 그려진 상태이며,
    ``clean_form.image`` 와 **픽셀 단위로 정렬**되어 있어 Verify 단계의
    잉크 비율 전/후 비교에 바로 쓸 수 있다.

    :returns: ``(SyntheticForm, (H, W, 3) uint8 BGR ndarray)``.
    """
    written = render_written(clean_form, checked_option=AGREE_LABEL, sign=True)
    return (clean_form, written)


@pytest.fixture(scope="module")
def sample_structure() -> DocumentStructure:
    """이미지 없이 즉시 생성되는 정답 :class:`DocumentStructure`.

    Agent·PII 테스트처럼 이미지가 필요 없는 경우에 쓴다(렌더링 비용 0).

    :returns: 필수 CHOICE 1개, APPLICANT SIGNATURE 1개, PRIVATE TEXT_INPUT
        2개 이상(성명·주민등록번호)을 포함하고 ``clause_text`` 가 채워진 구조.
    """
    return build_truth(
        FormSpec(
            document_id="fixture_structure",
            dpi=200,
            include_representative=True,
            seed=DEFAULT_SEED,
        )
    )
