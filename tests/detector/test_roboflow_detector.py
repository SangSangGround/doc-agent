"""Roboflow RF-DETR 탐지 어댑터(:mod:`docagent.detector.roboflow_detector`) 테스트.

네트워크·API 키 없이 돈다. 실제 Workflow 응답 구조(2026-10 확인)를 본뜬
가짜 클라이언트를 주입해 파싱·필터·mm 변환·오류 처리를 검증한다.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from docagent.contracts import FieldType
from docagent.detector.roboflow_detector import (
    API_KEY_ENV,
    DEFAULT_MODEL_ID,
    SOURCE_ROBOFLOW,
    RawPrediction,
    RoboflowDetector,
    format_results,
    main,
    parse_workflow_result,
)
from docagent.errors import AdapterUnavailable, VisionError
from docagent.interfaces import Detector

#: A4 250dpi 페이지 크기(px). 실제 테스트 이미지와 같은 해상도다.
PAGE_W, PAGE_H = 2067, 2924


def _workflow_response(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """실제 Workflow 응답과 같은 형태를 만든다."""
    empty = not predictions
    return [
        {
            "predictions": {
                "image": {
                    "width": None if empty else PAGE_W,
                    "height": None if empty else PAGE_H,
                },
                "predictions": predictions,
            },
            "inference_id": "00000000-0000-0000-0000-000000000000",
            "model_id": DEFAULT_MODEL_ID,
        }
    ]


SAMPLE_PREDICTIONS: list[dict[str, Any]] = [
    {"x": 717.5, "y": 2007.5, "width": 67.0, "height": 79.0,
     "confidence": 0.7626, "class_id": 0, "class": "check_box"},
    {"x": 1500.0, "y": 2700.0, "width": 300.0, "height": 80.0,
     "confidence": 0.81, "class_id": 1, "class": "signature_field"},
    {"x": 532.5, "y": 2010.0, "width": 69.0, "height": 78.0,
     "confidence": 0.31, "class_id": 0, "class": "check_box"},
]


class FakeClient:
    """``run_workflow`` 만 흉내 내는 가짜 클라이언트."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def run_workflow(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _blank_page() -> np.ndarray:
    return np.full((PAGE_H, PAGE_W, 3), 255, dtype=np.uint8)


# --------------------------------------------------------------------------
# parse_workflow_result
# --------------------------------------------------------------------------


def test_parse_real_response_shape() -> None:
    parsed = parse_workflow_result(_workflow_response(SAMPLE_PREDICTIONS))
    assert len(parsed) == 3
    assert parsed[0] == RawPrediction("check_box", 0.7626, 717.5, 2007.5, 67.0, 79.0)


def test_parse_empty_predictions_with_null_image_size() -> None:
    assert parse_workflow_result(_workflow_response([])) == []


@pytest.mark.parametrize(
    "response",
    [{}, [], None, [{"foo": 1}], [{"predictions": []}], [{"predictions": {"image": {}}}]],
)
def test_parse_rejects_unexpected_structure(response: Any) -> None:
    with pytest.raises(VisionError):
        parse_workflow_result(response)


def test_parse_rejects_malformed_prediction() -> None:
    with pytest.raises(VisionError):
        parse_workflow_result(_workflow_response([{"x": 1.0}]))


def test_raw_prediction_center_to_top_left_px() -> None:
    box = RawPrediction("check_box", 0.9, 100.0, 200.0, 20.0, 40.0).to_box_px()
    assert (box.x, box.y, box.w, box.h) == (90, 180, 20, 40)


# --------------------------------------------------------------------------
# RoboflowDetector
# --------------------------------------------------------------------------


def test_satisfies_detector_protocol() -> None:
    assert isinstance(RoboflowDetector(client=FakeClient()), Detector)


def test_predict_filters_by_confidence_and_sorts_desc() -> None:
    client = FakeClient(_workflow_response(SAMPLE_PREDICTIONS))
    predictions = RoboflowDetector(client=client, conf=0.5).predict("page.jpg")
    assert [p.class_name for p in predictions] == ["signature_field", "check_box"]
    assert client.calls[0]["parameters"] == {"model_id": DEFAULT_MODEL_ID}
    assert client.calls[0]["images"] == {"image": "page.jpg"}


def test_detect_returns_a4_mm_detections() -> None:
    client = FakeClient(_workflow_response(SAMPLE_PREDICTIONS))
    detections = RoboflowDetector(client=client).detect(_blank_page())

    assert [d.type for d in detections] == [FieldType.CHECKBOX, FieldType.SIGNATURE]
    assert all(d.source == SOURCE_ROBOFLOW for d in detections)
    checkbox = detections[0]
    # 2067px 폭 = A4 210mm → 1mm ≈ 9.843px
    px_per_mm = PAGE_W / 210.0
    assert checkbox.box_mm.x_mm == pytest.approx((717.5 - 33.5) / px_per_mm, abs=0.2)
    assert checkbox.box_mm.w_mm == pytest.approx(67.0 / px_per_mm, abs=0.2)
    assert checkbox.box_mm.center().y_mm == pytest.approx(2007.5 / px_per_mm, abs=0.2)


def test_detect_skips_unknown_class() -> None:
    response = _workflow_response(
        [{"x": 10.0, "y": 10.0, "width": 5.0, "height": 5.0, "confidence": 0.9, "class": "object"}]
    )
    assert RoboflowDetector(client=FakeClient(response)).detect(_blank_page()) == []


def test_detect_rejects_non_array() -> None:
    with pytest.raises(VisionError):
        RoboflowDetector(client=FakeClient()).detect("page.jpg")


def test_api_failure_is_wrapped_as_vision_error() -> None:
    detector = RoboflowDetector(client=FakeClient(error=RuntimeError("401 Unauthorized")))
    with pytest.raises(VisionError, match="Roboflow API 호출에 실패"):
        detector.predict("page.jpg")


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=API_KEY_ENV):
        RoboflowDetector()


def test_missing_sdk_raises_adapter_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "inference_sdk", None)
    with pytest.raises(AdapterUnavailable, match="inference-sdk"):
        RoboflowDetector(api_key="dummy")


def test_invalid_conf_rejected() -> None:
    with pytest.raises(ValueError):
        RoboflowDetector(client=FakeClient(), conf=1.5)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_format_results_matches_spec() -> None:
    text = format_results([RawPrediction("check_box", 0.812, 521.0, 430.0, 24.0, 25.0)])
    assert "class      : check_box" in text
    assert "confidence : 81.2%" in text
    assert "bbox       : x=521.0, y=430.0, width=24.0, height=25.0" in text
    assert text.endswith("Total objects: 1")


def test_cli_missing_image_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["no/such/image.jpg"]) == 1
    assert "이미지 파일이 없습니다" in capsys.readouterr().err
