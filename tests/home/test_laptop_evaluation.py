"""합성 평가가 실패를 성공처럼 요약하지 않고 빈 분모를 보존하는지 검사한다."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_laptop_capture.py"
spec = importlib.util.spec_from_file_location("laptop_capture_evaluation", path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_rejecting_every_image_is_not_reported_as_full_success():
    cases = [{"expected_usable": True, "accepted": False},
             {"expected_usable": False, "accepted": False}]
    metrics = evaluation.classification_metrics(cases, "accepted")
    assert metrics["bad_image_pass_rate"] == {"numerator": 0, "denominator": 1, "rate": 0}
    assert metrics["normal_image_pass_rate"]["rate"] == 0
    assert metrics["accuracy"]["rate"] == .5


def test_empty_denominators_are_unknown_not_zero():
    assert evaluation.classification_metrics([], "accepted")["bad_image_pass_rate"] == {
        "numerator": 0, "denominator": 0, "rate": None,
    }


def test_scenarios_are_reproducible_and_white_cover_is_not_claimed_as_glare():
    page = np.full((297, 210, 3), 255, np.uint8)
    first, second = evaluation.make_scenarios(page), evaluation.make_scenarios(page)
    assert [case["id"] for case in first] == ["normal", "tilted", "clipped", "blurred", "white_occlusion"]
    assert all(np.array_equal(a["image"], b["image"]) for a, b in zip(first, second))
    assert first[-1]["transformation"]["physical_glare_simulation"] is False
    assert sum(case["expected_usable"] for case in first) == 1


def test_existing_results_are_not_overwritten(tmp_path):
    (tmp_path / "report.json").write_text("previous run")
    with pytest.raises(ValueError, match="비어 있는"):
        evaluation.evaluate(None, tmp_path)
    assert (tmp_path / "report.json").read_text() == "previous run"
