"""합성 정답셋의 분모와 오완료 집계. 실제 사용자 성능을 주장하지 않는다."""
import copy
import json

import pytest

from docagent.home.demo import create_demo
from docagent.home.evaluation import evaluate_manifest


@pytest.fixture
def dataset(tmp_path):
    create_demo(tmp_path)
    path = tmp_path / "evaluation.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_demo_reports_counts_with_explicit_denominators(dataset):
    path, _ = dataset
    result = evaluate_manifest(path, demo=True)
    assert result["dataset_kind"] == "synthetic"
    assert result["sample_count"] == 5
    assert result["metrics"]["correct_completion"] == {"count": 1, "denominator": 1, "rate": 1.0}
    assert result["metrics"]["problem_detection"] == {"count": 3, "denominator": 3, "rate": 1.0}
    assert result["metrics"]["false_completion"] == {"count": 0, "denominator": 4, "rate": 0.0}
    assert result["metrics"]["uncertain"] == {"count": 1, "denominator": 5, "rate": .2}
    assert "합성" in result["notice"] and "증명하지" in result["notice"]
    assert {row["id"] for row in result["cases"]} == {"correct", "opposite", "missing", "duplicate", "blurred"}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("keep,undefined", [
    ("correct", ["false_completion", "problem_detection"]),
    ("missing", ["correct_completion"]),
])
def test_zero_denominator_is_null_not_zero_or_perfect(dataset, keep, undefined):
    path, data = dataset
    data["cases"] = [case for case in data["cases"] if case["id"] == keep]
    save(path,data)
    result = evaluate_manifest(path,demo=True)
    for metric in undefined:
        assert result["metrics"][metric] == {"count": 0, "denominator": 0, "rate": None}


@pytest.mark.parametrize("field", ["expected_complete", "expected_problem", "choices"])
def test_missing_ground_truth_or_user_intent_is_rejected(dataset, field):
    path, data = dataset
    data["cases"][0].pop(field)
    save(path,data)
    with pytest.raises(ValueError):
        evaluate_manifest(path,demo=True)


@pytest.mark.parametrize("bad", [True, 0, "1", None])
def test_schema_version_must_be_integer_one(dataset, bad):
    path, data = dataset
    data["schema_version"] = bad
    save(path,data)
    with pytest.raises(ValueError):
        evaluate_manifest(path,demo=True)


def test_root_must_be_an_object(dataset):
    path, _ = dataset
    save(path,[])
    with pytest.raises(ValueError):
        evaluate_manifest(path,demo=True)


def test_real_data_requires_review_information(dataset):
    path,data = dataset
    data["dataset_kind"] = "real"
    save(path,data)
    with pytest.raises(ValueError):
        evaluate_manifest(path,demo=True)


def test_demo_template_cannot_be_reported_as_real_data(dataset):
    path,data = dataset
    data.update(dataset_kind="real",reviewed_by="테스트 검토자",dataset_version="1")
    save(path,data)
    with pytest.raises(ValueError):
        evaluate_manifest(path,demo=True)


@pytest.mark.parametrize("flags", [(False,False,False),(True,True,False),(True,False,True),(False,True,True)])
def test_missing_or_conflicting_expected_outcomes_are_rejected(dataset, flags):
    path,data = dataset
    data["cases"][0].update(zip(("expected_complete","expected_problem","expected_uncertain"),flags))
    save(path,data)
    with pytest.raises(ValueError):
        evaluate_manifest(path,demo=True)


def test_duplicate_case_ids_are_rejected(dataset):
    path,data = dataset
    data["cases"] = [data["cases"][0],copy.deepcopy(data["cases"][0])]
    save(path,data)
    with pytest.raises(ValueError):
        evaluate_manifest(path,demo=True)


def test_image_path_cannot_escape_dataset_directory(dataset):
    path,data = dataset
    data["cases"][0]["image"] = "../private.png"
    save(path,data)
    with pytest.raises(ValueError):
        evaluate_manifest(path,demo=True)


def test_false_completion_metric_counts_an_incorrect_success(dataset,monkeypatch):
    from docagent.home import evaluation
    path,data = dataset
    data["cases"] = [case for case in data["cases"] if case["id"] == "missing"]
    save(path,data)

    def mistaken_complete(reference,captured,groups,**kwargs):
        return {"status":"ok","reason":"test_false_success","groups":[{
            "id":group["id"],"state":"single","marked_option_ids":["disagree"],
            "options":[{"id":option["id"],"state":"marked" if option["id"]=="disagree" else "unmarked"}
                       for option in group["options"]]} for group in groups]}

    monkeypatch.setattr(evaluation,"verify_checkbox_groups",mistaken_complete)
    result = evaluate_manifest(path,demo=True)
    assert result["metrics"]["false_completion"] == {"count":1,"denominator":1,"rate":1.0}
    assert result["metrics"]["problem_detection"] == {"count":0,"denominator":1,"rate":0.0}
