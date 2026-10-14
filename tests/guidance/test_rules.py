import copy
import json
from dataclasses import replace

import pytest

from docagent import guide
from docagent.guidance.rules import FALLBACK, Observation, RuleCatalog, content_digest


def entry(**changes):
    e = dict(document_id="fictional", document_version="1", page=1, field_id="consent",
             field_type="checkbox", surrounding_text="가상 설문 참여 동의 동의함 거부함",
             keywords=["설문 참여", "동의함", "거부함"], forbidden_keywords=["철회"],
             instruction="  가상 설문 참여 여부를 선택하는 항목입니다.\n동의함 또는 거부함을 선택하세요.  ")
    e.update(changes)
    e["approval"] = dict(status="approved", reviewed_by="fixture", reviewed_at="2026-10-15",
                         content_sha256=content_digest(e))
    return e


def catalog(*entries, purpose="production"):
    return RuleCatalog(dict(schema_version=1, purpose=purpose, fields=list(entries or [entry()])))


def observation(**changes):
    o = Observation("fictional", "1", 1, "checkbox", "설 문 참 여 / 동의함 / 거부함", .99, .96, .95, True, False)
    return replace(o, **changes)


def test_returns_approved_bytes_without_paraphrasing_or_llm():
    e = entry()
    r = catalog(e).match(observation())
    assert r.status == "matched"
    assert r.instruction == e["instruction"]
    assert r.content_sha256 == content_digest(e)
    assert r.field_id == "consent"


@pytest.mark.parametrize("changes,status", [
    ({"document_id":"unknown"}, "unsupported_document"),
    ({"document_version":"2"}, "unsupported_document"),
    ({"page":2}, "unsupported_document"),
    ({"identity_verified":False}, "unverified_document"),
    ({"identity_verified":"true"}, "unverified_document"),
    ({"needs_review":True}, "needs_review"),
    ({"needs_review":"false"}, "invalid_input"),
    ({"page":True}, "invalid_input"),
    ({"field_type":"signature"}, "no_match"),
    ({"surrounding_text":"설문 참여 동의함"}, "no_match"),
    ({"surrounding_text":"설문 참여 동의함 거부함 철회"}, "no_match"),
    ({"ocr_confidence":float("nan")}, "low_confidence"),
    ({"document_confidence":float("inf")}, "low_confidence"),
    ({"detection_confidence":.84}, "low_confidence"),
    ({"ocr_confidence":True}, "low_confidence"),
])
def test_uncertainty_never_returns_instruction(changes, status):
    r = catalog().match(observation(**changes))
    assert r.status == status
    assert r.instruction == FALLBACK
    assert r.field_id is None


def test_draft_and_ambiguous_matches_are_blocked():
    e = entry(); e["approval"] = {"status":"draft"}
    assert catalog(e).match(observation()).status == "unapproved"
    assert catalog(entry(), entry(field_id="other")).match(observation()).status == "ambiguous"
    # An unapproved competing field must still cause ambiguity.
    assert catalog(entry(field_id="other"), e).match(observation()).status == "ambiguous"


def test_demo_requires_explicit_mode():
    c = catalog(purpose="demo")
    assert c.match(observation()).status == "demo_only"
    assert c.match(observation(), demo=True).status == "matched"


@pytest.mark.parametrize("key,value", [("instruction", "새 안내"), ("keywords", ["다른 항목"]), ("page",2)])
def test_approval_invalidated_when_content_changes(key, value):
    e = entry(); e[key] = value
    with pytest.raises(ValueError, match="다시 검토"):
        catalog(e)


def test_rejects_missing_review_metadata_and_duplicate_keys():
    e = entry(); e["approval"]["reviewed_by"] = ""
    with pytest.raises(ValueError): catalog(e)
    with pytest.raises(ValueError): catalog(entry(), entry())


def test_caller_cannot_mutate_loaded_catalog():
    e = entry(); c = catalog(e)
    e["instruction"] = "몰래 바꿈"
    assert c.match(observation()).instruction != e["instruction"]


def test_cli_demo_and_invalid_input(tmp_path, capsys):
    c = tmp_path / "catalog.json"; o = tmp_path / "observation.json"
    c.write_text(json.dumps(dict(schema_version=1, purpose="demo", fields=[entry()])))
    from dataclasses import asdict
    o.write_text(json.dumps(asdict(observation())))
    args = ["--catalog", str(c), "--observation", str(o)]
    assert guide.main(args) == 2
    capsys.readouterr()
    assert guide.main(args + ["--demo"]) == 0
    assert json.loads(capsys.readouterr().out)["instruction"] == entry()["instruction"]
    o.write_text('{"secret_phone":"test-secret", broken}')
    assert guide.main(args) == 1
    assert "test-secret" not in capsys.readouterr().err
