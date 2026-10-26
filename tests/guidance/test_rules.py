import copy
import json
from dataclasses import replace

import pytest

from docagent import guide
from docagent.guidance.rules import FALLBACK, Observation, RuleCatalog, content_digest, page_content_digest


def entry(**changes):
    e = dict(document_id="fictional", document_version="1", page=1, field_id="consent",
             field_type="checkbox", surrounding_text="가상 설문 참여 동의 동의함 거부함",
             keywords=["설문 참여", "동의함", "거부함"], forbidden_keywords=["철회"],
             instruction="  가상 설문 참여 동의\n□ 동의함 □ 거부함  ",
             reading_mode="verbatim", source_text="  가상 설문 참여 동의\n□ 동의함 □ 거부함  ",
             source={"reference": "synthetic-test-source", "sha256": "a" * 64})
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
    assert r.source_text == e["source_text"]
    assert r.reading_mode == "verbatim"


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


def page_entry(number=1, count=1, text="  문서 원문\n항목·주의사항 전체.  ", *, approved=True):
    p = dict(document_id="fictional", document_version="1", page=number, page_count=count,
             reading_mode="verbatim", source_text=text,
             source={"reference": "synthetic-page-source", "sha256": "b" * 64})
    p["approval"] = ({"status": "approved", "reviewed_by": "synthetic-reviewer",
                      "reviewed_at": "2026-10-27", "content_sha256": page_content_digest(p)}
                     if approved else {"status": "draft"})
    return p


def page_catalog(*pages, purpose="production"):
    return RuleCatalog(dict(schema_version=1, purpose=purpose, fields=[], pages=list(pages)))


def test_production_legacy_approval_cannot_speak_without_original_source():
    e = entry()
    for key in ("reading_mode", "source_text", "source"):
        del e[key]
    e["approval"]["content_sha256"] = content_digest(e)
    with pytest.raises(ValueError, match="source_text"):
        catalog(e)
    # 미전사 초안은 편집·검수할 수 있지만 원문 없이 승인·재생할 수 없다.
    e["approval"] = {"status": "draft"}
    assert catalog(e).match(observation()).status == "unapproved"
    e["approval"] = dict(status="approved", reviewed_by="fixture", reviewed_at="2026-10-27",
                         content_sha256=content_digest(e))
    with pytest.raises(ValueError):
        catalog(e)


def test_legacy_demo_remains_explicitly_demo_only():
    c = RuleCatalog.load("examples/guidance/catalog.demo.json")
    observed = Observation("fictional-survey", "demo-1", 1, "checkbox",
                           "가상 설문 참여 동의함 거부함", .99, .99, .99, True, False)
    assert c.match(observed).status == "demo_only"
    result = c.match(observed, demo=True)
    assert result.status == "matched"
    assert result.source_text is None
    assert result.reading_mode is None


@pytest.mark.parametrize("changes", [
    {"source_text": "원문 일부"},
    {"instruction": "요약한 새 안내"},
    {"reading_mode": "summary"},
    {"source": {"reference": "changed", "sha256": "a" * 64}},
])
def test_source_and_mode_changes_invalidate_approval(changes):
    e = entry()
    e.update(changes)
    with pytest.raises(ValueError, match="다시 검토"):
        catalog(e)


@pytest.mark.parametrize("draft", [True, False])
def test_even_rehashed_paraphrase_cannot_replace_declared_source(draft):
    e = entry()
    e["instruction"] = "동의 여부를 선택하세요."
    e["approval"] = ({"status": "draft"} if draft else
                     dict(status="approved", reviewed_by="fixture", reviewed_at="2026-10-27",
                          content_sha256=content_digest(e)))
    with pytest.raises(ValueError, match="정확히 같아야"):
        catalog(e)


@pytest.mark.parametrize("source", [None, {}, {"reference": "source", "sha256": "bad"},
                                      {"reference": "", "sha256": "a" * 64}])
def test_production_verbatim_requires_source_provenance(source):
    e = entry(source=source)
    with pytest.raises(ValueError, match="출처"):
        catalog(e)


def test_complete_page_is_returned_without_ocr_or_field_assembly():
    text = "  신청서\n\n[ ] 선택 항목\n예외·주의: 동의하지 않아도 됩니다.\n(서명 또는 인)  "
    p = page_entry(text=text)
    c = page_catalog(p)
    result = c.read_page("fictional", "1", 1)
    assert result.status == "matched"
    assert result.instruction == result.source_text == text
    assert result.reading_mode == "verbatim"
    assert result.field_id is None
    assert result.content_sha256 == page_content_digest(p)
    # 페이지 원문을 등록하지 않은 항목 목록을 합쳐 전체 읽기를 만들지 않는다.
    assert catalog().read_page("fictional", "1", 1).status == "unsupported_document"
    p["source_text"] = "호출자가 원문을 바꿔도 로드한 자료는 바뀌지 않음"
    assert c.read_page("fictional", "1", 1).instruction == text


@pytest.mark.parametrize("key,value", [
    ("source_text", "변경"), ("page_count", 2),
    ("source", {"reference": "other-file", "sha256": "b" * 64}),
    ("transcription_note", "new unreviewed metadata"),
])
def test_page_approval_binds_entire_source_and_metadata(key, value):
    p = page_entry()
    p[key] = value
    with pytest.raises(ValueError, match="다시 검토"):
        page_catalog(p)


def test_full_document_preserves_all_pages_and_page_order():
    first = page_entry(1, 2, "  첫 장\n첫 장의 마지막 줄.\n")
    second = page_entry(2, 2, "\n둘째 장\n본문 끝.  ")
    c = page_catalog(second, first)
    result = c.read_document("fictional", "1")
    assert result.status == "matched"
    assert result.source_text == result.instruction == first["source_text"] + "\n\n" + second["source_text"]
    assert result.page is None and result.field_id is None
    assert len(result.content_sha256) == 64
    second["source"]["sha256"] = "c" * 64
    second["approval"]["content_sha256"] = page_content_digest(second)
    assert page_catalog(first, second).read_document("fictional", "1").content_sha256 != result.content_sha256


def test_document_rejects_missing_or_unapproved_pages_without_partial_text():
    for pages, status in [
        ([page_entry(1, 2)], "incomplete_document"),
        ([page_entry(1, 3), page_entry(3, 3)], "incomplete_document"),
        ([page_entry(1, 2), page_entry(2, 2, approved=False)], "unapproved"),
    ]:
        result = page_catalog(*pages).read_document("fictional", "1")
        assert result.status == status
        assert result.instruction == FALLBACK and result.source_text is None
    p = page_entry()
    del p["page_count"]
    p["approval"]["content_sha256"] = page_content_digest(p)
    c = page_catalog(p)
    assert c.read_page("fictional", "1", 1).status == "matched"
    assert c.read_document("fictional", "1").status == "incomplete_document"


def test_page_and_document_reading_fail_closed_for_unknown_or_demo_input():
    c = page_catalog(page_entry(approved=False))
    assert c.read_page("fictional", "1", 1).status == "unapproved"
    assert c.read_page("fictional", "unknown", 1).status == "unsupported_document"
    assert c.read_document("unknown", "1").status == "unsupported_document"
    assert c.read_page("fictional", "1", True).status == "invalid_input"
    assert c.read_document("", "1").status == "invalid_input"
    demo = page_catalog(page_entry(), purpose="demo")
    assert demo.read_page("fictional", "1", 1).status == "demo_only"
    assert demo.read_document("fictional", "1").status == "demo_only"
    assert demo.read_page("fictional", "1", 1, demo=True).status == "matched"
    assert demo.read_document("fictional", "1", demo=True).status == "matched"


@pytest.mark.parametrize("change", [
    lambda p: p.update(page=True), lambda p: p.update(page_count=True),
    lambda p: p.update(source_text="  "), lambda p: p.update(reading_mode="summary"),
    lambda p: p.update(source={"reference": "source", "sha256": "invalid"}),
    lambda p: p["approval"].update(reviewed_by=""),
])
def test_invalid_full_page_contract_is_rejected(change):
    p = page_entry()
    change(p)
    with pytest.raises(ValueError):
        page_catalog(p)


def test_duplicate_page_identity_is_rejected():
    with pytest.raises(ValueError, match="중복"):
        page_catalog(page_entry(), page_entry())


def test_large_declared_page_count_is_rejected_without_materializing_page_range():
    assert page_catalog(page_entry(1, 10 ** 100)).read_document("fictional", "1").status == "incomplete_document"


@pytest.mark.parametrize("text", ["원문\x00잘리는 문장", "원문\ud800잘못된 문자"])
def test_unreadable_source_text_is_rejected_even_before_draft_approval(text):
    p = page_entry(approved=False)
    p["source_text"] = text
    with pytest.raises(ValueError):
        page_catalog(p)
    e = entry()
    e.update(source_text=text, instruction=text, approval={"status": "draft"})
    with pytest.raises(ValueError):
        catalog(e)
