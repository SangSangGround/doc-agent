import copy
import json
import os

import pytest

from docagent.home.session import HomeSession
from docagent.home.templates import load_template
from tests.home.test_templates import write_bundle


@pytest.fixture
def template(tmp_path):
    path, _ = write_bundle(tmp_path)
    return load_template(path, demo=True)


def observation(marked=(), *, state=None, status="ok"):
    state = state or ("unmarked" if not marked else "single" if len(marked) == 1 else "multiple")
    return {"status": status, "groups": [
        {"id": "consent", "state": state, "marked_option_ids": list(marked), "options": [
            {"id": option, "state": "uncertain" if state == "uncertain" else "marked" if option in marked else "unmarked"}
            for option in ("yes", "no")
        ]}
    ]}


def choose(session, ids=("no",)):
    session.propose_selection("consent", list(ids))
    session.confirm_selection("consent")


def test_original_reading_preserves_whitespace(template):
    session = HomeSession(template)
    assert session.read_page() == template.manifest["pages"][0]["source_text"]
    assert session.read_current() == template.manifest["groups"][0]["source_text"]
    assert session.move("previous")["current_index"] == 0
    assert session.move("next")["current_index"] == 0
    with pytest.raises(ValueError):
        session.move("anything")


@pytest.mark.parametrize("marked,state,status,expected", [
    (("no",), None, "ok", "matched"),
    (("yes",), None, "ok", "mismatch"),
    ((), None, "ok", "unmarked"),
    (("yes", "no"), None, "ok", "multiple"),
    ((), "uncertain", "ok", "uncertain"),
    (("no",), None, "uncertain", "uncertain"),
])
def test_comparison_outcomes(template, marked, state, status, expected):
    session = HomeSession(template)
    choose(session)
    view = session.apply_verification(observation(marked, state=state, status=status))
    assert view["groups"][0]["verification"]["status"] == expected
    assert view["complete"] is (expected == "matched")


def test_pending_and_unconfirmed_choices_never_complete(template):
    session = HomeSession(template)
    assert session.apply_verification(observation(("no",)))["groups"][0]["verification"]["status"] == "unconfirmed"
    session.propose_selection("consent", ["no"])
    assert not session.apply_verification(observation(("no",)))["complete"]
    session.confirm_selection("consent")
    assert not session.view()["complete"]
    assert session.apply_verification(observation(("no",)))["complete"]
    session.propose_selection("consent", ["yes"])
    assert not session.view()["complete"]
    session.cancel_selection("consent")
    assert session.view()["groups"][0]["selected_option_ids"] == ["no"]
    assert not session.view()["complete"]


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(groups=[]),
    lambda r: r["groups"].append(copy.deepcopy(r["groups"][0])),
    lambda r: r["groups"][0].update(id="unknown"),
    lambda r: r["groups"][0].update(marked_option_ids=["other"]),
    lambda r: r["groups"][0].update(marked_option_ids=["no", "no"]),
    lambda r: r["groups"][0].update(state="unmarked"),
    lambda r: r["groups"][0].update(options=[]),
    lambda r: r["groups"][0]["options"][0].update(id="no"),
    lambda r: r["groups"][0]["options"][1].update(state="uncertain"),
    lambda r: r.update(status="complete"),
])
def test_malformed_observation_discards_previous_complete(template, mutate):
    session = HomeSession(template)
    choose(session)
    assert session.apply_verification(observation(("no",)))["complete"]
    result = observation(("no",))
    mutate(result)
    with pytest.raises(ValueError):
        session.apply_verification(result)
    assert not session.view()["complete"]


def test_retake_invalidates_prior_verification(template):
    session = HomeSession(template)
    choose(session)
    session.apply_verification(observation(("no",)))
    assert not session.invalidate_verification()["complete"]
    assert session.view()["groups"][0]["confirmed"]


def test_allowed_multiple_and_zero_selection(template):
    template.manifest["groups"][0].update(min_selected=0, max_selected=2)
    session = HomeSession(template)
    choose(session, ("yes", "no"))
    assert session.apply_verification(observation(("yes", "no")))["complete"]
    choose(session, ())
    assert session.apply_verification(observation())["complete"]


def test_all_groups_must_be_confirmed_and_matched(template):
    second = copy.deepcopy(template.manifest["groups"][0])
    second["id"] = "other_consent"
    second["source_text"] = "두 번째 항목의 원문"
    template.manifest["groups"].append(second)
    session = HomeSession(template)
    choose(session)
    evidence = observation(("no",))
    other = copy.deepcopy(evidence["groups"][0])
    other["id"] = "other_consent"
    evidence["groups"].append(other)
    assert not session.apply_verification(evidence)["complete"]
    session.propose_selection("other_consent", ["no"])
    session.confirm_selection("other_consent")
    assert session.apply_verification(evidence)["complete"]
    assert session.move("next")["current_group"]["id"] == "other_consent"
    assert session.read_current() == "두 번째 항목의 원문"


def test_view_mutation_cannot_change_session_selections_or_original(template):
    session = HomeSession(template)
    choose(session)
    view = session.view()
    view["groups"][0]["selected_option_ids"].append("yes")
    view["current_group"]["source_text"] = "수정한 안내"
    assert session.view()["groups"][0]["selected_option_ids"] == ["no"]
    assert session.read_current() == template.manifest["groups"][0]["source_text"]


def test_state_roundtrip_excludes_document_text_and_requires_new_verification(template, tmp_path):
    session = HomeSession(template)
    choose(session)
    session.set_speech_rate(1.5)
    assert session.apply_verification(observation(("no",)))["complete"]
    path = tmp_path / "state.json"
    session.save(path)
    raw = path.read_text(encoding="utf-8")
    assert "수집" not in raw and "verification" not in raw and "reference.png" not in raw
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0
    restored = HomeSession.load(path, template)
    assert restored.view()["groups"][0]["confirmed"]
    assert restored.preferences["speech_rate"] == 1.5
    assert not restored.view()["complete"]


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(template_sha256="0" * 64),
    lambda d: d.update(current_index=True),
    lambda d: d.update(current_index=1),
    lambda d: d.update(selections={}),
    lambda d: d["selections"].update(consent=["wrong"]),
    lambda d: d["preferences"].update(speech_rate=float("nan")),
    lambda d: d.update(verification="complete"),
])
def test_corrupt_or_wrong_document_saved_state_rejected(template, tmp_path, mutate):
    path = tmp_path / "state.json"
    HomeSession(template).save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        HomeSession.load(path, template)


def test_failed_atomic_save_keeps_original(template, tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    session = HomeSession(template)
    session.save(path)
    original = path.read_bytes()
    choose(session)

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        session.save(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".docagent-session-*"))


@pytest.mark.parametrize("ids", [["other"], ["yes", "no"], ["no", "no"], [], "no", [None]])
def test_invalid_choices_rejected(template, ids):
    with pytest.raises(ValueError):
        HomeSession(template).propose_selection("consent", ids)
