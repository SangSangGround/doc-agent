r"""STEP 3: Read-only audit of a real paper-form workflow.

Run from hand-tracking root using .venv_scan Python:
    .\.venv_scan\Scripts\python.exe -u .\step3_audit_workflow.py

Requires detection/refinement_output/document1/{workflow/document_workflow.json,
 writable_output/navigation_manifest.json,writable_output/navigation_targets.json}.
Does NOT change production code/JSON; does NOT use camera, STT or API keys.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DETECTION = ROOT / "detection"
BASE = DETECTION / "refinement_output" / "document1"
WORKFLOW = BASE / "workflow" / "document_workflow.json"
MANIFEST = BASE / "writable_output" / "navigation_manifest.json"
TARGETS = BASE / "writable_output" / "navigation_targets.json"
OUTPUT = ROOT / "step3_output"
BLOCKED = {"F033"}  # keep the safety block already used in the live runner

sys.path.insert(0, str(DETECTION))


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def field_simulator(linked, field_id, controller_class, item_class):
    """Isolated state machine; never invokes a camera/voice client."""
    d = controller_class(linked)
    d.steps = [item_class("field", [field_id])]
    d.index = 0
    return d


def has_track(events):
    return any(x.kind == "START_TRACKING" for x in events)


def main():
    from workflow_link_v8 import link_workflow
    from workflow_dialogue_v8 import WorkflowDialogueV8, Item, nav_allowed

    linked = link_workflow(read_json(WORKFLOW), read_json(MANIFEST), read_json(TARGETS))
    print(f"[STEP3] document={linked.document_id} fields={len(linked.fields)}", flush=True)
    problems, cautions, results, csv_rows = [], [], [], []

    applications = list(enumerate(linked.app_types, 1)) if linked.app_types else [(None, None)]
    for idx, app in applications:
        app_id, app_name = (app.id, app.spoken_name) if app else (None, "전체")
        d = WorkflowDialogueV8(linked)
        with redirect_stdout(io.StringIO()):
            d.start()
            if app is not None:
                d.on_user(str(idx))
        expected_state = "WAIT_CONSENT"
        passed = bool(d.steps) and d.state == expected_state and d.selected_type == app_id
        print(f"[TYPE] {app_name}: steps={len(d.steps)} state={d.state} {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed:
            problems.append(f"{app_name}: request selection did not load a nonempty plan")
        if not d.steps:
            continue
        official = []
        for position, item in enumerate(d.steps, 1):
            for fid in item.ids:
                f = linked.fields.get(fid)
                if f is None:
                    problems.append(f"{app_name} step {position}: unknown field {fid}")
                    continue
                if 'official' in (f.role, f.ai_role, f.reviewed_role):
                    official.append(fid)
                allowed = nav_allowed(f) and fid not in BLOCKED
                if allowed and f.target_mm is None:
                    problems.append(f"{fid}: navigable but target_mm is missing")
                csv_rows.append({
                    "application_type": app_id or "all", "step": position,
                    "step_kind": item.kind, "field_id": fid, "field_name": f.name,
                    "class_name": f.class_name, "role": f.role,
                    "requirement": f.requirement, "condition": f.condition_text,
                    "nav_status": f.nav_status, "safe_to_navigate": "YES" if allowed else "NO",
                    "target_mm": str(f.target_mm or ""),
                    "reason": f.nav_reason,
                })
        if official:
            problems.append(f"{app_name}: office-only fields in applicant plan: {sorted(set(official))}")
        results.append(f"{app_name}: {len(d.steps)} steps, {sum(r['safe_to_navigate']=='YES' for r in csv_rows if r['application_type'] == (app_id or 'all'))} safe navigation fields")

    # Independent real-field behavior probes (no physical completion claimed).
    for f in linked.fields.values():
        if f.id in BLOCKED:
            if nav_allowed(f):
                cautions.append(f"{f.id}: controller considers field navigable; live --block-field must remain enabled")
            continue
        if not nav_allowed(f):
            d = field_simulator(linked, f.id, WorkflowDialogueV8, Item)
            d.state = "ASK_NAV"
            with redirect_stdout(io.StringIO()):
                e = d.on_user("네 진행해 주세요")
            if has_track(e):
                problems.append(f"{f.id}: unsafe field started finger tracking")
        if f.role == "third_party":
            d = field_simulator(linked, f.id, WorkflowDialogueV8, Item)
            e = d._present()
            if d.state != "WAIT_EXTERNAL" or has_track(e):
                problems.append(f"{f.id}: third-party field was not held for external signer")
        if f.class_name == "signature_field" and f.role == "applicant":
            d = field_simulator(linked, f.id, WorkflowDialogueV8, Item)
            e = d._present()
            if d.state != "ASK_SIGNER" or has_track(e):
                problems.append(f"{f.id}: signer identity not checked")
            else:
                with redirect_stdout(io.StringIO()):
                    e = d.on_user("아니요")
                if d.status[f.id] != "needs_third_party" or has_track(e):
                    problems.append(f"{f.id}: refusing signer identity did not block guidance")
        if (f.requirement == "conditional" or f.condition_text) and f.role == "applicant" and f.class_name != "signature_field":
            d = field_simulator(linked, f.id, WorkflowDialogueV8, Item)
            e = d._present()
            if d.state != "ASK_CONDITION" or has_track(e):
                problems.append(f"{f.id}: conditional field did not ask whether condition applies")
            else:
                with redirect_stdout(io.StringIO()):
                    e = d.on_user("아니요")
                if d.status[f.id] != "not_applicable" or has_track(e):
                    problems.append(f"{f.id}: 'condition does not apply' was not honored")

    # This is a design gap, not a reason to invent missing coordinate values.
    unverified = [f.id for f in linked.fields.values() if f.role != "official" and not nav_allowed(f)]
    if unverified:
        cautions.append(f"{len(unverified)} applicant/third-party fields lack a safe tracking target: {', '.join(unverified)}")
    if not csv_rows:
        problems.append("No application fields were added to report")

    OUTPUT.mkdir(exist_ok=True)
    csv_path = OUTPUT / "document1_step3_fields.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "application_type", "step", "step_kind", "field_id", "field_name",
            "class_name", "role", "requirement", "condition", "nav_status",
            "safe_to_navigate", "target_mm", "reason"])
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [f"Document: {linked.document_id}", f"Fields: {len(linked.fields)}", *results,
             "", f"FAILURES ({len(problems)}):", *(problems or ["none"]),
             "", f"REVIEW NOTES ({len(cautions)}):", *(cautions or ["none"]),
             "", "This audit is not proof of camera stability, content correctness, or completed handwriting."]
    txt_path = OUTPUT / "document1_step3_report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"[FIELD CSV] {csv_path}")
    print(f"[REPORT] {txt_path}")
    print(f"[RESULT] errors={len(problems)} review_notes={len(cautions)}")
    for p in problems:
        print(f"[FAIL] {p}")
    for p in cautions:
        print(f"[REVIEW] {p}")
    if problems:
        raise SystemExit(1)
    print("[AUDIT PASS] Safety and dialogue state checks passed. Review notes still require attention.")


if __name__ == "__main__":
    main()
