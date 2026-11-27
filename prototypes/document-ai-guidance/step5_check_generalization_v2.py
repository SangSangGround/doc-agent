r"""STEP 5 v2 -- second-document workflow generalization smoke test (read-only).

Run in the hand-tracking root:
  .\.venv_scan\Scripts\python.exe -u .\step5_check_generalization.py --doc-id document2

This intentionally does NOT generate document-specific labels, field ids, geometry,
or invent a manifest. It validates only the artifacts that actually exist.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def is_tracking(events):
    return any(getattr(e, 'kind', '') == 'START_TRACKING' for e in events)


def sample_fixture():
    """Synthetic unfamiliar form; never interpreted as a real document2 test."""
    source_sha = 'c' * 64
    rows = [
        ('X01', 'check_box', '가입 신청', 'applicant', False, 'signup'),
        ('X02', 'check_box', '해지 신청', 'applicant', False, 'cancel'),
        ('X03', 'input_field', '이용자 번호', 'applicant', False, None),
        ('X04', 'input_field', '접수 번호', 'official', False, None),
        ('X05', 'signature_field', '대리인 서명', 'third_party', True, None),
    ]
    fields = [{
        'id': fid, 'class_name': cls, 'detected_label': name,
        'name': name, 'role': role, 'requirement': 'unknown',
        'condition_text': '', 'description': '원문을 확인하세요',
        'source_quote': '', 'aliases': [], 'confidence': 'high',
        'uncertain': uncertain, 'choice_group_id': None,
        'selects_application_type': selector
    } for fid, cls, name, role, uncertain, selector in rows]
    wf = {
        'schema': 'document_workflow_v1', 'document_id': 'test_synthetic_other_form',
        'title': '합성 이용 신청서', 'description': '다른 유형의 양식',
        'selection_question': '무엇을 진행할까요?',
        'provenance': {'image_sha256': source_sha, 'ai_inferred': True, 'human_reviewed': False},
        'fields': fields,
        'application_types': [
            {'id': 'signup', 'spoken_name': '가입 신청', 'aliases': ['가입'],
             'description': '', 'selector_field_ids': ['X01'], 'section_ids': ['common']},
            {'id': 'cancel', 'spoken_name': '해지 신청', 'aliases': ['해지'],
             'description': '', 'selector_field_ids': ['X02'], 'section_ids': ['common']},
        ],
        'sections': [{'id': 'common', 'title': '입력', 'description': '',
                      'applies_to': ['all'], 'field_ids': ['X01', 'X02', 'X03', 'X04', 'X05']}],
        'choice_groups': [], 'validation': {'warnings': []}
    }
    mf = {'schema': 'a4_navigation_manifest_v1', 'source_image_sha256': source_sha,
          'fields': [
              {'id': 'X01', 'class_name': 'check_box', 'label': '가입 신청', 'role': 'applicant',
               'target_mm': {'x': 30, 'y': 30}, 'approval_method': 'manually_adjusted'},
              {'id': 'X03', 'class_name': 'input_field', 'label': '이용자 번호', 'role': 'applicant',
               'target_mm': {'x': 60, 'y': 80}, 'approval_method': 'manually_adjusted'},
          ]}
    return wf, mf


def expected_grouped_plan(linked, type_id):
    """Independent audit of the linker plan after explicit multi-choice grouping.

    The linker emits one raw step per member of a multi-choice group, whereas
    the dialogue controller asks one question for the whole group. Count both
    correctly; do not treat two fields as two separate dialogue steps.
    """
    raw = linked.plan_for(type_id)
    eligible = []
    for step in raw:
        ids = [fid for fid in step.field_ids
               if linked.fields[fid].ai_role != 'official'
               and linked.fields[fid].reviewed_role != 'official'
               and linked.fields[fid].role != 'official']
        if ids:
            eligible.append((step.group_id, list(ids)))

    group_members = {}
    for gid, ids in eligible:
        if (gid and gid in linked.groups
                and linked.groups[gid].get('type') == 'multi_choice'):
            group_members.setdefault(gid, []).extend(ids)

    used, expected = set(), []
    for gid, ids in eligible:
        if gid in group_members:
            if gid in used:
                continue
            used.add(gid)
            expected.append(tuple(dict.fromkeys(group_members[gid])))
        else:
            expected.append(tuple(ids))
    return raw, expected, group_members


def run_checks(wf, mf, tg, det):
    from workflow_dialogue_v8 import WorkflowDialogueV8, Item, nav_allowed
    from workflow_link_v8 import link_workflow

    errors, notes = [], []
    linked = link_workflow(wf, mf, tg)
    plan_info = []
    app_types = list(enumerate(linked.app_types, start=1)) if linked.app_types else [(None, None)]
    for index, app in app_types:
        ctrl = WorkflowDialogueV8(linked)
        quiet(ctrl.start)
        if app:
            quiet(ctrl.on_user, str(index))
        # Do NOT compare dialogue steps against raw linker steps: a
        # multi_choice with 2 checkboxes correctly becomes ONE voice question.
        raw, expected_ids, grouped = expected_grouped_plan(
            linked, app.id if app else None)
        expected = len(expected_ids)
        actual_ids = [tuple(item.ids) for item in ctrl.steps]
        ok = (bool(ctrl.steps) and ctrl.state == 'WAIT_CONSENT' and
              (not app or ctrl.selected_type == app.id))
        if not ok:
            errors.append(f'type={app.id if app else "default"}: selection/plan failed')
        if actual_ids != expected_ids:
            first_bad = next((i for i,(a,b) in enumerate(zip(actual_ids,expected_ids),1) if a!=b), None)
            errors.append(f'type={app.id if app else "default"}: '
                          f'plan field/order differs; actual_steps={len(actual_ids)} '
                          f'expected_steps={expected}, first_mismatch={first_bad}')
        if len(raw) != expected:
            summary = ', '.join(f'{gid}={len(dict.fromkeys(ids))} fields->1 step'
                                for gid, ids in grouped.items())
            print(f'[GROUP COLLAPSE] {app.id if app else "default"}: '
                  f'raw={len(raw)} grouped={expected} ({summary or "official-only excluded"})', flush=True)
        plan_info.append({'type_id': app.id if app else 'default',
                          'spoken': app.spoken_name if app else '(단일 문서)',
                          'steps': len(ctrl.steps), 'state': ctrl.state, 'ok': ok})
        print(f'[TYPE] {app.spoken_name if app else "default"}: steps={len(ctrl.steps)} '
              f'state={ctrl.state} {"PASS" if ok else "FAIL"}', flush=True)
        for item in ctrl.steps:
            for fid in item.ids:
                if fid not in linked.fields:
                    errors.append(f'Unknown field in plan: {fid}')
                elif linked.fields[fid].role == 'official' or linked.fields[fid].ai_role == 'official':
                    errors.append(f'Official-only field entered user plan: {fid}')
        # Make sure users can safely skip through a plan, without camera or guessed positions.
        if ok:
            ev = quiet(ctrl.on_user, '네 진행해 주세요')
            steps = 0
            while ctrl.state != 'FINISHED' and steps < 300:
                steps += 1
                if is_tracking(ev):
                    errors.append(f'{app.id if app else "default"}: tracking started while skipping')
                    break
                ev = quiet(ctrl.on_user, '건너뛰기')
            if ctrl.state != 'FINISHED':
                errors.append(f'{app.id if app else "default"}: skip-all did not finish, state={ctrl.state}')
            else:
                print(f'[FLOW] {app.id if app else "default"}: skip_all_finished=True no_tracking=True')

    safe = [f for f in linked.fields.values() if nav_allowed(f)]
    unsafe = [f for f in linked.fields.values() if not nav_allowed(f)]
    for f in safe:
        if f.target_mm is None:
            errors.append(f'{f.id}: safe to navigate but target missing')
        else:
            x, y = f.target_mm
            if not (0 <= x <= 210 and 0 <= y <= 297):
                errors.append(f'{f.id}: reviewed target outside A4 bounds')
    for f in unsafe:
        if f.role == 'official':
            continue
        ctrl = WorkflowDialogueV8(linked)
        ctrl.steps = [Item('field', [f.id])]
        ctrl.state = 'ASK_NAV'
        if is_tracking(quiet(ctrl.on_user, '네 진행해 주세요')):
            errors.append(f'{f.id}: unsafe field started tracking')
    if safe:
        sample = safe[0]
        ctrl = WorkflowDialogueV8(linked)
        ctrl.steps = [Item('field', [sample.id])]
        ctrl.state = 'ASK_NAV'
        if not is_tracking(quiet(ctrl.on_user, '네 진행해 주세요')):
            errors.append(f'{sample.id}: verified field could not start guidance')
        else:
            print(f'[POSITIVE] {sample.id}: verified field can start tracking (simulated event only)')
    else:
        notes.append('No safe reviewed navigation target in this document')
    if mf is None:
        if safe:
            errors.append('Navigation allowed without human-reviewed manifest')
        notes.append('No reviewed navigation manifest; physical guidance not validated')
    print(f'[FIELDS] total={len(linked.fields)} safe_navigation={len(safe)} blocked_or_unverified={len(unsafe)}')
    return {'document_id': linked.document_id, 'title': linked.title,
            'application_types': [a.id for a in linked.app_types],
            'plans': plan_info, 'fields': len(linked.fields),
            'safe_navigation_fields': len(safe), 'held_fields': len(unsafe),
            'manifest_available': mf is not None, 'errors': errors, 'notes': notes,
            'ai_inferred': linked.ai_inferred}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--doc-id', default='document2', help='Document folder ID (default document2)')
    ap.add_argument('--selftest', action='store_true', help='Synthetic other-form engine test (NOT document2 evidence)')
    args = ap.parse_args()

    root = Path.cwd()
    det = root / 'detection'
    if not (det / 'workflow_dialogue_v8.py').is_file() or not (det / 'workflow_link_v8.py').is_file():
        print('[STOP] Run from hand-tracking root containing detection/workflow_dialogue_v8.py')
        return 2
    sys.path.insert(0, str(det.resolve()))
    print('[STEP5] Generic second-document smoke test (does not modify production files)')
    if args.selftest:
        wf, mf = sample_fixture()
        print('[SELFTEST] Synthetic NON-document1 workflow, not a real document2 result')
        try:
            report = run_checks(wf, mf, None, det)
        except Exception as ex:
            print(f'[SELFTEST FAIL] {type(ex).__name__}: {ex}')
            return 1
        if report['errors'] or report['safe_navigation_fields'] == 0:
            print(f'[SELFTEST FAIL] {report["errors"]}')
            return 1
        print('[SELFTEST PASS] Distinct form/type/field IDs are handled by the engine')
        print('[LIMIT] Still requires a real second document to claim document generalization')
        return 0

    doc = args.doc_id
    if not doc or any(v in doc for v in ('/', '\\', '..')):
        print('[STOP] Invalid --doc-id; use a folder name such as document2')
        return 2
    if doc == 'document1':
        print('[STOP] STEP5 requires a second document; document1 cannot prove generalization')
        return 2
    base = det / 'refinement_output' / doc
    workflow_path = base / 'workflow' / 'document_workflow.json'
    manifest_path = base / 'writable_output' / 'navigation_manifest.json'
    targets_path = base / 'writable_output' / 'navigation_targets.json'
    image_path = det / 'test_images' / f'{doc}.jpg'
    detection_path = det / 'gemini_output' / doc / 'detection_result.json'
    for name, path in [('image',image_path),('detection',detection_path),('workflow',workflow_path),('manifest',manifest_path),('targets',targets_path)]:
        print(f'[FILE] {name}: {"FOUND" if path.is_file() else "MISSING"} | {path.as_posix()}')
    if not workflow_path.is_file():
        print('[NOT READY] Second-document workflow JSON is missing. This is NOT a Step 5 pass.')
        if detection_path.is_file() and image_path.is_file():
            print('[NEXT] You can generate a document2 workflow from the EXISTING detection:')
            print('  .\\.venv_scan\\Scripts\\python.exe detection\\document_workflow_v8.py '
                  f'--detection "{detection_path.as_posix()}" --image "{image_path.as_posix()}" '
                  '--gemini --model gemini-3.8-flash')
            print('  (Requires a valid Gemini API key; this audit does NOT call Gemini.)')
        else:
            print('[NEXT] Create actual document2 detection_result.json before generating v8 workflow.')
            print('[NOTE] Do NOT copy document1 workflow/manifest or invent coordinates for document2.')
        return 2
    try:
        from document_workflow_v8 import load_workflow
        wf = load_workflow(workflow_path)
        if wf.get('document_id') != doc:
            raise ValueError(f'workflow document_id={wf.get("document_id")!r} expected {doc!r}')
        wsha = wf.get('provenance', {}).get('image_sha256')
        if not wsha:
            raise ValueError('Workflow has no source-image SHA256; cannot prove document identity')
        if image_path.is_file() and sha(image_path) != wsha:
            raise ValueError('Second-document reference image SHA256 does not match workflow; refuse to mix data')
        mf = read_json(manifest_path) if manifest_path.is_file() else None
        tg = read_json(targets_path) if targets_path.is_file() else None
        if mf is not None and mf.get('source_image_sha256') != wsha:
            raise ValueError('Manifest/workflow source image hashes differ; navigation is unsafe')
        report = run_checks(wf, mf, tg, det)
        # Distinct document, not document1 renamed.
        doc1_path = det / 'refinement_output' / 'document1' / 'workflow' / 'document_workflow.json'
        if doc != 'document1' and doc1_path.is_file():
            wf1 = load_workflow(doc1_path)
            if wf1.get('provenance', {}).get('image_sha256') == wsha:
                report['errors'].append('Second document has identical image SHA256 to document1')
        mode = 'FULL' if mf and report['safe_navigation_fields'] else 'MEANING_ONLY'
        if not report['ai_inferred']:
            report['notes'].append('Workflow is label-only fallback, not AI document understanding')
        if report['errors']:
            outcome = 'FAIL'
        elif mode == 'MEANING_ONLY' or not report['ai_inferred']:
            outcome = 'PARTIAL'
        else:
            outcome = 'PASS'
        report['outcome'] = outcome
        report['scope'] = mode
    except Exception as ex:
        print(f'[STEP5 FAIL] {type(ex).__name__}: {ex}')
        return 1
    out = root / 'step5_output'
    out.mkdir(exist_ok=True)
    report_file = out / f'{doc}_generalization_report.json'
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[REPORT] {report_file.as_posix()}')
    for err in report['errors']:
        print(f'[ERROR] {err}')
    for note in report['notes']:
        print(f'[NOTE] {note}')
    print(f'[STEP5 {outcome}] document={doc} scope={mode} '
          f'types={len(report["application_types"])} fields={report["fields"]} '
          f'reviewed_safe={report["safe_navigation_fields"]}')
    if outcome == 'PASS':
        print('[LIMIT] This is a structural smoke test, NOT a real-camera/iPhone usability test or proof of universal accuracy.')
        return 0
    if outcome == 'PARTIAL':
        print('[NEXT] Need a reviewed manifest and AI-derived workflow for full second-document verification.')
        return 3
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
