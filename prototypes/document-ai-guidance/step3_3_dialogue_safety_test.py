r"""STEP 3-3. Read-only safety and dialogue smoke test for document1.

PowerShell, from hand-tracking root:
    .\.venv_scan\Scripts\python.exe -u .\step3_3_dialogue_safety_test.py

No production file modifications, no camera, microphone, credentials, or network calls.
Only writes a text report to step3_output/document1_step3_3_report.txt.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DET = ROOT / 'detection'
BASE = DET / 'refinement_output' / 'document1'
WF = BASE / 'workflow' / 'document_workflow.json'
MF = BASE / 'writable_output' / 'navigation_manifest.json'
TG = BASE / 'writable_output' / 'navigation_targets.json'
OUT = ROOT / 'step3_output' / 'document1_step3_3_report.txt'


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f'필요한 파일이 없습니다: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def has_track(events):
    return any(e.kind == 'START_TRACKING' for e in events)


def isolated(linked, Item, Controller, fid):
    d = Controller(linked)
    d.steps = [Item('field', [fid])]
    d.index = 0
    return d


def main():
    sys.path.insert(0, str(DET.resolve()))
    from workflow_link_v8 import link_workflow
    from workflow_dialogue_v8 import WorkflowDialogueV8, Item, nav_allowed
    from document_workflow_v8 import load_workflow

    wf, mf = load_workflow(WF), load_json(MF)
    tg = load_json(TG) if TG.exists() else None
    linked = link_workflow(wf, mf, tg)
    problems = []
    reviews = []
    results = []

    def check(ok, message):
        if not ok:
            problems.append(message)

    print(f'[STEP3-3] document={linked.document_id} fields={len(linked.fields)}')
    print(f'[CONTROLLER] {sys.modules["workflow_dialogue_v8"].__file__}')
    print(f'[LINKER] {sys.modules["workflow_link_v8"].__file__}')

    # Verify actual voice-selection route and complete *simulated* non-navigation flow.
    for idx, app in enumerate(linked.app_types, 1):
        d = WorkflowDialogueV8(linked)
        quiet(d.start)
        quiet(d.on_user, str(idx))
        selected_ok = d.selected_type == app.id and bool(d.steps) and d.state == 'WAIT_CONSENT'
        check(selected_ok, f'{app.id}: 유형 선택/계획 로딩 실패 ({len(d.steps)} steps, {d.state})')
        if not selected_ok:
            continue
        initial_count = len(d.steps)
        emitted = quiet(d.on_user, '진행해 주세요')
        no_track = not has_track(emitted)
        transitions = 0
        while d.state not in ('FINISHED', 'PAUSED') and transitions < 150:
            transitions += 1
            state = d.state
            if state in ('ASK_CONDITION', 'ASK_SIGNER', 'SELECT_CHOICE', 'WAIT_EXTERNAL', 'ASK_NAV'):
                emitted = quiet(d.on_user, '건너뛰기')
            else:
                problems.append(f'{app.id}: 모의 보류 진행 중 예상하지 못한 상태 {state}')
                break
            if has_track(emitted):
                no_track = False
                problems.append(f'{app.id}: 보류/건너뛰기 중 START_TRACKING 발생')
                break
        finished = d.state == 'FINISHED'
        check(finished, f'{app.id}: 모의 대화가 끝나지 않음 ({d.state}, {transitions} transitions)')
        check(no_track, f'{app.id}: 모의 보류 경로에서 손가락 추적 시작됨')
        print(f'[FLOW] {app.spoken_name}: initial_steps={initial_count} skip_all_finished={finished} no_tracking={no_track}')
        results.append(f'{app.id}: {initial_count} steps, safe-skip-finished={finished}, no-tracking={no_track}')

    # Field-level probes: unsafe guidance must never emit START_TRACKING.
    held = []
    for fid, f in linked.fields.items():
        if f.role == 'official':
            continue
        if nav_allowed(f):
            if f.target_mm is None:
                problems.append(f'{fid}: safe navigation granted without target coordinates')
            continue
        held.append(fid)
        d = isolated(linked, Item, WorkflowDialogueV8, fid)
        d.state = 'ASK_NAV'
        events = quiet(d.on_user, '네 진행해 주세요')
        check(not has_track(events) and d.state != 'TRACKING', f'{fid}: 미검수 좌표인데 추적 시작')
        check(d.status.get(fid) == 'unverified', f'{fid}: 미검수 상태가 보류 기록되지 않음 ({d.status.get(fid)})')
        events = quiet(d.on_user, '건너뛰기')
        check(d.state == 'FINISHED' and not has_track(events), f'{fid}: 건너뛰기 후 안전하게 종료되지 않음 ({d.state})')
        check(d.status.get(fid) == 'unverified', f'{fid}: 건너뛰기 후 미검수 상태 유실')

    for fid in ('F011', 'F019', 'F033'):
        f = linked.fields.get(fid)
        check(f is not None, f'{fid}: 검증 대상이 문서에 없음')
        if f is None:
            continue
        check(not nav_allowed(f), f'{fid}: 안전 차단이 풀려 있음')
        if fid in ('F011', 'F019'):
            check(bool(f.uncertain), f'{fid}: 재검수 전에 불확실성 표시가 해제됨')
        if fid == 'F033':
            check(f.nav_status == 'blocked_by_review', f'F033: 검수 차단이 반영되지 않음 ({f.nav_status})')
            check(f.target_mm is None, 'F033: 차단된 필드에 추적 좌표가 남아 있음')

    # Signature and conditional decisions are not inferred by the system.
    for fid, f in linked.fields.items():
        if f.role == 'official':
            continue
        if f.class_name == 'signature_field' and f.role == 'applicant':
            d = isolated(linked, Item, WorkflowDialogueV8, fid)
            ev = quiet(d._present)
            check(d.state == 'ASK_SIGNER' and not has_track(ev), f'{fid}: 서명자 질문 누락')
            if d.state == 'ASK_SIGNER':
                ev = quiet(d.on_user, '아니요')
                check(d.status.get(fid) == 'needs_third_party' and not has_track(ev), f'{fid}: 본인 서명 아님 응답 무시')
        if f.class_name != 'signature_field' and f.role == 'applicant' and (f.requirement == 'conditional' or f.condition_text):
            d = isolated(linked, Item, WorkflowDialogueV8, fid)
            ev = quiet(d._present)
            check(d.state == 'ASK_CONDITION' and not has_track(ev), f'{fid}: 조건부 작성 질문 누락')
            if d.state == 'ASK_CONDITION':
                ev = quiet(d.on_user, '아니요')
                check(d.status.get(fid) == 'not_applicable' and not has_track(ev), f'{fid}: 조건 미해당 응답 무시')

    # One positive path to detect accidental regression of verified guidance.
    positives = [f for f in linked.fields.values() if nav_allowed(f)
                 and f.role == 'applicant' and f.class_name != 'signature_field'
                 and not (f.requirement == 'conditional' or f.condition_text)]
    if positives:
        f = positives[0]
        d = isolated(linked, Item, WorkflowDialogueV8, f.id)
        d.state = 'ASK_NAV'
        ev = quiet(d.on_user, '네 진행해 주세요')
        check(has_track(ev) and d.state == 'TRACKING' and d.current_track == f.id,
              f'{f.id}: 검증된 필드에서도 정상 추적을 시작하지 못함')
        if d.state == 'TRACKING':
            quiet(d.on_arrived, f.id)
            check(d.state == 'WAIT_WRITING', f'{f.id}: 도착 후 작성 완료 확인 상태로 넘어가지 않음')
            quiet(d.on_user, '완료')
            check(d.status.get(f.id) == 'user_claimed', f'{f.id}: 사용자 완료 보고 기록 실패')
        print(f'[POSITIVE] {f.id}: verified-field guidance and self-reported completion tested')
    else:
        reviews.append('긍정 추적 경로를 테스트할 검수·확정 필드가 없음')

    print(f'[HELD] unsafe/unverified applicant-or-third-party fields={len(held)}: {", ".join(held) or "none"}')
    if held:
        reviews.append(f'실제 위치 안내 보류 {len(held)}개: {", ".join(held)}')
    reviews.append('사진/카메라/Deepgram을 사용하지 않은 상태 전환 시험임; 실제 작성 검증은 별도')

    lines = [f'Document: {linked.document_id}', f'Controller: {sys.modules["workflow_dialogue_v8"].__file__}',
             f'Linker: {sys.modules["workflow_link_v8"].__file__}', '', 'FLOW:',
             *results, '', f'UNSAFE / HELD ({len(held)}):', ', '.join(held) or 'none',
             '', f'ERRORS ({len(problems)}):', *(problems or ['none']), '',
             f'REVIEW ({len(reviews)}):', *reviews,
             '', 'This test does not validate camera recovery, field meaning, or actual handwriting.']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines), encoding='utf-8-sig')
    print(f'[REPORT] {OUT}')
    for p in problems:
        print(f'[FAIL] {p}')
    for r in reviews:
        print(f'[REVIEW] {r}')
    print(f'[RESULT] errors={len(problems)} review_notes={len(reviews)}')
    if problems:
        raise SystemExit(1)
    print('[STEP3-3 PASS] Unsafe fields held; dialogue skip/condition/signer paths safe. Camera not tested.')


if __name__ == '__main__':
    main()
