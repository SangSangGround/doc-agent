# -*- coding: utf-8 -*-
r"""Document5 demonstration stabilization patch V2 for v8.
Run in hand-tracking root with .venv_scan Python.
Changes ONLY dialogue + live runner; copies both originals, atomic-ish rollback.
Keeps human-reviewed positions and tracking safety requirements unchanged.
"""
from __future__ import annotations

import ast
import importlib.util
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DIALOGUE = ROOT / 'detection' / 'workflow_dialogue_v8.py'
LIVE = ROOT / 'detection' / 'multi_field_voice_tracking_v8.py'
MARKER = 'DEMO_V8_RECOVERY_V1'


def change_once(src, old, new, label):
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f'[STOP] {label}: expected one exact anchor; found {n}. No changes made.')
    return src.replace(old, new, 1)


def patch_dialogue(src):
    if MARKER in src:
        return src
    # Distinct ASK_NAV hook: do not depend on the exact formatting of the
    # existing `if v in YES or ...` expression. Never widen YES globally.
    # The already supported phrase in YES is used only *inside* ASK_NAV.
    anchor = "        if self.state == 'ASK_NAV':\n"
    inserted = anchor + ("            # DEMO_V8_RECOVERY_V1: navigation request aliases, not legal consent.\n"
        "            if v in {norm(x) for x in ('해주세요', '해 주세요', '네 해주세요',\n"
        "                                               '안내해주세요', '위치 안내해주세요',\n"
        "                                               '진행해주세요')}:\n"
        "                v = norm('네 진행해 주세요')\n")
    src = change_once(src, anchor, inserted, 'ASK_NAV state hook')
    # Do not accept '네' as a declaration of having written something.
    src = change_once(src, "        if self.state == 'WAIT_WRITING':\n            if v in DONE or v in YES_PLAIN:",
        "        if self.state == 'WAIT_WRITING':\n            if v in DONE:", 'explicit writing completion')
    src = change_once(src,
        "return self.say('작성하셨다면 완료, 작성하지 않았다면 건너뛰기라고 말씀해 주세요.')",
        "return self.say('작성하셨다면 작성 완료, 작성하지 않았다면 건너뛰기라고 말씀해 주세요.')",
        'writing reply prompt')
    src = change_once(src,
        "작성한 후 완료라고 말씀해 주세요.",
        "작성한 후 작성 완료라고 말씀해 주세요.",
        'arrival completion prompt')
    # Third party reporting same word choice for consistency (not role authorization).
    src = src.replace('확인하셨다면 완료, 아니면 건너뛰기라고 말씀해 주세요.',
                      '확인하셨다면 작성 완료, 아니면 건너뛰기라고 말씀해 주세요.')
    compile(src, str(DIALOGUE), 'exec')
    return src


def patch_live(src):
    if MARKER in src:
        return src
    # Secondary independent FULL-PAGE matching is attempted only if the normal
    # close-range tracker fails, not as a bypass of homography/target checks.
    anchor = "    if not args.model.is_file():"
    new = '''    # DEMO_V8_RECOVERY_V1: a second, independent reference matcher helps recover
    # when close-view ORB cannot find enough distributed features. Both matchers
    # still have their own geometric acceptance tests; no fake homographies.
    recovery_tracker = None
    if not args.strict_full_page:
        try:
            recovery_tracker = PaperTracker(manifest, args.reference,
                                            max_dim=args.max_feature_size)
            print('[RECOVERY READY] Extra full-page feature matcher is available.', flush=True)
        except Exception as exc:
            print(f'[RECOVERY NOTE] Full-page fallback unavailable: {exc}', flush=True)
'''+anchor
    src = change_once(src, anchor, new, 'secondary tracker creation')
    anchor = "    recovery_since = None\n    recovery_frames = 0\n"
    replacement = anchor+'''    recovery_started = None
    recovery_last_reset = 0.0
    recovery_last_hint = 0.0
    recovery_fallback_reported = False
'''
    src = change_once(src, anchor, replacement, 'recovery counters')
    old = '''                    if op == 'retry':
                        if args.tts and not voice.busy:
                            voice.say('다시 말씀해 주세요.')'''
    new = '''                    if op == 'retry':
                        # Ignore empty STT transcripts while tracking/reacquiring.
                        # Those are NOT requests from the user, and spoken retries
                        # otherwise loop over the microphone and drown out guidance.
                        if controller.state in ('TRACKING', 'PAUSED', 'WAIT_WRITING'):
                            print(f'[STT RETRY MUTED] state={controller.state}', flush=True)
                        elif args.tts and not voice.busy:
                            voice.say('다시 말씀해 주세요.')'''
    src = change_once(src, old, new, 'STT retry muting')
    old = "                result = tracker.update(frame) if camera_frame_ok else None\n"
    new = '''                # Try both detectors while searching for the SAME reviewed page.
                # Do NOT accept a fallback unless it independently returns a
                # valid transform. The normal 1s + 6 frames + target-visible
                # restore requirement remains unchanged below.
                result = tracker.update(frame) if camera_frame_ok else None
                used_fallback = False
                if (result is None and camera_frame_ok and recovery_tracker is not None
                        and controller.state in ('TRACKING', 'PAUSED')
                        and (active_field_id is not None
                             or getattr(controller, 'auto_resume_tracking', False))):
                    fallback_result = recovery_tracker.update(frame)
                    if fallback_result is not None:
                        result = fallback_result
                        used_fallback = True
                        if not recovery_fallback_reported:
                            print('[RECOVERY FULLPAGE MATCH] fallback geometric match acquired; checking target and stability.', flush=True)
                            recovery_fallback_reported = True
                if result is not None and not used_fallback:
                    recovery_fallback_reported = False
                # Never say 'recovered' merely because a camera frame exists.
                if (controller.state == 'PAUSED'
                        and getattr(controller, 'auto_resume_tracking', False)):
                    if recovery_started is None:
                        recovery_started = now
                    if now - recovery_last_hint >= 15.0:
                        msg = ('문서를 다시 찾고 있습니다. 휴대전화를 종이에서 조금 멀리 들어, '
                               '종이 전체가 카메라에 들어오도록 비춰 주세요.'
                               if camera_frame_ok else
                               '카메라 화면을 받을 수 없습니다. 휴대전화 카메라 연결을 확인해 주세요.')
                        print('[RECOVERY GUIDANCE]', msg, flush=True)
                        if args.tts and not voice.busy:
                            voice.say(msg)
                        recovery_last_hint = now
                    if (result is None and now-recovery_started >= 10.0
                            and now-recovery_last_reset >= 10.0):
                        # Drop stale tracking candidates; never loosen a match threshold.
                        for tr in (tracker, recovery_tracker):
                            if tr is not None:
                                for attr, value in (('last_good', None), ('pending', None),
                                                    ('pending_count', 0), ('fail_frames', 0)):
                                    if hasattr(tr, attr):
                                        setattr(tr, attr, value)
                        recovery_last_reset = now
                        print('[RECOVERY RESET] Cleared stale tracking candidates; retrying.', flush=True)
                else:
                    recovery_started = None
'''
    src = change_once(src, old, new, 'fallback tracking logic')
    src = change_once(src, "emit(controller.on_user('완료'))", "emit(controller.on_user('작성 완료'))", 'C keyboard completion')
    compile(src, str(LIVE), 'exec')
    return src


def test_dialogue_code(src):
    # Exercise installed controller from a temporary module without camera/mic.
    import sys
    import tempfile
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'workflow_dialogue_v8.py'
        p.write_text(src, encoding='utf-8')
        spec = importlib.util.spec_from_file_location('dialogue_demopatch_test', p)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            f = SimpleNamespace(id='F001', name='성명', detected_label='성명',
                 class_name='input_field', nav_status='verified', navigable=True,
                 role='applicant', reviewed_role='applicant', ai_role='applicant',
                 uncertain=False, confidence='high', target_mm=(20., 50.),
                 requirement='required', condition_text='', description='성명 입력',
                 source_quote='', aliases=[], choice_group_id=None)
            linked = SimpleNamespace(fields={'F001': f}, title='샘플 문서',
                 app_types=[], description='테스트', sections={}, groups={},
                 plan_for=lambda x: [SimpleNamespace(field_ids=['F001'], kind='field',
                                                     group_id=None, question='')])
            c = module.WorkflowDialogueV8(linked)
            c.start(); c.on_user('네 진행해 주세요')
            assert c.state == 'ASK_NAV', c.state
            ev = c.on_user('해주세요')
            assert c.state == 'TRACKING' and any(e.kind == 'START_TRACKING' for e in ev)
            ev = c.on_arrived('F001')
            assert c.state == 'WAIT_WRITING' and '작성 완료' in ' '.join(e.text for e in ev)
            c.on_user('네')
            assert c.state == 'WAIT_WRITING', 'A plain yes must NOT mark writing completed'
            c.on_user('작성 완료')
            assert c.status['F001'] == 'user_claimed'
            c2 = module.WorkflowDialogueV8(linked)
            c2.start(); c2.on_user('네 진행해 주세요'); c2.on_user('네 진행해주세요')
            assert c2.state == 'TRACKING'
            c2.on_tracking_lost()
            assert c2.state == 'PAUSED' and c2.auto_resume_tracking
            c2.on_user('해주세요')
            assert c2.state == 'PAUSED', 'Speech must not bypass geometry during recovery'
            ev = c2.on_tracking_restored()
            assert c2.state == 'TRACKING' and any(e.kind == 'START_TRACKING' for e in ev)
        finally:
            sys.modules.pop(spec.name, None)
    print('[TEST PASS] 해주세요 accepted for navigation; 작성 완료 prompt and command; yes not mistaken for completion.', flush=True)
    print('[TEST PASS] PAUSED cannot be overridden with 해주세요; automatic restore requires caller geometry check.', flush=True)


def main():
    if not (DIALOGUE.is_file() and LIVE.is_file()):
        raise SystemExit('[STOP] Run this script from hand-tracking root. Required detection files missing.')
    dialog_old = DIALOGUE.read_text(encoding='utf-8-sig')
    live_old = LIVE.read_text(encoding='utf-8-sig')
    try:
        dialog_new = patch_dialogue(dialog_old)
        live_new = patch_live(live_old)
        ast.parse(dialog_new); ast.parse(live_new)
        test_dialogue_code(dialog_new)
        assert ('RECOVERY FULLPAGE MATCH' in live_new and 'STT RETRY MUTED' in live_new
                and 'recovery_tracker.update(frame)' in live_new), 'Missing fallback recovery logic'
    except Exception as exc:
        raise SystemExit(f'[STOP] Validation failed BEFORE modifying files: {exc}')
    if dialog_old == dialog_new and live_old == live_new:
        print('[ALREADY PATCHED] Files unchanged.')
        return
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    backups = []
    for path in (DIALOGUE, LIVE):
        backup = path.with_name(path.name + '.backup_demo_' + suffix)
        shutil.copy2(path, backup)
        backups.append((path, backup))
    try:
        DIALOGUE.write_text(dialog_new, encoding='utf-8')
        LIVE.write_text(live_new, encoding='utf-8')
        for path in (DIALOGUE,LIVE):
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    except Exception:
        for path, backup in backups:
            shutil.copy2(backup, path)
        raise
    print('[PATCH OK] Demo voice prompts + document recovery fallback installed.')
    print('[CHANGED]', DIALOGUE, LIVE, sep='\n')
    print('[BACKUPS]', *(str(x[1]) for x in backups), sep='\n')
    print('[LIMIT] Real-camera stability must be tested; the patch does not bypass geometry thresholds.')


if __name__ == '__main__':
    main()
