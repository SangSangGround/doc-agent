"""Safely diagnose and, only if needed, repair the SELECT_TYPE routing block in
'detection/workflow_dialogue_v8.py'. This preserves document data, the other
conversation states, and reviewed-coordinate navigation safety gates.

Run from the hand-tracking project root using .venv_scan Python.
"""
from __future__ import annotations

import ast
import importlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DETECTION = ROOT / 'detection'
CONTROLLER = DETECTION / 'workflow_dialogue_v8.py'
BASE = DETECTION / 'refinement_output' / 'document1'


def load_linked():
    sys.path.insert(0, str(DETECTION.resolve()))
    from document_workflow_v8 import load_workflow
    from workflow_link_v8 import link_workflow
    workflow = load_workflow(BASE / 'workflow' / 'document_workflow.json')
    manifest = json.loads((BASE / 'writable_output' / 'navigation_manifest.json').read_text(encoding='utf-8-sig'))
    targets_file = BASE / 'writable_output' / 'navigation_targets.json'
    targets = json.loads(targets_file.read_text(encoding='utf-8-sig')) if targets_file.exists() else None
    return link_workflow(workflow, manifest, targets)


def test_routes(linked):
    import workflow_dialogue_v8
    print(f'[LOADED MODULE] {Path(workflow_dialogue_v8.__file__).resolve()}', flush=True)
    checks = []
    for n, app in enumerate(linked.app_types, 1):
        raw_count = len(linked.plan_for(app.id))
        direct = workflow_dialogue_v8.WorkflowDialogueV8(linked)
        direct._load_plan(app.id)
        direct_count = len(direct.steps)
        ctl = workflow_dialogue_v8.WorkflowDialogueV8(linked)
        ctl.start()
        events = ctl.on_user(str(n))
        reached = (ctl.selected_type == app.id and len(ctl.steps) > 0 and ctl.state == 'WAIT_CONSENT')
        print(f'[ROUTE TEST] {app.id}: raw={raw_count} direct={direct_count} '
              f'via_speech={len(ctl.steps)} type={ctl.selected_type!r} state={ctl.state} '
              f'{"PASS" if reached else "FAIL"}', flush=True)
        checks.append((app.id, raw_count, direct_count, reached))
    return checks


def _is_select_type_test(node):
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    c = node.test
    return (len(c.ops) == 1 and isinstance(c.ops[0], ast.Eq)
            and len(c.comparators) == 1 and isinstance(c.comparators[0], ast.Constant)
            and c.comparators[0].value == 'SELECT_TYPE'
            and isinstance(c.left, ast.Attribute) and c.left.attr == 'state'
            and isinstance(c.left.value, ast.Name) and c.left.value.id == 'self')


REPLACEMENT = '''        if self.state == 'SELECT_TYPE':
            choice = self._select_type(v)
            result = choice.id if choice is not None else 'NO_UNIQUE_MATCH'
            print(f'[TYPE MATCH] heard={spoken!r} normalized={v!r} result={result}', flush=True)
            if choice is None:
                suggestion = self._suggest_type(v)
                if suggestion is not None:
                    self.pending_type_id = suggestion.id
                    self.state = 'CONFIRM_TYPE'
                    return self.say(
                        f'{suggestion.spoken_name} 신청을 말씀하신 건가요? '
                        '네 맞아요 또는 아니요로 확인해 주세요.'
                    )
                return self.say(
                    '신청 유형을 정확히 확인하지 못했습니다. '
                    '번호 또는 신청 유형 이름으로 다시 말씀해 주세요.'
                )
            self._load_plan(choice.id)
            print(f'[TYPE PLAN] selected={self.selected_type!r} steps={len(self.steps)} '
                  f'raw={len(self.linked.plan_for(choice.id))}', flush=True)
            return self._ask_consent(
                f'{choice.spoken_name}을 선택하셨습니다. {choice.description}'
            )
'''


def patch_source(text):
    tree = ast.parse(text, filename=str(CONTROLLER))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorkflowDialogueV8']
    if len(classes) != 1:
        raise ValueError('Expected exactly one WorkflowDialogueV8 class')
    funcs = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == 'on_user']
    if len(funcs) != 1:
        raise ValueError('Expected exactly one on_user method')
    branches = [n for n in funcs[0].body if _is_select_type_test(n)]
    if len(branches) != 1:
        raise ValueError('Expected exactly one top-level SELECT_TYPE branch')
    node = branches[0]
    lines = text.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = [s+'\n' for s in REPLACEMENT.rstrip('\n').split('\n')]
    new_text = ''.join(lines)
    ast.parse(new_text, filename=str(CONTROLLER))
    compile(new_text, str(CONTROLLER), 'exec')
    return new_text


def main():
    if not CONTROLLER.is_file():
        raise SystemExit(f'[STOP] Controller not found: {CONTROLLER}')
    try:
        linked = load_linked()
    except Exception as exc:
        raise SystemExit(f'[STOP] Could not load the real document JSON: {type(exc).__name__}: {exc}')
    if not linked.app_types:
        raise SystemExit('[STOP] No application types. Do not patch.')
    print('[BEFORE] Checking actual document plan and current controller...', flush=True)
    try:
        before = test_routes(linked)
    except Exception as exc:
        raise SystemExit(f'[STOP] Baseline test raised {type(exc).__name__}: {exc}')
    if all(ok for _, _, _, ok in before):
        print('[ALREADY OK] No files changed. The type route is working.', flush=True)
        return
    if any(raw == 0 or direct == 0 for _, raw, direct, _ in before):
        print('[STOP] Direct plan loading also fails. No files changed.', flush=True)
        print('[NEXT] Inspect workflow plan/field role gates, not SELECT_TYPE routing.', flush=True)
        return
    original = CONTROLLER.read_text(encoding='utf-8-sig')
    try:
        updated = patch_source(original)
    except Exception as exc:
        raise SystemExit(f'[STOP] Safe source patch was not possible: {type(exc).__name__}: {exc}. No files changed.')
    backup = CONTROLLER.with_name(CONTROLLER.name + '.backup_route_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    shutil.copy2(CONTROLLER, backup)
    try:
        CONTROLLER.write_text(updated, encoding='utf-8')
        importlib.invalidate_caches()
        sys.modules.pop('workflow_dialogue_v8', None)
        print('[AFTER] Verifying repaired route against actual document...', flush=True)
        after = test_routes(linked)
        if not all(ok for _, _, _, ok in after):
            raise AssertionError('Route did not pass all application types')
        print(f'[PATCH OK] {CONTROLLER}', flush=True)
        print(f'[BACKUP] {backup}', flush=True)
        print('[READY] Type selection and plan loading pass. STT/camera untouched.', flush=True)
    except Exception as exc:
        shutil.copy2(backup, CONTROLLER)
        print(f'[ROLLBACK] Restored original file after {type(exc).__name__}: {exc}', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
