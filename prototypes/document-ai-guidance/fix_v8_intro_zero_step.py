"""Minimal, guarded patch for v8 speech dialogue: clean introductory title and reject empty plan.
Run in hand-tracking root: .\\.venv_scan\\Scripts\\python.exe fix_v8_intro_zero_step.py
Does not edit workflow JSON or remove any navigation safety gates.
"""
from __future__ import annotations
import ast
import shutil
from pathlib import Path
from datetime import datetime

p = Path('detection/workflow_dialogue_v8.py')
if not p.is_file():
    raise SystemExit(f'[STOP] File not found: {p.resolve()}')
src = p.read_text(encoding='utf-8-sig')
tree = ast.parse(src, filename=str(p))
classes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'WorkflowDialogueV8']
if len(classes) != 1:
    raise SystemExit('[STOP] WorkflowDialogueV8 class not found exactly once; no edits made')
methods = {m.name:m for m in classes[0].body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
if not {'intro', '_ask_consent'}.issubset(methods):
    raise SystemExit('[STOP] intro/_ask_consent missing; no edits made')

# Replace only the intro() method; leave all other dialogue decisions unchanged.
intro = '''    def intro(self):
        title = str(self.linked.title or '제목이 확인되지 않은 문서').strip()
        desc = self.linked.description or '문서의 용도는 별도 확인이 필요합니다.'
        # Omit selectable title checkboxes only in the overall introduction.
        # No form-specific names or application types are hardcoded here.
        marker = r'(?:\\[\\s*\\]|[☐☑□▢]|체크박스)'
        if re.search(marker, title):
            stem = re.split(marker, title, maxsplit=1)[0].strip(' ,.:;\\u00b7\\u318d')
            suffix = re.search(r'(신청서|신고서|동의서|확인서|청구서|요청서|서식)\\s*$', title)
            if stem:
                title = (stem + (' ' + suffix.group(1) if suffix and not stem.endswith(suffix.group(1)) else ''))
        return (f'이 문서는 {title}입니다. {desc} '
                '안내 내용은 인공지능 분석 결과로, 작성 조건을 확정한 것은 아닙니다.')
'''

# Do not let an empty plan be described as having been completed.
empty_guard = '''        if not self.steps:
            self.state = 'FINISHED'
            self.current_track = None
            return [*self.say(
                '신청 유형은 선택됐지만, 연결된 작성 안내 항목이 0개입니다. '
                '아직 문서 안내를 시작할 수 없습니다. '
                '문서 워크플로와 작성 항목 연결을 확인해야 합니다.'
            ), Event('FINISHED')]
'''
lines = src.splitlines(keepends=True)
# Work bottom-up so line offsets remain accurate.
intro_node = methods['intro']
consent_node = methods['_ask_consent']
if intro_node.lineno > consent_node.lineno:
    raise SystemExit('[STOP] Unexpected function ordering; no edits made')
# Insert guard at start of consent method, before self.state = ...
first_stmt = consent_node.body[0]
consent_lines_idx = first_stmt.lineno - 1
if not ('신청 유형은 선택됐지만, 연결된 작성 안내 항목이 0개입니다.' in src):
    lines[consent_lines_idx:consent_lines_idx] = [x+'\n' for x in empty_guard.rstrip('\n').split('\n')]
lines[intro_node.lineno-1:intro_node.end_lineno] = [x+'\n' for x in intro.rstrip('\n').split('\n')]
updated = ''.join(lines)
# Only fix duplicated generic '신청' suffix in the spoken choice announcement.
updated = updated.replace("f'{choice.spoken_name} 신청을 선택하셨습니다.", "f'{choice.spoken_name}을 선택하셨습니다.")
updated = updated.replace("f'{proposed.spoken_name} 신청을 선택하셨습니다.", "f'{proposed.spoken_name}을 선택하셨습니다.")
ast.parse(updated, filename=str(p))
backup = p.with_name(p.name + '.bak_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
shutil.copy2(p, backup)
p.write_text(updated, encoding='utf-8')
print('[PATCH OK]', p)
print('[BACKUP]', backup)
print('[CHANGES] Initial title removes checkbox options; empty plan is an error, not a completed guide; duplicated 신청 removed.')
print('[NOTE] This does not invent fields or repair missing workflow sections. Run the workflow report below to find why plan is empty.')

# Read-only diagnosis of the same workflow/manifest used by the live demo.
try:
    import json
    import sys
    root = Path('detection')
    sys.path.insert(0, str(root.resolve()))
    from document_workflow_v8 import load_workflow
    from workflow_link_v8 import link_workflow
    from workflow_dialogue_v8 import WorkflowDialogueV8
    base = root / 'refinement_output' / 'document1'
    wf_path = base / 'workflow' / 'document_workflow.json'
    mf_path = base / 'writable_output' / 'navigation_manifest.json'
    target_path = base / 'writable_output' / 'navigation_targets.json'
    wf = load_workflow(wf_path)
    mf = json.loads(mf_path.read_text(encoding='utf-8-sig'))
    targets = json.loads(target_path.read_text(encoding='utf-8-sig')) if target_path.exists() else None
    linked = link_workflow(wf, mf, targets)
    print(f'[DIAG] document_id={linked.document_id} fields={len(linked.fields)} sections={len(linked.sections)} types={len(linked.app_types)}')
    for t in linked.app_types:
        raw = linked.plan_for(t.id)
        controller = WorkflowDialogueV8(linked)
        controller._load_plan(t.id)
        verified = sum(f.navigable for f in linked.fields.values())
        print(f'[DIAG TYPE] {t.id} ({t.spoken_name}) raw_steps={len(raw)} controller_steps={len(controller.steps)} reviewed_positions={verified}')
        print(f'            selectors={t.selector_field_ids} sections={t.section_ids}')
        if not raw:
            print('    [CAUSE] LinkedWorkflow.plan_for returned no steps. Check application type selectors/section_ids and source JSON.')
        elif not controller.steps:
            print('    [CAUSE] Controller safety filter removed every step; examine field roles (do NOT bypass review gates).')
            for step in raw[:8]:
                for fid in step.field_ids:
                    f = linked.fields[fid]
                    print(f'        {fid} name={f.name!r} role={f.role} ai_role={f.ai_role} reviewed_role={f.reviewed_role} nav_status={f.nav_status}')
        else:
            print('    [OK] This workflow has usable steps. If live app still reports zero, confirm file paths and actual loaded code.')
    for w in linked.warnings[:10]:
        print('[DIAG WARN]', w)
except Exception as e:
    print(f'[DIAG ERROR] {type(e).__name__}: {e}')
    print('[DIAG HINT] Patch applied; diagnosis requires the same workflow/manifest JSON files as the live program.')
