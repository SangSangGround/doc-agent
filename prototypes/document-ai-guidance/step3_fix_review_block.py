"""STEP 3-2: Persist a reviewed navigation exclusion for document1/F033.

Run from hand-tracking project root:
    .\\.venv_scan\\Scripts\\python.exe -u .\\step3_fix_review_block.py

Safely adds a generic blocked_field_ids gate to the v8 workflow linker and
records the document-specific reviewed F033 block in navigation_manifest.json.
Does not modify the camera, dialogue controller, voice, image, or AI workflow.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
DET = ROOT / 'detection'
LINKER = DET / 'workflow_link_v8.py'
BASE = DET / 'refinement_output' / 'document1'
MANIFEST = BASE / 'writable_output' / 'navigation_manifest.json'
WORKFLOW = BASE / 'workflow' / 'document_workflow.json'
TARGETS = BASE / 'writable_output' / 'navigation_targets.json'
DOC_BLOCK_ID = 'F033'  # Only this known reviewed block; future forms use their own IDs.


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def check_source(text):
    """Modify just two locations in linker. Never use a hard-coded F033 gate."""
    if 'blocked_field_ids' in text and 'blocked_by_review' in text:
        return text
    if 'blocked_field_ids' in text or 'blocked_by_review' in text:
        raise ValueError('Existing review-block feature is partial/unknown; refusing to overwrite.')
    first_anchor = "    fields = {}\n    for row in workflow['fields']:\n"
    final_anchor = "        fields[fid] = info\n"
    if text.count(first_anchor) != 1 or text.count(final_anchor) != 1:
        raise ValueError('Linker layout differs from expected v8; refusing automatic edit.')
    new_text = text.replace(first_anchor, (
        '    # Manifest-level manual exclusion applies even to an otherwise reviewed field.\n'
        "    reviewed_blocks = manifest.get('blocked_field_ids', []) if manifest is not None else []\n"
        '    if not isinstance(reviewed_blocks, list) or any(not isinstance(v, str) for v in reviewed_blocks):\n'
        "        raise ValueError('manifest.blocked_field_ids must be a list of field IDs')\n"
        '    blocked_ids = set(reviewed_blocks)\n'
        + first_anchor), 1)
    new_text = new_text.replace(final_anchor, (
        '        if fid in blocked_ids:\n'
        "            info.nav_status = 'blocked_by_review'\n"
        "            info.nav_reason = 'manual navigation block recorded in document manifest'\n"
        '            info.target_mm = None\n'
        + final_anchor), 1)
    compile(new_text, str(LINKER), 'exec')
    return new_text


def updated_manifest_text(old_text):
    obj = json.loads(old_text.lstrip('\ufeff'))
    if obj.get('schema') != 'a4_navigation_manifest_v1':
        raise ValueError('Expected a4_navigation_manifest_v1')
    ids = obj.get('blocked_field_ids', [])
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
        raise ValueError('Existing blocked_field_ids is not a string array')
    approved_ids = {f.get('id') for f in obj.get('fields', [])}
    if DOC_BLOCK_ID not in approved_ids and DOC_BLOCK_ID not in ids:
        raise ValueError('F033 not found among reviewed fields; expected original manifest differs')
    if DOC_BLOCK_ID in ids:
        return old_text
    # Preserve the original JSON body with a minimal new top-level metadata entry.
    trailing = len(old_text) - len(old_text.rstrip())
    trimmed = old_text.rstrip()
    if not trimmed.endswith('}'):
        raise ValueError('Unexpected JSON ending')
    return (trimmed[:-1].rstrip() + ',\n  "blocked_field_ids": ["F033"]\n}'
            + old_text[-trailing:] if trailing else
            trimmed[:-1].rstrip() + ',\n  "blocked_field_ids": ["F033"]\n}\n')


def verify(expected):
    sys.path.insert(0, str(DET.resolve()))
    for key in ('workflow_link_v8', 'workflow_dialogue_v8'):
        sys.modules.pop(key, None)
    importlib.invalidate_caches()
    from document_workflow_v8 import load_workflow
    from workflow_link_v8 import link_workflow
    from workflow_dialogue_v8 import WorkflowDialogueV8, nav_allowed

    w = load_workflow(WORKFLOW)
    m = read_json(MANIFEST)
    t = read_json(TARGETS) if TARGETS.is_file() else None
    linked = link_workflow(w, m, t)
    if DOC_BLOCK_ID not in linked.fields:
        raise AssertionError('F033 not in workflow fields')
    f = linked.fields[DOC_BLOCK_ID]
    assert f.nav_status == 'blocked_by_review', f.nav_status
    assert not nav_allowed(f), 'F033 remains navigable'
    assert f.target_mm is None, 'F033 still has a tracking target'
    # Do not clear uncertainty just to make a target navigable.
    for fid in ('F011', 'F019'):
        assert fid in linked.fields, f'{fid} missing from document'
        assert linked.fields[fid].uncertain, f'{fid}: uncertainty unexpectedly removed'
        assert not nav_allowed(linked.fields[fid]), f'{fid}: unsafe navigation allowed'

    counts = []
    for idx, app in enumerate(linked.app_types, 1):
        controller = WorkflowDialogueV8(linked)
        controller.start()
        controller.on_user(str(idx))
        actual = len(controller.steps)
        assert controller.selected_type == app.id, f'{app.id}: selected type mismatch'
        assert controller.state == 'WAIT_CONSENT', (app.id, controller.state)
        assert actual == expected.get(app.id), (app.id, actual, expected)
        counts.append(f'{app.id}={actual}')

    # The controller must refuse to emit START_TRACKING for F033 even if asked.
    controller = WorkflowDialogueV8(linked)
    from workflow_dialogue_v8 import Item
    controller.steps = [Item('field', [DOC_BLOCK_ID])]
    controller.index = 0
    controller.state = 'ASK_NAV'
    events = controller.on_user('네 진행해 주세요')
    assert not any(e.kind == 'START_TRACKING' for e in events), 'F033 started tracking!'
    print('[VERIFY PASS] F033: blocked_by_review, no tracking target, no START_TRACKING')
    print('[VERIFY PASS] F011 and F019 remain uncertain and blocked')
    print('[VERIFY PASS] Plans unchanged:', ', '.join(counts))


def main():
    for path in (LINKER, MANIFEST, WORKFLOW):
        if not path.is_file():
            raise SystemExit(f'[STOP] Missing required file: {path}')
    current = read_json(MANIFEST)
    if current.get('schema') != 'a4_navigation_manifest_v1':
        raise SystemExit('[STOP] Unexpected manifest schema')
    w = read_json(WORKFLOW)
    if w.get('schema') != 'document_workflow_v1':
        raise SystemExit('[STOP] Unexpected workflow schema')
    if w.get('document_id') != 'document1':
        raise SystemExit('[STOP] This patch is only for document1; refusing to modify a different form')
    if w.get('provenance', {}).get('image_sha256') != current.get('source_image_sha256'):
        raise SystemExit('[STOP] Workflow/manifest image hashes differ')

    sys.path.insert(0, str(DET.resolve()))
    from workflow_link_v8 import link_workflow
    orig = link_workflow(w, current, read_json(TARGETS) if TARGETS.is_file() else None)
    from workflow_dialogue_v8 import WorkflowDialogueV8
    expected = {}
    for a in orig.app_types:
        d = WorkflowDialogueV8(orig)
        d._load_plan(a.id)
        expected[a.id] = len(d.steps)
    if not all(expected.values()):
        raise SystemExit(f'[STOP] Unexpected empty plan(s): {expected}')
    for fid in ('F011', 'F019', 'F033'):
        if fid not in orig.fields:
            raise SystemExit(f'[STOP] Missing field {fid}')
    print('[BEFORE] document1 plans:', expected, 'F033 nav=', orig.fields['F033'].nav_status)

    old_linker = LINKER.read_text(encoding='utf-8-sig')
    old_manifest = MANIFEST.read_text(encoding='utf-8-sig')
    try:
        new_linker = check_source(old_linker)
        new_manifest = updated_manifest_text(old_manifest)
    except Exception as exc:
        raise SystemExit(f'[STOP] Cannot safely prepare patch: {exc}')
    if old_linker == new_linker and old_manifest == new_manifest:
        verify(expected)
        print('[ALREADY OK] Review block already applied; no files changed.')
        return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backups = {}
    for p in (LINKER, MANIFEST):
        backup = p.with_name(p.name + f'.backup_review_{stamp}')
        shutil.copy2(p, backup)
        backups[p] = backup
    try:
        LINKER.write_text(new_linker, encoding='utf-8')
        MANIFEST.write_text(new_manifest, encoding='utf-8')
        verify(expected)
    except Exception as exc:
        for original, backup in backups.items():
            shutil.copy2(backup, original)
        raise SystemExit(f'[ROLLBACK] Verification failed; restored backups: {type(exc).__name__}: {exc}')

    print('[PATCH OK] F033 blocked permanently in document1 navigation manifest + generic linker.')
    for original, backup in backups.items():
        print(f'[BACKUP] {original} -> {backup.name}')
    print('[NOTE] Re-generating navigation_manifest.json can remove the manual block; reapply review before reuse.')
    print('[NEXT] Re-run step3_audit_workflow.py; F033 warning should disappear.')


if __name__ == '__main__':
    main()
