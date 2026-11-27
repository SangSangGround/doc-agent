"""Safe, narrowly-scoped v8 patch: automatic document tracking recovery.

Place at hand-tracking root and execute with .venv_scan Python. Patches ONLY:
  detection/workflow_dialogue_v8.py
  detection/multi_field_voice_tracking_v8.py
It backs up both, checks syntax and behavior, and rolls back on failure.
"""
from __future__ import annotations

import ast
import importlib.util
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
CONTROLLER = ROOT / "detection" / "workflow_dialogue_v8.py"
LIVE = ROOT / "detection" / "multi_field_voice_tracking_v8.py"
MARKER_C = "def on_tracking_restored(self):"
MARKER_L = "[AUTO DOCUMENT RESTORED]"


def only_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise ValueError(f"{label}: expected exactly one anchor, found {n}; no files changed")
    return text.replace(old, new, 1)


def replace_method(text: str, method: str, code: str) -> str:
    tree = ast.parse(text)
    cls = next((x for x in tree.body if isinstance(x, ast.ClassDef)
                and x.name == "WorkflowDialogueV8"), None)
    if cls is None:
        raise ValueError("WorkflowDialogueV8 class missing")
    candidates = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == method]
    if len(candidates) != 1:
        raise ValueError(f"Expected one {method} method, found {len(candidates)}")
    node = candidates[0]
    parts = text.splitlines(keepends=True)
    parts[node.lineno - 1:node.end_lineno] = [s + "\n" for s in code.rstrip("\n").split("\n")]
    output = "".join(parts)
    ast.parse(output)
    return output


def replace_paused_branch(text: str) -> str:
    tree = ast.parse(text)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "WorkflowDialogueV8")
    on_user = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "on_user")
    branches = [x for x in on_user.body if isinstance(x, ast.If)
                and isinstance(x.test, ast.Compare) and ast.unparse(x.test) == "self.state == 'PAUSED'"]
    if len(branches) != 1:
        raise ValueError("PAUSED branch cannot be identified safely")
    node = branches[0]
    code = '''        if self.state == 'PAUSED':
            if v in PAUSE:
                # Explicit user pause overrides camera-based automatic resumption.
                self.auto_resume_tracking = False
                return self.say('안내를 일시 중단했습니다. 다시 시작하려면 계속이라고 말씀해 주세요.')
            if v in RESUME:
                self.state = self.resume_state or 'ASK_NAV'
                self.resume_state = None
                self.auto_resume_tracking = False
                if self.state == 'TRACKING' and self.current_track:
                    return [*self.say('손가락 안내를 재개합니다.'),
                            Event('START_TRACKING', field_id=self.current_track)]
                return self.say('안내를 재개합니다. ' + self.last_spoken)
            if self.auto_resume_tracking:
                return self.say('문서를 자동으로 다시 찾고 있습니다. 인식되면 손가락 안내를 재개하겠습니다.')
            return self.say('안내가 일시 중단된 상태입니다. 계속이라고 말씀해 주세요.')
'''
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [s + "\n" for s in code.rstrip("\n").split("\n")]
    result = "".join(lines)
    ast.parse(result)
    return result


def patch_controller(src: str) -> str:
    if MARKER_C in src:
        raise ValueError("Controller already has automatic restore method; refusing a duplicate patch")
    src = only_once(src, "        self.resume_state = None\n", "        self.resume_state = None\n        self.auto_resume_tracking = False\n", "init marker")
    src = replace_paused_branch(src)
    src = only_once(src,
        "        if v in PAUSE and self.state != 'FINISHED':\n            self.resume_state = self.state\n",
        "        if v in PAUSE and self.state != 'FINISHED':\n            self.auto_resume_tracking = False\n            self.resume_state = self.state\n",
        "explicit pause marker")
    src = replace_method(src, "on_tracking_lost", '''    def on_tracking_lost(self):
        if self.state != 'TRACKING':
            return []
        self.resume_state = 'TRACKING'
        self.state = 'PAUSED'
        self.auto_resume_tracking = True
        return [Event('STOP_TRACKING'), *self.say(
            '문서 인식이 잠시 끊어졌습니다. 자동으로 문서를 다시 찾고 있습니다.')]

    def on_tracking_restored(self):
        """Called by live camera loop only after stable, same-field re-acquisition."""
        if (self.state != 'PAUSED' or self.resume_state != 'TRACKING'
                or not self.auto_resume_tracking or not self.current_track):
            return []
        field = self.linked.fields.get(self.current_track)
        if field is None or not nav_allowed(field) or field.target_mm is None:
            return []
        self.state = 'TRACKING'
        self.resume_state = None
        self.auto_resume_tracking = False
        return [Event('START_TRACKING', field_id=field.id),
                *self.say('문서가 인식되었습니다. 안내를 시작합니다. 손가락을 종이 위에 올려주세요.', field.id)]
''')
    compile(src, str(CONTROLLER), "exec")
    return src


def patch_live(src: str) -> str:
    if MARKER_L in src:
        raise ValueError("Live camera already has the automatic recovery patch; refusing duplicate")
    src = only_once(src,
        "    stable_since = None\n    target_visible = False\n",
        "    stable_since = None\n    target_visible = False\n    recovery_since = None\n    recovery_frames = 0\n",
        "initial recovery counters")
    src = only_once(src,
        "                if result is None:\n                    target_visible = False\n",
        "                if result is None:\n                    recovery_since = None\n                    recovery_frames = 0\n                    target_visible = False\n",
        "reset-on-lost-frame")
    src = only_once(src,
        "                        put_line(vis, f'DIALOGUE {controller.state}', 61, (0,210,255), .58)\n",
        """                        # Auto-resume ONLY after consecutive stable paper matches and
                        # the SAME approved target remains inside the camera image.
                        if (controller.state == 'PAUSED'
                                and controller.resume_state == 'TRACKING'
                                and getattr(controller, 'auto_resume_tracking', False)
                                and target_visible):
                            if recovery_since is None:
                                recovery_since = now
                            recovery_frames += 1
                            if now - recovery_since >= 1.0 and recovery_frames >= 6:
                                fid = controller.current_track
                                print(f'[AUTO DOCUMENT RESTORED] target={fid} stable_for={now-recovery_since:.2f}s', flush=True)
                                emit(controller.on_tracking_restored())
                                recovery_since = None
                                recovery_frames = 0
                        else:
                            recovery_since = None
                            recovery_frames = 0
                        put_line(vis, f'DIALOGUE {controller.state}', 61, (0,210,255), .58)
""",
        "camera reacquisition gate")
    src = src.replace(
        '추적이 아직 안정되지 않았거나 목표가 화면 밖에 있습니다. 종이가 선명해진 다음 계속이라고 말씀해 주세요.',
        '문서 위치를 아직 안정적으로 찾지 못했습니다. 잠시 기다려 주세요.')
    compile(src, str(LIVE), "exec")
    return src


def test_controller(path: Path):
    # Import the current, patched controller using its exact path.
    spec = importlib.util.spec_from_file_location("test_auto_restore_v8", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from types import SimpleNamespace
    field = SimpleNamespace(id="F001", navigable=True, role="applicant", reviewed_role="applicant",
                            ai_role="applicant", uncertain=False, confidence="high", target_mm=(54.9, 28.7))
    linked = SimpleNamespace(fields={"F001": field})
    def controller():
        c = module.WorkflowDialogueV8(linked)
        c.state = "TRACKING"
        c.current_track = "F001"
        return c
    c = controller()
    ev = c.on_tracking_lost()
    assert c.state == "PAUSED" and c.auto_resume_tracking and ev[0].kind == "STOP_TRACKING"
    assert "계속이라고 말씀" not in ev[1].text
    ev = c.on_tracking_restored()
    assert c.state == "TRACKING" and not c.auto_resume_tracking
    assert [x.kind for x in ev] == ["START_TRACKING", "SPEAK"]
    assert "문서가 인식되었습니다" in ev[1].text
    assert c.on_tracking_restored() == []
    # Cannot resume if target was not reviewed or if blocked.
    c2 = controller()
    c2.on_tracking_lost()
    field.navigable = False
    assert c2.on_tracking_restored() == [] and c2.state == "PAUSED"
    field.navigable = True
    # Manual user pause is distinct from automatic camera recovery.
    c3 = controller()
    c3.on_user("중단")
    assert c3.state == "PAUSED" and not c3.auto_resume_tracking
    assert c3.on_tracking_restored() == []
    c4 = controller()
    c4.on_tracking_lost()
    c4.on_user("중단")
    assert not c4.auto_resume_tracking and c4.on_tracking_restored() == []
    print("[TEST PASS] Lost -> stable restore -> same reviewed target. Manual pause stays paused.")


def main():
    if not CONTROLLER.is_file() or not LIVE.is_file():
        raise SystemExit("[STOP] Both detection/workflow_dialogue_v8.py and detection/multi_field_voice_tracking_v8.py must exist.")
    a = CONTROLLER.read_text(encoding="utf-8-sig")
    b = LIVE.read_text(encoding="utf-8-sig")
    try:
        new_a = patch_controller(a)
        new_b = patch_live(b)
    except Exception as exc:
        raise SystemExit(f"[STOP] Cannot safely apply patch: {type(exc).__name__}: {exc}. No files changed.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = [(CONTROLLER, CONTROLLER.with_name(CONTROLLER.name + ".backup_auto_" + stamp)),
               (LIVE, LIVE.with_name(LIVE.name + ".backup_auto_" + stamp))]
    for original, backup in backups:
        shutil.copy2(original, backup)
    try:
        CONTROLLER.write_text(new_a, encoding="utf-8")
        LIVE.write_text(new_b, encoding="utf-8")
        test_controller(CONTROLLER)
        print("[PATCH OK] Automatic document reacquisition enabled.")
        print("[FILES] detection/workflow_dialogue_v8.py")
        print("[FILES] detection/multi_field_voice_tracking_v8.py")
        print("[BACKUP] Both originals saved as .backup_auto_<timestamp>.")
        print("[NOTE] Camera stability is checked in the live loop; physical camera is not simulated here.")
    except Exception as exc:
        for original, backup in backups:
            shutil.copy2(backup, original)
        raise SystemExit(f"[ROLLBACK] Files restored because {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
