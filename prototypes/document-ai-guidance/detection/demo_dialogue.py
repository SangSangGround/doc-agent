"""CLI-only simulation. Never starts real video, TTS, STT, or tracking."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from workflow_schema import load_workflow
from navigation_bridge import NavigationRegistry
from dialogue_controller import DialogueController, State


def main():
    p = argparse.ArgumentParser(description='공통 Dialogue Controller 데모 (텍스트 입력)')
    p.add_argument('--workflow', type=Path, required=True)
    p.add_argument('--targets', type=Path)
    p.add_argument('--links', type=Path)
    args = p.parse_args()
    if bool(args.targets) != bool(args.links):
        p.error('--targets and --links must be supplied together')
    workflow = load_workflow(args.workflow)
    nav = NavigationRegistry.from_files(workflow, args.targets, args.links) if args.targets else None
    controller = DialogueController(workflow, nav)

    def emit(events):
        for event in events:
            if event.kind == 'SPEAK':
                print('AI:', event.text)
            elif event.kind == 'START_TRACKING':
                print(f'[SIMULATION] START_TRACKING: field={event.field_id}, approved target={event.target_id}')
                print('도착 모의 신호를 주려면 /arrived 입력')
            elif event.kind == 'FINISHED':
                print('RESULT:', json.dumps(controller.summary(), ensure_ascii=False, indent=2))
            else:
                print(f'[EVENT] {event.kind}')
    emit(controller.start())
    while controller.state != State.FINISHED:
        try:
            utterance = input('사용자> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if utterance == '/arrived':
            emit(controller.on_arrived())
        elif utterance == '/lost':
            emit(controller.on_tracking_lost())
        elif utterance == '/summary':
            print(json.dumps(controller.summary(), ensure_ascii=False, indent=2))
        else:
            emit(controller.on_user(utterance))


if __name__ == '__main__':
    main()
