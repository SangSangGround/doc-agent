"""CLI-only conversation preview using the REAL v8 Workflow and native Manifest.
No camera, microphone, MediaPipe, Gemini API, or physical arrival confirmation.
/arrived creates a deliberately SIMULATED arrival, for testing dialogue only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from document_workflow_v8 import load_workflow
from workflow_link_v8 import link_workflow
from workflow_dialogue_v8 import WorkflowDialogueV8


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else None


def output(events):
    for e in events:
        if e.kind == 'SPEAK':
            print('AI:', e.text)
        elif e.kind == 'START_TRACKING':
            print('[SIMULATED TRACK START]', e.field_id, '(no actual tracking)')
        elif e.kind == 'STOP_TRACKING':
            print('[SIMULATED TRACK STOP]')
        elif e.kind == 'FINISHED':
            print('[SESSION FINISHED]')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workflow', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--targets', type=Path)
    args=ap.parse_args()
    linked=link_workflow(load_workflow(args.workflow), read(args.manifest), read(args.targets))
    ctl=WorkflowDialogueV8(linked)
    output(ctl.start())
    print('입력: 기록 / 정정·변경 / 삭제 / 예 / 아니요 / 완료 / 건너뛰기 / 이전 / 중단 / 계속')
    print('카메라 테스트가 아닙니다. 테스트 도착만 /arrived, 종료 /quit')
    while True:
        try:
            line=input('USER> ').strip()
        except (KeyboardInterrupt, EOFError):
            print('\n종료')
            return
        if line == '/quit':
            return
        if line == '/arrived':
            print('[MOCK ARRIVAL - NOT PHYSICALLY VERIFIED]')
            output(ctl.on_arrived(ctl.current_track))
        else:
            output(ctl.on_user(line))
        if ctl.state == 'FINISHED':
            print(json.dumps(ctl.summary(),ensure_ascii=False,indent=2))
            return

if __name__=='__main__':
    main()
