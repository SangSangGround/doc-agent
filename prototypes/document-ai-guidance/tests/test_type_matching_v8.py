"""Document-independent voice type-name regression tests (no microphone required)."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'detection'))
from workflow_dialogue_v8 import WorkflowDialogueV8, norm


class VoiceTypeTests(unittest.TestCase):
    def setUp(self):
        self.linked = SimpleNamespace(app_types=[
            SimpleNamespace(id='record', spoken_name='기록 신청', aliases=['기록','기록신청','신규기록','등록','기력']),
            SimpleNamespace(id='change', spoken_name='정정ㆍ변경 신청', aliases=['정정변경','정정','변경','수정']),
            SimpleNamespace(id='delete', spoken_name='삭제 신청', aliases=['삭제','삭제신청','말소','삭재']),
        ], fields={})
        self.controller = WorkflowDialogueV8(self.linked)

    def assert_choice(self, spoken, choice):
        with self.subTest(spoken=spoken, choice=choice):
            result = self.controller._select_type(norm(spoken))
            self.assertIsNotNone(result, f'{spoken!r} was rejected')
            self.assertEqual(result.id, choice)

    def test_record_natural_forms(self):
        for spoken in ('기록', '기록 신청', '기록이요', '기록으로 할게요', '기록할게요',
                       '기록으로 해주세요', '기록 신청할게요', '저는 기록이요',
                       '기록 기록', '기록 기록 기록', '등록', '기력', '1번', '1번이요'):
            self.assert_choice(spoken, 'record')

    def test_change_natural_forms(self):
        for spoken in ('정정변경', '정정ㆍ변경', '정정변경 신청', '정정', '변경',
                       '변경할게요', '변경이요', '2번', '수정'):
            self.assert_choice(spoken, 'change')

    def test_delete_natural_forms(self):
        for spoken in ('삭제', '삭제 신청', '삭제해주세요', '삭제로할게요', '말소', '삭재', '3번', '3번이요'):
            self.assert_choice(spoken, 'delete')

    def test_ambiguous_or_unrelated_not_auto_selected(self):
        for spoken in ('기록 또는 삭제', '기록 삭제', '기록이랑변경', '기록했다가삭제',
                       '신청', '아무거나', '기록하고 삭제', '새로운 문서', '기록 기록 삭제'):
            with self.subTest(spoken=spoken):
                self.assertIsNone(self.controller._select_type(norm(spoken)))

    def test_dynamic_type_names_no_hardcoding(self):
        linked = SimpleNamespace(app_types=[
            SimpleNamespace(id='issue', spoken_name='발급 신청', aliases=['발급']),
            SimpleNamespace(id='reissue', spoken_name='재발급 신청', aliases=['재발급']),
        ], fields={})
        ctl = WorkflowDialogueV8(linked)
        self.assertEqual(ctl._select_type(norm('발급이요')).id, 'issue')
        self.assertEqual(ctl._select_type(norm('재발급할게요')).id, 'reissue')
        self.assertIsNone(ctl._select_type(norm('신규')))
        # A shared alias must never be guessed.
        linked.app_types[1].aliases.append('발급')
        self.assertIsNone(ctl._select_type(norm('발급')))


if __name__ == '__main__':
    unittest.main()
