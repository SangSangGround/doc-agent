"""Run with: python -m unittest discover -s tests -v"""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'detection'))
from workflow_schema import load_workflow, validate_workflow, WorkflowError
from navigation_bridge import NavigationRegistry, LinkageError
from dialogue_controller import DialogueController, State


class WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = load_workflow(ROOT / 'examples' / 'sample_document_A.json')
        cls.b = load_workflow(ROOT / 'examples' / 'sample_document_B.json')
        cls.nav = NavigationRegistry.from_files(
            cls.a, ROOT / 'examples' / 'approved_targets_A.json', ROOT / 'examples' / 'field_links_A.json')

    def test_two_distinct_schemas_no_code_changes(self):
        self.assertEqual(len(self.a['application_types']), 2)
        self.assertEqual(len(self.b['application_types']), 0)
        self.assertEqual(DialogueController(self.b).start()[0].kind, 'SPEAK')

    def test_linkage_only_explicit_verified_applicant(self):
        self.assertEqual(self.nav.lookup('F001'), 'T001')
        self.assertEqual(self.nav.lookup('F002'), 'T002')
        self.assertIsNone(self.nav.lookup('F003'))  # official
        self.assertIsNone(self.nav.lookup('F004'))  # unverified target

    def test_refuse_ai_verified_navigation(self):
        c = copy.deepcopy(self.a)
        c['fields'][0]['navigation_target_id'] = 'T001'
        with self.assertRaises(WorkflowError):
            validate_workflow(c)

    def test_refuse_unmapped_navigation(self):
        with self.assertRaises(LinkageError):
            NavigationRegistry(self.a, {'targets': []}, {'F001': 'missing'})

    def test_application_type_flow_and_arrived(self):
        c = DialogueController(self.a, self.nav)
        c.start()
        self.assertEqual(c.state, State.SELECT_TYPE)
        c.on_user('기록')
        self.assertEqual(c.state, State.WAIT_CONSENT)
        c.on_user('예')
        self.assertEqual(c.field['id'], 'F001')
        self.assertEqual(c.state, State.ASK_TO_NAVIGATE)
        ev = c.on_user('응')
        self.assertEqual([x.kind for x in ev][-1], 'START_TRACKING')
        self.assertEqual(ev[-1].target_id, 'T001')
        c.on_user('완료')  # cannot claim done before arrived
        self.assertEqual(c.state, State.TRACKING)
        c.on_arrived()
        self.assertEqual(c.state, State.WAIT_WRITING)
        c.on_user('다 썼어')
        self.assertEqual(c.field_status['F001'], 'user_claimed')
        self.assertEqual(c.field['id'], 'F002')
        c.on_user('예')
        c.on_arrived()
        c.on_user('완료')
        self.assertEqual(c.field['id'], 'F004')
        c.on_user('예')
        self.assertEqual(c.state, State.ASK_TO_NAVIGATE)
        self.assertEqual(c.field_status['F004'], 'unverified')
        c.on_user('건너뛰기')
        self.assertEqual(c.state, State.FINISHED)
        self.assertEqual(c.field_status['F003'], 'pending')
        self.assertIn('F003', c.summary()['excluded_non_applicant_ids'])

    def test_no_type_flow_review_block(self):
        c = DialogueController(self.b, NavigationRegistry(
            self.b, {'targets': [{'id':'B01','verified': True,'kind': 'check_box'}]}, {'F002':'B01'}))
        c.start()
        self.assertEqual(c.state, State.WAIT_CONSENT)
        c.on_user('예')
        self.assertEqual(c.field['id'], 'F001')
        c.on_user('건너뛰기')
        self.assertEqual(c.field['id'], 'F002')
        c.on_user('예')
        self.assertEqual(c.state, State.ASK_TO_NAVIGATE)
        self.assertEqual(c.field_status['F002'], 'unverified')
        c.on_user('건너뛰기')
        self.assertEqual(c.state, State.FINISHED)

    def test_pause_lost_and_resume(self):
        c = DialogueController(self.a, self.nav)
        c.start(); c.on_user('삭제'); c.on_user('예')
        c.on_user('예')
        self.assertEqual(c.state, State.TRACKING)
        ev = c.on_tracking_lost()
        self.assertEqual(ev[0].kind, 'PAUSE_TRACKING')
        self.assertEqual(c.state, State.PAUSED)
        ev = c.on_user('계속')
        self.assertEqual(ev[-1].kind, 'RESUME_TRACKING')
        self.assertEqual(c.state, State.TRACKING)

    def test_silence_does_not_repeat(self):
        c = DialogueController(self.a)
        c.start()
        self.assertEqual(c.on_user('  '), [])

    def test_exclusive_choices_are_not_all_filled(self):
        cdata = copy.deepcopy(self.b)
        cdata['fields'].append({
            'id': 'F004', 'name': '거절', 'kind': 'check_box', 'section_id': 'section_001',
            'role': 'applicant', 'requirement': 'unknown', 'condition': '',
            'applicable_type_ids': [], 'explanation': '거절란', 'source_text': '거절',
            'review_required': True})
        cdata['sections'][0]['field_ids'].append('F004')
        cdata['choice_groups'] = [{'id': 'choice_001', 'field_ids': ['F002', 'F004'], 'exclusive': True}]
        validate_workflow(cdata)
        c = DialogueController(cdata)
        c.start()
        self.assertEqual(c.state, State.SELECT_CHOICE)
        c.on_user('동의')
        self.assertEqual(c.state, State.WAIT_CONSENT)
        self.assertEqual(c.field_status['F004'], 'not_applicable')
        self.assertNotIn('F004', [f['id'] for f in c.active_fields])

    def test_required_section_and_evidence_checks(self):
        cdata = copy.deepcopy(self.a)
        cdata['sections'][0]['field_ids'].remove('F001')
        with self.assertRaises(WorkflowError):
            validate_workflow(cdata)
        cdata = copy.deepcopy(self.a)
        cdata['fields'][0]['source_text'] = ''
        with self.assertRaises(WorkflowError):
            validate_workflow(cdata)

    def test_unknown_application_is_not_accepted(self):
        c = DialogueController(self.a)
        c.start(); c.on_user('변경')
        self.assertEqual(c.state, State.SELECT_TYPE)


if __name__ == '__main__':
    unittest.main()
