"""Offline tests: import the unmodified v8 linker; no camera, SDK, network or hardware."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'detection'))
# In this sandbox uploaded original v8 linker lives beside /mnt/data; in the user's
# hand-tracking project it already exists in detection/.
if (ROOT.parent / 'workflow_link_v8.py').is_file():
    sys.path.insert(1, str(ROOT.parent))
from workflow_link_v8 import link_workflow
from workflow_dialogue_v8 import WorkflowDialogueV8, nav_allowed


SHA = 'a' * 64


def sample():
    fields=[]
    items=[('F001','check_box','제목 기록','applicant','required',''),
           ('F002','check_box','제목 삭제','applicant','required',''),
           ('F004','input_field','접수번호','official','unknown',''),
           ('F008','check_box','등록 외국인','applicant','required',''),
           ('F009','check_box','외국국적동포','applicant','required',''),
           ('F010','input_field','로마자 이름','applicant','required',''),
           ('F017','input_field','관계','applicant','conditional','본인에게 해당할 때만 작성'),
           ('F019','signature_field','세대주 서명','applicant','required',''),
           ('F020','input_field','번호','applicant','unknown',''),
           ('F021','signature_field','법정대리인 서명','third_party','required',''),
           ('F025','check_box','성명 변경','applicant','optional',''),
           ('F026','check_box','번호 변경','applicant','optional',''),
           ('F033','check_box','하단 기록','applicant','required','')]
    for fid,kind,label,role,req,cond in items:
        fields.append({'id':fid, 'class_name':kind, 'detected_label':label,'name':label,
                       'role':role,'requirement':req, 'condition_text':cond,'source_quote':cond,
                       'description':f'{label}을 확인하세요', 'aliases':[], 'confidence':'medium',
                       'uncertain':False, 'choice_group_id':('rel' if fid in ('F008','F009') else 'changes' if fid in ('F025','F026') else None),
                       'selects_application_type': ('record' if fid in ('F001','F033') else 'delete' if fid=='F002' else None)})
    workflow={'schema':'document_workflow_v1','document_id':'sample','title':'샘플 문서',
              'description':'문서 설명', 'selection_question':'어떤 신청을 할까요?',
              'provenance':{'image_sha256': SHA, 'ai_inferred':True,'human_reviewed':False},
              'fields':fields,'sections':[
                  {'id':'app','field_ids':['F001','F002','F004'],'applies_to':['all']},
                  {'id':'record','field_ids':['F008','F009','F010','F017','F019','F020','F021'],'applies_to':['record']},
                  {'id':'change','field_ids':['F025','F026'],'applies_to':['record']},
                  {'id':'last','field_ids':['F033'],'applies_to':['all']}],
              'application_types':[
                  {'id':'record','spoken_name':'기록','aliases':['기록','등록'],'description':'기록 신청',
                   'selector_field_ids':['F001','F033'],'section_ids':['app','record','change','last']},
                  {'id':'delete','spoken_name':'삭제','aliases':['삭제'],'description':'삭제 신청',
                   'selector_field_ids':['F002'],'section_ids':['app']}],
              'choice_groups':[{'id':'rel','type':'single_choice','question':'신분 선택','field_ids':['F008','F009']},
                               {'id':'changes','type':'multi_choice','question':'변경 항목','field_ids':['F025','F026']}],
              'validation':{'warnings':[]}}
    manifest={'schema':'a4_navigation_manifest_v1','source_image_sha256': SHA,
              'fields':[{'id':f['id'],'class_name':f['class_name'],'label':f['detected_label'],
                         'role':'applicant','target_mm':{'x':50,'y':55},'approval_method':'manually_adjusted'}
                        for f in fields if f['id'] not in ('F002','F004','F020','F021')]}
    return workflow,manifest


def controller():
    wf,mf=sample()
    return WorkflowDialogueV8(link_workflow(wf,mf))


def kinds(events):
    return [e.kind for e in events]


class DialogueV8Tests(unittest.TestCase):
    def test_linked_orig_is_manifest_verified(self):
        c=controller()
        self.assertTrue(nav_allowed(c.linked.fields['F001']))
        self.assertFalse(nav_allowed(c.linked.fields['F004']))
        self.assertFalse(nav_allowed(c.linked.fields['F020']))

    def test_select_type_and_start(self):
        c=controller()
        self.assertEqual(c.state,'INIT')
        c.start()
        self.assertEqual(c.state,'SELECT_TYPE')
        c.on_user('기록')
        self.assertEqual(c.state,'WAIT_CONSENT')
        self.assertEqual(c.selected_type,'record')
        c.on_user('예')
        self.assertEqual(c.field.id,'F001')

    def test_cannot_spoof_arrival_or_complete(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예')
        c.on_user('예')
        self.assertEqual(c.state,'TRACKING')
        c.on_user('완료')
        self.assertEqual(c.state,'TRACKING')
        c.on_arrived('F999')
        self.assertEqual(c.state,'TRACKING')
        c.on_arrived('F001')
        self.assertEqual(c.state,'WAIT_WRITING')
        c.on_user('완료')
        self.assertEqual(c.status['F001'],'user_claimed')
        self.assertEqual(c.state,'SELECT_CHOICE')

    def test_single_choice_does_not_guide_both(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예')
        c.on_user('건너뛰기')  # skip heading
        self.assertEqual(c.state,'SELECT_CHOICE')
        c.on_user('2번')
        self.assertEqual(c.field.id,'F009')
        self.assertEqual(c.status['F008'],'not_applicable')
        self.assertNotIn('F008', [s.ids[0] for s in c.steps if s.kind=='field'])

    def test_multi_choice_collapsed_and_safe(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예')
        group=[s for s in c.steps if s.group_id=='changes']
        self.assertEqual(len(group),1)
        self.assertEqual(group[0].kind,'choice')
        c.index=c.steps.index(group[0]); c._present()
        c.on_user('1번 2번')
        self.assertEqual(c.field.id,'F025')
        self.assertEqual(c.choice_selections['changes'],['F025','F026'])

    def test_conditional_requires_yes(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예')
        c.index=next(i for i,s in enumerate(c.steps) if s.ids==['F017'])
        c._present()
        self.assertEqual(c.state,'ASK_CONDITION')
        c.on_user('아니요')
        self.assertEqual(c.status['F017'],'not_applicable')
        self.assertNotEqual(c.field.id,'F017')

    def test_conditional_yes_allows_ask(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예')
        c.index=next(i for i,s in enumerate(c.steps) if s.ids==['F017'])
        c._present(); c.on_user('예')
        self.assertEqual(c.state,'ASK_NAV')

    def test_signature_confirm_identity(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예')
        c.index=next(i for i,s in enumerate(c.steps) if s.ids==['F019'])
        c._present()
        self.assertEqual(c.state,'ASK_SIGNER')
        c.on_user('아니요')
        self.assertEqual(c.status['F019'],'needs_third_party')

    def test_signature_yes_then_navigation(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예')
        c.index=next(i for i,s in enumerate(c.steps) if s.ids==['F019'])
        c._present(); c.on_user('예')
        self.assertEqual(c.state,'ASK_NAV')
        self.assertEqual(c.on_user('예')[-1].field_id,'F019')

    def test_unverified_never_starts_tracking(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예')
        c.index=next(i for i,s in enumerate(c.steps) if s.ids==['F020'])
        c._present()
        self.assertNotIn('START_TRACKING',kinds(c.on_user('예')))

    def test_third_party_never_starts_tracking(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예')
        c.index=next(i for i,s in enumerate(c.steps) if s.ids==['F021'])
        c._present()
        self.assertEqual(c.state,'WAIT_EXTERNAL')
        self.assertNotIn('START_TRACKING',kinds(c.on_user('예')))

    def test_pause_resume_tracking(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예'); c.on_user('예')
        ev=c.on_user('중단')
        self.assertIn('STOP_TRACKING',kinds(ev))
        self.assertEqual(c.state,'PAUSED')
        ev=c.on_user('계속')
        self.assertIn('START_TRACKING',kinds(ev))
        self.assertEqual(c.state,'TRACKING')

    def test_tracking_loss_pause(self):
        c=controller(); c.start(); c.on_user('기록'); c.on_user('예'); c.on_user('예')
        self.assertEqual(c.state,'TRACKING')
        c.on_tracking_lost()
        self.assertEqual(c.state,'PAUSED')

    def test_photo_sha_mismatch_blocks(self):
        wf,mf=sample(); mf['source_image_sha256']='b'*64
        with self.assertRaises(ValueError):
            link_workflow(wf,mf)

    def test_official_ai_reviewed_applicant_still_blocked_by_runtime(self):
        wf,mf=sample()
        f=next(f for f in wf['fields'] if f['id']=='F004')
        mf['fields'].append({'id':'F004','class_name':f['class_name'],'label':f['detected_label'],
                             'role':'applicant','target_mm':{'x':50,'y':50}})
        linked=link_workflow(wf,mf)
        self.assertFalse(nav_allowed(linked.fields['F004']))
        ctl=WorkflowDialogueV8(linked)
        ctl.start(); ctl.on_user('기록')
        self.assertNotIn('F004', [fid for step in ctl.steps for fid in step.ids])

    def test_uncertain_meaning_blocks_navigation_even_with_position(self):
        wf,mf=sample()
        next(f for f in wf['fields'] if f['id']=='F010')['uncertain']=True
        linked=link_workflow(wf,mf)
        self.assertTrue(linked.fields['F010'].navigable)
        self.assertFalse(nav_allowed(linked.fields['F010']))

    def test_speech_alias_without_legacy_values(self):
        c=controller(); c.start(); c.on_user('신규')
        self.assertEqual(c.state,'SELECT_TYPE')
        c.on_user('등록')
        self.assertEqual(c.selected_type,'record')


if __name__=='__main__':
    unittest.main()
