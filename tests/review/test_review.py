import json

import cv2
import numpy as np
import pytest

from docagent import review
from docagent.guidance.rules import RuleCatalog
from docagent.guidance.session import validate_context
from docagent.pii.image_mask import draft_plan, image_digest, MaskPlan
from tests.guidance.test_analysis import report


def test_html_is_local_and_escapes_ocr():
    image=np.full((80,60,3),255,np.uint8)
    data=report(); data['image_sha256']=image_digest(image)
    data['ocr_words'][0]['text']='</script><img src=https://example.com onerror=alert(1)>'
    html=review.render_review(image, report=data)
    assert '</script><img' not in html
    assert '\\u003c/script' in html and "connect-src 'none'" in html
    assert '마스킹 미리보기' in html
    data['image_sha256']='other'
    with pytest.raises(ValueError): review.render_review(image, report=data)


def test_catalog_edit_requires_new_review_and_catches_changed_approval():
    data=review.read_json(__import__('pathlib').Path('examples/guidance/catalog.demo.json'))
    data['fields'][0]['instruction']='새로운 검토용 안내문'
    with pytest.raises(ValueError): RuleCatalog(data)
    draft=review.catalog_edit(data,'participation')
    assert draft['fields'][0]['approval']['status']=='draft'
    approved=review.catalog_edit(draft,'participation',reviewer='human-test')
    RuleCatalog(approved)
    assert approved['fields'][0]['approval']['reviewed_by']=='human-test'
    assert data['fields'][0]['approval']['status']=='approved'  # 입력 불변


def test_context_approval_binds_report_without_clearing_flags(tmp_path):
    data=report(); data['needs_review']=True
    draft=review.context_draft(data)
    draft.update(document_id='fictional-survey',document_version='demo-1',identity_verified=True,confidence=.99)
    a=tmp_path/'analysis.json';a.write_text(json.dumps(data),encoding='utf-8')
    c=tmp_path/'context.json';c.write_text(json.dumps(draft),encoding='utf-8')
    o=tmp_path/'approved.json'
    assert review.main(['context-approve',str(c),'--analysis',str(a),'--reviewer','human-test',
                        '--confirm-reviewed','--output',str(o)])==0
    validate_context(review.read_json(o),data)
    assert review.read_json(a)['needs_review'] is True
    assert review.main(['context-approve',str(c),'--analysis',str(a),'--reviewer','human-test',
                        '--confirm-reviewed','--output',str(o)])==1  # 덮어쓰기 금지


def test_mask_approval_checks_actual_image_and_nonempty_boxes(tmp_path):
    from docagent.vision.normalize import normalize_document
    raw=np.full((100,70,3),255,np.uint8)
    image=tmp_path/'image.png';image.write_bytes(cv2.imencode('.png',raw)[1].tobytes())
    normalized=normalize_document(raw,dpi=72).image
    plan=draft_plan(normalized);plan['boxes']=[[10,10,20,20]]
    p=tmp_path/'draft.json';p.write_text(json.dumps(plan),encoding='utf-8');out=tmp_path/'approved.json'
    args=['mask-approve',str(p),'--image',str(image),'--dpi','72','--reviewer','human-test',
          '--confirm-reviewed','--output',str(out)]
    assert review.main(args)==0
    assert np.all(MaskPlan.load(out).apply(normalized)[10:30,10:30]==0)
    plan['image_sha256']='changed';p.write_text(json.dumps(plan),encoding='utf-8');out.unlink()
    assert review.main(args)==1 and not out.exists()


def test_approval_requires_explicit_human_action(tmp_path):
    with pytest.raises(SystemExit) as e:
        review.main(['catalog-approve','examples/guidance/catalog.demo.json','--field-id','participation',
                     '--reviewer','human-test','--output',str(tmp_path/'approved.json')])
    assert e.value.code==2
