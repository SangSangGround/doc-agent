import copy
import json
from pathlib import Path

import pytest
from docagent import guide
from docagent.guidance.analysis import observation_from_analysis
from docagent.guidance.rules import RuleCatalog


def report():
    box=dict(x_mm=60,y_mm=100,w_mm=5,h_mm=5)
    return dict(document_id='arbitrary-file-name',coordinate_system='normalized_a4_mm',text_included=True,
                needs_review=False,capture_quality={'status':'pass'},
                fields=[dict(id='choice_01',type='choice',box_mm=box,confidence=.97,needs_review=False)],
                detections=[dict(type='checkbox',box_mm=box,confidence=.96)],
                ocr_words=[dict(text='가상 설문 참여 동의함 거부함',box_mm=dict(x_mm=66,y_mm=100,w_mm=40,h_mm=4),confidence=.95),
                           dict(text='unrelated-private-text',box_mm=dict(x_mm=180,y_mm=10,w_mm=20,h_mm=4),confidence=.2)],
                stages=[dict(stage='normalize',ok=True,confidence=.98),dict(stage='ocr',ok=True,confidence=.95)])


def context():
    return dict(document_id='fictional-survey',document_version='demo-1',page=1,confidence=.99,identity_verified=True)


def test_bridge_uses_verified_identity_local_words_and_preserves_confidence():
    o=observation_from_analysis(report(),context(),'choice_01')
    assert o.document_id=='fictional-survey'
    assert 'unrelated-private-text' not in o.surrounding_text
    assert (o.document_confidence,o.detection_confidence,o.ocr_confidence)==(.98,.96,.95)
    c=RuleCatalog.load('examples/guidance/catalog.demo.json')
    assert c.match(o,demo=True).status=='matched'


@pytest.mark.parametrize('change',[lambda r:r.update(needs_review=True),
                                  lambda r:r['fields'][0].update(needs_review=True),
                                  lambda r:r['stages'][0].update(ok=False),
                                  lambda r:r.update(capture_quality={'status':'not_checked'})])
def test_bridge_never_clears_review_flags(change):
    r=report();change(r)
    assert observation_from_analysis(r,context(),'choice_01').needs_review


def test_bridge_rejects_redacted_text_and_unknown_id():
    r=report();r['text_included']=False
    with pytest.raises(ValueError):observation_from_analysis(r,context(),'choice_01')
    with pytest.raises(ValueError):observation_from_analysis(report(),context(),'wrong')
    c=context();c.pop('identity_verified')
    assert not observation_from_analysis(report(),c,'choice_01').identity_verified


def test_cli_bridge(tmp_path,capsys):
    a=tmp_path/'analysis.json';c=tmp_path/'context.json'
    a.write_text(json.dumps(report()));c.write_text(json.dumps(context()))
    assert guide.main(['--analysis',str(a),'--document-context',str(c),'--field-id','choice_01',
                       '--catalog','examples/guidance/catalog.demo.json','--demo'])==0
    output=json.loads(capsys.readouterr().out)
    assert output['status']=='matched'
    assert 'unrelated-private-text' not in str(output)


def test_matching_phrase_keeps_row_order_when_ocr_word_tops_are_jittered():
    r=report()
    r['ocr_words']=[
        dict(text='사유',box_mm=dict(x_mm=78,y_mm=100,w_mm=8,h_mm=4),confidence=.95),
        dict(text='전입',box_mm=dict(x_mm=66,y_mm=100.4,w_mm=8,h_mm=4),confidence=.96),
        dict(text='직업',box_mm=dict(x_mm=66,y_mm=106,w_mm=8,h_mm=4),confidence=.94),
    ]
    o=observation_from_analysis(r,context(),'choice_01')
    assert o.surrounding_text=='전입 사유\n직업'
    assert o.ocr_confidence==.94
    assert r['ocr_words'][0]['text']=='사유'  # 증거 입력을 정렬·수정하지 않는다.


@pytest.mark.parametrize('value',[None,[],"private-invalid-input"])
def test_invalid_report_schema_is_rejected_without_echo(value):
    with pytest.raises(ValueError) as error:
        observation_from_analysis(value,context(),'choice_01')
    assert 'private-invalid-input' not in str(error.value)
