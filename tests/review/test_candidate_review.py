import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from docagent import candidate_review as cr
from docagent.evaluation.dataset import read_manifest
from docagent.guidance.rules import Observation, RuleCatalog
from docagent.pii.image_mask import MaskPlan, image_digest
from docagent.errors import PiiEgressBlocked
from docagent.review import draft_approval, read_json
from docagent.vision.normalize import normalize_document


@pytest.fixture
def inputs(tmp_path):
    source=tmp_path/'source'; analyses=tmp_path/'analyses'; output=tmp_path/'pack'
    source.mkdir(); (analyses/'sample').mkdir(parents=True)
    image=np.full((198,140,3),255,np.uint8)
    cv2.rectangle(image,(40,80),(48,88),(0,0,0),1)
    data=cv2.imencode('.png',image)[1].tobytes(); (source/'form.png').write_bytes(data)
    n=normalize_document(image,dpi=72)
    b=dict(x_mm=60,y_mm=120,w_mm=10,h_mm=10)
    report=dict(document_id='local-file',coordinate_system='normalized_a4_mm',text_included=True,
                source_image_sha256=image_digest(image),image_sha256=image_digest(n.image),
                normalized_image_shape=list(n.image.shape),source_to_normalized=n.homography.tolist(),
                needs_review=True,capture_quality={'status':'pass'},
                fields=[dict(id='choice_01',type='choice',box_mm=b,confidence=.5,needs_review=True)],
                detections=[dict(type='checkbox',box_mm=b,confidence=.6)],
                ocr_words=[dict(text='가상 동의함',box_mm=b,confidence=.7)],
                stages=[dict(stage='normalize',confidence=.7,ok=False)])
    (analyses/'sample'/'analysis.json').write_text(json.dumps(report),encoding='utf-8')
    entry=dict(document_id='fictional',document_version='v1',page=1,field_id='consent',
               field_type='checkbox',surrounding_text='가상 동의함',keywords=['동의함'],
               forbidden_keywords=[],instruction='가상 항목입니다.',approval=draft_approval())
    c=dict(id='sample',title='가상 문서',document_id='fictional',document_version='v1',page=1,
           image='form.png',file_sha256=hashlib.sha256(data).hexdigest(),
           catalog=dict(schema_version=1,purpose='production',fields=[entry]),
           reference=dict(image_size=[280,396],version_splits={'3':'test'},
                          boxes=[dict(id='a',**{'class':'checkbox'},center_xywh=[88,168,16,16])]),
           mask_regions=[[.1,.1,.1,.1]],ocr_regions=[dict(field_id='consent',bbox=[.1,.1,.5,.5])])
    m=dict(schema_version=1,dpi=72,roboflow_version=3,dataset_version='synthetic-candidate',
           analysis_kind='fake-test-only',candidates=[c])
    return m,source,analyses,output,report


def test_pack_preserves_failed_analysis_and_requires_real_review(inputs):
    m,source,analyses,out,report=inputs
    result=cr.prepare(m,source,analyses,out)
    assert read_json(out/'sample/analysis.json')==report
    context=read_json(out/'sample/context.draft.json')
    assert context['confidence']==0 and context['identity_verified'] is False
    assert context['approval']['status']=='draft'
    assert result['candidates'][0]['guidance_statuses']==['unverified_document']
    with pytest.raises(PiiEgressBlocked): MaskPlan.load(out/'sample/mask-plan.draft.json')
    catalog=RuleCatalog.load(out/'catalog.draft.json')
    assert not catalog.supports('fictional','v1',1)
    # 높은 신뢰도의 가상 관측도 실제 초안의 승인을 대신하지 못한다.
    observed=Observation('fictional','v1',1,'checkbox','가상 동의함',1,1,1,True,False)
    assert catalog.match(observed).status=='unapproved'
    data,samples=read_manifest(out/'evaluation.manifest.json')
    assert samples[0].truth is None and samples[0].training_membership=='unknown'
    assert samples[0].predictions is not None
    assert (out/'sample/masked.preview.png').exists()
    with pytest.raises(ValueError): cr.prepare(m,source,analyses,out)


@pytest.mark.parametrize('change', ['source','report_source','report_image','report_matrix','report_shape','report_text',
                                   'approved','identity','outside','version','split','bbox','mask'])
def test_mismatched_or_unreviewed_inputs_never_produce_pack(inputs,change):
    m,source,analyses,out,report=inputs
    c=m['candidates'][0]
    if change=='source': (source/'form.png').write_bytes(b'changed')
    if change=='report_source': report['source_image_sha256']='other'
    if change=='report_image': report['image_sha256']='other'
    if change=='report_matrix': report['source_to_normalized'][0][2]+=1
    if change=='report_shape': report['normalized_image_shape']=[1,2,3]
    if change=='report_text': report['text_included']=False
    if change=='approved':
        from docagent.review import catalog_edit
        c['catalog']=catalog_edit(c['catalog'],'consent',reviewer='synthetic-reviewer')
    if change=='identity': c['document_version']='different'
    if change=='outside': c['image']='../form.png'
    if change=='version': m['roboflow_version']=4
    if change=='split': c['reference']['version_splits']['3']='unknown'
    if change=='bbox': c['reference']['boxes'][0]['center_xywh']=[1,1,500,500]
    if change=='mask': c['mask_regions']=[[.9,.9,.5,.5]]
    (analyses/'sample/analysis.json').write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises((ValueError,OSError)):
        cr.prepare(m,source,analyses,out)
    assert not out.exists()


def test_reference_scaling_is_display_only_and_markup_is_escaped(inputs):
    m,source,analyses,out,_=inputs
    c=m['candidates'][0]
    c['title']='</title><script>alert(1)</script>'
    c['catalog']['fields'][0]['instruction']='<img src=https://example.com onerror=alert(1)>'
    assert cr._rectangles(c,140,198)[0]['bbox']==[40,80,8,8]
    cr.prepare(m,source,analyses,out)
    page=(out/'sample/reference.html').read_text(encoding='utf-8')
    assert '<script>' not in page and '<img src=https://' not in page
    assert '&lt;script&gt;' in page and "default-src 'none'" in page
    assert read_json(out/'sample/reference.draft.json')['status']=='alignment_unverified'


def test_manual_crop_diagnostic_does_not_replace_analysis_or_identity(inputs):
    from docagent.vision.ocr import StubOcr
    from docagent.contracts import OcrWord, BoxMm
    m,source,analyses,out,report=inputs
    reader=StubOcr([OcrWord('동의함',BoxMm(1,1,2,2),.9)])
    cr.prepare(m,source,analyses,out,crop_reader=reader)
    crop=read_json(out/'sample/crop-ocr.json')['regions'][0]
    assert crop['missing_keywords']==[] and crop['minimum_word_confidence']==.9
    assert crop['scope']=='manual_roi_diagnostic_only'
    assert read_json(out/'sample/analysis.json')==report
    assert not read_json(out/'sample/context.draft.json')['identity_verified']


@pytest.mark.parametrize(('split','expected','membership'), [('valid','val','unknown'),('train','train','included')])
def test_evaluation_keeps_observed_split(inputs,split,expected,membership):
    m,source,analyses,out,_=inputs
    m['candidates'][0]['reference']['version_splits']['3']=split
    cr.prepare(m,source,analyses,out)
    _,samples=read_manifest(out/'evaluation.manifest.json')
    assert (samples[0].split,samples[0].training_membership)==(expected,membership)


def test_real_candidate_config_has_no_approved_guidance():
    m=read_json(Path('examples/guidance/candidates/three-documents.draft.json'))
    assert len(m['candidates'])==3
    assert sum(len(c['catalog']['fields']) for c in m['candidates'])==4
    for c in m['candidates']:
        catalog=RuleCatalog(c['catalog'])
        assert not catalog.supports(c['document_id'],c['document_version'],c['page'])
        assert all(e['approval']['status']=='draft' for e in c['catalog']['fields'])
        assert c['reference']['version_splits']['3']=='test'
    assert read_json(Path('examples/guidance/catalog.production.json'))['fields']==[]


def official_inputs(m):
    m.pop('roboflow_version')
    m['source_kind']='official_pdf'
    c=m['candidates'][0]
    c['reference'].pop('version_splits')
    c['official_source']=dict(url='https://example.gov/form.pdf',pdf_sha256='a'*64,revision='2026-04-28')
    return c


def test_official_pdf_does_not_invent_roboflow_split_or_training_exclusion(inputs):
    m,source,analyses,out,report=inputs
    official_inputs(m)
    cr.prepare(m,source,analyses,out)
    _,samples=read_manifest(out/'evaluation.manifest.json')
    assert samples[0].split=='external' and samples[0].training_membership=='unknown'
    assert samples[0].truth is None
    assert read_json(out/'sample/analysis.json')==report
    assert read_json(out/'sample/context.draft.json')['approval']['status']=='draft'


def test_official_training_exclusion_requires_and_preserves_scoped_evidence(inputs):
    m,source,analyses,out,_=inputs
    c=official_inputs(m)
    c['training_membership']='excluded'
    with pytest.raises(ValueError): cr.prepare(m,source,analyses,out)
    assert not out.exists()
    evidence=dict(status='not_found_in_project_training',checked_versions=[1,2,3],
                  project_url='https://example.org/project',checked_on='2026-10-25',
                  scope='This project only; pretraining unknown',evidence='audit.md')
    c['training_audit']=evidence
    result=cr.prepare(m,source,analyses,out)
    s=result['candidates'][0]
    assert s['training_membership']=='excluded' and s['training_audit']==evidence
    assert s['independent_external_test'] is False and s['approved_fields']==0
    record=read_json(out/'evaluation.manifest.json')['samples'][0]
    assert record['training_audit']==evidence and record['labels'] is None


@pytest.mark.parametrize('change',[
    lambda m,c:m.update(source_kind='other'),
    lambda m,c:c['official_source'].update(url='file:///private/form.pdf'),
    lambda m,c:c['official_source'].update(url='https://user:secret@example.org/form.pdf'),
    lambda m,c:c['official_source'].update(pdf_sha256='wrong'),
    lambda m,c:c['official_source'].update(revision=''),
])
def test_invalid_official_provenance_rejected(inputs,change):
    m,source,analyses,out,_=inputs
    c=official_inputs(m);change(m,c)
    with pytest.raises(ValueError): cr.prepare(m,source,analyses,out)
    assert not out.exists()
