"""알려진 정답으로 지표·좌표 변환·평가 자료 분리를 검증한다."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from docagent import evaluate
from docagent.evaluation.dataset import ObjectBox, Sample, audit_dataset, read_manifest, read_predictions, read_truth
from docagent.evaluation.metrics import detection_counts, detection_metrics, guidance_metrics, mask_metrics, union_area
from docagent.guidance.rules import RuleCatalog
from docagent.pii.image_mask import image_digest


def write(path, data):
    path.write_text(json.dumps(data),encoding='utf-8')
    return path


def sample(**kw):
    base=dict(alias='sample_0001',width=100,height=100,variant='original',split='test',form_id='form-a',source_id='page-a',
              training_membership='excluded',image_sha256='x',truth=None,predictions=None,guidance_cases=[],mask=None)
    return Sample(**(base|kw))


def fixture(tmp_path):
    image=np.full((100,100,3),255,np.uint8);image[10:30,10:30]=0
    (tmp_path/'original.png').write_bytes(cv2.imencode('.png',image)[1].tobytes())
    (tmp_path/'labels.txt').write_text('0 0.2 0.2 0.2 0.2\n',encoding='utf-8')
    write(tmp_path/'pred.json',dict(coordinate_system='image_px',image_size=[100,100],image_sha256=image_digest(image),
                                  predictions=[dict(**{'class':'check_box'},bbox=[10,10,20,20],confidence=.9)]))
    data=dict(schema_version=1,dataset_version='synthetic-1',model_version='fake-detector',prediction_version='run-1',
              samples=[dict(id='private-original-file',image='original.png',image_size=[100,100],variant='original',split='test',
                            training_membership='excluded',form_id='private-form-name',source_id='page-a',
                            labels=dict(format='yolo',path='labels.txt',class_names=['check_box','signature_field']),predictions='pred.json')])
    path=write(tmp_path/'manifest.json',data)
    return path,data,image


def test_duplicate_predictions_are_false_positives_and_missing_class_is_fn():
    truth=[ObjectBox('checkbox',(10,10,20,20)),ObjectBox('signature',(50,50,20,10))]
    preds=[ObjectBox('checkbox',(10,10,20,20),.99),ObjectBox('checkbox',(10,10,20,20),.98)]
    counts=detection_counts(truth,preds,threshold=.5,confidence=.5)
    assert (counts['checkbox']['tp'],counts['checkbox']['fp'],counts['signature']['fn'])==(1,1,1)
    stats=detection_metrics([sample(truth=truth,predictions=preds)],threshold=.5,confidence=.5)
    assert stats['classes']['checkbox']['precision']==.5
    assert stats['classes']['checkbox']['f1']==pytest.approx(2/3)
    assert stats['classes']['signature']['recall']==0


def test_confidence_order_selects_one_to_one_match():
    truth=[ObjectBox('checkbox',(0,0,10,10)),ObjectBox('checkbox',(8,0,10,10))]
    pred=[ObjectBox('checkbox',(0,0,10,10),.7),ObjectBox('checkbox',(8,0,10,10),.9)]
    assert detection_counts(truth,pred,threshold=.5,confidence=.8)['checkbox']['fn']==1


def test_unlabeled_vs_negative_image_and_missing_predictions():
    assert detection_metrics([sample()],threshold=.5,confidence=.5)['classes']['checkbox']['tp'] is None
    negative=sample(truth=[],predictions=[ObjectBox('checkbox',(0,0,10,10),.99)])
    m=detection_metrics([negative],threshold=.5,confidence=.5)['classes']['checkbox']
    assert m['fp']==1 and m['precision']==0 and m['recall'] is None


def test_mask_union_and_excess_are_not_double_counted():
    s=sample(mask={'truth':[[0,0,10,10],[5,0,10,10]],'predicted':[[0,0,10,10],[0,0,10,10],[20,0,10,10]]})
    result=mask_metrics([s])
    assert result['truth_area']==150 and result['masked_area']==200
    assert result['uncovered_area']==50 and result['excess_area']==100
    assert result['uncovered_rate']==pytest.approx(1/3) and result['excess_mask_rate']==.5
    assert mask_metrics([sample()])['uncovered_rate'] is None


def test_guidance_matches_mismatches_rejections_and_unlabeled():
    catalog=RuleCatalog.load('examples/guidance/catalog.demo.json')
    obs=json.loads(Path('examples/guidance/observation.demo.json').read_text(encoding='utf-8'))
    expected=dict(document_id='fictional-survey',document_version='demo-1',page=1,field_id='participation')
    denied=obs|{'needs_review':True}
    cases=[dict(observation=obs,expected=expected),dict(observation=obs,expected=None),
           dict(observation=denied,expected=None),dict(observation=denied,expected=expected),dict(observation=obs)]
    m=guidance_metrics([sample(guidance_cases=cases)],catalog,demo=True)
    assert m['correct_match']==m['wrong_match']==m['correct_rejection']==m['missed_match']==m['unlabeled']==1
    assert m['matching_accuracy']==.5 and m['wrong_guidance_rate']==.5 and m['rejection_rate']==.4
    assert m['rejection_reasons']=={'needs_review':2}


def test_dataset_audit_groups_original_and_photos_across_splits():
    a=sample(split='train',training_membership='included')
    b=replace(a,alias='sample_0002',variant='photo',split='test',training_membership='unknown',image_sha256='y')
    c=replace(a,alias='sample_0003',split='external',training_membership='excluded')
    audit=audit_dataset([a,b,c])
    assert audit['evaluation_training_overlap']==['sample_0002','sample_0003']
    assert audit['unknown_training']==['sample_0002']
    assert audit['duplicate_groups']==[['sample_0001','sample_0003']]
    assert len(audit['cross_split_groups'])==3


def test_yolo_class_order_and_coco_fractional_coordinates(tmp_path):
    path,data,image=fixture(tmp_path)
    _,samples=read_manifest(path)
    assert samples[0].truth[0].kind=='checkbox'  # 프로젝트 내부의 class 0은 signature지만 외부 매핑은 checkbox
    coco=dict(images=[dict(id=3,file_name='original.png',width=100,height=100)],
              categories=[dict(id=8,name='check_box')],annotations=[dict(id=1,image_id=3,category_id=8,bbox=[10.25,11.5,20.5,21.75])])
    write(tmp_path/'coco.json',coco)
    truth=read_truth(dict(format='coco',path='coco.json',image_id=3),tmp_path,100,100)
    assert truth[0].bbox==(10.25,11.5,20.5,21.75)
    coco['annotations'][0]['iscrowd']=1;write(tmp_path/'coco.json',coco)
    with pytest.raises(ValueError): read_truth(dict(format='coco',path='coco.json',image_id=3),tmp_path,100,100)


def test_inverse_homography_converts_a4_mm_to_source_pixels(tmp_path):
    _,_,image=fixture(tmp_path)
    # 원본 [10,20,20,10] -> 정합 픽셀 [25,47,40,20], A4 폭 210px 높이297px
    data=dict(coordinate_system='normalized_a4_mm',source_image_sha256=image_digest(image),
              normalized_image_shape=[297,210,3],source_to_normalized=[[2,0,5],[0,2,7],[0,0,1]],
              detections=[dict(type='checkbox',confidence=.9,box_mm=dict(x_mm=25,y_mm=47,w_mm=40,h_mm=20))])
    path=write(tmp_path/'analysis.json',data)
    assert read_predictions(path,image)[0].bbox==pytest.approx((10,20,20,10))
    data['source_to_normalized']=[[0,0,0]]*3;write(path,data)
    with pytest.raises(np.linalg.LinAlgError): read_predictions(path,image)


@pytest.mark.parametrize('change',[
    lambda d:d['samples'][0].update(image_size=[10,10]),
    lambda d:d['samples'][0]['labels'].update(path='missing.txt'),
    lambda d:d['samples'][0]['labels'].update(class_names=['other','checkbox']),
    lambda d:d['samples'].append(copy.deepcopy(d['samples'][0])),
    lambda d:d['samples'][0].update(split='other'),
])
def test_invalid_manifest_is_rejected(tmp_path,change):
    path,data,_=fixture(tmp_path);change(data);write(path,data)
    with pytest.raises((ValueError,OSError)): read_manifest(path)


def test_prediction_fingerprint_and_coordinate_system_are_required(tmp_path):
    path,data,_=fixture(tmp_path)
    pred=json.loads((tmp_path/'pred.json').read_text(encoding='utf-8'));pred['image_sha256']='wrong';write(tmp_path/'pred.json',pred)
    with pytest.raises(ValueError): read_manifest(path)


def test_cli_reports_no_images_or_ocr_and_prevents_overwrite(tmp_path,capsys):
    path,_,_=fixture(tmp_path);out=tmp_path/'report'
    assert evaluate.main([str(path),'--output-dir',str(out)])==0
    j=json.loads((out/'evaluation.json').read_text(encoding='utf-8'));md=(out/'evaluation.md').read_text(encoding='utf-8')
    assert j['overall']['detection']['classes']['checkbox']['f1']==1
    assert '측정 불가' in md
    for text in ('private-original-file','private-form-name','original.png','base64','ocr_words'):
        assert text not in json.dumps(j) and text not in md
    assert evaluate.main([str(path),'--output-dir',str(out)])==1
    assert 'private-original-file' not in capsys.readouterr().err
    b=copy.deepcopy(j);b['overall']['detection']['classes']['checkbox']['recall']=.5
    assert evaluate.compare_reports(j,b)['detection_delta']['checkbox']['recall']==.5
    b['settings']['iou_threshold']=.75
    assert not evaluate.compare_reports(j,b)['comparable']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,2,True])
def test_invalid_thresholds_rejected(tmp_path,value):
    path,_,_=fixture(tmp_path)
    with pytest.raises(ValueError): evaluate.evaluate_manifest(path,iou_threshold=value)
