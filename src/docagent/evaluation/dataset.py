"""COCO·YOLO 정답, 원본/촬영본 관계, 저장된 예측 결과를 읽는다."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from docagent.guidance.session import digest
from docagent.pii.image_mask import image_digest
from docagent.vision.dataset import CLASS_MAP, read_yolo_labels

ALIASES = {'checkbox':'checkbox', 'check_box':'checkbox', 'signature':'signature', 'signature_field':'signature'}


def load_json(path: Path) -> Any:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('JSON 객체가 필요합니다.')
    return data


def box(values: Any, width: int, height: int) -> tuple[float, float, float, float]:
    if (not isinstance(values,(tuple,list)) or len(values)!=4
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)):
        raise ValueError('좌표는 유한한 숫자 네 개여야 합니다.')
    x,y,w,h=values
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > width+1e-3 or y+h > height+1e-3:
        raise ValueError('박스가 이미지 범위를 벗어났습니다.')
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class ObjectBox:
    kind: str
    bbox: tuple[float,float,float,float]
    confidence: float = 1.0


def object_box(item: dict, width: int, height: int) -> ObjectBox:
    if item.get('class') not in ALIASES:
        raise ValueError('지원하지 않는 클래스입니다.')
    c=item.get('confidence',1.0)
    if type(c) not in (int,float) or not math.isfinite(c) or not 0<=c<=1:
        raise ValueError('신뢰도 범위가 잘못되었습니다.')
    return ObjectBox(ALIASES[item['class']],box(item['bbox'],width,height),float(c))


def _check_label_review(payload: dict, spec: dict, width: int, height: int,
                        source_file_sha256: str | None) -> None:
    """우리 도구의 승인 표식이 있으면 내용 변경·다른 원본 연결을 거절한다.

    일반 COCO에는 승인 표식을 요구하지 않는다. 파일 이름 변경은 허용하되,
    read_manifest에서 실제 입력 파일의 바이트 지문을 전달받으면 이를 대조한다.
    """
    if payload.get('kind') == 'docagent_coco_draft':
        raise ValueError('미승인 라벨 초안을 평가 정답으로 사용할 수 없습니다.')
    info = payload.get('info')
    if not isinstance(info, dict) or 'docagent_label_review' not in info:
        return
    review = info['docagent_label_review']
    if (not isinstance(review, dict) or type(review.get('schema_version')) is not int
            or review['schema_version'] != 1 or review.get('status') != 'approved'
            or review.get('coordinate_system') != 'image_xywh_px'
            or review.get('source_kind') not in ('roboflow', 'official_pdf')
            or any(not isinstance(review.get(k), str) or not review[k].strip()
                   for k in ('candidate_id', 'reviewed_by', 'reviewed_at'))
            or any(not isinstance(review.get(k), str) or not re.fullmatch(r'[0-9a-f]{64}', review[k])
                   for k in ('content_sha256', 'source_file_sha256'))):
        raise ValueError('라벨 검수 승인 기록이 유효하지 않습니다.')
    reviewed_at = datetime.fromisoformat(review['reviewed_at'])
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError('라벨 검수 시각에는 시간대가 필요합니다.')
    core = {key: value for key, value in payload.items() if key != 'info'}
    canonical = json.dumps(core, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    if hashlib.sha256(canonical.encode('utf-8')).hexdigest() != review['content_sha256']:
        raise ValueError('승인 후 COCO 정답 내용이 변경되었습니다.')
    images = payload['images']
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError('검수 승인은 원본 이미지 한 장에만 적용됩니다.')
    image = images[0]
    if (not isinstance(image, dict) or type(image.get('id')) is not int or image['id'] != spec['image_id']
            or any(type(image.get(k)) is not int or image[k] <= 0 for k in ('width', 'height'))
            or [image['width'], image['height']] != [width, height]
            or image.get('file_sha256') != review['source_file_sha256']
            or (source_file_sha256 is not None and source_file_sha256 != review['source_file_sha256'])
            or type(review.get('background_only')) is not bool
            or review['background_only'] != (not payload['annotations'])):
        raise ValueError('라벨 승인과 평가 원본 이미지의 지문·크기가 다릅니다.')


def read_truth(spec: dict | None, root: Path, width: int, height: int, *,
               source_file_sha256: str | None = None) -> list[ObjectBox] | None:
    if spec is None: return None
    if spec['format']=='yolo':
        # 기존 YOLO 리더는 내부 클래스 순서를 사용한다. 외부 순서를 반드시 명시해 재매핑한다.
        mapping=spec.get('class_names')
        if (not isinstance(mapping,list) or len(mapping)!=len(CLASS_MAP)
                or any(k not in ALIASES for k in mapping)
                or len({ALIASES[k] for k in mapping})!=2):
            raise ValueError('YOLO의 두 클래스 이름과 순서를 명시하세요.')
        labels=read_yolo_labels(root/spec['path'])
        result=[]
        for label in labels:
            b=((label.cx-label.w/2)*width,(label.cy-label.h/2)*height,label.w*width,label.h*height)
            # 정규화 텍스트 반올림 오차만 보정하고 범위 초과는 거절한다.
            b=tuple(0 if -1e-3<v<0 else v for v in b)
            result.append(ObjectBox(ALIASES[mapping[label.class_id]],box(b,width,height)))
        return result
    if spec['format']=='coco':
        payload=load_json(root/spec['path'])
        _check_label_review(payload,spec,width,height,source_file_sha256)
        images=payload['images']; categories=payload['categories']; annotations=payload['annotations']
        for records in (images,categories,annotations):
            ids=[r['id'] for r in records]
            if len(set(ids))!=len(ids): raise ValueError('COCO ID가 중복되었습니다.')
        cats={c['id']:c['name'] for c in categories}
        if any(c not in ALIASES for c in cats.values()): raise ValueError('COCO 클래스 이름을 확인하세요.')
        found=[i for i in images if i['id']==spec['image_id']]
        if len(found)!=1 or (found[0]['width'],found[0]['height'])!=(width,height):
            raise ValueError('COCO 이미지 ID 또는 크기가 다릅니다.')
        # 변환기의 정수 반올림을 피하고 COCO 원래 실수 좌표를 보존한다.
        known={i['id'] for i in images}
        if any(a['image_id'] not in known or a['category_id'] not in cats for a in annotations):
            raise ValueError('COCO 참조가 유효하지 않습니다.')
        selected=[a for a in annotations if a['image_id']==spec['image_id']]
        if any(a.get('iscrowd',0) or a.get('ignore',0) for a in selected):
            raise ValueError('crowd/ignore 평가는 지원하지 않습니다. 별도 평가 자료를 사용하세요.')
        return [object_box({'class':cats[a['category_id']],'bbox':a['bbox']},width,height) for a in selected]
    raise ValueError('정답 형식은 coco 또는 yolo 입니다.')


def read_predictions(path: Path, image: np.ndarray) -> list[ObjectBox]:
    data=load_json(path); height,width=image.shape[:2]
    space=data['coordinate_system']
    if space=='image_px':
        if data.get('image_sha256')!=image_digest(image) or data.get('image_size')!=[width,height]:
            raise ValueError('예측의 이미지 지문/크기가 다릅니다.')
        if any('confidence' not in p for p in data['predictions']):
            raise ValueError('예측에는 신뢰도가 필요합니다.')
        return [object_box(p,width,height) for p in data['predictions']]
    if space!='normalized_a4_mm' or data.get('source_image_sha256')!=image_digest(image):
        raise ValueError('예측 좌표계 또는 원본 이미지 지문을 확인하세요.')
    shape=data['normalized_image_shape']; matrix=np.asarray(data['source_to_normalized'],dtype=float)
    if (len(shape)<2 or any(type(n) is not int or n<=0 for n in shape)
            or matrix.shape!=(3,3) or not np.all(np.isfinite(matrix))):
        raise ValueError('좌표 변환 메타데이터가 잘못되었습니다.')
    inverse=np.linalg.inv(matrix)
    result=[]
    for item in data['detections']:
        b=item['box_mm']; x,y,w,h=box([b[k] for k in ('x_mm','y_mm','w_mm','h_mm')],210,297)
        points=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],dtype=float)*[shape[1]/210,shape[0]/297]
        projected=np.c_[points,np.ones(4)]@inverse.T
        if np.any(np.abs(projected[:,2])<1e-10): raise ValueError('유효하지 않은 원근 좌표입니다.')
        p=projected[:,:2]/projected[:,2,None]
        lo=np.maximum(p.min(axis=0),[0,0]); hi=np.minimum(p.max(axis=0),[width,height])
        result.append(object_box({'class':item['type'],'confidence':item['confidence'],
                                  'bbox':[lo[0].item(),lo[1].item(),(hi[0]-lo[0]).item(),(hi[1]-lo[1]).item()]},width,height))
    return result


@dataclass
class Sample:
    alias: str
    width: int
    height: int
    variant: str
    split: str
    form_id: str | None
    source_id: str | None
    training_membership: str
    image_sha256: str
    truth: list[ObjectBox] | None
    predictions: list[ObjectBox] | None
    guidance_cases: list[dict]
    mask: dict | None


def read_manifest(path: Path) -> tuple[dict, list[Sample]]:
    data=load_json(path);root=path.parent
    if type(data.get('schema_version')) is not int or data['schema_version']!=1:
        raise ValueError('평가 명세 버전은 1입니다.')
    for key in ('dataset_version','model_version','prediction_version'):
        if not isinstance(data.get(key),str) or not data[key].strip(): raise ValueError('평가 버전이 필요합니다.')
    records=data['samples']; ids=[r['id'] for r in records]
    if not records or any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('중복 없는 샘플 ID가 필요합니다.')
    samples=[]
    for index,s in enumerate(records,1):
        if s['variant'] not in ('original','photo') or s['split'] not in ('train','val','test','external','unknown'):
            raise ValueError('자료 종류와 분할을 확인하세요.')
        membership=s.get('training_membership','unknown')
        if membership not in ('included','excluded','unknown') or (s['split']=='train' and membership=='excluded'):
            raise ValueError('학습 포함 여부가 잘못되었습니다.')
        if s['split']=='train': membership='included'
        for key in ('form_id','source_id'):
            if s.get(key) is not None and (not isinstance(s[key],str) or not s[key].strip()):
                raise ValueError('자료 그룹 ID가 잘못되었습니다.')
        image_bytes=(root/s['image']).read_bytes()
        image=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise ValueError('이미지를 읽을 수 없습니다.')
        height,width=image.shape[:2]
        if [width,height]!=s['image_size']: raise ValueError('명세와 이미지 크기가 다릅니다.')
        truth=read_truth(s.get('labels'),root,width,height,
                         source_file_sha256=hashlib.sha256(image_bytes).hexdigest())
        preds=read_predictions(root/s['predictions'],image) if s.get('predictions') else None
        samples.append(Sample(f'sample_{index:04}',width,height,s['variant'],s['split'],s.get('form_id'),s.get('source_id'),
                              membership,image_digest(image),truth,preds,s.get('guidance_cases',[]),s.get('mask')))
    # 파일 경로/내용 대신 정답과 픽셀 지문을 묶어 기준 자료 변경을 추적한다.
    data['_dataset_fingerprint']=digest([dict(image=s.image_sha256,truth=[vars(x) for x in s.truth] if s.truth is not None else None,
        variant=s.variant,split=s.split,form=s.form_id,source=s.source_id,membership=s.training_membership,
        guidance=[dict(labeled='expected' in c,expected=c.get('expected')) for c in s.guidance_cases], mask_truth=s.mask.get('truth') if s.mask else None) for s in samples])
    data['_prediction_fingerprint']=digest([dict(predictions=[vars(x) for x in s.predictions] if s.predictions is not None else None,
        observations=[c['observation'] for c in s.guidance_cases],mask_prediction=s.mask.get('predicted') if s.mask else None) for s in samples])
    return data,samples


def audit_dataset(samples: list[Sample]) -> dict:
    """분할·양식·원본 관계와 동일 픽셀 중복을 검사. 미상은 미상으로 유지한다."""
    groups={name:defaultdict(list) for name in ('form_id','source_id','image_sha256')}
    for s in samples:
        for key in groups:
            value=getattr(s,key)
            if value is not None: groups[key][value].append(s)
    duplicates=[];crossings=[];overlap=set()
    for key,values in groups.items():
        for members in values.values():
            if len(members)>1 and key=='image_sha256': duplicates.append([s.alias for s in members])
            splits={s.split for s in members}-{'unknown'}
            if len(splits)>1: crossings.append(dict(kind=key,samples=[s.alias for s in members],splits=sorted(splits)))
            if any(s.training_membership=='included' for s in members):
                overlap.update(s.alias for s in members if s.split in ('val','test','external'))
    return dict(duplicate_groups=duplicates,cross_split_groups=crossings,
                evaluation_training_overlap=sorted(overlap | {s.alias for s in samples if s.split in ('val','test','external') and s.training_membership=='included'}),
                unknown_training=[s.alias for s in samples if s.training_membership=='unknown'],
                unknown_group=[s.alias for s in samples if not s.form_id or not s.source_id],
                unknown_split=[s.alias for s in samples if s.split=='unknown'])
