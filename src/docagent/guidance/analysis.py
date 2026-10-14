"""로컬 OCR 결과를 안내 관측값으로 변환. 문서 식별은 호출자가 별도로 확인한다."""
from __future__ import annotations

import math
from typing import Any
from docagent.guidance.rules import Observation


def _box(data: dict) -> tuple[float,float,float,float]:
    values=tuple(data[k] for k in ('x_mm','y_mm','w_mm','h_mm'))
    if (any(type(x) not in (int,float) or not math.isfinite(x) for x in values)
            or values[2]<=0 or values[3]<=0):
        raise ValueError('유효하지 않은 OCR 좌표입니다.')
    return values


def _overlap(a,b) -> bool:
    return a[0]<b[0]+b[2] and b[0]<a[0]+a[2] and a[1]<b[1]+b[3] and b[1]<a[1]+a[3]


def _minimum(values) -> float:
    values=list(values)
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):
        return 0.0
    return min(values)


def observation_from_analysis(report: dict[str,Any], context: dict[str,Any], field_id: str) -> Observation:
    if not isinstance(report,dict) or not isinstance(context,dict):
        raise ValueError('분석 결과와 문서 확인 정보는 JSON 객체여야 합니다.')
    for key in ('fields','detections','ocr_words','stages'):
        if not isinstance(report.get(key),list) or any(not isinstance(x,dict) for x in report[key]):
            raise ValueError('분석 결과의 목록 형식이 잘못되었습니다.')
    if not isinstance(report.get('capture_quality',{}),dict):
        raise ValueError('촬영 검사 형식이 잘못되었습니다.')
    if report.get('coordinate_system')!='normalized_a4_mm' or report.get('text_included') is not True:
        raise ValueError('좌표가 확인된 로컬 원문 포함 분석 결과가 필요합니다.')
    matches=[f for f in report['fields'] if f['id']==field_id]
    if len(matches)!=1:
        raise ValueError('분석 항목을 하나로 특정할 수 없습니다.')
    f=matches[0]; box=_box(f['box_mm'])
    # Same padding as regional OCR, clipped to the supported portrait A4 space.
    x0=max(0,box[0]-45);y0=max(0,box[1]-15)
    crop=(x0,y0,min(210,box[0]+box[2]+45)-x0,min(297,box[1]+box[3]+8)-y0)
    words=[w for w in report['ocr_words'] if _overlap(_box(w['box_mm']),crop)]
    words.sort(key=lambda w:(w['box_mm']['y_mm'],w['box_mm']['x_mm']))
    kind='checkbox' if f['type']=='choice' else f['type']
    detections=[d for d in report['detections'] if d['type']==kind and _overlap(_box(d['box_mm']),box)]
    normalized=[s for s in report['stages'] if s['stage']=='normalize']
    quality=report.get('capture_quality',{})
    # Treat omitted/malformed flags as unresolved. No hidden override for regions mode.
    review=(report.get('needs_review') is not False or f.get('needs_review') is not False
            or not report['stages'] or any(s.get('ok') is not True for s in report['stages'])
            or quality.get('status')!='pass')
    return Observation(
        document_id=context['document_id'], document_version=context['document_version'],
        page=context['page'], field_type=kind,
        surrounding_text=' '.join(w['text'] for w in words),
        document_confidence=_minimum([context['confidence']]+[s['confidence'] for s in normalized]) if normalized else 0,
        detection_confidence=_minimum([f['confidence']]+[d['confidence'] for d in detections]) if detections else 0,
        ocr_confidence=_minimum(w['confidence'] for w in words),
        identity_verified=context.get('identity_verified') is True, needs_review=review,
    )
