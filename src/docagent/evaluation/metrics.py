"""박스 일대일 매칭, 안내 오매칭, 마스킹 면적을 독립적으로 평가한다."""
from __future__ import annotations

from collections import Counter
from typing import Any

from docagent.evaluation.dataset import ObjectBox, Sample, box
from docagent.guidance.rules import Observation, RuleCatalog


def iou(a: tuple, b: tuple) -> float:
    intersection=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])) * max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection)


def ratio(n: float, d: float) -> float | None:
    return n/d if d else None


def detection_counts(truth: list[ObjectBox], predictions: list[ObjectBox], *, threshold: float, confidence: float) -> dict:
    counts={kind:dict(tp=0,fp=0,fn=0,matched_iou=[]) for kind in ('checkbox','signature')}
    for kind,c in counts.items():
        targets=[x for x in truth if x.kind==kind]; available=set(range(len(targets)))
        proposed=sorted((x for x in predictions if x.kind==kind and x.confidence>=confidence),key=lambda p:-p.confidence)
        for p in proposed:
            ranked=sorted(((iou(p.bbox,targets[i].bbox),i) for i in available),key=lambda t:(-t[0],t[1]))
            if ranked and ranked[0][0]>=threshold:
                overlap,index=ranked[0];available.remove(index);c['tp']+=1;c['matched_iou'].append(overlap)
            else: c['fp']+=1
        c['fn']=len(available)
    return counts


def detection_metrics(samples: list[Sample], *, threshold: float, confidence: float) -> dict:
    totals={kind:dict(tp=0,fp=0,fn=0,matched_iou=[]) for kind in ('checkbox','signature')}
    evaluated=0
    for sample in samples:
        if sample.truth is None or sample.predictions is None: continue
        evaluated+=1
        counts=detection_counts(sample.truth,sample.predictions,threshold=threshold,confidence=confidence)
        for kind,c in counts.items():
            for k in ('tp','fp','fn'): totals[kind][k]+=c[k]
            totals[kind]['matched_iou']+=c['matched_iou']
    result={}
    for kind,c in totals.items():
        tp,fp,fn=(c[k] for k in ('tp','fp','fn'))
        result[kind]=dict(tp=tp,fp=fp,fn=fn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
                         f1=ratio(2*tp,2*tp+fp+fn),mean_matched_iou=ratio(sum(c['matched_iou']),len(c['matched_iou'])))
        if not evaluated: result[kind]={k:None for k in result[kind]}
    return dict(evaluated_samples=evaluated,unlabeled_samples=sum(s.truth is None for s in samples),
                missing_predictions=sum(s.predictions is None for s in samples),classes=result)


def union_area(boxes: list[tuple]) -> float:
    """겹친 사각형을 중복 계산하지 않는 연속 좌표 면적."""
    xs=sorted({x for b in boxes for x in (b[0],b[0]+b[2])})
    total=0.0
    for left,right in zip(xs,xs[1:]):
        intervals=sorted((b[1],b[1]+b[3]) for b in boxes if b[0]<right and b[0]+b[2]>left)
        length=0.0; end=float('-inf')
        for lo,hi in intervals:
            length+=max(0,hi-max(lo,end));end=max(end,hi)
        total+=(right-left)*length
    return total


def mask_metrics(samples: list[Sample]) -> dict:
    totals=dict(evaluated_samples=0,truth_area=0.,masked_area=0.,covered_area=0.,page_area=0.)
    for s in samples:
        if s.mask is None or s.mask.get('truth') is None or s.mask.get('predicted') is None: continue
        truth=[box(b,s.width,s.height) for b in s.mask['truth']]
        predicted=[box(b,s.width,s.height) for b in s.mask['predicted']]
        intersections=[]
        for a in truth:
            for b in predicted:
                lo=(max(a[0],b[0]),max(a[1],b[1])); hi=(min(a[0]+a[2],b[0]+b[2]),min(a[1]+a[3],b[1]+b[3]))
                if hi[0]>lo[0] and hi[1]>lo[1]: intersections.append((lo[0],lo[1],hi[0]-lo[0],hi[1]-lo[1]))
        totals['evaluated_samples']+=1;totals['page_area']+=s.width*s.height
        totals['truth_area']+=union_area(truth);totals['masked_area']+=union_area(predicted);totals['covered_area']+=union_area(intersections)
    t=totals
    missing=t['truth_area']-t['covered_area']; excess=t['masked_area']-t['covered_area']
    return dict(**t,uncovered_area=missing,excess_area=excess,
                uncovered_rate=ratio(missing,t['truth_area']),excess_mask_rate=ratio(excess,t['masked_area']),
                nonprivate_mask_rate=ratio(excess,t['page_area']-t['truth_area']))


def guidance_metrics(samples: list[Sample], catalog: RuleCatalog | None, *, demo: bool = False) -> dict:
    counts=Counter();reasons=Counter();total=0
    for s in samples:
        for case in s.guidance_cases:
            if catalog is None: raise ValueError('안내 평가에는 안내 목록이 필요합니다.')
            observed=Observation.from_dict(case['observation']);result=catalog.match(observed,demo=demo);total+=1
            if result.status!='matched': reasons[result.status]+=1
            if 'expected' not in case:
                counts['unlabeled']+=1;continue
            expected=case['expected']
            if expected is not None and (not isinstance(expected,dict) or set(expected)!={'document_id','document_version','page','field_id'}):
                raise ValueError('정답 안내 식별자 형식을 확인하세요.')
            if result.status=='matched':
                actual={k:getattr(result,k) for k in ('document_id','document_version','page','field_id')}
                counts['correct_match' if expected==actual else 'wrong_match']+=1
            else:
                counts['correct_rejection' if expected is None else 'missed_match']+=1
    labeled=total-counts['unlabeled']
    return dict(total=total,labeled=labeled,**{k:counts[k] for k in ('correct_match','wrong_match','correct_rejection','missed_match','unlabeled')},
                matching_accuracy=ratio(counts['correct_match']+counts['correct_rejection'],labeled),
                wrong_guidance_rate=ratio(counts['wrong_match'],counts['correct_match']+counts['wrong_match']),
                rejection_rate=ratio(sum(reasons.values()),total),rejection_reasons=dict(sorted(reasons.items())))
