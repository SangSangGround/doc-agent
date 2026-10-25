"""저장된 예측과 정답을 비교. API 호출·학습·원본 이미지 복사 없음."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from docagent.evaluation.dataset import audit_dataset, load_json, read_manifest
from docagent.evaluation.metrics import detection_metrics, guidance_metrics, mask_metrics
from docagent.guidance.rules import RuleCatalog
from docagent.guidance.session import digest


class UnseenEvaluationError(ValueError):
    """명세의 미학습 탐지 평가 조건을 충족하지 못했다."""


def evaluate_manifest(path: Path, *, iou_threshold: float = .5, confidence: float = .5,
                      demo: bool = False, require_unseen: bool = False) -> dict:
    for value in (iou_threshold,confidence):
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('임계값은 0~1 사이여야 합니다.')
    if iou_threshold==0: raise ValueError('IoU 임계값은 0보다 커야 합니다.')
    data,samples=read_manifest(path)
    audit = audit_dataset(samples)
    if require_unseen and (
            any(audit[k] for k in ('evaluation_training_overlap', 'unknown_training',
                                   'unknown_group', 'unknown_split'))
            or any(s.split not in ('test', 'external') for s in samples)
            or any(s.truth is None or s.predictions is None for s in samples)):
        raise UnseenEvaluationError('미학습 탐지 평가는 학습 제외·양식 그룹·정답·예측이 확인된 test/external 자료가 필요합니다.')
    catalog=RuleCatalog.load(path.parent/data['catalog']) if data.get('catalog') else None
    def metrics(subset):
        return dict(sample_count=len(subset),detection=detection_metrics(subset,threshold=iou_threshold,confidence=confidence),
                    guidance=guidance_metrics(subset,catalog,demo=demo),masking=mask_metrics(subset))
    versions={name:importlib.metadata.version(name) for name in ('numpy','Pillow','opencv-python-headless')}
    return dict(schema_version=1,generated_at=datetime.now(timezone.utc).isoformat(),
                dataset_version=data['dataset_version'],model_version=data['model_version'],prediction_version=data['prediction_version'],
                dataset_fingerprint=data['_dataset_fingerprint'],prediction_fingerprint=data['_prediction_fingerprint'],manifest_sha256=digest({k:v for k,v in data.items() if not k.startswith('_')}),
                catalog_sha256=digest(load_json(path.parent/data['catalog'])) if catalog else None,
                settings=dict(iou_threshold=iou_threshold,confidence_threshold=confidence,demo=demo,
                              matching='confidence_greedy_one_to_one',iou_space='original_image_axis_aligned_px',
                              require_unseen=require_unseen),
                runtime=dict(python=platform.python_version(),packages=versions),audit=audit,
                overall=metrics(samples),by_variant={k:metrics([s for s in samples if s.variant==k]) for k in ('original','photo')},
                by_split={k:metrics([s for s in samples if s.split==k]) for k in sorted({s.split for s in samples})})


def compare_reports(current: dict, baseline: dict) -> dict:
    if baseline.get('schema_version')!=1: raise ValueError('기준 보고서 버전이 다릅니다.')
    keys=('dataset_fingerprint','settings','catalog_sha256')
    comparable=all(current[k]==baseline.get(k) for k in keys)
    changes={}
    if comparable:
        for kind in ('checkbox','signature'):
            a=current['overall']['detection']['classes'][kind];b=baseline['overall']['detection']['classes'][kind]
            changes[kind]={k:(a[k]-b[k] if a[k] is not None and b[k] is not None else None)
                           for k in ('precision','recall','f1','mean_matched_iou')}
    return dict(comparable=comparable,reason='same_data_and_settings' if comparable else 'data_settings_or_catalog_changed',
                baseline_model_version=baseline['model_version'],baseline_prediction_version=baseline['prediction_version'],detection_delta=changes)


def render_markdown(report: dict) -> str:
    def cell(value): return '측정 불가' if value is None else f'{value:.4f}'
    def safe(value): return str(value).replace('|','\\|').replace('\n',' ').replace('<','&lt;').replace('>','&gt;')
    lines=['# 문서 안내 평가 보고서','',f"- 데이터: {safe(report['dataset_version'])}",
           f"- 모델: {safe(report['model_version'])} / 예측: {safe(report['prediction_version'])}",
           f"- IoU 기준: {report['settings']['iou_threshold']} / 탐지 신뢰도: {report['settings']['confidence_threshold']}",
           '- 정답 없는 지표는 측정 불가입니다. 합성 자료 결과는 실제 서비스 성능을 뜻하지 않습니다.',
           '- 학습 제외 검사는 입력 명세에 기록한 범위에 한정됩니다. 전체 학습 목록·사람의 라벨 검수·최종 테스트 미사용 여부를 자동으로 증명하지 않습니다.',
           '- 마스킹은 제공한 정답·예측 사각형 면적을 평가하며 개인정보 자동 검출 성능을 보장하지 않습니다.',
           '', '## 데이터 분리 검사','']
    audit=report['audit']
    for key,label in [('duplicate_groups','동일 픽셀 중복 그룹'),('cross_split_groups','분할 간 중복 그룹'),
                      ('evaluation_training_overlap','학습 자료와 겹치는 평가 샘플'),('unknown_training','학습 포함 여부 미상'),
                      ('unknown_group','양식·원본 그룹 미상'),('unknown_split','분할 미상')]:
        lines.append(f'- {label}: {len(audit[key])}건')
    lines+=['','## 탐지 결과','','| 자료 | 클래스 | TP / FP / FN | Precision | Recall | F1 | 일치 박스 평균 IoU |',
            '|---|---|---|---|---|---|---|']
    groups={'전체':report['overall'],'원본':report['by_variant']['original'],'촬영본':report['by_variant']['photo']}
    for name,group in groups.items():
        for kind,m in group['detection']['classes'].items():
            counts=' / '.join('측정 불가' if m[k] is None else str(m[k]) for k in ('tp','fp','fn'))
            lines.append('| '+' | '.join([name,kind,counts]+[cell(m[k]) for k in ('precision','recall','f1','mean_matched_iou')])+' |')
    lines+=['','## 안내 및 마스킹 결과','','| 자료 | 안내 정답률 | 잘못 반환한 안내 비율 | 거절률 | 미마스킹 비율 | 과도한 마스킹 비율 |',
            '|---|---|---|---|---|---|']
    for name,g in groups.items():
        a=g['guidance'];m=g['masking']
        lines.append('| '+' | '.join([name]+[cell(v) for v in (a['matching_accuracy'],a['wrong_guidance_rate'],a['rejection_rate'],m['uncovered_rate'],m['excess_mask_rate'])])+' |')
    lines+=['','- 안내 정답률: 정답 매칭과 올바른 거절 / 정답이 있는 전체 안내 사례',
            '- 잘못 반환한 안내 비율: 오매칭 / 반환한 안내 중 정답이 있는 사례',
            '- 미마스킹 비율: 가려지지 않은 정답 면적 / 전체 정답 면적',
            '- 과도한 마스킹 비율: 정답 밖에 가린 면적 / 전체 가린 면적',
            '- 매칭 IoU는 성공한 일대일 매칭만의 평균입니다. 누락은 Recall에 반영됩니다.',
            '', '거절 사유: '+json.dumps(report['overall']['guidance']['rejection_reasons'],ensure_ascii=False)]
    if 'comparison' in report:
        c=report['comparison'];lines+=['','## 이전 결과 비교','',
            '동일 자료·설정 기준 비교 가능' if c['comparable'] else '자료·설정·안내 목록이 달라 직접적인 개선 수치 비교를 생략했습니다.']
        if c['comparable']: lines+=['','```json',json.dumps(c['detection_delta'],ensure_ascii=False,indent=2),'```']
    return '\n'.join(lines)+'\n'


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description='저장된 정답·예측만 평가합니다. 외부 요청 없음.')
    parser.add_argument('manifest',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--iou',type=float,default=.5);parser.add_argument('--confidence',type=float,default=.5)
    parser.add_argument('--demo',action='store_true');parser.add_argument('--baseline',type=Path)
    parser.add_argument('--require-unseen',action='store_true',help='학습 중복·이력 미상·정답/예측 누락이 있으면 탐지 평가를 거절')
    args=parser.parse_args(argv)
    try:
        report=evaluate_manifest(args.manifest,iou_threshold=args.iou,confidence=args.confidence,
                                 demo=args.demo,require_unseen=args.require_unseen)
        if args.baseline: report['comparison']=compare_reports(report,load_json(args.baseline))
        args.output_dir.mkdir(parents=True,exist_ok=True)
        paths=[args.output_dir/'evaluation.json',args.output_dir/'evaluation.md']
        if any(p.exists() for p in paths): raise ValueError('결과 파일이 이미 존재합니다.')
        paths[0].write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        paths[1].write_text(render_markdown(report),encoding='utf-8')
        print('JSON·Markdown 평가 보고서를 저장했습니다. 데이터 분리 경고를 함께 확인하세요.')
        return 0
    except UnseenEvaluationError as exc:
        print(f'[ERROR] {exc}',file=sys.stderr)
        return 1
    except (OSError,ValueError,TypeError,KeyError,np.linalg.LinAlgError,cv2.error):
        print('[ERROR] 평가 명세·파일·클래스·좌표계·이미지 지문 또는 출력 중복을 확인하세요.',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
