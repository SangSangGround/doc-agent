"""네트워크 없는 로컬 박스 검수와 명시적인 사람 승인 절차."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from docagent.errors import DocAgentError
from docagent.guidance.rules import RuleCatalog, content_digest, page_content_digest
from docagent.guidance.session import context_digest, digest, validate_context
from docagent.pii.image_mask import MaskPlan, draft_plan, image_digest, mask_boxes, plan_digest
from docagent.vision.normalize import normalize_document


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('JSON 객체가 필요합니다.')
    return data


def write_new(path: Path, data: dict[str, Any] | str) -> None:
    """검토 중인 파일을 덮어쓰지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def approval(reviewer: str, content_sha256: str) -> dict[str, str]:
    if not reviewer.strip():
        raise ValueError('검토자가 필요합니다.')
    return dict(status='approved', reviewed_by=reviewer,
                reviewed_at=datetime.now(timezone.utc).isoformat(), content_sha256=content_sha256)


def draft_approval() -> dict[str, str]:
    return dict(status='draft', reviewed_by='', reviewed_at='', content_sha256='')


def context_draft(report: dict[str, Any]) -> dict[str, Any]:
    """순서는 초안일 뿐이며 사람이 확인해야 한다. 분석 플래그는 수정하지 않는다."""
    return dict(schema_version=1, document_id='', document_version='', page=1, confidence=0,
                identity_verified=False, analysis_sha256=digest(report),
                analysis_document_id=report['document_id'], source_image_sha256=report.get('source_image_sha256'),
                field_order=[f['id'] for f in report['fields'] if f['type'] in ('choice','checkbox','signature')],
                approval=draft_approval())


def catalog_edit(data: dict[str, Any], field_id: str, *, reviewer: str | None = None) -> dict[str, Any]:
    """단일 항목만 초안화/승인. 다른 항목의 잘못된 승인은 그대로 오류 처리한다."""
    result = copy.deepcopy(data)
    entries = [e for e in result['fields'] if e['field_id'] == field_id]
    if len(entries) != 1:
        raise ValueError('항목 ID를 하나로 특정할 수 없습니다. 문서별 파일을 사용하세요.')
    entry = entries[0]
    entry['approval'] = draft_approval()
    RuleCatalog(result)
    if reviewer is not None:
        entry['approval'] = approval(reviewer, content_digest(entry))
    RuleCatalog(result)
    return result


def page_edit(data: dict[str, Any], document_id: str, document_version: str, page: int, *,
              reviewer: str | None = None, source_file: Path | None = None) -> dict[str, Any]:
    """전체 원문을 초안화/승인한다. 승인 전 출처 파일 지문을 확인하며 문구는 바꾸지 않는다."""
    result = copy.deepcopy(data)
    matches = [p for p in result.get('pages', []) if
               (p['document_id'], p['document_version'], p['page']) == (document_id, document_version, page)]
    if len(matches) != 1:
        raise ValueError('페이지 원문을 하나로 특정할 수 없습니다.')
    original = matches[0]
    original['approval'] = draft_approval()
    RuleCatalog(result)
    if reviewer is not None:
        if source_file is None or hashlib.sha256(source_file.read_bytes()).hexdigest() != original['source']['sha256']:
            raise ValueError('검토한 원본 파일의 지문이 다릅니다.')
        original['approval'] = approval(reviewer, page_content_digest(original))
    RuleCatalog(result)
    return result


def render_review(image: np.ndarray, *, plan: dict[str, Any] | None = None,
                  report: dict[str, Any] | None = None) -> str:
    """정합 이미지·오버레이를 포함하는 로컬 전용 HTML. 외부 요청 없음."""
    plan = copy.deepcopy(plan) if plan is not None else draft_plan(image)
    if plan['image_sha256'] != image_digest(image) or plan['image_shape'] != list(image.shape):
        raise ValueError('검토 이미지가 계획과 다릅니다.')
    if plan['boxes']:
        mask_boxes(image, plan['boxes'], padding_px=plan['padding_px'])
    # HTML 편집 결과는 기존 승인을 재사용하지 않는다.
    plan['approval'] = draft_approval()
    overlays: dict[str, list] = {'detections': [], 'ocr_words': []}
    if report is not None:
        if report.get('coordinate_system') != 'normalized_a4_mm' or report.get('image_sha256') != image_digest(image):
            raise ValueError('같은 정합 이미지의 분석 결과가 필요합니다.')
        h, w = image.shape[:2]
        for key in overlays:
            for item in report[key]:
                b = item['box_mm']
                overlays[key].append(dict(box=[b['x_mm'] * w / 210, b['y_mm'] * h / 297,
                                               b['w_mm'] * w / 210, b['h_mm'] * h / 297],
                                          label=item.get('text', item.get('type', ''))))
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('이미지 변환 실패')
    payload = dict(plan=plan, overlays=overlays, image='data:image/png;base64,' + base64.b64encode(encoded).decode())
    # </script> 및 HTML 삽입 방지. OCR는 DOM textContent로만 표시한다.
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (Path(__file__).parent / 'review_assets' / 'review.html').read_text(encoding='utf-8')
    return template.replace('__REVIEW_DATA__', data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='로컬 검수. 승인 명령은 실제 검토 후에만 사용하세요.')
    commands = parser.add_subparsers(dest='command', required=True)
    mask = commands.add_parser('mask', help='드래그로 마스킹 영역을 지정하는 HTML 생성')
    mask.add_argument('image', type=Path)
    mask.add_argument('--plan', type=Path)
    mask.add_argument('--analysis', type=Path)
    mask.add_argument('--dpi', type=int, default=300)
    mask.add_argument('--output', type=Path, required=True)
    for name in ('catalog-check', 'catalog-draft', 'catalog-approve', 'page-draft', 'page-approve',
                 'context-draft', 'context-approve', 'mask-approve'):
        sub = commands.add_parser(name)
        sub.add_argument('input', type=Path)
        if name != 'catalog-check': sub.add_argument('--output', type=Path, required=True)
        if name in ('catalog-draft','catalog-approve'): sub.add_argument('--field-id', required=True)
        if name in ('page-draft', 'page-approve'):
            sub.add_argument('--document-id', required=True)
            sub.add_argument('--document-version', required=True)
            sub.add_argument('--page', type=int, required=True)
        if name == 'page-approve': sub.add_argument('--source-file', type=Path, required=True)
        if name == 'context-approve': sub.add_argument('--analysis', type=Path, required=True)
        if name == 'mask-approve':
            sub.add_argument('--image', type=Path, required=True)
            sub.add_argument('--dpi', type=int, default=300)
        if name.endswith('-approve'):
            sub.add_argument('--reviewer', required=True)
            sub.add_argument('--confirm-reviewed', action='store_true', required=True,
                             help='사람이 내용과 영역을 직접 확인했음을 명시')
    args = parser.parse_args(argv)
    try:
        if args.command in ('mask', 'mask-approve'):
            if not 72 <= args.dpi <= 600: raise ValueError('DPI 범위')
            raw = cv2.imdecode(np.frombuffer(args.image.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            image = normalize_document(raw, dpi=args.dpi).image
        if args.command == 'mask':
            write_new(args.output, render_review(image, plan=read_json(args.plan) if args.plan else None,
                                                report=read_json(args.analysis) if args.analysis else None))
        else:
            data = read_json(args.input)
            if args.command == 'catalog-check':
                catalog = RuleCatalog(data)
                counts = {s: sum(e['approval']['status'] == s for e in data['fields']) for s in ('draft','approved')}
                pages = {s: sum(e['approval']['status'] == s for e in data.get('pages', [])) for s in ('draft','approved')}
                print(json.dumps(dict(status='valid', purpose=catalog.purpose, **counts, pages=pages), ensure_ascii=False))
                return 0
            if args.command in ('catalog-draft','catalog-approve'):
                data = catalog_edit(data, args.field_id, reviewer=getattr(args, 'reviewer', None))
            elif args.command in ('page-draft', 'page-approve'):
                data = page_edit(data, args.document_id, args.document_version, args.page,
                                 reviewer=getattr(args, 'reviewer', None), source_file=getattr(args, 'source_file', None))
            elif args.command == 'context-draft':
                data = context_draft(data)
            elif args.command == 'context-approve':
                data['approval'] = approval(args.reviewer, context_digest(data))
                validate_context(data, read_json(args.analysis))
            elif args.command == 'mask-approve':
                data['approval'] = approval(args.reviewer, plan_digest(data))
                MaskPlan(data).apply(image)
            write_new(args.output, data)
        print('로컬 검토 파일을 저장했습니다. 자동 승인이나 원본 외부 전송은 수행하지 않았습니다.'
              if not args.command.endswith('-approve') else '명시한 검토자의 승인 기록을 저장했습니다.')
        return 0
    except (DocAgentError, OSError, ValueError, TypeError, KeyError, cv2.error):
        print('[ERROR] 검토 파일 형식·이미지 일치·승인 기록·출력 파일 중복을 확인하세요.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
