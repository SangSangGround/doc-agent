"""선정 후보의 로컬 원본·분석·라벨·안내 초안을 묶는다. 승인/API 호출 없음."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import cv2
import numpy as np

from docagent.errors import DocAgentError
from docagent.evaluation.dataset import box
from docagent.guidance.analysis import observation_from_analysis
from docagent.guidance.rules import RuleCatalog, normalize_text
from docagent.pii.image_mask import draft_plan, image_digest, mask_boxes
from docagent.review import context_draft, read_json, render_review, write_new
from docagent.vision.normalize import normalize_document


def _inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('입력 경로는 지정한 루트 안에 있어야 합니다.')
    return path


def _rectangles(candidate: dict, width: int, height: int) -> list[dict]:
    """참조 중심점 좌표를 표시용으로만 환산. 정답/예측으로 승격하지 않는다."""
    reference = candidate['reference']
    rw, rh = reference['image_size']
    if any(type(n) is not int or n <= 0 for n in (rw, rh)):
        raise ValueError('참조 이미지 크기가 잘못되었습니다.')
    result = []
    for item in reference['boxes']:
        if item['class'] not in ('checkbox', 'signature'):
            raise ValueError('참조 클래스가 잘못되었습니다.')
        cx, cy, bw, bh = item['center_xywh']
        x, y, bw, bh = box([cx-bw/2, cy-bh/2, bw, bh], rw, rh)
        result.append(dict(id=item['id'], **{'class': item['class']},
                           bbox=[x*width/rw, y*height/rh, bw*width/rw, bh*height/rh]))
    return result


def _source_info(manifest: dict, candidate: dict) -> tuple[str, str]:
    """외부 공식 서식과 Roboflow 자료의 출처·학습 제외 근거를 구별한다."""
    kind = manifest.get('source_kind', 'roboflow')
    if kind == 'roboflow':
        version = manifest.get('roboflow_version')
        if type(version) is not int or version < 1:
            raise ValueError('Roboflow 버전이 필요합니다.')
        split = candidate['reference']['version_splits'].get(str(version))
        if split not in ('train', 'valid', 'test'):
            raise ValueError('선택한 Roboflow 버전의 분할 근거가 필요합니다.')
        return split, 'included' if split == 'train' else 'unknown'
    if kind != 'official_pdf':
        raise ValueError('지원하지 않는 출처 종류입니다.')
    source = candidate['official_source']
    url = urlparse(source['url'])
    if (url.scheme != 'https' or not url.hostname or url.username or url.password
            or not re.fullmatch(r'[0-9a-f]{64}', source['pdf_sha256'])
            or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', source['revision'])):
        raise ValueError('공식 원본 URL·지문·개정일을 확인하세요.')
    membership = candidate.get('training_membership', 'unknown')
    if membership not in ('unknown', 'included', 'excluded'):
        raise ValueError('학습 포함 여부가 잘못되었습니다.')
    if membership == 'excluded':
        audit = candidate.get('training_audit', {})
        versions = audit.get('checked_versions')
        if (audit.get('status') != 'not_found_in_project_training'
                or not isinstance(versions, list) or not versions
                or any(type(v) is not int or v < 1 for v in versions)
                or any(not isinstance(audit.get(k), str) or not audit[k].strip()
                       for k in ('project_url', 'checked_on', 'evidence', 'scope'))):
            raise ValueError('학습 제외에는 범위와 대조 기록이 필요합니다.')
    return 'external', membership


def _fraction_box(values: list, width: int, height: int) -> tuple[float, ...]:
    x, y, w, h = box(values, 1, 1)
    return x*width, y*height, w*width, h*height


def _mask_plan(image: np.ndarray, normalized: Any, candidate: dict) -> dict:
    plan = draft_plan(normalized.image)
    height, width = image.shape[:2]
    nh, nw = normalized.image.shape[:2]
    for region in candidate.get('mask_regions', []):
        x, y, w, h = _fraction_box(region, width, height)
        points = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]], dtype=float)
        projected = np.c_[points, np.ones(4)] @ normalized.homography.T
        if np.any(np.abs(projected[:,2]) < 1e-10):
            raise ValueError('마스킹 좌표를 변환할 수 없습니다.')
        points = projected[:,:2] / projected[:,2,None]
        lo = np.maximum(points.min(axis=0), [0,0])
        hi = np.minimum(points.max(axis=0), [nw,nh])
        plan['boxes'].append(list(box([float(v) for v in (*lo, *(hi-lo))], nw, nh)))
    return plan


def _page(title: str, body: str) -> str:
    return ('<!doctype html><html lang="ko"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
            'img-src data:; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
            f'<title>{html.escape(title)}</title><style>'
            'body{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:auto;padding:24px;color:#172534}'
            'a{color:#135ab6}h1{font-size:28px}h2{font-size:21px}small{color:#536478}'
            'section{border:1px solid #ccd5df;border-radius:12px;padding:20px;margin:20px 0}'
            'svg{width:100%;height:auto;border:1px solid #ccd5df}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
            '.notice{background:#fff3d6;padding:16px;border-radius:8px}table{border-collapse:collapse;width:100%}'
            'td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd5df}'
            '</style><body>' + body + '</body></html>')


def _reference_html(image: np.ndarray, candidate: dict, rectangles: list[dict], crops: list[dict]) -> str:
    height, width = image.shape[:2]
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('이미지 인코딩 실패')
    data = base64.b64encode(encoded).decode('ascii')
    shapes = []
    for r in rectangles:
        x,y,w,h = r['bbox']
        label = html.escape(f"{r['id']} · {r['class']}")
        shapes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" '
                      f'stroke="#d31832" stroke-width="4"><title>{label}</title></rect>'
                      f'<text x="{x}" y="{y-8}" fill="#b00020" font-size="25">{label}</text>')
    fields = []
    for entry in candidate['catalog']['fields']:
        crop = next((c for c in crops if c['field_id'] == entry['field_id']), None)
        source = entry.get('source_text')
        fields.append('<section><h2>' + html.escape(entry['field_id']) + '</h2><b>검색용 주변 문구</b><p>'
                      + html.escape(entry['surrounding_text']) + '</p><b>낭독 원문 초안 · 미승인</b><pre>'
                      + html.escape(source if source is not None else '원문 전사 전입니다. 기존 설명 초안을 낭독하지 않습니다.')
                      + '</pre><b>필수 키워드</b><p>'
                      + html.escape(' / '.join(entry['keywords'])) + '</p><b>주변 영역 OCR 진단</b><pre>'
                      + html.escape(json.dumps(crop, ensure_ascii=False, indent=2) if crop else '미실행')
                      + '</pre></section>')
    pages = ''.join('<section><h2>페이지 원문 전체 · 미승인</h2><p>원본과 대조해 누락·읽기 순서를 검토하세요.</p><pre>'
                    + html.escape(p['source_text']) + '</pre></section>' for p in candidate['catalog'].get('pages', []))
    return _page(candidate['title'], '<h1>' + html.escape(candidate['title']) + '</h1>'
                 '<p class="notice">후보 검수 전용 · 서비스 지원 확정/사람 승인 아님. 빨간 박스는 출처 명세의 '
                 '참조 좌표를 로컬 원본 해상도에 비례 표시한 검토 초안입니다. 이미지 동일성·라벨 기준·정렬을 검토하기 '
                 '전에는 평가 정답으로 사용하지 않습니다. 주변 OCR은 미리 지정한 영역의 진단이며 자동 탐지 결과가 아닙니다.</p>'
                 '<p><a href="review.html">정합 이미지의 탐지·OCR·마스킹 검수</a> · '
                 '<a href="catalog.draft.json">안내 초안 JSON</a> · <a href="../index.html">목록</a></p>'
                 f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
                 f'role="img" aria-label="원본과 참조 라벨"><image width="{width}" height="{height}" '
                 f'href="data:image/png;base64,{data}"/>{"".join(shapes)}</svg>' + pages + ''.join(fields))


def prepare(manifest: dict, source_root: Path, analysis_root: Path, output: Path, *, crop_reader=None) -> dict:
    """기존 분석의 실패/신뢰도는 보존한다. 모든 입력을 확인한 후 새 폴더에만 쓴다."""
    if output.exists():
        raise ValueError('새 출력 폴더가 필요합니다. 검토 중인 파일은 덮어쓰지 않습니다.')
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1
            or type(manifest.get('dpi')) is not int or not 72 <= manifest['dpi'] <= 600):
        raise ValueError('후보 명세 버전/DPI가 잘못되었습니다.')
    candidates = manifest['candidates']
    ids = [c['id'] for c in candidates]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', i) for i in ids):
        raise ValueError('중복 없는 안전한 후보 ID가 필요합니다.')
    prepared = []
    for candidate in candidates:
        source = _inside(source_root, candidate['image'])
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != candidate['file_sha256']:
            raise ValueError('선정한 원본과 파일 지문이 다릅니다.')
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('원본을 읽을 수 없습니다.')
        report = read_json(_inside(analysis_root, candidate['id'] + '/analysis.json'))
        normalized = normalize_document(image, dpi=manifest['dpi'])
        if (report.get('source_image_sha256') != image_digest(image)
                or report.get('image_sha256') != image_digest(normalized.image)
                or report.get('normalized_image_shape') != list(normalized.image.shape)
                or report.get('source_to_normalized') != normalized.homography.tolist()
                or report.get('text_included') is not True):
            raise ValueError('원본·정합·DPI·분석 지문이 일치하지 않습니다.')
        catalog_data = candidate['catalog']
        catalog = RuleCatalog(catalog_data)
        entries = catalog_data['fields']
        identity = (candidate['document_id'], candidate['document_version'], candidate['page'])
        if (catalog.purpose != 'production' or not entries
                or any(e['approval']['status'] != 'draft' for e in entries)
                or any(p['approval']['status'] != 'draft'
                       or (p['document_id'],p['document_version'],p['page']) != identity
                       for p in catalog_data.get('pages', []))
                or any((e['document_id'],e['document_version'],e['page']) != identity for e in entries)):
            raise ValueError('후보에는 같은 문서·버전·페이지의 미승인 초안만 넣으세요.')
        h,w = image.shape[:2]
        split, membership = _source_info(manifest, candidate)
        rectangles = _rectangles(candidate,w,h)
        plan = _mask_plan(image,normalized,candidate)
        context = context_draft(report)
        context.update(document_id=identity[0],document_version=identity[1],page=identity[2])
        # 사람이 승인한 것으로 취급하지 않는다. 0/false/draft를 그대로 유지한다.
        observations = [observation_from_analysis(report,context,f['id']) for f in report['fields']
                        if f['type'] in ('choice','checkbox','signature')]
        crops = []
        for region in candidate.get('ocr_regions', []):
            entry = next(e for e in entries if e['field_id'] == region['field_id'])
            x,y,bw,bh = _fraction_box(region['bbox'],w,h)
            x0,y0 = int(np.floor(x)),int(np.floor(y))
            x1,y1 = min(w,int(np.ceil(x+bw))),min(h,int(np.ceil(y+bh)))
            words = crop_reader.read(image[y0:y1,x0:x1]) if crop_reader is not None else None
            text = ' '.join(word.text for word in words) if words is not None else None
            crops.append(dict(field_id=entry['field_id'],source_box_px=[x0,y0,x1-x0,y1-y0],
                              scope='manual_roi_diagnostic_only',text=text,
                              minimum_word_confidence=min((word.confidence for word in words),default=0) if words is not None else None,
                              missing_keywords=[k for k in entry['keywords'] if normalize_text(k) not in normalize_text(text)] if text is not None else None))
        summary = dict(id=candidate['id'],title=candidate['title'],status='draft',
                       analysis_kind=manifest['analysis_kind'],detection_count=len(report['detections']),
                       field_count=len(report['fields']),ocr_word_count=len(report['ocr_words']),
                       reference_box_count=len(rectangles),needs_review=report['needs_review'],
                       failed_stages=[s['stage'] for s in report['stages'] if s.get('ok') is not True],
                       guidance_statuses=[catalog.match(o).status for o in observations],
                       approved_fields=0,independent_external_test=False,
                       dataset_split=split,
                       training_membership=membership,
                       training_audit=copy.deepcopy(candidate.get('training_audit')),
                       official_source=copy.deepcopy(candidate.get('official_source')),
                       reference_labels_status='alignment_unverified')
        # render_review also checks the report's coordinate system and image binding.
        review_html = render_review(normalized.image,plan=plan,report=report)
        reference_html = _reference_html(image,candidate,rectangles,crops)
        prepared.append((candidate,source,image,normalized,report,plan,context,crops,summary,review_html,reference_html))
    combined = dict(schema_version=1,purpose='production',
                    fields=[copy.deepcopy(e) for c,*_ in prepared for e in c['catalog']['fields']],
                    pages=[copy.deepcopy(p) for c,*_ in prepared for p in c['catalog'].get('pages', [])])
    RuleCatalog(combined)
    output.mkdir(parents=True,exist_ok=False)
    summaries = []
    samples = []
    for c,source,image,normalized,report,plan,context,crops,summary,review_html,reference_html in prepared:
        folder = output/c['id']
        for name,payload in [('analysis.json',report),('catalog.draft.json',c['catalog']),
                             ('context.draft.json',context),('mask-plan.draft.json',plan),
                             ('reference.draft.json',dict(status='alignment_unverified',source=c['reference'])),
                             ('crop-ocr.json',dict(scope='manual_roi_diagnostic_only',regions=crops)),
                             ('readiness.json',summary),('review.html',review_html),('reference.html',reference_html)]:
            write_new(folder/name,payload)
        # 미승인 계획의 시각 검토용 사본. 외부 전송에는 사용할 수 없다.
        if plan['boxes']:
            preview = mask_boxes(normalized.image,plan['boxes'],padding_px=plan['padding_px'])
            (folder/'masked.preview.png').write_bytes(cv2.imencode('.png',preview)[1].tobytes())
        summaries.append(summary)
        samples.append(dict(id=c['id'],image=str(source),image_size=[image.shape[1],image.shape[0]],
                            variant='original',split={'valid':'val'}.get(summary['dataset_split'],summary['dataset_split']),
                            form_id=c['document_id'],source_id=c['id'],
                            training_membership=summary['training_membership'],
                            training_audit=summary['training_audit'],
                            labels=None,predictions=f"{c['id']}/analysis.json"))
    write_new(output/'catalog.draft.json',combined)
    write_new(output/'evaluation.manifest.json',dict(schema_version=1,dataset_version=manifest['dataset_version'],
              model_version=manifest['analysis_kind'],prediction_version='local-candidate-baseline-1',
              samples=samples,catalog='catalog.draft.json'))
    result = dict(schema_version=1,candidates=summaries,approved_fields=0,
                  note='정답 좌표의 로컬 원본 정렬 미검증. 모델 성능·안내 정확도·마스킹 누락률을 주장할 수 없음.')
    write_new(output/'summary.json',result)
    rows = ''.join(f'<tr><td><a href="{s["id"]}/reference.html">{html.escape(s["title"])}</a></td>'
                   f'<td>{html.escape(s["analysis_kind"])}</td>'
                   f'<td>{s["detection_count"]} / {s["reference_box_count"]}</td>'
                   f'<td>{s["ocr_word_count"]}</td><td>{html.escape(", ".join(s["failed_stages"]))}</td></tr>' for s in summaries)
    write_new(output/'index.html',_page('문서 후보 검수',
              '<h1>문서 후보 검수</h1><p class="notice">로컬 전용 · 원본과 OCR 포함 · 외부 공유 전 별도 검토 필요. '
              '모든 안내·문서 식별·마스킹은 미승인 초안입니다. 검수 팩 생성 단계는 저장된 분석만 읽으며 '
              '추가 학습이나 외부 API 호출을 하지 않습니다.</p>'
              '<p>표의 탐지 수와 참조 라벨 수는 단순 개수이며 정확도 비교가 아닙니다. '
              '각 문서에서 원문, 안내 초안, 참조 박스, 주변 OCR을 확인할 수 있습니다.</p>'
              '<table><thead><tr><th>후보</th><th>분석 종류</th><th>탐지 / 참조 라벨</th><th>OCR 단어</th><th>재확인 단계</th></tr></thead>'
              '<tbody>'+rows+'</tbody></table>'))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--analysis-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--crop-ocr',action='store_true',help='미리 지정한 영역을 로컬 Tesseract로 진단')
    args = parser.parse_args(argv)
    try:
        reader = None
        if args.crop_ocr:
            from docagent.vision.ocr import TesseractOcr
            reader = TesseractOcr(lang='kor+eng',config='--psm 6')
            reader.check_available()
        prepare(read_json(args.manifest),args.source_root,args.analysis_root,args.output,crop_reader=reader)
    except (DocAgentError,OSError,ValueError,TypeError,KeyError,StopIteration,cv2.error):
        print('[ERROR] 후보 명세·원본 지문·분석·초안 상태 또는 새 출력 폴더를 확인하세요.',file=sys.stderr)
        return 1
    print('로컬 후보 검수 자료를 생성했습니다. 사람 승인·서비스 등록·외부 전송은 수행하지 않았습니다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
