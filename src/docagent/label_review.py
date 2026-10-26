"""원본 픽셀 좌표의 정답 박스를 로컬에서 검수하고 명시적으로 승인한다.

초안의 COCO 자료는 ``coco`` 키 안에 보관한다. 따라서 사람이 확인하기 전에
기존 평가기의 일반 COCO 입력으로 연결할 수 없다. 이 모듈은 추론이나 학습,
외부 통신을 수행하지 않는다.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from docagent.evaluation.dataset import box
from docagent.review import draft_approval, read_json, write_new

_CATEGORIES = [{'id': 1, 'name': 'checkbox'}, {'id': 2, 'name': 'signature'}]
_CLASS_IDS = {c['name']: c['id'] for c in _CATEGORIES}


def _read_image(path: Path) -> tuple[bytes, np.ndarray]:
    raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('원본 이미지를 읽을 수 없습니다.')
    return raw, image


def _size(value: Any) -> tuple[int, int]:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(n) is not int or n <= 0 for n in value)):
        raise ValueError('양수 정수의 이미지 크기가 필요합니다.')
    return value[0], value[1]


def _source_candidate(manifest: dict, source_root: Path, candidate_id: str) -> tuple[dict, Path]:
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('후보 명세 버전은 1입니다.')
    kind = manifest.get('source_kind', 'roboflow')
    if kind not in ('roboflow', 'official_pdf'):
        raise ValueError('지원하지 않는 후보 출처입니다.')
    candidates = manifest['candidates']
    if not isinstance(candidates, list):
        raise ValueError('후보 목록이 필요합니다.')
    found = [c for c in candidates if c['id'] == candidate_id]
    if len(found) != 1 or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', candidate_id):
        raise ValueError('후보 ID를 하나로 특정하세요.')
    candidate = found[0]
    if not isinstance(candidate['image'], str) or not candidate['image']:
        raise ValueError('원본 이미지 경로가 필요합니다.')
    root = source_root.resolve()
    image_path = (root / candidate['image']).resolve()
    if not image_path.is_relative_to(root):
        raise ValueError('원본은 지정한 루트 안에 있어야 합니다.')
    return candidate, image_path


def _coco_from_reference(candidate: dict, image_path: Path, raw: bytes, image: np.ndarray) -> dict:
    height, width = image.shape[:2]
    reference = candidate['reference']
    if reference.get('coordinate_system', 'image_center_xywh_px') != 'image_center_xywh_px':
        raise ValueError('참조 좌표는 원본 중심점 XYWH 픽셀이어야 합니다.')
    rw, rh = _size(reference['image_size'])
    annotations = []
    references = reference['boxes']
    if not isinstance(references, list):
        raise ValueError('참조 박스 목록이 필요합니다.')
    ids = [item['id'] for item in references]
    if (any(not isinstance(i, str) or not i.strip() for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('참조 박스 ID는 중복 없는 문자열이어야 합니다.')
    for index, item in enumerate(references, 1):
        if item['class'] not in _CLASS_IDS:
            raise ValueError('클래스는 checkbox 또는 signature입니다.')
        values = item['center_xywh']
        # 중심점도 bool/문자열/NaN 등을 허용하지 않는다.
        if (not isinstance(values, list) or len(values) != 4
                or any(type(v) not in (int, float) for v in values)):
            raise ValueError('참조 박스는 숫자 네 개여야 합니다.')
        cx, cy, bw, bh = values
        x, y, bw, bh = box([cx-bw/2, cy-bh/2, bw, bh], rw, rh)
        bbox = list(box([x*width/rw, y*height/rh, bw*width/rw, bh*height/rh], width, height))
        annotations.append(dict(id=index, image_id=1, category_id=_CLASS_IDS[item['class']],
                                bbox=bbox, area=bbox[2]*bbox[3], iscrowd=0,
                                source_reference_id=item['id']))
    return dict(images=[dict(id=1, file_name=image_path.name, width=width, height=height,
                             file_sha256=hashlib.sha256(raw).hexdigest())],
                categories=copy.deepcopy(_CATEGORIES), annotations=annotations)


def validate_draft(data: dict, image_path: Path) -> dict:
    """이미지 동일성과 모든 박스를 검증하고 정규화한 COCO를 반환한다. 승인은 하지 않는다."""
    if (type(data.get('schema_version')) is not int or data['schema_version'] != 1
            or data.get('kind') != 'docagent_coco_draft' or data.get('status') != 'draft'
            or data.get('coordinate_system') != 'image_xywh_px'
            or data.get('approval') != draft_approval()):
        raise ValueError('미승인 라벨 검수 초안이 필요합니다.')
    for name in ('candidate_id', 'title', 'source_kind'):
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise ValueError('후보 식별 정보가 필요합니다.')
    if data['source_kind'] not in ('roboflow', 'official_pdf'):
        raise ValueError('지원하지 않는 후보 출처입니다.')
    raw, image = _read_image(image_path)
    height, width = image.shape[:2]
    coco = copy.deepcopy(data['coco'])
    if (not isinstance(coco, dict) or coco.get('categories') != _CATEGORIES
            or any(type(c['id']) is not int for c in coco['categories'])
            or not isinstance(coco.get('images'), list) or len(coco['images']) != 1):
        raise ValueError('한 장의 이미지와 고정된 두 클래스가 필요합니다.')
    record = coco['images'][0]
    if (type(record['id']) is not int or record['id'] != 1
            or _size([record['width'], record['height']]) != (width, height)
            or record.get('file_sha256') != hashlib.sha256(raw).hexdigest()
            or not isinstance(record.get('file_name'), str) or not record['file_name']
            or Path(record['file_name']).name != record['file_name']):
        raise ValueError('초안과 원본 이미지의 지문·크기가 다릅니다.')
    annotations = coco['annotations']
    if not isinstance(annotations, list):
        raise ValueError('박스 목록이 필요합니다.')
    ids = []
    for item in annotations:
        if (not isinstance(item, dict) or type(item['id']) is not int or item['id'] <= 0
                or type(item['image_id']) is not int or item['image_id'] != 1
                or type(item['category_id']) is not int or item['category_id'] not in (1, 2)
                or type(item.get('iscrowd', 0)) is not int or item.get('iscrowd', 0) != 0
                or item.get('ignore', 0) != 0):
            raise ValueError('박스 ID·클래스·이미지 참조가 잘못되었습니다.')
        ids.append(item['id'])
        item['bbox'] = list(box(item['bbox'], width, height))
        item['area'] = item['bbox'][2] * item['bbox'][3]
        item['iscrowd'] = 0
    if len(set(ids)) != len(ids):
        raise ValueError('박스 ID가 중복되었습니다.')
    return coco


def render_labels(data: dict, image: np.ndarray) -> str:
    """외부 요청 없이 동작하는 편집 화면. 저장은 언제나 미승인 초안이다."""
    height, width = image.shape[:2]
    record = data['coco']['images'][0]
    if [record['width'], record['height']] != [width, height]:
        raise ValueError('검수 이미지 크기가 다릅니다.')
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('이미지 변환에 실패했습니다.')
    payload = dict(draft=data, image='data:image/png;base64,' + base64.b64encode(encoded).decode('ascii'))
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    serialized = serialized.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    nonce = secrets.token_hex(16)
    template = (Path(__file__).parent/'review_assets'/'labels.html').read_text(encoding='utf-8')
    return template.replace('__SCRIPT_NONCE__', nonce).replace('__LABEL_DATA__', serialized)


def prepare(manifest: dict, source_root: Path, candidate_id: str, output: Path, *,
            draft: dict | None = None) -> dict:
    """참조 좌표를 새 검수 폴더에만 복사한다. 원본·기존 검수본은 덮어쓰지 않는다."""
    if output.exists():
        raise ValueError('새 출력 폴더가 필요합니다.')
    candidate, image_path = _source_candidate(manifest, source_root, candidate_id)
    raw, image = _read_image(image_path)
    if candidate['file_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('후보 명세와 원본 파일의 지문이 다릅니다.')
    if draft is None:
        draft = dict(schema_version=1, kind='docagent_coco_draft', status='draft',
                     coordinate_system='image_xywh_px', candidate_id=candidate_id,
                     title=candidate['title'], source_kind=manifest.get('source_kind', 'roboflow'),
                     approval=draft_approval(),
                     coco=_coco_from_reference(candidate, image_path, raw, image),
                     note='참조에서 시작한 미승인 초안입니다. 전체 페이지의 누락·잘못된 박스를 사람이 검수해야 합니다.')
    else:
        draft = copy.deepcopy(draft)
        if draft.get('candidate_id') != candidate_id:
            raise ValueError('저장한 초안의 후보 ID가 다릅니다.')
    draft['coco'] = validate_draft(draft, image_path)
    page = render_labels(draft, image)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/'labels.draft.json', draft)
    write_new(output/'labels.html', page)
    return dict(candidate_id=candidate_id, status='draft',
                box_count=len(draft['coco']['annotations']), image_size=list(image.shape[1::-1]),
                html=str(output/'labels.html'), draft=str(output/'labels.draft.json'))


def approve(data: dict, image_path: Path, *, reviewer: str, confirm_reviewed: bool) -> dict:
    """사람의 확인 진술을 기록한다. 검토자의 신원이나 정답 품질을 자동 보증하지 않는다."""
    if (confirm_reviewed is not True or not isinstance(reviewer, str)
            or not reviewer.strip() or len(reviewer.strip()) > 200):
        raise ValueError('사람의 검수 확인과 검토자 이름이 필요합니다.')
    coco = validate_draft(data, image_path)
    # 초안에 부가 설명이 들어 있더라도 생성하는 승인 info는 내용 지문에서 제외한다.
    coco.pop('info', None)
    content = json.dumps(coco, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    coco['info'] = dict(docagent_label_review=dict(schema_version=1, status='approved',
                      candidate_id=data['candidate_id'], source_kind=data['source_kind'],
                      reviewed_by=reviewer.strip(), reviewed_at=datetime.now(timezone.utc).isoformat(),
                      content_sha256=hashlib.sha256(content.encode('utf-8')).hexdigest(),
                      source_file_sha256=coco['images'][0]['file_sha256'],
                      background_only=not coco['annotations'], coordinate_system='image_xywh_px',
                      scope='원본 이미지의 정답 박스 검수 진술. 학습 제외·독립 평가·안내문 승인을 의미하지 않습니다.'))
    return coco


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='원본 이미지의 정답 박스 검수와 사람 승인')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='후보 참조 박스를 편집할 로컬 HTML과 초안 생성')
    prep.add_argument('manifest', type=Path)
    prep.add_argument('--source-root', type=Path, required=True)
    prep.add_argument('--candidate', required=True)
    prep.add_argument('--draft', type=Path, help='HTML에서 저장한 초안을 다시 열 때 사용')
    prep.add_argument('--output', type=Path, required=True)
    accept = commands.add_parser('approve', help='사람 검수 후에만 평가용 COCO 정답 생성')
    accept.add_argument('draft', type=Path)
    accept.add_argument('--image', type=Path, required=True)
    accept.add_argument('--reviewer', required=True)
    accept.add_argument('--confirm-reviewed', action='store_true', required=True,
                        help='전체 페이지의 누락·오탐·클래스·박스 경계를 사람이 확인했음을 명시')
    accept.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(read_json(args.manifest), args.source_root, args.candidate, args.output,
                             draft=read_json(args.draft) if args.draft else None)
            print(json.dumps(result, ensure_ascii=False))
        else:
            result = approve(read_json(args.draft), args.image, reviewer=args.reviewer,
                             confirm_reviewed=args.confirm_reviewed)
            write_new(args.output, result)
            print('검토자의 확인을 기록한 COCO 정답을 새 파일에 저장했습니다.')
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError, cv2.error):
        print('[ERROR] 후보·초안·원본 이미지 일치·사람 검수 확인·새 출력 경로를 확인하세요.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
