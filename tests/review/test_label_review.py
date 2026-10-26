import copy
import hashlib
import json
import re
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from docagent import label_review as lr
from docagent.evaluation.dataset import read_manifest, read_truth
from docagent.review import read_json


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    image = np.full((198, 140, 3), 255, np.uint8)
    cv2.rectangle(image, (40, 80), (48, 88), (0, 0, 0), 1)
    image_path = source/'blank.png'
    raw = cv2.imencode('.png', image)[1].tobytes()
    image_path.write_bytes(raw)
    candidate = dict(id='sample', title='가상 문서', image='blank.png',
                     file_sha256=hashlib.sha256(raw).hexdigest(),
                     reference=dict(image_size=[280, 396], coordinate_system='image_center_xywh_px',
                                    boxes=[dict(id='checkbox-a', **{'class': 'checkbox'},
                                                center_xywh=[88, 168, 16, 16])]))
    manifest = dict(schema_version=1, candidates=[candidate])
    output = tmp_path/'labels'
    return manifest, source, image_path, output


def prepared(inputs):
    manifest, source, image, output = inputs
    result = lr.prepare(manifest, source, 'sample', output)
    return result, read_json(output/'labels.draft.json')


@pytest.mark.parametrize('kind', ['roboflow', 'official_pdf'])
def test_scaled_reference_remains_draft_and_original_is_unchanged(inputs, kind):
    manifest, source, image, output = inputs
    manifest['source_kind'] = kind
    original = image.read_bytes()
    result, draft = prepared(inputs)
    assert result['box_count'] == 1 and result['image_size'] == [140, 198]
    assert draft['status'] == draft['approval']['status'] == 'draft'
    assert draft['source_kind'] == kind
    assert draft['coco']['annotations'][0]['bbox'] == [40, 80, 8, 8]
    assert draft['coco']['images'][0]['file_sha256'] == hashlib.sha256(original).hexdigest()
    assert image.read_bytes() == original
    assert (output/'labels.html').exists()
    # 일반 COCO 정답으로 연결하면 실패한다. 초안에서 자동으로 성능 수치가 나올 수 없다.
    with pytest.raises(ValueError):
        read_truth(dict(format='coco', path='labels.draft.json', image_id=1), output, 140, 198)


def test_approval_validates_image_and_edited_boxes_then_exports_usable_coco(inputs):
    _, _, image, output = inputs
    _, draft = prepared(inputs)
    draft['coco']['annotations'][0].update(bbox=[39.5, 79.25, 9, 10], area=1, category_id=2)
    accepted = lr.approve(draft, image, reviewer='  테스트 검토자  ', confirm_reviewed=True)
    assert accepted['annotations'][0]['area'] == 90
    metadata = accepted['info']['docagent_label_review']
    assert metadata['status'] == 'approved' and metadata['reviewed_by'] == '테스트 검토자'
    assert metadata['source_file_sha256'] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert metadata['background_only'] is False and metadata['reviewed_at']
    core = {k: v for k, v in accepted.items() if k != 'info'}
    canonical = json.dumps(core, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    assert metadata['content_sha256'] == hashlib.sha256(canonical.encode()).hexdigest()
    assert draft['approval']['status'] == 'draft'
    (output/'approved.json').write_text(json.dumps(accepted), encoding='utf-8')
    boxes = read_truth(dict(format='coco', path='approved.json', image_id=1), output, 140, 198)
    assert len(boxes) == 1 and boxes[0].kind == 'signature'
    assert boxes[0].bbox == (39.5, 79.25, 9, 10)


def test_empty_background_requires_same_explicit_human_confirmation(inputs):
    _, _, image, _ = inputs
    _, draft = prepared(inputs)
    draft['coco']['annotations'] = []
    with pytest.raises(ValueError):
        lr.approve(draft, image, reviewer='검토자', confirm_reviewed=False)
    result = lr.approve(draft, image, reviewer='검토자', confirm_reviewed=True)
    assert result['annotations'] == []
    assert result['info']['docagent_label_review']['background_only'] is True


@pytest.mark.parametrize('change', [
    lambda d: d.update(status='approved'),
    lambda d: d.update(coordinate_system='normalized_a4_mm'),
    lambda d: d['approval'].update(reviewed_by='다른 검토자'),
    lambda d: d['coco']['images'][0].update(file_sha256='0'*64),
    lambda d: d['coco']['images'][0].update(width=140.0),
    lambda d: d['coco']['images'][0].update(id=True),
    lambda d: d['coco']['images'][0].update(file_name='../other.png'),
    lambda d: d['coco']['categories'][0].update(name='signature_field'),
    lambda d: d['coco']['categories'][0].update(id=1.0),
    lambda d: d['coco']['annotations'].append(copy.deepcopy(d['coco']['annotations'][0])),
    lambda d: d['coco']['annotations'][0].update(id=True),
    lambda d: d['coco']['annotations'][0].update(image_id=True),
    lambda d: d['coco']['annotations'][0].update(category_id=True),
    lambda d: d['coco']['annotations'][0].update(category_id=3),
    lambda d: d['coco']['annotations'][0].update(iscrowd=1),
    lambda d: d['coco']['annotations'][0].update(ignore=1),
    lambda d: d['coco']['annotations'][0].update(bbox=[True, 1, 2, 2]),
    lambda d: d['coco']['annotations'][0].update(bbox=[float('nan'), 1, 2, 2]),
    lambda d: d['coco']['annotations'][0].update(bbox=[1, 1, float('inf'), 2]),
    lambda d: d['coco']['annotations'][0].update(bbox=[-1, 1, 2, 2]),
    lambda d: d['coco']['annotations'][0].update(bbox=[1, 1, 0, 2]),
    lambda d: d['coco']['annotations'][0].update(bbox=[139, 1, 2, 2]),
])
def test_invalid_or_reused_approval_never_becomes_labels(inputs, change):
    _, _, image, _ = inputs
    _, draft = prepared(inputs)
    change(draft)
    with pytest.raises((ValueError, TypeError, KeyError)):
        lr.approve(draft, image, reviewer='검토자', confirm_reviewed=True)


@pytest.mark.parametrize(('reviewer', 'confirm'), [('', True), ('  ', True), ('검토자', False), ('검토자', 1)])
def test_approval_needs_nonempty_reviewer_and_explicit_boolean(inputs, reviewer, confirm):
    _, _, image, _ = inputs
    _, draft = prepared(inputs)
    with pytest.raises(ValueError):
        lr.approve(draft, image, reviewer=reviewer, confirm_reviewed=confirm)


@pytest.mark.parametrize('change', [
    lambda m: m.update(schema_version=True),
    lambda m: m.update(source_kind='unrecognized'),
    lambda m: m['candidates'][0].update(image='../blank.png'),
    lambda m: m['candidates'][0].update(file_sha256='0'*64),
    lambda m: m['candidates'][0]['reference'].update(coordinate_system='normalized_a4_mm'),
    lambda m: m['candidates'][0]['reference'].update(image_size=[0, 396]),
    lambda m: m['candidates'][0]['reference']['boxes'][0].update(center_xywh=[1, 1, 50, 50]),
    lambda m: m['candidates'][0]['reference']['boxes'][0].update(center_xywh=[88, 168, True, 16]),
    lambda m: m['candidates'][0]['reference']['boxes'][0].update(**{'class': 'text'}),
])
def test_invalid_candidate_leaves_no_output(inputs, change):
    manifest, source, _, output = inputs
    change(manifest)
    with pytest.raises((ValueError, OSError)):
        lr.prepare(manifest, source, 'sample', output)
    assert not output.exists()


def test_duplicate_candidate_reference_ids_and_source_symlink_are_rejected(inputs, tmp_path):
    manifest, source, image, output = inputs
    manifest['candidates'].append(copy.deepcopy(manifest['candidates'][0]))
    with pytest.raises(ValueError):
        lr.prepare(manifest, source, 'sample', output)
    manifest['candidates'].pop()
    refs = manifest['candidates'][0]['reference']['boxes']
    refs.append(copy.deepcopy(refs[0]))
    with pytest.raises(ValueError):
        lr.prepare(manifest, source, 'sample', output)
    refs.pop()
    outside = tmp_path/'outside.png'
    outside.write_bytes(image.read_bytes())
    image.unlink()
    image.symlink_to(outside)
    with pytest.raises(ValueError):
        lr.prepare(manifest, source, 'sample', output)
    assert not output.exists()


def test_saved_draft_can_be_reopened_but_cannot_change_image_or_candidate(inputs, tmp_path):
    manifest, source, image, _ = inputs
    _, draft = prepared(inputs)
    draft['coco']['annotations'][0]['bbox'] = [42, 82, 6, 6]
    lr.prepare(manifest, source, 'sample', tmp_path/'resumed', draft=draft)
    assert read_json(tmp_path/'resumed/labels.draft.json')['coco']['annotations'][0]['bbox'] == [42, 82, 6, 6]
    draft['candidate_id'] = 'other'
    with pytest.raises(ValueError):
        lr.prepare(manifest, source, 'sample', tmp_path/'bad', draft=draft)
    draft['candidate_id'] = 'sample'
    # 해상도와 픽셀이 같아도 파일이 달라지면 동일 원본으로 취급하지 않는다.
    decoded = cv2.imdecode(np.frombuffer(image.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    altered = tmp_path/'reencoded.png'
    altered.write_bytes(cv2.imencode('.png', decoded, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes())
    assert altered.read_bytes() != image.read_bytes()
    with pytest.raises(ValueError):
        lr.approve(draft, altered, reviewer='검토자', confirm_reviewed=True)


def test_html_escapes_payload_and_blocks_external_requests(inputs):
    manifest, _, _, output = inputs
    manifest['candidates'][0]['title'] = '</script><script src="https://example.org/steal"></script> __SCRIPT_NONCE__'
    prepared(inputs)
    page = (output/'labels.html').read_text(encoding='utf-8')
    assert '<script src=' not in page
    assert '\\u003c/script\\u003e' in page
    assert "connect-src 'none'" in page and "default-src 'none'" in page
    assert 'fetch(' not in page and 'XMLHttpRequest' not in page and 'innerHTML' not in page
    assert '미승인 초안 JSON 저장' in page
    # 본문 데이터에 포함된 템플릿 문자열은 내용대로 보존한다.
    assert '__SCRIPT_NONCE__' in page


def test_prepare_and_approve_cli_never_overwrite_existing_files(inputs, tmp_path, capsys):
    manifest, source, image, output = inputs
    manifest_path = tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    arguments = ['prepare', str(manifest_path), '--source-root', str(source), '--candidate', 'sample', '--output', str(output)]
    assert lr.main(arguments) == 0
    before = (output/'labels.html').read_bytes()
    assert lr.main(arguments) == 1
    assert (output/'labels.html').read_bytes() == before
    accepted = tmp_path/'approved.json'
    approval = ['approve', str(output/'labels.draft.json'), '--image', str(image), '--reviewer', '검토자', '--confirm-reviewed', '--output', str(accepted)]
    assert lr.main(approval) == 0
    first = accepted.read_bytes()
    assert lr.main(approval) == 1
    assert accepted.read_bytes() == first
    assert '[ERROR]' in capsys.readouterr().err


def test_cli_requires_review_flag_and_reports_malformed_input_without_output(inputs, tmp_path):
    _, _, image, _ = inputs
    _, draft = prepared(inputs)
    path = tmp_path/'draft.json'
    path.write_text(json.dumps(draft), encoding='utf-8')
    out = tmp_path/'accepted.json'
    with pytest.raises(SystemExit):
        lr.main(['approve', str(path), '--image', str(image), '--reviewer', '검토자', '--output', str(out)])
    draft['coco']['annotations'][0]['bbox'] = [1, 1, -5, 2]
    path.write_text(json.dumps(draft), encoding='utf-8')
    assert lr.main(['approve', str(path), '--image', str(image), '--reviewer', '검토자', '--confirm-reviewed', '--output', str(out)]) == 1
    assert not out.exists()


@pytest.mark.parametrize('change', [
    lambda c: c['annotations'][0]['bbox'].__setitem__(0, 41),
    lambda c: c['annotations'][0].update(category_id=2),
    lambda c: c['annotations'][0].update(area=999),
    lambda c: c['images'][0].update(file_sha256='0'*64),
    lambda c: c['info']['docagent_label_review'].update(schema_version=True),
    lambda c: c['info']['docagent_label_review'].update(schema_version=2),
    lambda c: c['info']['docagent_label_review'].update(status='draft'),
    lambda c: c['info']['docagent_label_review'].update(content_sha256='0'*64),
    lambda c: c['info']['docagent_label_review'].update(source_file_sha256='0'*64),
    lambda c: c['info']['docagent_label_review'].update(background_only=True),
    lambda c: c['info']['docagent_label_review'].update(reviewed_by=''),
    lambda c: c['info']['docagent_label_review'].update(reviewed_at='2026-10-27T10:00:00'),
    lambda c: c['info'].__setitem__('docagent_label_review', None),
])
def test_evaluation_rejects_changed_approved_labels_or_invalid_marker(inputs, change):
    _, _, image, output = inputs
    _, draft = prepared(inputs)
    accepted = lr.approve(draft, image, reviewer='검토자', confirm_reviewed=True)
    change(accepted)
    (output/'approved.json').write_text(json.dumps(accepted), encoding='utf-8')
    with pytest.raises(ValueError):
        read_truth(dict(format='coco', path='approved.json', image_id=1), output, 140, 198)


def test_evaluation_manifest_binds_approved_labels_to_actual_image_bytes(inputs, tmp_path):
    _, _, image, output = inputs
    _, draft = prepared(inputs)
    accepted = lr.approve(draft, image, reviewer='검토자', confirm_reviewed=True)
    (output/'approved.json').write_text(json.dumps(accepted), encoding='utf-8')
    sample = dict(id='source', image=str(image), image_size=[140, 198], variant='original',
                  split='external', form_id='form', source_id='source', training_membership='excluded',
                  labels=dict(format='coco', path='approved.json', image_id=1), predictions=None)
    manifest = dict(schema_version=1, dataset_version='test', model_version='test',
                    prediction_version='test', samples=[sample])
    manifest_path = output/'evaluation.manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    _, samples = read_manifest(manifest_path)
    assert samples[0].truth[0].bbox == (40, 80, 8, 8)
    # 같은 크기와 픽셀이어도 다른 파일 인코딩을 사용하면 승인 바이트 지문은 다르다.
    decoded = cv2.imdecode(np.frombuffer(image.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    reencoded = tmp_path/'reencoded.png'
    reencoded.write_bytes(cv2.imencode('.png', decoded, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes())
    sample['image'] = str(reencoded)
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='지문'):
        read_manifest(manifest_path)


def test_approved_marker_checks_dimensions_and_explicit_file_hash(inputs):
    _, _, image, output = inputs
    _, draft = prepared(inputs)
    # 임의 부가 info가 있던 초안도 승인 시 생성하는 info를 제외한 내용으로 지문을 계산한다.
    draft['coco']['info'] = dict(description='검토 메모')
    accepted = lr.approve(draft, image, reviewer='검토자', confirm_reviewed=True)
    (output/'approved.json').write_text(json.dumps(accepted), encoding='utf-8')
    spec = dict(format='coco', path='approved.json', image_id=1)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    assert read_truth(spec, output, 140, 198, source_file_sha256=digest)
    with pytest.raises(ValueError):
        read_truth(spec, output, 141, 198, source_file_sha256=digest)
    with pytest.raises(ValueError):
        read_truth(spec, output, 140, 198, source_file_sha256='f'*64)


@pytest.mark.skipif(shutil.which('node') is None, reason='HTML 동작 회귀 검사는 Node.js가 있을 때 실행합니다.')
def test_html_retains_unapplied_edits_across_selection_and_drawing(inputs, tmp_path):
    manifest, _, _, output = inputs
    refs = manifest['candidates'][0]['reference']['boxes']
    refs.append(dict(id='signature-b', **{'class': 'signature'}, center_xywh=[180, 250, 40, 30]))
    prepared(inputs)
    page = (output/'labels.html').read_text(encoding='utf-8')
    payload = re.search(r'<script type="application/json"[^>]*>(.*?)</script>', page, re.S).group(1)
    script = re.search(r'<script nonce="[^"]+">(.*?)</script>', page, re.S).group(1)
    payload_path = tmp_path/'payload.json'
    script_path = tmp_path/'script.js'
    payload_path.write_text(payload, encoding='utf-8')
    script_path.write_text(script, encoding='utf-8')
    harness = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
class Element {
  constructor() { this.children=[];this.style={};this.dataset={};this.attributes={};this.handlers={};this.value='';this.clientWidth=1000; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children=[...items]; }
  setAttribute(key,value) { this.attributes[key]=value; }
  addEventListener(key,handler) { this.handlers[key]=handler; }
  closest() { return this.dataset.id ? this : null; }
  getBoundingClientRect() { return {left:0,top:0,width:parseFloat(this.style.width)||1000,height:parseFloat(this.style.height)||1000}; }
  setPointerCapture() {} remove() {} click() {}
}
const elements={}, el=id=>elements[id]||=(new Element());
el('payload').textContent=fs.readFileSync(process.argv[1],'utf8');
el('zoom').value='fit';el('new-class').value='1';el('class').value='1';
let saved=null;
const document={getElementById:el,createElement:()=>new Element(),createElementNS:()=>new Element(),body:new Element(),addEventListener(){}};
const context=vm.createContext({document,window:{addEventListener(){}},structuredClone,Blob,URL:{createObjectURL(blob){saved=blob;return 'blob:local';},revokeObjectURL(){}},setTimeout(fn){fn();}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const run=code=>vm.runInContext(code,context);
const clickRow=index=>el('box-list').children[index].children[0].handlers.click();
clickRow(0);el('x').value='41';el('class').value='2';
clickRow(1);assert.equal(el('selected-id').value,'1');assert.equal(el('x').value,'41');assert.equal(el('class').value,'2');
el('canvas').handlers.pointerdown({button:0,target:el('boxes').children[1]});
assert.equal(el('selected-id').value,'1');assert.equal(el('x').value,'41');
el('draw').handlers.click();assert.equal(run('drawing'),false);
el('save').handlers.click();assert.equal(saved,null);
el('apply').handlers.click();clickRow(1);assert.equal(el('selected-id').value,'2');
assert.equal(run('draft.coco.annotations[0].bbox[0]'),41);assert.equal(run('draft.coco.annotations[0].category_id'),2);
// 추가 모드에서도 기존 박스 입력을 바꾼 채 드래그하면 수정이 사라지지 않아야 한다.
el('draw').handlers.click();assert.equal(run('drawing'),true);el('width').value='19';
el('canvas').handlers.pointerdown({button:0,target:el('canvas'),pointerId:1,clientX:1,clientY:1,preventDefault(){}});
assert.equal(run('start'),null);assert.equal(el('width').value,'19');
el('draw').handlers.click();assert.equal(run('drawing'),true);
el('save').handlers.click();assert.equal(saved,null);
el('apply').handlers.click();el('draw').handlers.click();assert.equal(run('drawing'),false);
el('save').handlers.click();assert(saved);
saved.text().then(text=>{const data=JSON.parse(text);assert.equal(data.status,'draft');assert.equal(data.coco.annotations[0].bbox[0],41);assert.equal(data.coco.annotations[1].bbox[2],19);}).catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([shutil.which('node'), '-e', harness, str(payload_path), str(script_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
