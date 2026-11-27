"""v8 document understanding -> dynamic workflow JSON (schema document_workflow_v1).

Separates AI-inferred MEANING from human-reviewed POSITION:
  * The workflow stores only meaning: title, application types, sections,
    roles, conditions, spoken descriptions. It NEVER stores coordinates.
  * Navigable positions come only from the reviewed navigation_manifest.json,
    joined at runtime by workflow_link_v8.py using the shared field IDs
    (F001...) of gemini_output/<doc>/detection_result.json.

Sources (exactly one):
  --gemini              one Gemini call: photo + detected field list (+ optional OCR text)
  --from-json FILE      import a saved Gemini-format response (no API call)
  --fallback            label-only workflow, no AI meaning (offline / API failure)

Run from project root (PowerShell):
  .\\.venv_scan\\Scripts\\python.exe detection\\document_workflow_v8.py --detection detection\\gemini_output\\document1\\detection_result.json --gemini
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = 'document_workflow_v1'
ROLES = ('applicant', 'official', 'third_party', 'unknown')
REQUIREMENTS = ('required', 'optional', 'conditional', 'unknown')
CONFIDENCE = ('high', 'medium', 'low')
GROUP_TYPES = ('single_choice', 'multi_choice')
CLASSES = ('check_box', 'input_field', 'signature_field')

_STR = {'type': 'string'}
_STRS = {'type': 'array', 'items': {'type': 'string'}}
GEMINI_WORKFLOW_SCHEMA = {
    'type': 'object',
    'required': ['title', 'description', 'selection_question', 'application_types', 'sections',
                 'fields', 'choice_groups', 'dependencies', 'uncertain_items'],
    'properties': {
        'title': _STR,
        'description': _STR,
        'selection_question': _STR,
        'application_types': {'type': 'array', 'items': {
            'type': 'object',
            'required': ['id', 'spoken_name', 'aliases', 'description', 'selector_field_ids', 'section_ids'],
            'properties': {'id': _STR, 'spoken_name': _STR, 'aliases': _STRS, 'description': _STR,
                           'selector_field_ids': _STRS, 'section_ids': _STRS}}},
        'sections': {'type': 'array', 'items': {
            'type': 'object', 'required': ['id', 'title', 'description', 'applies_to', 'field_ids'],
            'properties': {'id': _STR, 'title': _STR, 'description': _STR,
                           'applies_to': _STRS, 'field_ids': _STRS}}},
        'fields': {'type': 'array', 'items': {
            'type': 'object',
            'required': ['id', 'name', 'role', 'requirement', 'condition_text', 'description',
                         'source_quote', 'aliases', 'confidence'],
            'properties': {'id': _STR, 'name': _STR,
                           'role': {'type': 'string', 'enum': list(ROLES)},
                           'requirement': {'type': 'string', 'enum': list(REQUIREMENTS)},
                           'condition_text': _STR, 'description': _STR, 'source_quote': _STR,
                           'aliases': _STRS,
                           'confidence': {'type': 'string', 'enum': list(CONFIDENCE)}}}},
        'choice_groups': {'type': 'array', 'items': {
            'type': 'object', 'required': ['id', 'type', 'question', 'field_ids'],
            'properties': {'id': _STR, 'type': {'type': 'string', 'enum': list(GROUP_TYPES)},
                           'question': _STR, 'field_ids': _STRS}}},
        'dependencies': {'type': 'array', 'items': {
            'type': 'object', 'required': ['field_id', 'depends_on', 'relation'],
            'properties': {'field_id': _STR, 'depends_on': _STR, 'relation': _STR}}},
        'uncertain_items': {'type': 'array', 'items': {
            'type': 'object', 'required': ['field_id', 'reason'],
            'properties': {'field_id': _STR, 'reason': _STR}}},
    },
}

PROMPT = '''You are helping a screen-reader assistant guide a BLIND applicant through a Korean paper form.
You get (1) a photo of the form and (2) a JSON list of writable regions ALREADY detected on that photo
(id, class_name, printed label, box_2d=[ymin,xmin,ymax,xmax] normalized 0..1000).
Return one JSON object describing how to fill the form. Rules:
- Use ONLY the given field ids. Never invent ids, boxes or coordinates. Every given id must appear exactly once in "fields".
- title: the exact printed form title. description: 1-2 plain Korean sentences on what the form is for, based only on printed text.
- application_types: mutually exclusive kinds of application the applicant picks (e.g. title checkboxes like "[ ]A [ ]B [ ]C 신청서").
  selector_field_ids = ALL checkbox ids that mark that kind (the same choice may be printed more than once).
  aliases = other ways a person may SAY the choice, including likely speech-recognition spellings. Return [] if the form has no such choice.
- selection_question: a short Korean question asking which application type to proceed with ("" if none).
- sections: groups of fields in the recommended filling order (normally top to bottom). Include selector checkboxes in the section where
  they are printed. applies_to = application type ids the section is used for, or ["all"].
- fields[].name: short Korean spoken name. role: applicant | official (filled by the receiving office, e.g. 접수번호, shaded office-only
  cells) | third_party (someone other than the applicant, e.g. another person's signature) | unknown.
- requirement: required | optional | conditional | unknown. Use "conditional" ONLY when printed text states a condition, and copy that
  printed sentence into source_quote. Never invent legal duties, documents or penalties that are not printed on the form.
- description: one short Korean sentence (under 80 characters) saying what to write or mark there. No legal advice.
- choice_groups: checkbox sets where only one may be marked (single_choice) or several may be marked (multi_choice).
  Do NOT put application-type selector checkboxes in choice_groups.
- dependencies: printed relations such as "only if X is not checked".
- uncertain_items: ids whose meaning, role or condition is unclear from the photo, with a short Korean reason.
All spoken strings must be Korean.
Detected fields:
'''


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def _s(value, limit=400):
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return re.sub(r'\s+', ' ', str(value)).strip()[:limit]
    return ''


def _slug(value, fallback):
    s = re.sub(r'[^0-9A-Za-z_]+', '_', _s(value)).strip('_')
    return s or fallback


def _list(value):
    return value if isinstance(value, list) else []


def norm_alias(text):
    return re.sub(r'[^0-9가-힣]', '', str(text).lower())


def expand_aliases(*names):
    """Normalized aliases incl. parts split at ·/ㆍ/,/또는 (e.g. 정정ㆍ변경 -> 정정, 변경)."""
    out = []
    for name in names:
        if not name:
            continue
        whole = norm_alias(name)
        if whole:
            out.append(whole)
        for part in re.split(r'[·ㆍ•,/]|\s또는\s', str(name)):
            p = norm_alias(part)
            if len(p) >= 2:
                out.append(p)
    seen, result = set(), []
    for a in out:
        if a not in seen:
            seen.add(a)
            result.append(a)
    return result


def detection_fields(detection):
    rows = detection.get('detections')
    if not isinstance(rows, list) or not rows:
        raise ValueError('detection_result.json has no detections')
    ids = [r.get('id') for r in rows]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError('Duplicate or missing detection ids')
    return rows


def prompt_payload(detection):
    return [{'id': r['id'], 'class_name': r['class_name'], 'label': r.get('label', ''),
             'box_2d': r.get('box_2d')} for r in detection_fields(detection)]


def normalize_workflow(raw, detection, *, document_id, provenance):
    """Validate/repair an AI (or imported) response into document_workflow_v1.

    Every reference is checked against detection ids. Nothing positional is copied.
    """
    if not isinstance(raw, dict):
        raise ValueError('Workflow response must be a JSON object')
    det = {r['id']: r for r in detection_fields(detection)}
    order = list(det)
    warnings = []

    # ---- fields (meaning only; class/label come from the detector, not the LLM)
    ai = {}
    for row in _list(raw.get('fields')):
        fid = _s(row.get('id')) if isinstance(row, dict) else ''
        if fid not in det:
            warnings.append(f'field: unknown id {fid!r} dropped')
            continue
        if fid in ai:
            warnings.append(f'field: duplicate {fid} ignored')
            continue
        ai[fid] = row
    uncertain = []
    for row in _list(raw.get('uncertain_items')):
        if isinstance(row, dict):
            fid = _s(row.get('field_id'))
            if fid and fid not in det:
                warnings.append(f'uncertain: unknown id {fid!r} dropped')
                continue
            uncertain.append({'field_id': fid, 'reason': _s(row.get('reason'))})
    fields = []
    for fid in order:
        d, row = det[fid], ai.get(fid)
        if row is None:
            warnings.append(f'field: {fid} missing from AI output, added as unknown')
            uncertain.append({'field_id': fid, 'reason': 'AI 분석 결과에 없는 항목'})
            row = {}
        def pick(key, allowed, default='unknown'):
            v = _s(row.get(key))
            if v and v not in allowed:
                warnings.append(f'field: {fid} invalid {key} {v!r} -> {default}')
            return v if v in allowed else default
        name = _s(row.get('name'), 120) or _s(d.get('label'), 120) or fid
        fields.append({
            'id': fid,
            'class_name': d['class_name'],
            'detected_label': _s(d.get('label'), 200),
            'name': name,
            'role': pick('role', ROLES),
            'requirement': pick('requirement', REQUIREMENTS),
            'condition_text': _s(row.get('condition_text'), 300),
            'description': _s(row.get('description'), 300),
            'source_quote': _s(row.get('source_quote'), 400),
            'aliases': expand_aliases(name, d.get('label'), *[_s(a, 80) for a in _list(row.get('aliases'))]),
            'confidence': pick('confidence', CONFIDENCE, 'low'),
            'choice_group_id': None,
            'selects_application_type': None,
            'uncertain': False,
        })
    by_id = {f['id']: f for f in fields}
    for u in uncertain:
        if u['field_id'] in by_id:
            by_id[u['field_id']]['uncertain'] = True

    # ---- application types
    types, type_ids = [], set()
    for i, row in enumerate(_list(raw.get('application_types')), 1):
        if not isinstance(row, dict):
            continue
        tid = _slug(row.get('id'), f'type_{i}')
        while tid in type_ids:
            tid += '_x'
        name = _s(row.get('spoken_name'), 80)
        if not name:
            warnings.append(f'application_type #{i}: no spoken_name, dropped')
            continue
        selectors = []
        for fid in _list(row.get('selector_field_ids')):
            fid = _s(fid)
            if fid not in det or det[fid]['class_name'] != 'check_box':
                warnings.append(f'type {tid}: selector {fid!r} is not a detected check_box, dropped')
                continue
            if by_id[fid]['selects_application_type']:
                warnings.append(f'type {tid}: selector {fid} already used by another type, dropped')
                continue
            by_id[fid]['selects_application_type'] = tid
            selectors.append(fid)
        type_ids.add(tid)
        types.append({'id': tid, 'spoken_name': name, 'description': _s(row.get('description'), 300),
                      'aliases': expand_aliases(name, *[_s(a, 60) for a in _list(row.get('aliases'))]),
                      'selector_field_ids': selectors,
                      '_declared_sections': [_s(x) for x in _list(row.get('section_ids'))]})
    # an alias shared by two types would make speech selection ambiguous
    counts = {}
    for t in types:
        for a in t['aliases']:
            counts[a] = counts.get(a, 0) + 1
    for t in types:
        full = norm_alias(t['spoken_name'])
        kept = [a for a in t['aliases'] if counts[a] == 1 or a == full]
        if len(kept) != len(t['aliases']):
            warnings.append(f'type {t["id"]}: ambiguous aliases removed {sorted(set(t["aliases"]) - set(kept))}')
        t['aliases'] = kept

    # ---- sections
    sections, section_ids, assigned = [], set(), {}
    for i, row in enumerate(_list(raw.get('sections')), 1):
        if not isinstance(row, dict):
            continue
        sid = _slug(row.get('id'), f'section_{i}')
        while sid in section_ids:
            sid += '_x'
        applies = [x for x in (_s(a) for a in _list(row.get('applies_to'))) if x == 'all' or x in type_ids]
        ids = []
        for fid in _list(row.get('field_ids')):
            fid = _s(fid)
            if fid not in det:
                warnings.append(f'section {sid}: unknown id {fid!r} dropped')
            elif fid in assigned:
                warnings.append(f'section {sid}: {fid} already in {assigned[fid]}, ignored')
            else:
                assigned[fid] = sid
                ids.append(fid)
        section_ids.add(sid)
        sections.append({'id': sid, 'title': _s(row.get('title'), 80) or sid,
                         'description': _s(row.get('description'), 300),
                         'applies_to': applies or ['all'], 'field_ids': ids})
    rest = [fid for fid in order if fid not in assigned and by_id[fid]['role'] != 'official']
    if rest:
        warnings.append(f'unassigned fields put in sec_unassigned: {rest}')
        sections.append({'id': 'sec_unassigned', 'title': '기타 작성란',
                         'description': 'AI가 작성 순서를 정하지 못한 항목입니다.',
                         'applies_to': ['all'], 'field_ids': rest})
        for fid in rest:
            by_id[fid]['uncertain'] = True
    for t in types:
        declared = set(t.pop('_declared_sections'))
        unknown = declared - {s['id'] for s in sections}
        if unknown:
            warnings.append(f'type {t["id"]}: unknown section ids {sorted(unknown)} ignored')
        t['section_ids'] = [s['id'] for s in sections
                            if s['id'] in declared or t['id'] in s['applies_to'] or 'all' in s['applies_to']]
        if not t['selector_field_ids'] and not t['section_ids']:
            warnings.append(f'type {t["id"]}: no selector and no sections')

    # ---- choice groups (never application selectors)
    groups, group_ids = [], set()
    for i, row in enumerate(_list(raw.get('choice_groups')), 1):
        if not isinstance(row, dict):
            continue
        gid = _slug(row.get('id'), f'group_{i}')
        while gid in group_ids:
            gid += '_x'
        gtype = _s(row.get('type'))
        if gtype not in GROUP_TYPES:
            warnings.append(f'group {gid}: invalid type {gtype!r}, dropped')
            continue
        members = []
        for fid in _list(row.get('field_ids')):
            fid = _s(fid)
            f = by_id.get(fid)
            if f is None or f['class_name'] != 'check_box':
                warnings.append(f'group {gid}: {fid!r} not a detected check_box, dropped')
            elif f['selects_application_type']:
                warnings.append(f'group {gid}: {fid} is an application selector, dropped')
            elif f['choice_group_id'] or fid in members:
                warnings.append(f'group {gid}: {fid} already grouped, dropped')
            else:
                members.append(fid)
        if len(members) < 2:
            warnings.append(f'group {gid}: fewer than two checkboxes, dropped')
            continue
        for fid in members:
            by_id[fid]['choice_group_id'] = gid
        group_ids.add(gid)
        groups.append({'id': gid, 'type': gtype, 'question': _s(row.get('question'), 200), 'field_ids': members})

    deps = []
    for row in _list(raw.get('dependencies')):
        if isinstance(row, dict):
            a, b = _s(row.get('field_id')), _s(row.get('depends_on'))
            if a in det and (b in det or b == ''):
                deps.append({'field_id': a, 'depends_on': b, 'relation': _s(row.get('relation'), 200)})
            else:
                warnings.append(f'dependency {a!r}->{b!r} dropped')

    return {
        'schema': SCHEMA,
        'document_id': document_id,
        'provenance': provenance,
        'title': _s(raw.get('title'), 300),
        'description': _s(raw.get('description'), 400),
        'selection_question': _s(raw.get('selection_question'), 200),
        'application_types': types,
        'sections': sections,
        'fields': fields,
        'choice_groups': groups,
        'dependencies': deps,
        'uncertain_items': uncertain,
        'validation': {'warnings': warnings},
    }


def fallback_raw(detection):
    """No AI meaning at all: one section, detector labels, top-to-bottom order."""
    rows = detection_fields(detection)
    def key(r):
        b = r.get('box_2d') or [0, 0, 0, 0]
        return (round(b[0] / 15), b[1])
    ordered = sorted(rows, key=key)
    return {
        'title': '', 'description': '', 'selection_question': '',
        'application_types': [],
        'sections': [{'id': 'sec_all', 'title': '전체 작성란', 'description': '',
                      'applies_to': ['all'], 'field_ids': [r['id'] for r in ordered]}],
        'fields': [{'id': r['id'], 'name': r.get('label', ''), 'role': 'unknown', 'requirement': 'unknown',
                    'condition_text': '', 'description': '', 'source_quote': '', 'aliases': [],
                    'confidence': 'low'} for r in rows],
        'choice_groups': [], 'dependencies': [],
        'uncertain_items': [{'field_id': '', 'reason': 'AI 문서 해석 없이 탐지 레이블만 사용'}],
    }


def api_generate(image_path, detection, model, ocr_text=None):
    """One Gemini call. Same SDK pattern as gemini_detection.py (interactions API)."""
    from google import genai
    from gemini_detection import read_image, encode_image
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('GEMINI_API_KEY is not set in this PowerShell session.')
    image = read_image(image_path)
    data, mime = encode_image(image, Path(image_path))
    text = PROMPT + json.dumps(prompt_payload(detection), ensure_ascii=False)
    if ocr_text:
        text += '\nOCR text of the same page (may contain errors):\n' + ocr_text[:20000]
    client = genai.Client(api_key=key)
    started = time.perf_counter()
    interaction = client.interactions.create(
        model=model,
        input=[{'type': 'text', 'text': text}, {'type': 'image', 'data': data, 'mime_type': mime}],
        response_format={'type': 'text', 'mime_type': 'application/json', 'schema': GEMINI_WORKFLOW_SCHEMA},
    )
    seconds = round(time.perf_counter() - started, 3)
    usage = getattr(interaction, 'usage', None)
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump(mode='json', exclude_none=True)
    elif usage is not None and not isinstance(usage, dict):
        usage = str(usage)
    raw = interaction.output_text
    if not raw:
        raise RuntimeError('Empty Gemini response. Check API status, quota and model settings.')
    return raw, usage, seconds


def load_workflow(path):
    wf = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if wf.get('schema') != SCHEMA:
        raise ValueError(f'Workflow schema must be {SCHEMA}')
    if not isinstance(wf.get('fields'), list) or not wf['fields']:
        raise ValueError('Workflow has no fields')
    return wf


def build(detection_path, mode, *, image=None, from_json=None, model=None, ocr_text_path=None,
          note='', fallback_on_error=False):
    detection_path = Path(detection_path).resolve()
    detection = json.loads(detection_path.read_text(encoding='utf-8-sig'))
    image_path = Path(image or detection.get('image', '')).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f'Source photo not found: {image_path}. Pass --image.')
    document_id = detection_path.parent.name
    ocr_text = Path(ocr_text_path).read_text(encoding='utf-8-sig') if ocr_text_path else None
    provenance = {
        'generator': mode, 'model': None, 'latency_seconds': None, 'usage': None,
        'generated_at': _dt.datetime.now().isoformat(timespec='seconds'),
        'image': str(image_path.resolve()), 'image_sha256': sha256_file(image_path),
        'detection_result': str(detection_path), 'detection_sha256': sha256_file(detection_path),
        'ocr_text': str(ocr_text_path) if ocr_text_path else None,
        'ai_inferred': mode in ('gemini', 'imported_json'), 'human_reviewed': False,
        'note': note,
    }
    raw_text = None
    if mode == 'gemini':
        try:
            raw_text, usage, seconds = api_generate(image_path, detection, model, ocr_text)
            provenance.update(model=model, latency_seconds=seconds, usage=usage)
            raw = json.loads(raw_text)
        except Exception as exc:
            if not fallback_on_error:
                raise
            print(f'[GEMINI FAILED] {exc!r} -> label-only fallback workflow', flush=True)
            provenance.update(generator='label_fallback', ai_inferred=False, error=repr(exc))
            raw = fallback_raw(detection)
    elif mode == 'imported_json':
        raw_text = Path(from_json).read_text(encoding='utf-8-sig')
        raw = json.loads(raw_text)
        provenance['imported_from'] = str(Path(from_json).resolve())
        if isinstance(raw, dict) and raw.get('_fixture_note'):
            provenance['note'] = (provenance['note'] + ' ' + str(raw['_fixture_note'])).strip()
    else:
        raw = fallback_raw(detection)
        provenance['ai_inferred'] = False
    wf = normalize_workflow(raw, detection, document_id=document_id, provenance=provenance)
    return wf, raw_text


def summarize(wf):
    lines = [f"[{wf['document_id']}] {wf['title'] or '(제목 없음)'}",
             f"  generator={wf['provenance']['generator']} ai_inferred={wf['provenance']['ai_inferred']} "
             f"human_reviewed={wf['provenance']['human_reviewed']}",
             f"  description: {wf['description']}"]
    for t in wf['application_types']:
        lines.append(f"  TYPE {t['id']} '{t['spoken_name']}' selectors={t['selector_field_ids']} "
                     f"sections={t['section_ids']} aliases={t['aliases']}")
    for s in wf['sections']:
        lines.append(f"  SECTION {s['id']} '{s['title']}' applies_to={s['applies_to']} fields={s['field_ids']}")
    for g in wf['choice_groups']:
        lines.append(f"  GROUP {g['id']} {g['type']} {g['field_ids']}")
    roles = {}
    for f in wf['fields']:
        roles[f['role']] = roles.get(f['role'], 0) + 1
    lines.append(f"  fields={len(wf['fields'])} roles={roles} uncertain={sum(f['uncertain'] for f in wf['fields'])}")
    for w in wf['validation']['warnings']:
        lines.append(f'  WARN {w}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--detection', required=True, type=Path, help='gemini_output/<doc>/detection_result.json')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--gemini', action='store_true')
    src.add_argument('--from-json', type=Path)
    src.add_argument('--fallback', action='store_true')
    ap.add_argument('--image', type=Path, help='Original photo (default: image recorded in detection JSON)')
    ap.add_argument('--model', default=os.environ.get('GEMINI_MODEL', 'gemini-4-hydrogen'))
    ap.add_argument('--ocr-text', type=Path, help='Optional plain-text OCR of the same page')
    ap.add_argument('--out', type=Path, help='Default: refinement_output/<doc>/workflow/document_workflow.json')
    ap.add_argument('--note', default='')
    ap.add_argument('--fallback-on-error', action='store_true', help='Write label-only workflow if Gemini fails')
    ap.add_argument('--print-prompt', action='store_true', help='Print the Gemini prompt and exit (no API call)')
    args = ap.parse_args()
    if args.print_prompt:
        det = json.loads(args.detection.read_text(encoding='utf-8-sig'))
        print(PROMPT + json.dumps(prompt_payload(det), ensure_ascii=False, indent=1))
        return
    mode = 'gemini' if args.gemini else ('imported_json' if args.from_json else 'label_fallback')
    if mode == 'gemini':
        print(f'Calling {args.model} for document workflow (one request, may take ~1-2 min) ...', flush=True)
    wf, raw_text = build(args.detection, mode, image=args.image, from_json=args.from_json, model=args.model,
                         ocr_text_path=args.ocr_text, note=args.note, fallback_on_error=args.fallback_on_error)
    out = args.out or HERE / 'refinement_output' / wf['document_id'] / 'workflow' / 'document_workflow.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if raw_text is not None:
        (out.parent / 'workflow_raw_response.json').write_text(raw_text, encoding='utf-8')
    out.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summarize(wf))
    print('Saved:', out)
    print('NOTE: meaning is AI-inferred and unreviewed; positions come only from the reviewed manifest.')
    if wf['provenance']['generator'] == 'label_fallback' and mode == 'gemini':
        sys.exit(2)


if __name__ == '__main__':
    main()
