"""Document-independent, conservative workflow contract. Python 3.12 stdlib only."""
from __future__ import annotations
from typing import Any
import json
from pathlib import Path

KINDS = {'check_box', 'input_field', 'signature_field', 'other'}
ROLES = {'applicant', 'official', 'third_party', 'unknown'}
REQUIREMENTS = {'required', 'optional', 'conditional', 'unknown'}

# JSON schema used to request structured output from Gemini; runtime checks below
# are authoritative and run even when the model returned JSON.
GENERATE_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'title': {'type': 'string'},
        'description': {'type': 'string'},
        'application_types': {'type': 'array', 'items': {
            'type': 'object', 'properties': {
                'id': {'type': 'string'}, 'name': {'type': 'string'},
                'aliases': {'type': 'array', 'items': {'type': 'string'}},
                'description': {'type': 'string'}},
            'required': ['id', 'name', 'aliases', 'description']}},
        'sections': {'type': 'array', 'items': {
            'type': 'object', 'properties': {
                'id': {'type': 'string'}, 'title': {'type': 'string'},
                'field_ids': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['id', 'title', 'field_ids']}},
        'fields': {'type': 'array', 'items': {
            'type': 'object', 'properties': {
                'id': {'type': 'string'}, 'name': {'type': 'string'},
                'kind': {'type': 'string', 'enum': sorted(KINDS)},
                'section_id': {'type': 'string'},
                'role': {'type': 'string', 'enum': sorted(ROLES)},
                'requirement': {'type': 'string', 'enum': sorted(REQUIREMENTS)},
                'condition': {'type': 'string'},
                'applicable_type_ids': {'type': 'array', 'items': {'type': 'string'}},
                'explanation': {'type': 'string'},
                'source_text': {'type': 'string'},
                'review_required': {'type': 'boolean'}},
            'required': ['id', 'name', 'kind', 'section_id', 'role',
                         'requirement', 'condition', 'applicable_type_ids',
                         'explanation', 'source_text', 'review_required']}},
        'choice_groups': {'type': 'array', 'items': {
            'type': 'object', 'properties': {
                'id': {'type': 'string'},
                'field_ids': {'type': 'array', 'items': {'type': 'string'}},
                'exclusive': {'type': 'boolean'}},
            'required': ['id', 'field_ids', 'exclusive']}}
    },
    'required': ['title', 'description', 'application_types', 'sections', 'fields', 'choice_groups']
}

class WorkflowError(ValueError):
    pass


def _unique_ids(records: list[dict], category: str) -> set[str]:
    seen: set[str] = set()
    for record in records:
        ident = record.get('id')
        if not isinstance(ident, str) or not ident.strip() or ident in seen:
            raise WorkflowError(f'{category}: ID missing or duplicated: {ident!r}')
        seen.add(ident)
    return seen


def validate_workflow(workflow: dict) -> dict:
    if not isinstance(workflow, dict):
        raise WorkflowError('workflow must be a JSON object')
    for key in ('title', 'description', 'application_types', 'sections', 'fields', 'choice_groups'):
        if key not in workflow:
            raise WorkflowError(f'missing key {key}')
    if not isinstance(workflow['title'], str) or not workflow['title'].strip():
        raise WorkflowError('title must be non-empty')
    for group in ('application_types', 'sections', 'fields', 'choice_groups'):
        if not isinstance(workflow[group], list) or not all(isinstance(x, dict) for x in workflow[group]):
            raise WorkflowError(f'{group} must be a list of objects')
    type_ids = _unique_ids(workflow['application_types'], 'application_types')
    section_ids = _unique_ids(workflow['sections'], 'sections')
    field_ids = _unique_ids(workflow['fields'], 'fields')
    _unique_ids(workflow['choice_groups'], 'choice_groups')
    for app in workflow['application_types']:
        if not isinstance(app.get('name'), str) or not app['name'].strip() or not isinstance(app.get('aliases'), list):
            raise WorkflowError(f"invalid application type {app['id']}")
    for section in workflow['sections']:
        ids = section.get('field_ids', [])
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or any(f not in field_ids for f in ids):
            raise WorkflowError(f"section {section['id']} has invalid field references")
    appearances = {fid: 0 for fid in field_ids}
    for section in workflow['sections']:
        for fid in section['field_ids']:
            appearances[fid] += 1
    if any(n != 1 for n in appearances.values()):
        raise WorkflowError('each field must appear exactly once in sections.field_ids')
    for field in workflow['fields']:
        fid = field['id']
        if field.get('kind') not in KINDS or field.get('role') not in ROLES or field.get('requirement') not in REQUIREMENTS:
            raise WorkflowError(f'{fid}: invalid kind, role or requirement')
        if field.get('section_id', '') not in section_ids:
            raise WorkflowError(f'{fid}: unknown section_id')
        ids = field.get('applicable_type_ids')
        if not isinstance(ids, list) or any(t not in type_ids for t in ids):
            raise WorkflowError(f'{fid}: unknown application_type ID')
        if not isinstance(field.get('review_required'), bool):
            raise WorkflowError(f'{fid}: review_required must be a boolean')
        if not isinstance(field.get('source_text'), str):
            raise WorkflowError(f'{fid}: source_text must be a string')
        if not field['source_text'].strip() and field['review_required'] is False:
            raise WorkflowError(f'{fid}: missing source_text must be flagged for review')
        for key in ('name', 'condition', 'explanation'):
            if not isinstance(field.get(key), str):
                raise WorkflowError(f'{fid}: {key} must be a string')
        # Do not let a model launder a guess into an approved navigation target.
        if field.get('navigation_verified') is True or field.get('navigation_target_id'):
            raise WorkflowError(f'{fid}: navigation info must be linked separately, not generated')
    for group in workflow['choice_groups']:
        refs = group.get('field_ids')
        if not isinstance(refs, list) or any(fid not in field_ids for fid in refs):
            raise WorkflowError(f"choice group {group['id']} has invalid field IDs")
        if not isinstance(group.get('exclusive'), bool):
            raise WorkflowError(f"choice group {group['id']} requires exclusive boolean")
        if group['exclusive'] and any(next(f for f in workflow['fields'] if f['id'] == fid)['kind'] != 'check_box' for fid in refs):
            raise WorkflowError(f"exclusive choice group {group['id']} must contain checkboxes")
    return workflow


def save_json(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_workflow(path: str | Path) -> dict:
    return validate_workflow(json.loads(Path(path).read_text(encoding='utf-8-sig')))
