"""Explicitly-reviewed navigation linkage. Never fuzzy-match coordinates.

The native v7.6 manifest layout is deliberately NOT assumed here.
An operator must export/copy approved targets to the normalized small format:
{"targets": [{"id": "T001", "verified": true, "kind": "check_box"}]}.
Links must be explicit: {"F001": "T001"}. Actual coordinates remain in native manifest.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from workflow_schema import validate_workflow


class LinkageError(ValueError):
    pass


class NavigationRegistry:
    def __init__(self, workflow: dict, approved_targets: dict, field_links: dict):
        validate_workflow(workflow)
        if not isinstance(approved_targets, dict) or not isinstance(approved_targets.get('targets'), list):
            raise LinkageError('approved_targets expects {"targets": [...]}; native manifest is not auto-parsed')
        if not isinstance(field_links, dict):
            raise LinkageError('field_links must be {field_id: target_id}')
        targets: dict[str, dict] = {}
        for target in approved_targets['targets']:
            if not isinstance(target, dict) or not isinstance(target.get('id'), str) or target['id'] in targets:
                raise LinkageError('approved target IDs missing or duplicated')
            targets[target['id']] = target
        fields = {f['id']: f for f in workflow['fields']}
        if any(fid not in fields for fid in field_links):
            raise LinkageError('field_links references field ID absent from generated workflow')
        used_targets: set[str] = set()
        validated: dict[str, str] = {}
        for fid, tid in field_links.items():
            target = targets.get(tid)
            field = fields[fid]
            if tid in used_targets:
                raise LinkageError(f'target reused for multiple fields: {tid}')
            used_targets.add(tid)
            if target is None:
                raise LinkageError(f'{fid} links to a missing target {tid}')
            # AI review_requested, unknown owner and non-approved targets cannot navigate.
            if target.get('verified') is not True:
                continue
            if target.get('kind') not in (None, field['kind']):
                raise LinkageError(f'{fid} kind mismatch ({field["kind"]} vs {target["kind"]})')
            if field['role'] != 'applicant' or field['review_required']:
                continue
            validated[fid] = tid
        self.targets = targets
        self.field_to_target = validated

    @classmethod
    def from_files(cls, workflow: dict, targets_file: str | Path, links_file: str | Path):
        with open(targets_file, encoding='utf-8-sig') as f:
            targets = json.load(f)
        with open(links_file, encoding='utf-8-sig') as f:
            links = json.load(f)
        return cls(workflow, targets, links)

    def lookup(self, field_id: str) -> str | None:
        return self.field_to_target.get(field_id)

    def report(self, workflow: dict) -> dict[str, Any]:
        return {'n_fields': len(workflow['fields']),
                'verified_links': len(self.field_to_target),
                'unlinked_or_blocked': [f['id'] for f in workflow['fields'] if f['id'] not in self.field_to_target]}
