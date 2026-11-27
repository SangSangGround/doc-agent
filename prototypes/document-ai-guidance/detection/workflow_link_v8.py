"""v8 linker: AI workflow (meaning) + reviewed manifest (position) -> guidance plan.

Safety gates (a field is NAVIGABLE only if all hold):
  * the field id is in the reviewed a4_navigation_manifest_v1 (human approved on paper),
  * manifest and workflow were built from the SAME photo (SHA256),
  * manifest label/class equal the detector label/class recorded in the workflow,
  * the resolved role is not 'official'.
Human review wins for role when present; AI role is used only for unreviewed fields.
Fields with role 'official' never enter any plan.

CLI report:
  .\\.venv_scan\\Scripts\\python.exe detection\\workflow_link_v8.py --workflow <document_workflow.json> [--manifest <navigation_manifest.json>] [--targets <navigation_targets.json>]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

NAV_VERIFIED = 'verified'
NAV_UNVERIFIED = 'unverified'
NAV_OFFICIAL = 'excluded_official'
NAV_MISMATCH = 'blocked_mismatch'


@dataclass
class FieldInfo:
    id: str
    class_name: str
    name: str
    detected_label: str
    role: str
    ai_role: str
    reviewed_role: str | None
    requirement: str
    condition_text: str
    description: str
    source_quote: str
    aliases: list
    confidence: str
    uncertain: bool
    choice_group_id: str | None
    selects_application_type: str | None
    nav_status: str = NAV_UNVERIFIED
    nav_reason: str = ''
    target_mm: tuple | None = None

    @property
    def navigable(self):
        return self.nav_status == NAV_VERIFIED


@dataclass
class Step:
    kind: str                      # 'field' or 'choice'
    field_ids: list
    group_id: str | None = None
    question: str = ''


@dataclass
class AppType:
    id: str
    spoken_name: str
    description: str
    aliases: list
    selector_field_ids: list
    section_ids: list


@dataclass
class LinkedWorkflow:
    document_id: str
    title: str
    description: str
    selection_question: str
    app_types: list
    sections: dict
    section_order: list
    fields: dict
    groups: dict
    ai_inferred: bool
    human_reviewed: bool
    manifest_loaded: bool
    warnings: list = field(default_factory=list)

    def type_by_id(self, tid):
        return next((t for t in self.app_types if t.id == tid), None)

    def plan_for(self, tid=None):
        """Ordered steps for one application type (or the whole form if tid is None)."""
        t = self.type_by_id(tid) if tid else None
        other_selectors = set()
        for x in self.app_types:
            if t is None or x.id != t.id:
                other_selectors.update(x.selector_field_ids)
        section_ids = t.section_ids if t else list(self.section_order)
        in_sections = [fid for sid in section_ids for fid in self.sections[sid]['field_ids']]
        ordered = ([fid for fid in t.selector_field_ids if fid not in in_sections] if t else []) + in_sections
        allowed = [fid for fid in ordered
                   if fid in self.fields and fid not in other_selectors
                   and self.fields[fid].nav_status != NAV_OFFICIAL]
        allowed_set = set(allowed)
        steps, seen = [], set()
        for fid in allowed:
            if fid in seen:
                continue
            f = self.fields[fid]
            g = self.groups.get(f.choice_group_id) if f.choice_group_id else None
            if g and g['type'] == 'single_choice':
                members = [m for m in g['field_ids'] if m in allowed_set and m not in seen]
                seen.update(members)
                if len(members) >= 2:
                    steps.append(Step('choice', members, g['id'], g.get('question', '')))
                    continue
                steps.append(Step('field', members or [fid], g['id']))
                continue
            seen.add(fid)
            steps.append(Step('field', [fid], g['id'] if g else None))
        return steps

    def excluded_official(self):
        return [f for f in self.fields.values() if f.nav_status == NAV_OFFICIAL]

    def report(self):
        lines = [f'LINK [{self.document_id}] {self.title or "(제목 없음)"}',
                 f'  manifest loaded: {self.manifest_loaded} | ai_inferred: {self.ai_inferred} '
                 f'| human_reviewed: {self.human_reviewed}']
        counts = {}
        for f in self.fields.values():
            counts[f.nav_status] = counts.get(f.nav_status, 0) + 1
        lines.append(f'  nav status counts: {counts}')
        for f in self.fields.values():
            pos = f' ({f.target_mm[0]:.1f},{f.target_mm[1]:.1f})mm' if f.target_mm else ''
            lines.append(f'    {f.id} {f.class_name:<15} {f.nav_status:<17} role={f.role:<11} {f.name}{pos}'
                         + (f'  [{f.nav_reason}]' if f.nav_reason else ''))
        types = self.app_types or [None]
        for t in types:
            steps = self.plan_for(t.id if t else None)
            nav = sum(any(self.fields[x].navigable for x in s.field_ids) for s in steps)
            name = t.spoken_name if t else '(전체)'
            lines.append(f'  PLAN {name}: {len(steps)} steps, {nav} with verified position')
            for i, s in enumerate(steps, 1):
                marks = ','.join(f'{x}{"*" if self.fields[x].navigable else ""}' for x in s.field_ids)
                lines.append(f'    {i:2}. {s.kind:<6} {marks}  {self.fields[s.field_ids[0]].name if s.kind == "field" else (s.question or s.group_id)}')
        for w in self.warnings:
            lines.append(f'  WARN {w}')
        lines.append('  (* = human-reviewed position; only these can start finger guidance)')
        return '\n'.join(lines)


def _norm(s):
    return ' '.join(str(s or '').split())


def link_workflow(workflow, manifest=None, targets=None):
    if workflow.get('schema') != 'document_workflow_v1':
        raise ValueError('workflow schema must be document_workflow_v1')
    warnings = list(workflow.get('validation', {}).get('warnings', []))
    prov = workflow.get('provenance', {})
    mf = {}
    if manifest is not None:
        if manifest.get('schema') != 'a4_navigation_manifest_v1':
            raise ValueError('manifest schema must be a4_navigation_manifest_v1')
        wf_sha, mf_sha = prov.get('image_sha256'), manifest.get('source_image_sha256')
        if not wf_sha or not mf_sha or wf_sha != mf_sha:
            raise ValueError('Workflow and manifest come from different source photos (SHA256 mismatch). '
                             'Field IDs cannot be trusted to refer to the same boxes.')
        mf = {f['id']: f for f in manifest.get('fields', [])}
    gates = {}
    if targets is not None:
        gates = {t['id']: str(t.get('role_gate', '')) for t in targets.get('targets', [])}
    # Manifest-level manual exclusion applies even to an otherwise reviewed field.
    reviewed_blocks = manifest.get('blocked_field_ids', []) if manifest is not None else []
    if not isinstance(reviewed_blocks, list) or any(not isinstance(v, str) for v in reviewed_blocks):
        raise ValueError('manifest.blocked_field_ids must be a list of field IDs')
    blocked_ids = set(reviewed_blocks)
    fields = {}
    for row in workflow['fields']:
        fid = row['id']
        m = mf.get(fid)
        reviewed = m.get('role') if m else None
        info = FieldInfo(
            id=fid, class_name=row['class_name'], name=row.get('name') or row.get('detected_label') or fid,
            detected_label=row.get('detected_label', ''), role=row.get('role', 'unknown'),
            ai_role=row.get('role', 'unknown'), reviewed_role=reviewed,
            requirement=row.get('requirement', 'unknown'), condition_text=row.get('condition_text', ''),
            description=row.get('description', ''), source_quote=row.get('source_quote', ''),
            aliases=list(row.get('aliases', [])), confidence=row.get('confidence', 'low'),
            uncertain=bool(row.get('uncertain')), choice_group_id=row.get('choice_group_id'),
            selects_application_type=row.get('selects_application_type'))
        if m is not None:
            if m.get('class_name') != row['class_name'] or _norm(m.get('label')) != _norm(row.get('detected_label')):
                info.nav_status, info.nav_reason = NAV_MISMATCH, 'manifest label/class differs from workflow'
                warnings.append(f'{fid}: manifest label/class mismatch -> navigation blocked')
            else:
                # Human review is authoritative for permission to guide. A third-party
                # hint from the AI is kept only for the spoken explanation.
                info.role = 'third_party' if info.ai_role == 'third_party' else 'applicant'
                if info.ai_role == 'official':
                    warnings.append(f'{fid}: AI says official but a human approved it ({reviewed}); reviewed approval kept')
                p = m['target_mm']
                info.nav_status, info.target_mm = NAV_VERIFIED, (float(p['x']), float(p['y']))
                info.nav_reason = f"reviewed ({m.get('approval_method', '')})"
        else:
            if info.ai_role == 'official' or gates.get(fid, '').startswith('excluded_'):
                info.role = 'official'
                info.nav_status = NAV_OFFICIAL
                info.nav_reason = 'office-only field (AI role or writable_targets gate)'
            else:
                info.nav_reason = 'no reviewed position in manifest' if manifest is not None else 'no manifest loaded'
        if fid in blocked_ids:
            info.nav_status = 'blocked_by_review'
            info.nav_reason = 'manual navigation block recorded in document manifest'
            info.target_mm = None
        fields[fid] = info
    if manifest is not None:
        extra = set(mf) - set(fields)
        if extra:
            warnings.append(f'manifest ids not in workflow (ignored): {sorted(extra)}')
    sections = {s['id']: s for s in workflow.get('sections', [])}
    groups = {g['id']: g for g in workflow.get('choice_groups', [])}
    types = [AppType(t['id'], t['spoken_name'], t.get('description', ''), list(t.get('aliases', [])),
                     list(t.get('selector_field_ids', [])), [s for s in t.get('section_ids', []) if s in sections])
             for t in workflow.get('application_types', [])]
    return LinkedWorkflow(
        document_id=workflow.get('document_id', ''), title=workflow.get('title', ''),
        description=workflow.get('description', ''), selection_question=workflow.get('selection_question', ''),
        app_types=types, sections=sections, section_order=[s['id'] for s in workflow.get('sections', [])],
        fields=fields, groups=groups, ai_inferred=bool(prov.get('ai_inferred', True)),
        human_reviewed=bool(prov.get('human_reviewed', False)), manifest_loaded=manifest is not None,
        warnings=warnings)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workflow', required=True, type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--targets', type=Path)
    args = ap.parse_args()
    from document_workflow_v8 import load_workflow
    linked = link_workflow(load_workflow(args.workflow), _read(args.manifest), _read(args.targets))
    print(linked.report())


if __name__ == '__main__':
    main()
