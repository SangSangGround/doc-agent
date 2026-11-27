"""Review AI-extracted workflow before connecting to approved navigation targets.

This module never auto-approves AI fields or navigation coordinates.
CSV is a human-readable review worksheet only and is not accepted as verified
input by NavigationRegistry.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from workflow_schema import load_workflow

COLUMNS = [
    'field_id', 'section', 'field_name', 'kind', 'role', 'requirement',
    'applicable_types', 'condition', 'source_text', 'explanation',
    'AI_review_required', 'human_review_status', 'review_notes',
    'approved_target_id_for_review_only',
]


def write_review(workflow_path: Path, out_path: Path) -> None:
    workflow = load_workflow(workflow_path)
    sections = {s['id']: s['title'] for s in workflow['sections']}
    types = {t['id']: t['name'] for t in workflow['application_types']}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for f in workflow['fields']:
            names = [types[t] for t in f['applicable_type_ids']]
            writer.writerow({
                'field_id': f['id'],
                'section': sections.get(f['section_id'], f['section_id']),
                'field_name': f['name'],
                'kind': f['kind'],
                'role': f['role'],
                'requirement': f['requirement'],
                'applicable_types': ', '.join(names) if names else '공통 (AI 추정)',
                'condition': f['condition'],
                'source_text': f['source_text'],
                'explanation': f['explanation'],
                'AI_review_required': 'YES' if f['review_required'] else 'NO',
                'human_review_status': '',
                'review_notes': '',
                'approved_target_id_for_review_only': '',
            })
    print(f'[OK] 검수표 저장: {out_path}')
    print(f'문서: {workflow["title"]}')
    print(f'신청 유형 ({len(workflow["application_types"])}): ' + (
        ', '.join(t['name'] for t in workflow['application_types']) or '신청 유형 구분 없음'))
    print(f'작성 항목 후보: {len(workflow["fields"])}개')
    print('작성 주체 분포:', dict(Counter(f['role'] for f in workflow['fields'])))
    print('영역 유형 분포:', dict(Counter(f['kind'] for f in workflow['fields'])))
    print('필수 여부 분포:', dict(Counter(f['requirement'] for f in workflow['fields'])))
    print('AI가 검수 필요로 표시:', sum(f['review_required'] for f in workflow['fields']))
    print(f'선택 그룹: {len(workflow["choice_groups"])}개')
    print('[SAFE] 이 검수표는 좌표 승인이나 법률·작성 의무 확인을 자동으로 수행하지 않습니다.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Workflow 검수용 Excel 호환 CSV 생성')
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    write_review(args.workflow, args.out)


if __name__ == '__main__':
    main()
