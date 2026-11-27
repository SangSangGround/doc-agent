"""Build a manifest-constrained narration guide (NOT Gemini or live OCR).

Allows the working v7.5 tracker to announce newly reviewed fields without
claiming they are required for every applicant or generating policy advice.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from multi_field_voice_tracking_v6 import load_manifest


def build_guide(manifest, base_guide):
    fields=manifest['fields']
    ids={f['id'] for f in fields}
    descriptions={}
    if base_guide:
        for fid, desc in base_guide.get('field_descriptions',{}).items():
            if fid in ids:
                descriptions[fid]=str(desc)
    for f in fields:
        if f['id'] in descriptions:
            continue
        cls=f.get('class_name')
        # Avoid guessing exact completion instructions and conditional rules.
        if cls=='signature_field':
            descriptions[f['id']]='서명 또는 날인 위치입니다. 이 항목의 적용 조건은 실제 문서의 안내를 확인해 주세요.'
        elif cls=='input_field':
            descriptions[f['id']]='실제 필기 위치로 안내합니다. 작성 내용은 문서의 해당 항목을 확인해 주세요.'
        else:
            descriptions[f['id']]='선택 위치입니다. 신청 목적에 맞는 항목인지 확인해 주세요.'
    selection_groups=[]
    for group in (base_guide or {}).get('selection_groups',[]):
        group_ids=list(group.get('field_ids',[]))
        if group.get('type')=='single_choice' and group_ids and set(group_ids).issubset(ids):
            selection_groups.append(group)
    return {
        'schema':'navigation_guide_v1',
        'document_title':str((base_guide or {}).get('document_title','')),
        'document_summary':str((base_guide or {}).get('document_summary','')),
        'field_descriptions':descriptions,
        'selection_groups':selection_groups,
        'unverified_notice':'검증되지 않았거나 기관 담당자용인 항목은 안내하지 않습니다.',
        'source_note':'Document title/summary from existing human-authored document2 guide; all navigable positions from explicit reviewed manifest. This is NOT live Gemini or Apple Vision OCR.'
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--base-guide',type=Path,default=None)
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    manifest=load_manifest(args.manifest)
    base={}
    if args.base_guide:
        base=json.loads(args.base_guide.read_text(encoding='utf-8-sig'))
        if base.get('schema')!='navigation_guide_v1':
            ap.error('Wrong guide schema')
    guide=build_guide(manifest,base)
    output=args.out or args.manifest.parent/'navigation_guide_reviewed_v7_6.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(guide,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Guide saved:',output,'| fields:',len(manifest['fields']))
    print('Reviewed fields:',[(f['id'],f.get('label'),f.get('class_name')) for f in manifest['fields']])
    print('WARNING: not live OCR/Gemini narration; read only reviewed fields.')

if __name__=='__main__':
    main()
