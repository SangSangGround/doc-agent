"""List UNAPPROVED candidate form fields for human review; NEVER auto-approve.

Use original writable_targets v2 JSON and existing navigation_review.json.
Print/export descriptive candidates to help review general input/signature fields.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets',required=True,type=Path)
    ap.add_argument('--review',required=True,type=Path)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    data=json.loads(args.targets.read_text(encoding='utf-8-sig'))
    review=json.loads(args.review.read_text(encoding='utf-8-sig'))
    ref={x['id']:x for x in review.get('fields',[])}
    output=[]
    for field in data.get('targets',[]):
        entry=ref.get(field['id'],{})
        point=field.get('target_px')
        if isinstance(point,dict):
            coord=f"{point.get('x')}, {point.get('y')}"
        else:
            coord='(없음 — 수동 좌표 지정 필요)'
        output.append({
            'id':field['id'],
            'class':field.get('class_name',''),
            'label':field.get('label',''),
            'candidate_status':field.get('candidate_status',''),
            'role_gate':field.get('role_gate',''),
            'candidate_position_px':coord,
            'approved':str(bool(entry.get('approved'))),
            'verified_on_paper':str(bool(entry.get('verified_on_paper'))),
            'role':entry.get('role','unknown'),
        })
    if not output:
        print('No candidates found')
        return
    print('\n===== WRITABLE CANDIDATES: HUMAN REVIEW REQUIRED =====')
    for f in output:
        status='APPROVED' if f['approved']=='True' else 'REVIEW'
        print(f"{f['id']:>4} {f['class']:<17} {f['label']:<22} [{status}] "
              f"{f['candidate_status']}  target={f['candidate_position_px']}")
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(output[0].keys()))
            writer.writeheader();writer.writerows(output)
        print('Saved REVIEW LIST (no approvals changed):',args.out)
    print('\nALWAYS check visualization + real paper before setting approved/verified_on_paper. '
          'Institutional and conditional fields must remain excluded until reviewed.')


if __name__=='__main__':
    main()
