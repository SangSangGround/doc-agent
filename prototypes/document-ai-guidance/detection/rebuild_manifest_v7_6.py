"""Safely rebuild nav manifest using the previously accepted ORIGINAL-photo corners.

Requires the same exact original photo and navigation_targets.json as the old
manifest; otherwise refuses to reuse calibration. Saves backup automatically.
No OpenCV clicking, Gemini or OCR calls.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from shutil import copy2
import os
from navigation_adapter import _sha256, make_manifest


def rebuild(old_manifest: Path, targets: Path, image: Path, review: Path | None = None,
            allow_fewer: bool = False):
    old=json.loads(old_manifest.read_text(encoding='utf-8-sig'))
    if old.get('schema')!='a4_navigation_manifest_v1':
        raise ValueError('Not an approved v1 manifest')
    if old.get('source_image_sha256') != _sha256(image):
        raise ValueError('IMAGE CONTENT CHANGED. Refusing to reuse old paper corners.')
    if old.get('source_targets_sha256') != _sha256(targets):
        raise ValueError('TARGET FILE CHANGED. Old calibration/field ids may be invalid.')
    corners=old.get('reference_corners_px_TL_TR_BR_BL')
    if not isinstance(corners,list) or len(corners)!=4:
        raise ValueError('Old manifest lacks 4 verified reference corners')
    review=review or targets.parent/'navigation_review.json'
    tmp=old_manifest.with_name(old_manifest.stem+'.v7_6_new.tmp.json')
    backup=old_manifest.with_name(old_manifest.stem+'.before_v7_6.bak.json')
    try:
        new=make_manifest(targets, review, image, corners,
                          (float(old['paper_width_mm']),float(old['paper_height_mm'])),tmp)
        old_count=len(old.get('fields',[]))
        new_count=len(new.get('fields',[]))
        if not new_count:
            raise ValueError('No approved fields in new manifest: original preserved.')
        if not allow_fewer and new_count<old_count:
            raise ValueError(f'New manifest has fewer fields ({new_count}) than old ({old_count}); '
                             'use --allow-fewer if the revocations were deliberate.')
        if old_manifest.exists():
            copy2(old_manifest,backup)
        os.replace(tmp,old_manifest)
        return new,backup
    finally:
        if tmp.exists():
            tmp.unlink()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--targets',type=Path,required=True)
    ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--review',type=Path)
    ap.add_argument('--allow-fewer',action='store_true')
    a=ap.parse_args()
    result,backup=rebuild(a.manifest,a.targets,a.image,a.review,a.allow_fewer)
    print('APPROVED FIELDS:',len(result['fields']),'| backup:',backup)
    print('SKIPPED REASONS:',result['skipped_approval_counts'])
    for f in result['fields']:
        print(' ',f['id'],f['class_name'],f['label'], f['target_mm'])
    print('Updated:',a.manifest)
    print('No new camera calibration needed; exact same original photo was verified by SHA256.')

if __name__=='__main__':
    main()
