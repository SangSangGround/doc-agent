import copy
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from docagent.pii.image_mask import MaskedDetector, MaskPlan, draft_plan, image_digest, mask_boxes, plan_digest
from docagent.errors import PiiEgressBlocked


def approved(image, boxes=None):
    p = draft_plan(image)
    p['boxes'] = boxes if boxes is not None else [[10,20,20,10]]
    p['approval'] = dict(status='approved', reviewed_by='test', reviewed_at='2026-10-15', content_sha256=plan_digest(p))
    return p


def test_mask_is_opaque_clamped_and_does_not_modify_source():
    image = np.full((80,100,4),255,np.uint8)
    result = mask_boxes(image,[[-2.2,5.1,15.4,10.1],[80,70,30,30]],padding_px=2)
    assert np.all(result[3:18,:16,:3] == 0)
    assert np.all(result[3:18,:16,3] == 255)
    assert np.all(result[68:,78:,:3] == 0)
    assert np.all(image == 255)
    assert np.all(result[30:40,30:40] == 255)


@pytest.mark.parametrize('boxes', [[],[[0,0,0,1]],[[100,0,20,20]],[[0,0,float('nan'),2]],[[True,0,2,2]],[[0,0,-1,2]]])
def test_invalid_mask_never_calls_detector(boxes):
    image = np.full((100,100,3),255,np.uint8)
    calls = []
    class Detector:
        def detect(self, image): calls.append(image); return []
    with pytest.raises((PiiEgressBlocked,ValueError)):
        MaskedDetector(Detector(), MaskPlan(approved(image,boxes))).detect(image)
    assert calls == []


def test_changed_image_or_coordinates_block_upload():
    image = np.full((100,100,3),255,np.uint8)
    plan = approved(image)
    p = MaskPlan(plan)
    changed = image.copy(); changed[50,50] = 1
    with pytest.raises(PiiEgressBlocked): p.apply(changed)
    plan['boxes'][0][0] += 1
    with pytest.raises(PiiEgressBlocked): MaskPlan(plan)
    with pytest.raises(PiiEgressBlocked): MaskPlan(draft_plan(image))


def test_mask_failure_does_not_fallback_to_original():
    image = np.full((100,100,3),255,np.uint8)
    seen = []
    class Fails:
        def detect(self, input_image):
            seen.append(input_image.copy())
            raise RuntimeError('network failed')
    with pytest.raises(RuntimeError):
        MaskedDetector(Fails(),MaskPlan(approved(image))).detect(image)
    assert len(seen) == 1
    assert np.all(seen[0][20:30,10:30] == 0)
    assert np.all(image == 255)


def test_plan_defensive_copy_and_grayscale():
    image = np.full((100,100),255,np.uint8)
    raw = approved(image); plan = MaskPlan(raw)
    raw['boxes'] = []
    assert np.all(plan.apply(image)[20:30,10:30] == 0)


def test_corrupt_file_error_does_not_echo_contents(tmp_path):
    p=tmp_path/'plan.json';p.write_text('SECRET-personal-value')
    with pytest.raises(PiiEgressBlocked) as e: MaskPlan.load(p)
    assert 'SECRET' not in str(e.value)
