import json

import cv2
import numpy as np
import pytest

from docagent.home.capture import CaptureCoach


def document():
    image = np.full((1000, 800, 3), 255, np.uint8)
    for y in range(100, 900, 35):
        cv2.putText(image, 'SYNTHETIC FORM 123 ABC', (80, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    return image


def test_ready_requires_distinct_stable_frames_and_elapsed_time():
    coach = CaptureCoach(mode='scan')
    image = document()
    first = coach.update(image, timestamp=0.)
    assert first['status'] == 'stabilizing'
    assert first['beep_state'] == 'stabilizing' and first['stability_progress'] == 0
    assert not first['should_announce']
    for _ in range(5):
        assert not coach.update(image, timestamp=0.)['ready_to_capture']
    assert not coach.update(image, timestamp=.3)['ready_to_capture']
    ready = coach.update(image, timestamp=.7)
    assert ready['ready_to_capture']
    assert ready['status'] == 'ready'
    assert ready['beep_state'] == 'ready' and ready['stability_progress'] == 1
    json.dumps(ready, allow_nan=False)


def test_failures_revoke_ready_immediately_and_require_stability_again():
    coach = CaptureCoach(mode='scan')
    image = document()
    for stamp in (0., .35, .7):
        result = coach.update(image, timestamp=stamp)
    assert result['ready_to_capture']
    blurred = cv2.GaussianBlur(image, (31, 31), 8)
    rejected = coach.update(blurred, timestamp=.8)
    assert not rejected['ready_to_capture']
    assert rejected['status'] == 'retake'
    assert rejected['beep_state'] == 'adjusting' and rejected['stability_progress'] == 0
    assert 'blurred' in rejected['quality']['reasons']
    assert not coach.update(image, timestamp=.9)['ready_to_capture']


def test_voice_cooldown_hysteresis_and_repeat_interval():
    coach = CaptureCoach(mode='scan')
    dark = np.zeros((1000, 800), np.uint8)
    assert not coach.update(dark, timestamp=0.)['should_announce']
    assert not coach.update(dark, timestamp=.2)['should_announce']
    assert coach.update(dark, timestamp=1.6)['should_announce']
    for stamp in (2., 8., 13., 25.):
        assert not coach.update(dark, timestamp=stamp)['should_announce']
    assert coach.update(dark, timestamp=25.6)['should_announce']
    # 상태가 잠깐 바뀌거나 음성 간격 안에 바뀌면 발화를 추가하지 않는다.
    assert not coach.update(document(), timestamp=26.)['should_announce']
    assert not coach.update(document(), timestamp=26.1)['should_announce']


def test_motion_and_camera_pause_reset_stability():
    coach = CaptureCoach(mode='scan', motion_threshold=.2)
    image = document()
    for stamp in (0., .35, .7):
        result = coach.update(image, timestamp=stamp)
    assert result['ready_to_capture']
    moved = np.roll(image, 70, axis=1)
    moving = coach.update(moved, timestamp=.8)
    assert moving['motion_score'] > .2
    assert not moving['ready_to_capture']
    assert not coach.update(moved, timestamp=8.)['ready_to_capture']
    coach.reset()
    assert not coach.update(image, timestamp=0.)['ready_to_capture']


def test_uncertain_document_boundary_is_never_ready():
    coach = CaptureCoach(mode='photo')
    for stamp in (0., 1., 2., 3.):
        result = coach.update(document(), timestamp=stamp)
        assert not result['ready_to_capture']
    assert 'boundary_unknown' in result['quality']['reasons']


def test_invalid_frame_and_timestamps():
    coach = CaptureCoach(mode='scan')
    assert coach.update(None, timestamp=1.)['quality']['reasons'] == ('invalid_image',)
    for timestamp in (float('nan'), float('inf'), .5, True):
        with pytest.raises(ValueError):
            coach.update(document(), timestamp=timestamp)
    assert not coach.update(document(), timestamp=1.1)['ready_to_capture']


@pytest.mark.parametrize('kwargs', [
    {'mode': 'live'}, {'stable_frames': True}, {'stable_frames': 1},
    {'stable_seconds': 0}, {'repeat_interval': float('nan')}, {'motion_threshold': -1},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CaptureCoach(**kwargs)
