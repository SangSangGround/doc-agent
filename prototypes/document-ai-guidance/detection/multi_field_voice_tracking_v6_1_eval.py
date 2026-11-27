"""Standalone MediaPipe + A4 paper-coordinate navigation PoC v6.1 evaluation.

Use only with a developer-reviewed navigation_manifest.json produced by
navigation_adapter.py. No Gemini calls. Fixed camera and flat paper REQUIRED.
This is NOT an iOS build and is NOT a verbatim port of the user's older v5,
which was not available for source-level merging.

Controls: C recalibrate live paper corners, N/P next/previous field,
          V speak selected label/direction, Q or Esc quit.
Optional: --beep (Windows), --tts (pyttsx3).
Press M only while physically touching the current box center to log an unverified
finger-to-target discrepancy, NOT independent physical accuracy/ground truth.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
import math
import sys
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def load_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema') != 'a4_navigation_manifest_v1':
        raise ValueError('Manifest must be created with navigation_adapter.py')
    fields = manifest.get('fields', [])
    if not fields or manifest.get('approved_fields_count') != len(fields):
        raise ValueError('No reviewed fields or manifest count mismatch')
    w, h = float(manifest.get('paper_width_mm', 0)), float(manifest.get('paper_height_mm', 0))
    if w <= 0 or h <= 0:
        raise ValueError('Invalid paper dimensions')
    seen = set()
    for t in fields:
        ident = str(t.get('id', ''))
        if not ident or ident in seen:
            raise ValueError('Duplicate or missing approved field ID')
        seen.add(ident)
        p = t['target_mm']
        if not (2 <= float(p['x']) <= w - 2 and 2 <= float(p['y']) <= h - 2):
            raise ValueError(f'Approved field outside paper boundary: {ident}')
    return manifest


def validate_corners(pts, width, height):
    a = np.asarray(pts, dtype=np.float32)
    if a.shape != (4, 2) or not np.isfinite(a).all():
        raise ValueError('4 paper corners required (TL TR BR BL)')
    if (a[:, 0] < 0).any() or (a[:, 0] >= width).any() or (a[:, 1] < 0).any() or (a[:, 1] >= height).any():
        raise ValueError('Corners outside camera view')
    if not cv2.isContourConvex(a) or cv2.contourArea(a) < .07 * width * height:
        raise ValueError('Nonconvex or small paper area; click TL TR BR BL')
    return a


def live_calibration(frame):
    """Developer-only GUI calibration: 4 corners of PAPER, not page text box."""
    h, w = frame.shape[:2]
    scale = min(1., 1080 / w, 850 / h)
    dw, dh = round(w * scale), round(h * scale)
    base = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
    sx, sy = dw / w, dh / h
    pts = []
    window = 'Live calibration: CLICK TL TR BR BL | ENTER OK | R reset | ESC cancel'
    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            shown = base.copy()
            for i, p in enumerate(pts):
                cv2.circle(shown, p, 6, (0, 200, 255), -1)
                cv2.putText(shown, str(i+1), (p[0] + 6, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            if len(pts) > 1:
                cv2.polylines(shown, [np.asarray(pts, np.int32)], len(pts) == 4, (0, 230, 0), 2)
            cv2.imshow(window, shown)
            key = cv2.waitKey(30) & 0xff
            if key in (10, 13) and len(pts) == 4:
                original = [[x/sx, y/sy] for x, y in pts]
                return validate_corners(original, w, h)
            if key in (ord('r'), ord('R')):
                pts.clear()
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyWindow(window)


def homography_to_paper(corners, width_mm, height_mm):
    world = np.array([[0.,0.],[width_mm,0.],[width_mm,height_mm],[0.,height_mm]], np.float32)
    return cv2.getPerspectiveTransform(np.asarray(corners, np.float32), world)


def warp_points(points, H):
    a = np.asarray(points, dtype=np.float64).reshape(-1,1,2)
    return cv2.perspectiveTransform(a, H.astype(np.float64)).reshape(-1,2)


def movement(finger_mm, target_mm):
    dx, dy = (target_mm[0] - finger_mm[0], target_mm[1] - finger_mm[1])
    distance = float(math.hypot(dx, dy))
    if abs(dx) >= abs(dy):
        direction = 'right' if dx >= 0 else 'left'
        primary = abs(dx)
    else:
        direction = 'down' if dy >= 0 else 'up'
        primary = abs(dy)
    return distance, direction, primary


class Sound:
    def __init__(self, enabled):
        self.enabled = enabled and sys.platform.startswith('win')
        self.busy = False
        if enabled and not self.enabled:
            print('Beeper requires Windows winsound, disabled here.')
    def play(self, hz=950, milliseconds=35):
        if not self.enabled or self.busy:
            return
        self.busy = True
        def work():
            try:
                import winsound
                winsound.Beep(hz, milliseconds)
            except Exception as exc:
                print('Sound error:', exc)
            finally:
                self.busy = False
        threading.Thread(target=work, daemon=True).start()


class Speaker:
    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self.queue = deque(maxlen=3)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        if self.enabled:
            threading.Thread(target=self._run, daemon=True).start()
    def say(self, text):
        print('[VOICE]', text)
        if self.enabled:
            with self.lock:
                self.queue.append(str(text))
    def _run(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
        except Exception as exc:
            print('TTS disabled:', exc)
            self.enabled = False
            return
        while not self.stop.is_set():
            with self.lock:
                text = self.queue.popleft() if self.queue else None
            if text is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as exc:
                    print('TTS error:', exc)
            else:
                time.sleep(.06)


def create_landmarker(model_path):
    import mediapipe as mp
    base = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=.6,
        min_hand_presence_confidence=.6,
        min_tracking_confidence=.6)
    return mp, vision.HandLandmarker.create_from_options(options)



EVAL_COLUMNS = [
    'timestamp', 'field_id', 'label', 'target_x_mm', 'target_y_mm',
    'finger_x_mm_smoothed', 'finger_y_mm_smoothed',
    'finger_x_mm_raw', 'finger_y_mm_raw',
    'distance_mm_display', 'distance_mm_raw', 'calibration_age_s',
    'camera_index', 'camera_mirrored', 'operator_touch_confirmed'
]


def save_touch_sample(path: Path, sample: dict):
    """Append operator-declared physical touch sample; never treat as ground-truth accuracy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8-sig' if not existing else 'utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=EVAL_COLUMNS)
        if not existing:
            writer.writeheader()
        writer.writerow({c: sample.get(c, '') for c in EVAL_COLUMNS})


def put_line(im, msg, y, color=(230,230,230), scale=.63):
    cv2.putText(im, msg, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (15,15,15), 4, cv2.LINE_AA)
    cv2.putText(im, msg, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent.parent/'hand_landmarker.task')
    parser.add_argument('--eval-csv', type=Path, default=None,
                        help='CSV for operator touch discrepancy. Default: manifest folder / tracking_touch_samples.csv')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--beep', action='store_true', help='Enable audible distance beep on Windows')
    parser.add_argument('--tts', action='store_true', help='Enable optional Korean TTS (pyttsx3/Windows voice)')
    parser.add_argument('--mirror', action='store_true', help='Only for mirror-preview camera; changes calibration view')
    parser.add_argument('--hit-mm', type=float, default=8., help='Developer-only proximity threshold, not empirically verified')
    parser.add_argument('--frame-width', type=int, default=1280)
    parser.add_argument('--frame-height', type=int, default=720)
    parser.add_argument('--check-manifest', action='store_true', help='Validate manifest and exit, no camera or MediaPipe')
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    eval_path = args.eval_csv or args.manifest.expanduser().resolve().parent / 'tracking_touch_samples.csv'
    targets = manifest['fields']
    width_mm, height_mm = float(manifest['paper_width_mm']), float(manifest['paper_height_mm'])
    print('Developer PoC only. Paper AND camera must stay fixed; click C after movement.')
    print('Tip is above paper; parallax can produce significant physical error.')
    print(f'Reviewed fields: {len(targets)} | paper {width_mm:.1f} x {height_mm:.1f} mm')
    print(f'Operator-touch sample CSV: {eval_path} (press M while finger actually touches target)')
    for t in targets:
        print(f"{t['id']}: {t['label']} | {t['target_mm']}")
    if args.check_manifest:
        print('Manifest schema and coordinates OK. No motion accuracy has been measured.')
        return
    if args.hit_mm < 3 or args.hit_mm > 25:
        parser.error('--hit-mm must be between 3 and 25 mm')
    if not args.model.is_file():
        parser.error(f'MediaPipe model not found: {args.model}')
    try:
        mp, landmarker = create_landmarker(args.model)
    except (ModuleNotFoundError, AttributeError, ImportError) as e:
        parser.error(f'MediaPipe not installed/compatible in this interpreter: {e}')

    if sys.platform == 'win32':
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        landmarker.close()
        parser.error(f'Cannot open camera index {args.camera}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    sound = Sound(args.beep)
    speaker = Speaker(args.tts)
    selection = 0
    corners = None
    H_frame_to_mm = None
    H_mm_to_frame = None
    smooth_mm = None
    last_seen = 0.
    arrive_frames = 0
    hit_fired = False
    next_beep = 0.
    previous_voice = None
    last_voice = 0.
    last_ms = -1
    calibrated_at = None
    start = time.monotonic()
    window = 'A4 Finger Navigation PoC v6.1 (C corners | N/P fields | M log touch | Q exit)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        with landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print('Camera read failed. Guidance stopped.')
                    break
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                h,w = frame.shape[:2]
                now = time.monotonic()
                chosen = targets[selection]
                current_touch_sample = None  # Never reuse coordinates from a prior video frame.
                target_mm = (float(chosen['target_mm']['x']), float(chosen['target_mm']['y']))
                view = frame.copy()
                put_line(view, f"FIELD {selection+1}/{len(targets)} {chosen['id']} | C=calibrate N/P=select V=voice Q=quit", 28)
                if H_frame_to_mm is None:
                    put_line(view, 'NOT CALIBRATED - NO GUIDANCE. Press C, click 4 paper corners.', 59, (0, 200, 255))
                else:
                    cv2.polylines(view, [corners.astype(np.int32)], True, (0, 180, 255), 2)
                    camera_target = warp_points([target_mm], H_mm_to_frame)[0]
                    if np.isfinite(camera_target).all():
                        tx, ty = map(int, np.rint(camera_target))
                        cv2.drawMarker(view, (tx,ty), (0,0,255), cv2.MARKER_CROSS, 25, 2)
                        cv2.circle(view,(tx,ty),round(args.hit_mm*max(w/width_mm,h/height_mm)/2), (0,0,255), 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb = np.ascontiguousarray(rgb)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    stamp = max(last_ms + 1, int((now - start) * 1000))
                    last_ms = stamp
                    result = landmarker.detect_for_video(image, stamp)
                    is_inside = False
                    if result.hand_landmarks:
                        landmark = result.hand_landmarks[0][8]
                        fp = (float(landmark.x*w), float(landmark.y*h))
                        if 0 <= fp[0] < w and 0 <= fp[1] < h:
                            x,y = map(int,np.rint(fp))
                            cv2.circle(view,(x,y),8,(255,130,10),-1)
                            is_inside = cv2.pointPolygonTest(corners, fp, False) >= 0
                            if is_inside:
                                raw_mm = warp_points([fp],H_frame_to_mm)[0]
                                if np.isfinite(raw_mm).all() and 0 <= raw_mm[0] <= width_mm and 0 <= raw_mm[1] <= height_mm:
                                    if smooth_mm is None or now-last_seen > .5:
                                        smooth_mm = raw_mm
                                    else:
                                        smooth_mm = .35*raw_mm + .65*smooth_mm
                                    last_seen = now
                                    dist, direction, primary = movement(smooth_mm,target_mm)
                                    raw_distance = float(np.linalg.norm(raw_mm - np.asarray(target_mm)))
                                    current_touch_sample = {
                                        'timestamp': datetime.now().astimezone().isoformat(timespec='seconds'),
                                        'field_id': chosen['id'], 'label': chosen['label'],
                                        'target_x_mm': round(target_mm[0], 3),
                                        'target_y_mm': round(target_mm[1], 3),
                                        'finger_x_mm_smoothed': round(float(smooth_mm[0]), 3),
                                        'finger_y_mm_smoothed': round(float(smooth_mm[1]), 3),
                                        'finger_x_mm_raw': round(float(raw_mm[0]), 3),
                                        'finger_y_mm_raw': round(float(raw_mm[1]), 3),
                                        'distance_mm_display': round(dist, 3),
                                        'distance_mm_raw': round(raw_distance, 3),
                                        'calibration_age_s': round(now - calibrated_at, 2) if calibrated_at is not None else '',
                                        'camera_index': args.camera,
                                        'camera_mirrored': args.mirror,
                                        'operator_touch_confirmed': True,
                                    }
                                    put_line(view,f'FINGER ({smooth_mm[0]:.1f},{smooth_mm[1]:.1f}) mm  DIST {dist:.1f} mm  {direction}', 61, (255,255,255))
                                    if dist <= args.hit_mm:
                                        arrive_frames += 1
                                        if arrive_frames >= 8:
                                            put_line(view, 'NEAR TARGET (not handwriting verification)', 93, (0,255,0))
                                            if not hit_fired:
                                                sound.play(1200,140)
                                                speaker.say('목표 위치에 가까워졌습니다. 종이에 손끝을 대고 위치를 다시 확인해 주세요.')
                                                hit_fired = True
                                    else:
                                        arrive_frames = 0
                                        hit_fired = False
                                        if now >= next_beep:
                                            # Distance-adaptive beep: closer means faster.
                                            interval = float(np.interp(np.clip(dist,8,150),[8,150],[.18,1.35]))
                                            next_beep = now+interval
                                            sound.play(780,45)
                                        if args.tts and (direction != previous_voice) and now-last_voice>5:
                                            kor = {'right':'오른쪽','left':'왼쪽','up':'위쪽','down':'아래쪽'}[direction]
                                            speaker.say(f'{kor}으로 약 {primary/10:.1f}센티미터 이동하세요.')
                                            previous_voice = direction
                                            last_voice = now
                    if not is_inside:
                        smooth_mm = None
                        arrive_frames = 0
                        hit_fired = False
                        next_beep = now+.3
                        put_line(view, 'Finger not reliably on paper - guidance paused', 62, (0,170,255))
                put_line(view, 'M = log discrepancy ONLY when physically touching printed target; Q = quit', h - 16, (255,255,255), .5)
                cv2.imshow(window,view)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q'), ord('Q')):
                    break
                if key in (ord('c'), ord('C')):
                    # Stop all guidance until re-calibration is completed.
                    H_frame_to_mm = None
                    H_mm_to_frame = None
                    calibrated_at = None
                    corners = None
                    smooth_mm = None
                    new_corners = live_calibration(frame)
                    if new_corners is not None:
                        corners = new_corners
                        H_frame_to_mm = homography_to_paper(corners,width_mm,height_mm)
                        H_mm_to_frame = np.linalg.inv(H_frame_to_mm)
                        calibrated_at = time.monotonic()
                        print('Camera PAPER corners calibrated. Recalibrate if paper/camera moves.')
                        speaker.say('카메라 좌표 보정이 완료되었습니다.')
                        next_beep = time.monotonic()+1.
                elif key in (ord('n'), ord('N'), ord('p'), ord('P')):
                    selection = (selection + (1 if chr(key).lower()=='n' else -1)) % len(targets)
                    smooth_mm = None
                    arrive_frames = 0
                    hit_fired = False
                    previous_voice = None
                    next_beep = time.monotonic()+1.
                    print('Selected:',targets[selection]['id'],targets[selection]['label'])
                    speaker.say(f"{targets[selection]['label']} 항목을 안내합니다.")
                elif key in (ord('m'), ord('M')):
                    if current_touch_sample is None:
                        print('[MEASURE] Skipped: finger not detected on calibrated paper THIS frame.')
                    else:
                        save_touch_sample(eval_path, current_touch_sample)
                        print(f"[MEASURE] {chosen['id']} distance={current_touch_sample['distance_mm_display']:.1f}mm "
                              f"(raw={current_touch_sample['distance_mm_raw']:.1f}mm) saved -> {eval_path}")
                elif key in (ord('v'), ord('V')):
                    speaker.say(f"{chosen['label']} 항목입니다. 종이가 움직이면 C 키로 좌표를 다시 맞춰 주세요.")
    finally:
        speaker.stop.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
