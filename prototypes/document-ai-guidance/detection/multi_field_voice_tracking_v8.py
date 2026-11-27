"""v8 dynamic Korean document dialogue + existing v7.5 paper/fingertip/audio tracking.

NEW FILE: does not replace or modify multi_field_voice_tracking_v7_5.py.
Safe to execute --check-setup without camera. Real navigation REQUIRES the reviewed
manifest, source-image SHA match, and v8 link_workflow approval gates.

Keyboard for local prototype: 1/2/3/4 choice, Y=yes, U=no, C=completed,
K=skip, P=previous, O=overview, I=repeat, SPACE=pause/resume, R=reset tracker,
Q/Esc=quit. --stt uses raw Korean speech (not legacy 신규/변경/해지 parser).
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from document_workflow_v8 import load_workflow
from workflow_link_v8 import link_workflow
from workflow_dialogue_v8 import WorkflowDialogueV8, nav_allowed
from workflow_speech_v8 import WorkflowSpeechCommands
from camera_recovery_v8 import CameraRecovery
from multi_field_voice_tracking_v6 import (
    load_manifest, warp_points, movement, create_landmarker, put_line,
)
from multi_field_voice_tracking_v7_5 import open_camera, _in_image
from document_tracker_v7 import PaperTracker
from document_tracker_v7_5 import ElbowClosePaperTracker
from narration_v7_1 import Narrator, voice_diagnostic
from proximity_audio_v7_4 import (
    GuidanceBeeper, beep_interval, direction_phrase, direction_zone, direction_signature,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def setup_linked(args):
    """Link native v8 workflow to original v7.6 manifest; reject mismatched photos."""
    if not args.workflow or not args.manifest or not args.reference:
        raise ValueError('--workflow, --manifest, --reference are required')
    raw_manifest = read_json(args.manifest)
    linked = link_workflow(load_workflow(args.workflow), raw_manifest,
                           read_json(args.targets) if args.targets else None)
    # Reference photo bytes MUST match the workflow provenance; don't accidentally
    # calibrate live tracking on a similarly named but different picture.
    from document_workflow_v8 import sha256_file
    expected = raw_manifest.get('source_image_sha256')
    if not expected or sha256_file(args.reference) != expected:
        raise ValueError('Reference image SHA256 differs from reviewed Manifest photo; navigation blocked.')
    tracking_manifest = load_manifest(args.manifest)
    return linked, tracking_manifest


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workflow', type=Path, help='v8 document_workflow_v1 JSON')
    ap.add_argument('--manifest', type=Path, help='reviewed a4_navigation_manifest_v1 JSON')
    ap.add_argument('--targets', type=Path, help='optional original navigation_targets.json (role gate)')
    ap.add_argument('--block-field', action='append', default=[], metavar='FID', help='operator block navigation for a questionable field ID; repeatable')
    ap.add_argument('--reference', type=Path, help='EXACT original reference photo used for manifest')
    ap.add_argument('--model', type=Path, default=Path(__file__).resolve().parent.parent/'hand_landmarker.task')
    ap.add_argument('--camera', type=int, default=1)
    ap.add_argument('--camera-backend', choices=('msmf','dshow','auto'), default='msmf')
    ap.add_argument('--frame-width', type=int, default=1280)
    ap.add_argument('--frame-height', type=int, default=720)
    ap.add_argument('--beep', action='store_true')
    ap.add_argument('--tts', action='store_true')
    ap.add_argument('--tts-rate', type=int, default=4)
    ap.add_argument('--tts-backend', choices=('powershell','pyttsx3'), default='powershell')
    ap.add_argument('--stt', action='store_true')
    ap.add_argument('--mic-index', type=int)
    ap.add_argument('--list-mics', action='store_true')
    ap.add_argument('--test-voice', action='store_true')
    ap.add_argument('--check-setup', action='store_true', help='Validate JSON and reference; no camera or STT')
    ap.add_argument('--experimental-live', action='store_true', help='Required consent to operate prototype on real paper; sighted supervision only')
    ap.add_argument('--strict-full-page', action='store_true')
    ap.add_argument('--max-feature-size', type=int, default=1080)
    ap.add_argument('--near-ref-size', type=int, default=1600)
    ap.add_argument('--hit-mm', type=float, default=8.0)
    ap.add_argument('--hold-sec', type=float, default=0.65)
    ap.add_argument('--mirror', action='store_true')
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.list_mics:
        WorkflowSpeechCommands.list_microphones()
        return
    if args.test_voice:
        raise SystemExit(voice_diagnostic(args.tts_backend, args.tts_rate))
    if not (3 <= args.hit_mm <= 20) or not (0.35 <= args.hold_sec <= 2):
        ap.error('--hit-mm must be 3..20 and --hold-sec 0.35..2')
    try:
        linked, manifest = setup_linked(args)
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))
    for fid in args.block_field:
        if fid not in linked.fields:
            ap.error(f'Unknown --block-field ID: {fid}')
        field = linked.fields[fid]
        field.nav_status = 'blocked_operator'
        field.nav_reason = 'operator temporarily blocked suspicious position'
        field.target_mm = None
    print(linked.report(), flush=True)
    print('[RUNTIME NAV] Safety-eligible targets:', sum(nav_allowed(f) for f in linked.fields.values()), '/', len(linked.fields))
    print('IMPORTANT: verified coordinates are human APPROVALS, not physical contact proof.')
    if args.check_setup:
        print('[CHECK-SETUP OK] v8 linker + same-photo SHA validated. Camera/STT NOT tested.')
        return
    if not args.experimental_live:
        ap.error('Live prototype requires --experimental-live and sighted supervision. Review F033 and other suspicious targets first.')
    print('[WARNING] AI document meaning not human-reviewed; user completion is self-report only.')

    try:
        if args.strict_full_page:
            tracker = PaperTracker(manifest, args.reference, max_dim=args.max_feature_size)
        else:
            tracker = ElbowClosePaperTracker(manifest, args.reference,
                        max_dim=args.max_feature_size, detail_max_dim=args.near_ref_size)
    except Exception as exc:
        ap.error('Cannot initialize paper tracker: ' + str(exc))
    # DEMO_V8_RECOVERY_V1: a second, independent reference matcher helps recover
    # when close-view ORB cannot find enough distributed features. Both matchers
    # still have their own geometric acceptance tests; no fake homographies.
    recovery_tracker = None
    if not args.strict_full_page:
        try:
            recovery_tracker = PaperTracker(manifest, args.reference,
                                            max_dim=args.max_feature_size)
            print('[RECOVERY READY] Extra full-page feature matcher is available.', flush=True)
        except Exception as exc:
            print(f'[RECOVERY NOTE] Full-page fallback unavailable: {exc}', flush=True)
    if not args.model.is_file():
        ap.error('MediaPipe model not found: ' + str(args.model))
    try:
        mp, landmarker = create_landmarker(args.model)
    except Exception as exc:
        ap.error('MediaPipe initialization failed: ' + str(exc))
    def cam_open():
        return open_camera(args.camera, args.frame_width, args.frame_height, args.camera_backend)

    cap = cam_open()
    camera = CameraRecovery(cap, cam_open, retry_interval=2.5)
    if not cap.isOpened():
        print('[CAMERA WAIT] Camo capture did not open; keep dialogue running and retry the SAME camera.', flush=True)
    else:
        print('[CAMERA OPEN] Waiting for first valid frame from Camo (MSMF).', flush=True)

    voice = Narrator(args.tts, args.tts_backend, args.tts_rate)
    sound = GuidanceBeeper(args.beep)
    speech = WorkflowSpeechCommands(args.stt, voice, args.mic_index)
    controller = WorkflowDialogueV8(linked)
    active_field_id = None
    active_generation = 0
    arrival_queue = queue.Queue()
    arrived_since = None
    arrived_latched = False
    finger_smooth = None
    last_finger_seen = 0.0
    last_mp_ms = -1
    next_beep = 0.0
    last_direction_voice = 0.0
    prev_voice_direction = None
    prev_voice_band = None
    lost_since = None
    target_offscreen_since = None
    stable_since = None
    target_visible = False
    recovery_since = None
    recovery_frames = 0
    recovery_started = None
    recovery_last_reset = 0.0
    recovery_last_hint = 0.0
    recovery_fallback_reported = False
    last_near_at = -1e9
    logged_finished = False
    last_debug_print = 0.0
    last_camera_message = 0.0
    started = time.monotonic()
    fps_last = started
    fps = 0.0

    def clear_audio():
        nonlocal arrived_since, arrived_latched, finger_smooth, next_beep
        nonlocal last_direction_voice, prev_voice_direction, prev_voice_band
        sound.cancel()
        arrived_since = None
        arrived_latched = False
        finger_smooth = None
        next_beep = time.monotonic() + 0.45
        last_direction_voice = 0.
        prev_voice_direction = None
        prev_voice_band = None

    def emit(events):
        nonlocal active_field_id, active_generation, last_near_at
        for e in events:
            if e.kind == 'SPEAK':
                print('[GUIDE]', e.text, flush=True)
                if args.tts:
                    voice.say(e.text)
            elif e.kind == 'START_TRACKING':
                f = linked.fields.get(e.field_id)
                if not f or not nav_allowed(f) or f.target_mm is None:
                    # Controller has independent guards too; fail closed here.
                    print('[SAFETY BLOCK] No approved applicant navigation for', e.field_id, flush=True)
                    active_field_id = None
                    clear_audio()
                    raise RuntimeError('Controller requested navigation without reviewed applicant target')
                active_generation += 1
                last_near_at = -1e9
                active_field_id = f.id
                clear_audio()
                print(f'[TRACK START] {f.id} {f.name} @ {f.target_mm} mm', flush=True)
            elif e.kind in ('STOP_TRACKING', 'FINISHED'):
                active_generation += 1
                last_near_at = -1e9
                active_field_id = None
                clear_audio()
            else:
                print('[UNHANDLED EVENT]', e, flush=True)

    def resume_allowed(now):
        if controller.state != 'PAUSED' or controller.resume_state != 'TRACKING':
            return True
        return stable_since is not None and now-stable_since >= 0.6 and target_visible

    def accept_user(text, now):
        # Guard BOTH microphone and keyboard resume paths. Lost-paper recovery
        # requires a fresh, stable homography AND visibility of the same target.
        if str(text).strip() in ('계속', '재개', '다시 시작', '다시시작') and not resume_allowed(now):
            emit(controller.say('문서 위치를 아직 안정적으로 찾지 못했습니다. 잠시 기다려 주세요.'))
            return
        emit(controller.on_user(text))

    def arrived_from_beep(fid, gen):
        # Called by sound callback, possibly on a background thread.
        # Main camera thread is the only thread allowed to change dialogue state.
        arrival_queue.put((fid, gen))

    speech.start()
    window = 'v8 Dynamic Workflow | iPhone Camo | Q exit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print('[KEYS] 1-4 select, Y yes, U no, C completed, K skip, P previous, O overview, I repeat, Space pause/resume, R reset, Q quit')
    emit(controller.start())
    try:
        with landmarker:
            while True:
                live_frame, camera_status = camera.read()
                camera_frame_ok = live_frame is not None
                now = time.monotonic()
                if camera_status == 'first_frame':
                    print('[CAMERA FRAME OK] Live Camo image received. STT remains independent.', flush=True)
                elif camera_status == 'recovered':
                    print('[CAMERA RESTORED] Live Camo image recovered. Reacquiring homography.', flush=True)
                    # A new camera stream cannot inherit an old homography.
                    tracker.last_good = None
                    tracker.pending = None
                    tracker.pending_count = 0
                    tracker.fail_frames = 0
                elif camera_status == 'lost':
                    print('[CAMERA LOST] Frame missing; navigation paused, dialogue/STT kept alive.', flush=True)
                elif camera_status == 'retry_open':
                    print(f'[CAMERA RETRY] Reopening camera {args.camera} ({args.camera_backend}), attempt {camera.reopen_count} '
                          f'{camera.last_error}', flush=True)
                elif not camera_frame_ok and now-last_camera_message >= 6.0:
                    print('[CAMERA WAIT] No usable frames yet; check Camo Studio source and virtual camera.', flush=True)
                    last_camera_message = now

                if not camera_frame_ok:
                    # Never pass a blank/stale frame to ORB, MediaPipe or the nav engine.
                    target_visible = False
                    stable_since = None
                    clear_audio()
                    if active_field_id and controller.state == 'TRACKING':
                        emit(controller.on_tracking_lost())
                    frame = np.zeros((args.frame_height, args.frame_width, 3), dtype=np.uint8)
                    time.sleep(0.08)  # avoid a CPU-spinning loop on disconnected camera
                else:
                    frame = live_frame
                    if args.mirror:
                        frame = cv2.flip(frame, 1)
                dt = max(0.001, now - fps_last)
                fps = (fps*0.86 + 0.14/dt) if fps else 1/dt
                fps_last = now
                h, w = frame.shape[:2]

                # Beeper completion event only applies to the exact same active field/generation.
                while True:
                    try:
                        fid, gen = arrival_queue.get_nowait()
                    except queue.Empty:
                        break
                    if (fid == active_field_id and gen == active_generation and target_visible
                            and now-last_near_at <= 0.5):
                        emit(controller.on_arrived(fid))
                    elif fid == active_field_id and gen == active_generation:
                        # Beep callback arrived after finger left or geometry was lost.
                        arrived_latched = False
                        arrived_since = None

                cmd = speech.get()
                if cmd:
                    op, val = cmd
                    if op == 'retry':
                        # Ignore empty STT transcripts while tracking/reacquiring.
                        # Those are NOT requests from the user, and spoken retries
                        # otherwise loop over the microphone and drown out guidance.
                        if controller.state in ('TRACKING', 'PAUSED', 'WAIT_WRITING'):
                            print(f'[STT RETRY MUTED] state={controller.state}', flush=True)
                        elif args.tts and not voice.busy:
                            voice.say('다시 말씀해 주세요.')
                    elif op == 'utterance':
                        accept_user(val, now)

                vis = frame.copy()
                # Try both detectors while searching for the SAME reviewed page.
                # Do NOT accept a fallback unless it independently returns a
                # valid transform. The normal 1s + 6 frames + target-visible
                # restore requirement remains unchanged below.
                result = tracker.update(frame) if camera_frame_ok else None
                used_fallback = False
                if (result is None and camera_frame_ok and recovery_tracker is not None
                        and controller.state in ('TRACKING', 'PAUSED')
                        and (active_field_id is not None
                             or getattr(controller, 'auto_resume_tracking', False))):
                    fallback_result = recovery_tracker.update(frame)
                    if fallback_result is not None:
                        result = fallback_result
                        used_fallback = True
                        if not recovery_fallback_reported:
                            print('[RECOVERY FULLPAGE MATCH] fallback geometric match acquired; checking target and stability.', flush=True)
                            recovery_fallback_reported = True
                if result is not None and not used_fallback:
                    recovery_fallback_reported = False
                # Never say 'recovered' merely because a camera frame exists.
                if (controller.state == 'PAUSED'
                        and getattr(controller, 'auto_resume_tracking', False)):
                    if recovery_started is None:
                        recovery_started = now
                    if now - recovery_last_hint >= 15.0:
                        msg = ('문서를 다시 찾고 있습니다. 휴대전화를 종이에서 조금 멀리 들어, '
                               '종이 전체가 카메라에 들어오도록 비춰 주세요.'
                               if camera_frame_ok else
                               '카메라 화면을 받을 수 없습니다. 휴대전화 카메라 연결을 확인해 주세요.')
                        print('[RECOVERY GUIDANCE]', msg, flush=True)
                        if args.tts and not voice.busy:
                            voice.say(msg)
                        recovery_last_hint = now
                    if (result is None and now-recovery_started >= 10.0
                            and now-recovery_last_reset >= 10.0):
                        # Drop stale tracking candidates; never loosen a match threshold.
                        for tr in (tracker, recovery_tracker):
                            if tr is not None:
                                for attr, value in (('last_good', None), ('pending', None),
                                                    ('pending_count', 0), ('fail_frames', 0)):
                                    if hasattr(tr, attr):
                                        setattr(tr, attr, value)
                        recovery_last_reset = now
                        print('[RECOVERY RESET] Cleared stale tracking candidates; retrying.', flush=True)
                else:
                    recovery_started = None
                if result is None:
                    recovery_since = None
                    recovery_frames = 0
                    target_visible = False
                    stable_since = None
                    if active_field_id:
                        if lost_since is None:
                            lost_since = now
                        # Immediate silence even for brief homography failure.
                        clear_audio()
                        if now-lost_since >= 1.5 and controller.state == 'TRACKING':
                            emit(controller.on_tracking_lost())
                    else:
                        lost_since = None
                    target_offscreen_since = None
                    if now-last_debug_print > 2:
                        print('[TRACK LOST]', tracker.last_reason, flush=True)
                        last_debug_print = now
                    if camera_frame_ok:
                        put_line(vis, 'SEARCHING: GUIDANCE SILENCED', 29, (0,60,255), .65)
                        put_line(vis, str(tracker.last_reason)[:85], 60, (0,200,255), .52)
                    else:
                        put_line(vis, 'CAMERA NOT READY - GUIDANCE PAUSED', 29, (0,60,255), .65)
                        put_line(vis, 'Check Camo source / virtual camera. Retry is automatic.', 60, (0,200,255), .52)
                else:
                    if stable_since is None:
                        stable_since = now
                    lost_since = None
                    q = result.corners
                    if np.max(np.abs(q)) < 30000:
                        cv2.polylines(vis, [np.round(q).astype(np.int32)], True, (0,210,35), 2)
                    put_line(vis, f'TRACKING [{result.method}] {result.detail[:40]}', 29, (0,240,80), .57)
                    if not active_field_id:
                        target_visible = False
                        if controller.state == 'PAUSED' and controller.resume_state == 'TRACKING' and controller.current_track:
                            resume_target = linked.fields[controller.current_track].target_mm
                            if resume_target is not None:
                                resume_loc = warp_points([resume_target], result.H_mm_to_live)[0]
                                target_visible = _in_image(resume_loc, frame.shape, pad=12)
                        # Auto-resume ONLY after consecutive stable paper matches and
                        # the SAME approved target remains inside the camera image.
                        if (controller.state == 'PAUSED'
                                and controller.resume_state == 'TRACKING'
                                and getattr(controller, 'auto_resume_tracking', False)
                                and target_visible):
                            if recovery_since is None:
                                recovery_since = now
                            recovery_frames += 1
                            if now - recovery_since >= 1.0 and recovery_frames >= 6:
                                fid = controller.current_track
                                print(f'[AUTO DOCUMENT RESTORED] target={fid} stable_for={now-recovery_since:.2f}s', flush=True)
                                emit(controller.on_tracking_restored())
                                recovery_since = None
                                recovery_frames = 0
                        else:
                            recovery_since = None
                            recovery_frames = 0
                        put_line(vis, f'DIALOGUE {controller.state}', 61, (0,210,255), .58)
                    else:
                        f = linked.fields[active_field_id]
                        target_mm = f.target_mm
                        loc = warp_points([target_mm], result.H_mm_to_live)[0]
                        if not _in_image(loc, frame.shape, pad=12):
                            target_visible = False
                            clear_audio()
                            if target_offscreen_since is None:
                                target_offscreen_since = now
                            put_line(vis, 'TARGET OUT OF VIEW - NO GUIDANCE', 61, (0,160,255), .58)
                            if now-target_offscreen_since >= 3.0 and controller.state == 'TRACKING':
                                emit(controller.on_tracking_lost())
                        else:
                            target_visible = True
                            target_offscreen_since = None
                            cv2.drawMarker(vis, tuple(np.rint(loc).astype(int)), (0,0,255), cv2.MARKER_CROSS, 27, 3)
                            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            stamp = max(last_mp_ms+1, int((now-started)*1000))
                            last_mp_ms = stamp
                            detected = landmarker.detect_for_video(mp_img, stamp)
                            finger_valid = False
                            if detected.hand_landmarks:
                                tip = detected.hand_landmarks[0][8]
                                px, py = float(tip.x*w), float(tip.y*h)
                                if 0 <= px < w and 0 <= py < h:
                                    cv2.circle(vis, (round(px),round(py)), 8, (255,120,0), -1)
                                    world = warp_points([(px,py)], result.H_live_to_mm)[0]
                                    pw, ph = float(manifest['paper_width_mm']), float(manifest['paper_height_mm'])
                                    if np.isfinite(world).all() and 0 <= world[0] <= pw and 0 <= world[1] <= ph:
                                        finger_valid = True
                                        if finger_smooth is None or now-last_finger_seen > .5:
                                            finger_smooth = world
                                        else:
                                            finger_smooth = .55*world + .45*finger_smooth
                                        last_finger_seen = now
                                        dist, direction, primary = movement(finger_smooth, target_mm)
                                        put_line(vis, f'TIP ({finger_smooth[0]:.0f},{finger_smooth[1]:.0f})mm DIST {dist:.1f}mm {direction}',
                                                 61, (245,245,245), .56)
                                        dx = float(target_mm[0]-finger_smooth[0])
                                        dy = float(target_mm[1]-finger_smooth[1])
                                        if dist <= args.hit_mm:
                                            last_near_at = now
                                            if arrived_since is None:
                                                arrived_since = now
                                            if not arrived_latched and now-arrived_since >= args.hold_sec:
                                                arrived_latched = True
                                                if not args.beep:
                                                    emit(controller.on_arrived(active_field_id))
                                                else:
                                                    fid, generation = active_field_id, active_generation
                                                    # Cancelled if target changes before completion.
                                                    if sound.arrival(on_complete=lambda f=fid,g=generation: arrived_from_beep(f,g)):
                                                        print('[ARRIVAL] long beep queued', flush=True)
                                                    else:
                                                        arrived_latched = False
                                            if not arrived_latched:
                                                interval = beep_interval(dist, args.hit_mm)
                                                if interval is not None and now >= next_beep:
                                                    sound.pulse(dist,args.hit_mm)
                                                    next_beep = now+interval
                                        else:
                                            arrived_since = None
                                            if arrived_latched and dist > args.hit_mm*1.65:
                                                arrived_latched = False
                                            if not arrived_latched:
                                                interval = beep_interval(dist,args.hit_mm)
                                                if interval is not None and now >= next_beep:
                                                    sound.pulse(dist,args.hit_mm)
                                                    next_beep = now+interval
                                                band = direction_zone(dist)
                                                heading = direction_signature(dx,dy,dist)
                                                cadence = {'far':3.7, 'near':4.5, 'closer':4.2,'very_close':5.2}[band]
                                                changed = heading != prev_voice_direction or band != prev_voice_band
                                                elapsed = now - last_direction_voice
                                                if args.tts and not voice.busy and (elapsed >= cadence or (changed and elapsed >= 2)):
                                                    message = direction_phrase(dx,dy,dist)
                                                    if message:
                                                        voice.say(message)
                                                        last_direction_voice = now
                                                        prev_voice_direction, prev_voice_band = heading, band
                                        stage = direction_zone(dist)
                                        put_line(vis, f'AUDIO: {stage.upper()}', 91, (190,235,245), .5)
                            if not finger_valid:
                                finger_smooth = None
                                arrived_since = None
                                sound.cancel()
                                put_line(vis, 'FINGER NOT DETECTED - NO GUIDANCE', 61, (0,180,255), .54)
                label = linked.fields[active_field_id].name if active_field_id else ''
                put_line(vis, (f'{controller.state} | {label}')[:65], h-37, (240,240,240), .54)
                put_line(vis, f'CAM {args.camera} | {fps:.0f}fps | STT {"ON" if speech.ready else "OFF"}', h-12, (240,240,240), .47)
                cv2.imshow(window,vis)
                key = cv2.waitKey(1) & 255
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif ord('1') <= key <= ord('4'):
                    emit(controller.on_user(f'{key-ord("0")}번'))
                elif key in (ord('y'),ord('Y'),13):
                    emit(controller.on_user('예'))
                elif key in (ord('u'),ord('U')):
                    emit(controller.on_user('아니요'))
                elif key in (ord('c'),ord('C')):
                    emit(controller.on_user('작성 완료'))
                elif key in (ord('k'),ord('K')):
                    emit(controller.on_user('건너뛰기'))
                elif key in (ord('p'),ord('P')):
                    emit(controller.on_user('이전'))
                elif key in (ord('o'),ord('O')):
                    emit(controller.on_user('전체설명'))
                elif key in (ord('i'),ord('I')):
                    emit(controller.on_user('다시설명'))
                elif key == 32:
                    accept_user('계속' if controller.state=='PAUSED' else '중단', now)
                elif key in (ord('r'),ord('R')):
                    tracker.last_good = None
                    tracker.pending = None
                    tracker.pending_count = 0
                    tracker.fail_frames = 0
                    clear_audio()
                    print('[RESET] Tracking will re-acquire paper')
                if controller.state == 'FINISHED' and not logged_finished:
                    logged_finished = True
                    print('[SESSION]', controller.summary(), flush=True)
                    # Remain open for operator review until Q is pressed.
    finally:
        speech.stop.set()
        sound.close()
        voice.stop.set()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
