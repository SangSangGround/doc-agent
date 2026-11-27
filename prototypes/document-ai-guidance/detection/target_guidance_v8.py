"""v8 single-target finger guidance, extracted from multi_field_voice_tracking_v7_5.

Same thresholds and helpers as v7.5 (proximity_audio_v7_4, movement, warp_points):
  direction voice + distance beeps, rapid beeps near target, ~0.65 s hold ->
  arrival chirps + long beep. v7.5 itself is NOT modified or replaced.

Difference from v7.5: instead of speaking '도달했습니다' itself, arrival calls
on_arrival(field_id) so the DialogueController decides what to say next.
Guidance runs only between start(field_id) and stop(). A target must come from
the reviewed manifest; unknown ids raise KeyError.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from multi_field_voice_tracking_v6 import movement, put_line, warp_points
from proximity_audio_v7_4 import beep_interval, direction_phrase, direction_signature, direction_zone


def in_image(point, shape, pad=10):
    h, w = shape[:2]
    x, y = map(float, point)
    return math.isfinite(x) and math.isfinite(y) and pad <= x < w - pad and pad <= y < h - pad


class TargetGuidance:
    def __init__(self, manifest, sound, voice, mp, landmark, *, hit_mm=8., hold_sec=.65,
                 beep=True, tts=True, on_arrival=None, started=0.):
        self.pw = float(manifest['paper_width_mm'])
        self.ph = float(manifest['paper_height_mm'])
        self.targets = {f['id']: f for f in manifest['fields']}
        self.sound, self.voice, self.mp, self.landmark = sound, voice, mp, landmark
        self.hit_mm, self.hold_sec = float(hit_mm), float(hold_sec)
        self.beep = bool(beep) and bool(getattr(sound, 'enabled', beep))
        self.tts = bool(tts)
        self.on_arrival = on_arrival
        self.started = started
        self.last_mp_ms = -1
        self.field = None
        self.last_target_offscreen = False
        self.last_offscreen_alert = 0.
        self.arrival_sent = False
        self.reset_audio(started)

    @property
    def active(self):
        return self.field is not None

    @property
    def field_id(self):
        return self.field['id'] if self.field else None

    def start(self, field_id, now):
        self.field = self.targets[field_id]          # KeyError for unreviewed ids
        self.last_target_offscreen = False
        self.reset_audio(now)

    def stop(self, now=0.):
        self.field = None
        self.reset_audio(now)

    def reset_audio(self, now):
        self.sound.cancel()                          # suppress pending arrival callback
        self.arrived_since = None
        self.arrived_latched = False
        self.arrival_sent = False
        self.next_beep = now + .45
        self.last_direction_voice = 0.
        self.prev_voice_direction = None
        self.prev_voice_band = None
        self.finger_smooth = None
        self.last_finger_seen = 0.

    def tracking_lost(self):
        self.finger_smooth = None
        self.arrived_since = None
        self.last_target_offscreen = False

    def _emit_arrival(self, fid):
        if self.arrival_sent or self.field_id != fid:
            return
        self.arrival_sent = True
        if self.on_arrival is not None:
            self.on_arrival(fid)

    def update(self, frame, result, now, vis):
        """One frame of guidance; `result` must be a current, valid TrackResult."""
        if not self.active or result is None:
            return
        h, w = frame.shape[:2]
        field = self.field
        fid = field['id']
        target = field['target_mm']
        target_mm = (float(target['x']), float(target['y']))
        loc = warp_points([target_mm], result.H_mm_to_live)[0]
        if not in_image(loc, frame.shape, pad=12):
            self.finger_smooth = None
            self.arrived_since = None
            if not self.last_target_offscreen and now - self.last_offscreen_alert > 2:
                self.voice.say('목표 작성란이 현재 카메라 화면 밖에 있습니다. 휴대폰 위치를 조금 이동해 주세요.')
                self.last_offscreen_alert = now
            self.last_target_offscreen = True
            put_line(vis, 'TARGET OUTSIDE CAMERA VIEW - NO GUIDANCE', 62, (0, 160, 255), .58)
            return
        self.last_target_offscreen = False
        cv2.drawMarker(vis, tuple(np.rint(loc).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 27, 3)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        stamp = max(self.last_mp_ms + 1, int((now - self.started) * 1000))
        self.last_mp_ms = stamp
        detected = self.landmark.detect_for_video(mp_img, stamp)
        finger_valid = False
        if detected.hand_landmarks:
            tip = detected.hand_landmarks[0][8]
            px, py = float(tip.x * w), float(tip.y * h)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(vis, (round(px), round(py)), 8, (255, 120, 0), -1)
                world = warp_points([(px, py)], result.H_live_to_mm)[0]
                if np.isfinite(world).all() and 0 <= world[0] <= self.pw and 0 <= world[1] <= self.ph:
                    finger_valid = True
                    self._guide(world, target_mm, fid, now, vis)
        if not finger_valid:
            self.finger_smooth = None
            self.arrived_since = None
            self.sound.cancel()
            put_line(vis, 'NO VALID FINGER ON PAPER - GUIDANCE PAUSED', 61, (0, 180, 255), .54)

    def _guide(self, world, target_mm, fid, now, vis):
        if self.finger_smooth is None or now - self.last_finger_seen > .5:
            self.finger_smooth = world
        else:
            self.finger_smooth = .55 * world + .45 * self.finger_smooth
        self.last_finger_seen = now
        fs = self.finger_smooth
        dist, direction, _ = movement(fs, target_mm)
        put_line(vis, f'TIP ({fs[0]:.0f},{fs[1]:.0f})mm  DIST {dist:.1f}mm  {direction}', 61, (245, 245, 245), .56)
        dx = float(target_mm[0] - fs[0])           # +: move right on paper
        dy = float(target_mm[1] - fs[1])           # +: move down on paper
        if dist <= self.hit_mm:
            if self.arrived_since is None:
                self.arrived_since = now
            if not self.arrived_latched and now - self.arrived_since >= self.hold_sec:
                put_line(vis, 'NEAR TARGET - TOUCH NOT VERIFIED', 91, (0, 255, 100), .58)
                if not self.beep:
                    self.arrived_latched = True
                    self._emit_arrival(fid)
                elif self.sound.arrival(on_complete=lambda fid=fid: self._emit_arrival(fid)):
                    self.arrived_latched = True
                    print('[ARRIVAL] quick chirps -> long beep -> controller', flush=True)
            if not self.arrived_latched:
                interval = beep_interval(dist, self.hit_mm)
                if self.beep and interval is not None and now >= self.next_beep:
                    self.sound.pulse(dist, self.hit_mm)
                    self.next_beep = now + interval
        else:
            self.arrived_since = None
            if self.arrived_latched and dist > self.hit_mm * 1.65:
                self.arrived_latched = False
                self.arrival_sent = False
            if not self.arrived_latched:
                interval = beep_interval(dist, self.hit_mm)
                if self.beep and interval is not None and now >= self.next_beep:
                    self.sound.pulse(dist, self.hit_mm)
                    self.next_beep = now + interval
                band = direction_zone(dist)
                heading = direction_signature(dx, dy, dist)
                cadence = {'far': 3.7, 'near': 4.5, 'closer': 4.2, 'very_close': 5.2}[band]
                changed = heading != self.prev_voice_direction or band != self.prev_voice_band
                elapsed = now - self.last_direction_voice
                if self.tts and not self.voice.busy and (elapsed >= cadence or (changed and elapsed >= 2.0)):
                    text = direction_phrase(dx, dy, dist)
                    if text:
                        self.voice.say(text)
                        self.last_direction_voice = now
                        self.prev_voice_direction = heading
                        self.prev_voice_band = band
        stage = direction_zone(dist)
        put_line(vis, f'AUDIO: {stage.upper()}  {"beep+voice" if dist <= 100 else "voice only"}', 91, (190, 235, 245), .5)
