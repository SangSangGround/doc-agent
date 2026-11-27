"""Audio feedback for reviewed paper navigation targets.

Distance is estimated from projected index-tip position to the target ON THE
PAPER PLANE. This is an experimental proxy, not measured touch/contact.

Audio bands:
  >100 mm  : spoken direction only
  35-100 mm: spaced beep PLUS periodic spoken direction
   12-35 mm: quicker beep PLUS concise spoken direction
    8-12 mm: very quick beep PLUS optional near-target cue
  <=8 mm  : hold sufficiently long, then quick chirps + long beep + completion TTS
No sound is emitted from this module when tracking is lost. Call cancel() as
soon as the caller loses the homography, cannot see the target, or pauses.
"""
from __future__ import annotations
import math
import sys
import threading
import time


def beep_interval(distance_mm, hit_mm=8.0):
    """seconds between beeps, or None beyond speech-only range."""
    if not math.isfinite(distance_mm) or distance_mm < 0:
        return None
    if distance_mm > 100:
        return None
    if distance_mm > 60:
        return 0.85
    if distance_mm > 35:
        return 0.55
    if distance_mm > 20:
        return 0.32
    if distance_mm > 12:
        return 0.18
    if distance_mm > hit_mm:
        return 0.105
    return 0.085  # rapid approach tone while hold is accumulating


def direction_zone(distance_mm):
    if distance_mm > 100:
        return 'far'
    if distance_mm > 35:
        return 'near'
    if distance_mm > 12:
        return 'closer'
    return 'very_close'


def direction_phrase(dx_mm, dy_mm, distance_mm):
    """Use paper-coordinate axes; never pretend the finger is actually touching."""
    if not all(math.isfinite(v) for v in (dx_mm,dy_mm,distance_mm)):
        return ''
    adx, ady = abs(dx_mm), abs(dy_mm)
    # On a near target, few-millimetre jitter must not flip directions constantly.
    if distance_mm <= 10:
        return '거의 다 왔습니다. 손가락을 천천히 움직여 주세요.'
    horiz='오른쪽' if dx_mm > 0 else '왼쪽'
    vert='아래' if dy_mm > 0 else '위'
    diag = min(adx,ady) >= max(11.0, 0.38*max(adx,ady))
    if diag:
        heading=f'{horiz} {vert} 대각선으로'
    elif adx >= ady:
        heading=f'{horiz}으로'
    else:
        heading=f'{vert}로'
    if distance_mm > 100:
        return heading + ' 이동하세요.'
    if distance_mm > 35:
        return '계속 '+heading+' 움직이세요.'
    return '조금 더 '+heading+' 움직이세요.'


def direction_signature(dx,dy,distance):
    """Coarse heading allows extra feedback when the user changes direction."""
    if distance <= 10:
        return 'almost'
    def s(v):
        return 1 if v > 0 else -1
    ax,ay=abs(dx),abs(dy)
    if min(ax,ay) >= max(11.0,0.38*max(ax,ay)):
        return ('diag',s(dx),s(dy))
    return ('horizontal',s(dx)) if ax >= ay else ('vertical',s(dy))


class GuidanceBeeper:
    """Nonblocking beep producer with cancelable arrival pattern.

    On Windows it uses winsound.Beep; a custom beep_impl can be injected for
    deterministic tests on other operating systems.
    """
    def __init__(self, enabled=True, beep_impl=None, sleep_impl=None):
        self.enabled=bool(enabled) and (sys.platform=='win32' or beep_impl is not None)
        if self.enabled and beep_impl is None:
            import winsound
            beep_impl=winsound.Beep
        self._beep=beep_impl
        self._sleep=sleep_impl or time.sleep
        self._lock=threading.Lock()
        self._busy=False
        self._token=0
        self._closed=False

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def _run_async(self, notes, on_complete=None):
        if not self.enabled:
            return False
        with self._lock:
            if self._busy or self._closed:
                return False
            self._busy=True
            self._token+=1
            token=self._token

        def worker():
            success=True
            try:
                for hz,duration_ms,gap_ms in notes:
                    with self._lock:
                        if token != self._token or self._closed:
                            success=False
                            break
                    try:
                        self._beep(int(hz),int(duration_ms))
                    except Exception as exc:
                        print('[BEEP ERROR]', repr(exc), flush=True)
                        success=False
                        break
                    if gap_ms:
                        self._sleep(gap_ms/1000.)
            finally:
                with self._lock:
                    valid = token == self._token and not self._closed
                    self._busy=False
            if success and valid and on_complete is not None:
                try:
                    on_complete()
                except Exception as exc:
                    print('[ARRIVAL CALLBACK ERROR]',repr(exc),flush=True)
        threading.Thread(target=worker,daemon=True).start()
        return True

    def pulse(self, distance_mm, hit_mm=8.0):
        # Pulse *regardless of TTS busy*: it is meant to accompany speech.
        if beep_interval(distance_mm,hit_mm) is None:
            return False
        hz=950 if distance_mm>35 else (1050 if distance_mm>12 else 1160)
        duration=65 if distance_mm>12 else 48
        return self._run_async([(hz,duration,0)])

    def arrival(self, on_complete=None):
        # user-requested: 삐 삐 삐 비 비 ... 삐~~~ -> '도달했습니다'
        pattern=[(960,85,45),(960,85,45),(960,85,55),
                 (1130,65,38),(1130,65,100),(1420,800,0)]
        return self._run_async(pattern,on_complete=on_complete)

    def cancel(self):
        """Invalidate completion callback; interrupt between discrete notes."""
        with self._lock:
            self._token+=1

    def close(self):
        with self._lock:
            self._closed=True
            self._token+=1
