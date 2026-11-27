"""Recover transient Camo/OpenCV frame failures without using stale frames.

Uses only the originally selected camera/index/backend. It never silently chooses
another video device. No fake frame is ever returned as a valid camera frame.
"""
from __future__ import annotations
import time
from typing import Callable


class CameraRecovery:
    def __init__(self, capture, opener: Callable, *, retry_interval: float = 2.5, clock=time.monotonic):
        self.cap = capture
        self.opener = opener
        self.retry_interval = retry_interval
        self.clock = clock
        self.last_reopen_at = clock()
        self.was_live = False
        self.ever_live = False
        self.failures = 0
        self.last_error = ''
        self.reopen_count = 0

    def read(self):
        """Return (real_frame_or_None, status). None MUST disable navigation.

        Status: first_frame, ok, recovered, waiting, lost, retry_open.
        """
        now = self.clock()
        frame = None
        try:
            if self.cap is not None and self.cap.isOpened():
                valid, candidate = self.cap.read()
                if valid and candidate is not None and getattr(candidate, 'size', 0) > 0:
                    frame = candidate
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
        if frame is not None:
            status = 'ok' if self.was_live else 'recovered' if self.ever_live else 'first_frame'
            self.was_live = True
            self.ever_live = True
            self.failures = 0
            self.last_error = ''
            return frame, status
        status = 'lost' if self.was_live else 'waiting'
        self.was_live = False
        self.failures += 1
        if now - self.last_reopen_at >= self.retry_interval:
            self.last_reopen_at = now
            self.reopen_count += 1
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception as exc:
                self.last_error = f'Camera release: {type(exc).__name__}: {exc}'
            try:
                self.cap = self.opener()
            except Exception as exc:
                self.cap = None
                self.last_error = f'Camera reopen: {type(exc).__name__}: {exc}'
            status = 'retry_open'
        return None, status

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
