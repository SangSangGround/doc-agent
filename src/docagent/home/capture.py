"""연속 카메라 프레임의 로컬 품질 검사와 음성 안내 빈도 조절."""
from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from docagent.vision.quality import QualityPolicy, assess_capture


class CaptureCoach:
    """여러 선명한 프레임이 유지된 뒤에만 촬영 준비 상태를 제공한다.

    실제 카메라 접근·음성 재생은 호출자가 담당한다. `should_announce`가 참일
    때만 `instruction`을 읽으면 오류 안내가 매 프레임 반복되지 않는다.
    품질 임계값은 휴리스틱이며 OCR 정확도나 개인정보 제거를 보장하지 않는다.
    """

    def __init__(self, *, mode: str = "photo", stable_frames: int = 3,
                 stable_seconds: float = .6, announcement_interval: float = 3.,
                 repeat_interval: float = 8., motion_threshold: float = 4.,
                 policy: QualityPolicy | None = None):
        if mode not in {"photo", "scan"}:
            raise ValueError("촬영 모드는 photo 또는 scan 이어야 합니다.")
        if type(stable_frames) is not int or stable_frames < 2:
            raise ValueError("안정 프레임 수는 2 이상이어야 합니다.")
        for value in (stable_seconds, announcement_interval, repeat_interval, motion_threshold):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("촬영 안내 설정은 유한한 양수여야 합니다.")
        self.mode = mode
        self.policy = policy
        self.stable_frames = stable_frames
        self.stable_seconds = stable_seconds
        self.announcement_interval = announcement_interval
        self.repeat_interval = max(repeat_interval, announcement_interval)
        self.motion_threshold = motion_threshold
        self.reset()

    def reset(self) -> None:
        """카메라·대상 문서가 바뀌었을 때 이전 안정 상태를 지운다."""
        self._last_time: float | None = None
        self._stable_since: float | None = None
        self._good_frames = 0
        self._last_frame: np.ndarray | None = None
        self._last_announcement_time: float | None = None
        self._last_announcement_key: str | None = None
        self._candidate_key: str | None = None
        self._candidate_frames = 0

    def update(self, image: Any, *, timestamp: float | None = None) -> dict[str, Any]:
        """한 프레임을 검사한다. 시간 역행은 이전 상태를 변경하지 않고 거절한다."""
        now = time.monotonic() if timestamp is None else timestamp
        if (type(now) not in (int, float) or not math.isfinite(now)
                or (self._last_time is not None and now < self._last_time)):
            raise ValueError("프레임 시간은 단조 증가하는 유한한 수여야 합니다.")
        previous_time = self._last_time
        self._last_time = now
        quality = assess_capture(image, mode=self.mode, policy=self.policy)
        motion = 0.0
        if quality.can_continue:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            thumb = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            if self._last_frame is not None:
                motion = float(np.mean(cv2.absdiff(thumb, self._last_frame)))
            # 긴 중단 후의 두 프레임을 연속 관측으로 취급하지 않는다.
            interrupted = previous_time is not None and now - previous_time > max(2., self.stable_seconds * 3)
            if self._stable_since is None or motion > self.motion_threshold or interrupted:
                self._stable_since = now
                self._good_frames = 1
            elif previous_time is None or now > previous_time:
                self._good_frames += 1
            self._last_frame = thumb
        else:
            self._stable_since = None
            self._good_frames = 0
            self._last_frame = None
        ready = (quality.can_continue and self._good_frames >= self.stable_frames
                 and self._stable_since is not None and now - self._stable_since >= self.stable_seconds)
        status = "ready" if ready else "stabilizing" if quality.can_continue else quality.status
        if ready:
            instruction = "촬영할 수 있습니다. 문서를 그대로 유지해주세요."
        elif quality.can_continue:
            instruction = "문서가 보입니다. 카메라를 잠시 고정해주세요."
        else:
            instruction = quality.instruction
        key = status + ":" + (quality.reasons[0] if quality.reasons else "")
        if key == self._candidate_key:
            self._candidate_frames += 1
        else:
            self._candidate_key, self._candidate_frames = key, 1
        first = self._last_announcement_time is None
        elapsed = float("inf") if first else now - self._last_announcement_time
        changed = key != self._last_announcement_key
        announce = first or (
            self._candidate_frames >= 2 and elapsed >= self.announcement_interval
            and (changed or elapsed >= self.repeat_interval)
        )
        if announce:
            self._last_announcement_key = key
            self._last_announcement_time = now
        return {
            "status": status, "instruction": instruction,
            "should_announce": announce, "ready_to_capture": bool(ready),
            "stable_frames": self._good_frames, "motion_score": round(motion, 3),
            "quality": quality.to_dict(),
        }
