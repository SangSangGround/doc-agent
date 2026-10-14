"""문서 종류와 무관한 촬영 품질 휴리스틱. 판정 불가를 통과시키지 않는다."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from docagent.vision.normalize import detect_document_quad_detailed

MESSAGES = {
    "invalid_image": "이미지를 읽을 수 없습니다. 다시 촬영해주세요.",
    "low_resolution": "문서의 글자가 더 크게 보이도록 가까이에서 다시 촬영해주세요.",
    "low_contrast": "문서의 글자가 잘 보이도록 조명을 조절해 다시 촬영해주세요.",
    "too_dark": "문서가 어둡게 촬영되었습니다. 밝은 곳에서 다시 촬영해주세요.",
    "blurred": "문서가 흐리게 촬영되었습니다. 카메라를 고정하고 초점을 맞춰 다시 촬영해주세요.",
    "boundary_unknown": "문서 전체가 담겼는지 확인하기 어렵습니다. 네 모서리가 보이게 다시 촬영해주세요.",
    "edge_cutoff": "문서가 사진 가장자리에 닿아 있습니다. 문서 주변에 여백을 두고 다시 촬영해주세요.",
    "tilted": "문서가 기울어져 있습니다. 카메라를 문서와 평행하게 놓고 다시 촬영해주세요.",
}
PASS_MESSAGE = "기본 촬영 품질 검사를 통과했습니다. 문서와 작성 영역은 추가 확인이 필요합니다."


@dataclass(frozen=True)
class QualityPolicy:
    min_short_side: int = 600
    min_sharpness: float = 60.0
    min_contrast: float = 12.0
    min_brightness: float = 45.0
    max_tilt_degrees: float = 12.0
    min_boundary_confidence: float = .85

    def __post_init__(self):
        if type(self.min_short_side) is not int or self.min_short_side < 8:
            raise ValueError("최소 해상도가 잘못되었습니다.")
        for value in (self.min_sharpness, self.min_contrast, self.min_brightness,
                      self.max_tilt_degrees, self.min_boundary_confidence):
            if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                raise ValueError("품질 검사 임계값이 잘못되었습니다.")
        if self.min_boundary_confidence > 1 or self.max_tilt_degrees > 90:
            raise ValueError("품질 검사 임계값 범위가 잘못되었습니다.")


@dataclass(frozen=True)
class CaptureQuality:
    status: str
    reasons: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    mode: str = "photo"

    @property
    def can_continue(self) -> bool:
        return self.status == "pass"

    @property
    def instruction(self) -> str:
        return PASS_MESSAGE if self.can_continue else MESSAGES[self.reasons[0]]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "can_continue":self.can_continue, "instruction":self.instruction}


def assess_capture(image: Any, *, mode: str = "photo",
                   policy: QualityPolicy | None = None) -> CaptureQuality:
    if mode not in ("photo", "scan"):
        raise ValueError("촬영 모드는 photo 또는 scan 이어야 합니다.")
    policy = policy or QualityPolicy()
    if (not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2,3)
            or image.size == 0 or min(image.shape[:2]) < 8
            or (image.ndim == 3 and image.shape[2] != 3)):
        return CaptureQuality("retake",("invalid_image",),mode=mode)
    h,w = image.shape[:2]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0,1200/max(h,w))
    sample = cv2.resize(gray,(max(8,round(w*scale)),max(8,round(h*scale))),interpolation=cv2.INTER_AREA)
    brightness = float(np.mean(sample))
    contrast = float(np.std(sample))
    sharpness = float(cv2.Laplacian(sample,cv2.CV_64F).var())
    metrics = dict(brightness=brightness,contrast=contrast,sharpness=sharpness,short_side=float(min(h,w)))
    failures = []
    if min(h,w) < policy.min_short_side: failures.append("low_resolution")
    if brightness < policy.min_brightness: failures.append("too_dark")
    if contrast < policy.min_contrast: failures.append("low_contrast")
    if sharpness < policy.min_sharpness: failures.append("blurred")
    unknown = False
    if mode == "photo":
        boundary = detect_document_quad_detailed(image)
        metrics["boundary_confidence"] = boundary.confidence
        if boundary.fallback or boundary.confidence < policy.min_boundary_confidence:
            unknown = True
        else:
            q = boundary.quad
            margin = max(2,min(h,w)*.01)
            if (np.any(q[:,0] <= margin) or np.any(q[:,0] >= w-1-margin)
                    or np.any(q[:,1] <= margin) or np.any(q[:,1] >= h-1-margin)):
                failures.append("edge_cutoff")
            angle = abs(math.degrees(math.atan2(float(q[1,1]-q[0,1]),float(q[1,0]-q[0,0]))))
            angle = min(angle,abs(180-angle))
            metrics["tilt_degrees"] = angle
            if angle > policy.max_tilt_degrees: failures.append("tilted")
    if failures:
        return CaptureQuality("retake",tuple(failures + (["boundary_unknown"] if unknown else [])),metrics,mode)
    if unknown:
        return CaptureQuality("review",("boundary_unknown",),metrics,mode)
    return CaptureQuality("pass",metrics=metrics,mode=mode)
