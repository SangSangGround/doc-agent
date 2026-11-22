"""macOS Vision OCR. 원본은 PNG 바이트로 표준 입력에만 전달한다."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


class NativeOcrUnavailable(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def build_helper() -> Path:
    if platform.system() != 'Darwin':
        raise NativeOcrUnavailable('macOS에서만 Apple Vision을 사용할 수 있습니다.')
    source = Path(__file__).with_name('native') / 'RecognizeText.swift'
    digest = hashlib.sha256(source.read_bytes() + platform.release().encode()).hexdigest()[:16]
    folder = Path(tempfile.gettempdir()) / f'docagent-vision-{os.getuid()}'
    folder.mkdir(mode=0o700, exist_ok=True)
    executable = folder / f'ocr-{digest}'
    if not executable.is_file():
        # 병렬 실행에서도 부분적으로 컴파일된 파일을 실행하지 않는다.
        with tempfile.TemporaryDirectory(prefix='build-', dir=folder) as temporary:
            target = Path(temporary) / 'ocr'
            try:
                subprocess.run(['xcrun', 'swiftc', str(source), '-O', '-module-cache-path',
                                str(folder / 'modules'), '-o', str(target)],
                               capture_output=True, timeout=120, check=True)
                target.chmod(0o700)
                os.replace(target, executable)
            except (OSError, subprocess.SubprocessError) as exc:
                raise NativeOcrUnavailable('Apple Vision OCR 준비에 실패했습니다. macOS Command Line Tools를 확인해 주세요.') from exc
    return executable


class MacVision:
    name = 'apple_vision'

    def __init__(self, binary: Path | None = None):
        self.binary = binary or build_helper()

    def _run(self, image, *arguments, timeout=35):
        success, encoded = cv2.imencode('.png', image)
        if not success:
            raise NativeOcrUnavailable('이미지를 준비하지 못했습니다.')
        try:
            result = subprocess.run([str(self.binary), *arguments], input=encoded.tobytes(),
                                    capture_output=True, timeout=timeout, check=True)
            if len(result.stdout) > 8 * 1024 * 1024:
                raise ValueError('OCR output too large')
            return json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise NativeOcrUnavailable('macOS 문자 인식을 실행하지 못했습니다. 앱 서버를 다시 시작해 주세요.') from exc

    def read(self, image):
        value = self._run(image)
        if not isinstance(value, dict) or value.get('engine') != self.name or not isinstance(value.get('lines'), list):
            raise NativeOcrUnavailable('문자 인식 결과 형식을 확인하지 못했습니다.')
        lines = []
        for line in value['lines']:
            if not isinstance(line.get('text'), str) or not line['text'].strip():
                continue
            box = np.asarray(line['box'], dtype=float)
            confidence = float(line['confidence'])
            if box.shape != (4,) or not np.isfinite(box).all() or not 0 <= confidence <= 1:
                raise NativeOcrUnavailable('문자 인식 좌표를 확인하지 못했습니다.')
            lines.append({**line, 'box': box.tolist(), 'confidence': confidence,
                          'uncertain': confidence < .7})
        return lines

    def boundary(self, image):
        """미리보기는 800px로 제한한다. 검출 실패를 화면 전체 경계로 대체하지 않는다."""
        h, w = image.shape[:2]
        scale = min(1., 800 / max(h, w))
        sample = cv2.resize(image, (round(w * scale), round(h * scale)))
        result = self._run(sample, '--document', timeout=8)
        return boundary_result(result, (h, w), scale)


def boundary_result(observations, shape, scale):
    h, w = shape
    candidates = []
    for observation in observations:
        q = np.asarray(observation.get('quad'), dtype=np.float32) / scale
        if q.shape != (4, 2) or not np.isfinite(q).all() or observation.get('confidence', 0) < .7:
            continue
        area = float(abs(cv2.contourArea(q))) / (w * h)
        if not .1 <= area <= 1.05:
            continue
        candidates.append((area, q, observation['confidence']))
    if not candidates:
        return {'status':'searching', 'quad':None, 'area_ratio':0., 'engine':'apple_vision',
                'instruction':'문서를 찾고 있어요. 네 모서리가 보이도록 종이와 카메라 사이에 여유를 주세요.'}
    area, q, confidence = max(candidates, key=lambda row: row[0])
    margin = max(2, min(h, w) * .008)
    clipped = bool(np.any(q[:, 0] <= margin) or np.any(q[:, 1] <= margin)
                   or np.any(q[:, 0] >= w - 1 - margin) or np.any(q[:, 1] >= h - 1 - margin))
    return {'status':'clipped' if clipped else 'framed', 'quad':q.tolist(), 'area_ratio':round(area,3),
            'confidence':confidence, 'engine':'apple_vision',
            'instruction':'문서 가장자리에 여백을 조금 더 주세요.' if clipped else '문서가 들어왔어요. 잠시 그대로 유지해 주세요.'}
