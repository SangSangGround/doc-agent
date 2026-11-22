"""짧은 마이크 질문을 설치된 Whisper 모델로 로컬에서 인식한다."""
from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path

import numpy as np

from docagent.home.questions import QuestionUnavailable

DEFAULT_SPEECH_MODEL = Path("outputs/models/whisper-small")
MAX_AUDIO_BYTES = 4 * 1024 * 1024
MAX_SECONDS = 35


class LocalSpeech:
    def __init__(self, model_path: Path = DEFAULT_SPEECH_MODEL) -> None:
        self.model_path = Path(model_path)
        self._model = None

    def transcribe(self, encoded: str) -> str:
        """인식 결과는 사용자가 듣고 수정할 수 있도록 질문 입력란으로 반환한다."""
        if not isinstance(encoded, str) or len(encoded) > MAX_AUDIO_BYTES * 4 // 3 + 512:
            raise ValueError("음성 파일이 너무 큽니다. 30초 이내로 질문해 주세요.")
        if encoded.startswith("data:audio/"):
            encoded = encoded.split(",", 1)[-1]
        try:
            blob = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ValueError("음성을 읽을 수 없습니다. 다시 녹음해 주세요.") from exc
        if not blob or len(blob) > MAX_AUDIO_BYTES:
            raise ValueError("음성 파일 크기를 확인하세요.")
        try:
            import av
            import onnxruntime
            from faster_whisper import WhisperModel
            onnxruntime.disable_telemetry_events()
        except ImportError as exc:
            raise QuestionUnavailable("음성 인식 모듈을 설치해 주세요. 키보드로도 질문할 수 있습니다.") from exc
        if not self.model_path.joinpath("model.bin").is_file():
            raise QuestionUnavailable("Whisper 음성 모델이 설치되지 않았습니다. 키보드로 질문할 수 있습니다.")
        try:
            samples = []
            total = 0
            # 디코딩하는 동안에도 길이를 제한해 긴 압축 오디오를 통째로 메모리에 올리지 않는다.
            with av.open(io.BytesIO(blob)) as container:
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
                for frame in container.decode(audio=0):
                    for converted in resampler.resample(frame):
                        chunk = converted.to_ndarray().ravel()
                        total += len(chunk)
                        if total > MAX_SECONDS * 16000:
                            raise ValueError("음성이 너무 깁니다. 30초 이내로 다시 질문해 주세요.")
                        samples.append(chunk)
                for converted in resampler.resample(None):
                    chunk = converted.to_ndarray().ravel()
                    total += len(chunk)
                    if total > MAX_SECONDS * 16000:
                        raise ValueError("음성이 너무 깁니다.")
                    samples.append(chunk)
            if not samples or total < 1600:
                raise ValueError("녹음된 음성이 너무 짧습니다. 다시 말씀해 주세요.")
            if self._model is None:
                self._model = WhisperModel(str(self.model_path), device="cpu", compute_type="int8",
                                          cpu_threads=4, local_files_only=True)
            segments, _ = self._model.transcribe(
                np.concatenate(samples), language="ko", beam_size=3, vad_filter=True,
                condition_on_previous_text=False, hotwords="서명란, 신청인, 체크박스, 동의, 수수료",
            )
            text = " ".join(segment.text.strip() for segment in segments
                            if segment.no_speech_prob < .6 and segment.avg_logprob > -1.).strip()
        except ValueError:
            raise
        except Exception as exc:
            raise QuestionUnavailable("녹음된 음성을 인식하지 못했습니다. 다시 말하거나 키보드로 입력해 주세요.") from exc
        if not text:
            raise QuestionUnavailable("질문을 듣지 못했습니다. 마이크 가까이에서 다시 말씀해 주세요.")
        if len(text) > 500:
            raise QuestionUnavailable("질문이 너무 깁니다. 짧게 나눠 말씀해 주세요.")
        return text
