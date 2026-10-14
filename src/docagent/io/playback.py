"""승인된 고정 문구의 로컬 재생. 생성형 설명과 클라우드 TTS를 호출하지 않는다."""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from docagent.errors import AdapterUnavailable, ToolExecutionError
from docagent.guidance.rules import GuidanceResult, Observation, RuleCatalog


def _has_audio_frames(path: Path) -> bool:
    """say may return 0 with an empty AIFF when the OS speech service is unavailable."""
    try:
        length = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b"FORM" or header[8:] not in (b"AIFF", b"AIFC"):
                return False
            frames = False
            samples = False
            while stream.tell() + 8 <= length:
                kind, size = struct.unpack(">4sI", stream.read(8))
                start = stream.tell()
                if start + size > length:
                    return False
                if kind == b"COMM" and size >= 6:
                    channels, count = struct.unpack(">HI", stream.read(6))
                    frames = channels > 0 and count > 0
                if kind == b"SSND" and size > 8:
                    offset, _ = struct.unpack(">II", stream.read(8))
                    samples = size > 8 + offset
                stream.seek(start + size + (size % 2))
            return frames and samples
    except (OSError, ValueError, struct.error):
        return False


class PlaybackBackend(Protocol):
    @property
    def state(self) -> str: ...
    def play(self, text: str) -> None: ...
    def stop(self) -> None: ...
    def wait(self) -> None: ...


class SayBackend:
    """macOS에 이미 설치된 음성으로 비동기 재생/AIFF 저장. 추가 패키지 없음."""
    def __init__(self, *, voice: str = "Yuna", output_file: str | Path | None = None):
        if sys.platform != "darwin" or not Path('/usr/bin/say').is_file():
            raise AdapterUnavailable("로컬 TTS는 현재 macOS의 say를 지원합니다.")
        if not isinstance(voice,str) or not voice.strip():
            raise ValueError("음성 이름이 필요합니다.")
        self.voice = voice
        self.output_file = Path(output_file) if output_file is not None else None
        if self.output_file is not None and self.output_file.suffix.lower() != '.aiff':
            raise ValueError("로컬 오디오 출력 확장자는 .aiff여야 합니다.")
        self._process = None
        self._state = "idle"

    @property
    def state(self) -> str:
        if self._process is not None:
            code = self._process.poll()
            if code is not None:
                valid_output = self.output_file is None or _has_audio_frames(self.output_file)
                self._state = "completed" if code == 0 and valid_output else "failed"
                self._process = None
        return self._state

    def play(self, text: str) -> None:
        self.stop()
        if not isinstance(text,str) or not text.strip():
            raise ValueError("읽을 문장이 없습니다.")
        command = ['/usr/bin/say','-v',self.voice]
        if self.output_file is not None:
            self.output_file.parent.mkdir(parents=True,exist_ok=True)
            command += ['-o',str(self.output_file)]
        try:
            self._process = subprocess.Popen(command,stdin=subprocess.PIPE,
                                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            # stdin prevents a leading '-' or shell syntax in approved text becoming options.
            self._process.stdin.write(text.encode('utf-8'))
            self._process.stdin.close()
            self._state = "playing"
        except (OSError,ValueError):
            self.stop()
            self._state = "failed"
            raise ToolExecutionError("음성을 시작하지 못했습니다. 설치된 음성을 확인하세요.") from None

    def stop(self) -> None:
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            self._process = None
            self._state = "stopped"

    def wait(self) -> None:
        if self._process is not None:
            self._process.wait()
        if self.state == "failed":
            raise ToolExecutionError("음성 재생에 실패했습니다. 음성 설치와 출력 장치를 확인하세요.")


class GuidancePlayer:
    """검색 결과를 내부에서 만들고 검증된 문구 또는 고정 실패 문구만 재생한다."""
    def __init__(self, catalog: RuleCatalog, backend: PlaybackBackend):
        self.catalog = catalog
        self.backend = backend
        self._last: GuidanceResult | None = None

    def play(self, observation: Observation, *, demo: bool = False) -> GuidanceResult:
        self.backend.stop()
        self._last = None
        result = self.catalog.match(observation,demo=demo)
        self.backend.play(result.instruction)
        self._last = result
        return result

    def replay(self) -> GuidanceResult:
        if self._last is None:
            raise ToolExecutionError("다시 들을 안내가 없습니다.")
        self.backend.play(self._last.instruction)
        return self._last

    def stop(self) -> None:
        self.backend.stop()

    def wait(self) -> None:
        self.backend.wait()
