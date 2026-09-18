"""음성 입출력 구현 — :class:`docagent.interfaces.SpeechIO` 프로토콜.

이번 범위에서 실제 STT/TTS 연동은 하지 않는다. 콘솔·스크립트 구현으로
**에이전트 로직 전체**를 검증할 수 있게 하고, 클라우드 어댑터는 지연 import
골격만 둔다(설치되지 않은 상태에서 import·생성이 실패하지 않아야 한다).

구현 3종
--------
============================ ==================================================
:class:`ConsoleSpeechIO`     데모용. 발화를 콘솔에 태그와 함께 출력하고,
                             입력은 미리 정한 스크립트 또는 stdin 에서 읽는다.
:class:`ScriptedSpeechIO`    테스트용. 출력은 리스트에 모으고 입력은 스크립트에서만
                             읽는다. 콘솔에 아무것도 쓰지 않는다.
:class:`GoogleSpeechIO`      Google Cloud STT/TTS 어댑터. 패키지가 없으면
                             :class:`~docagent.errors.AdapterUnavailable`.
============================ ==================================================

스크립트가 소진되면 :meth:`listen` 은 빈 문자열을 돌려준다. 프로토콜이
"침묵·인식 실패 시 빈 문자열"로 규정하고 있으므로, 스크립트 소진도 같은
방식으로 다뤄 오케스트레이터가 별도 분기를 갖지 않게 한다.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Sequence, TextIO

from docagent.errors import AdapterUnavailable, ToolExecutionError

__all__ = [
    "AGENT_TAG",
    "USER_TAG",
    "HARDWARE_TAG",
    "VERIFY_TAG",
    "SYSTEM_TAG",
    "ConsoleSpeechIO",
    "ScriptedSpeechIO",
    "GoogleSpeechIO",
]


#: 콘솔 로그 태그(데모 출력 규약). 다른 모듈도 같은 태그를 쓴다.
AGENT_TAG: str = "[에이전트]"
USER_TAG: str = "[사용자]"
HARDWARE_TAG: str = "[하드웨어]"
VERIFY_TAG: str = "[검증]"
SYSTEM_TAG: str = "[시스템]"


class ScriptedSpeechIO:
    """발화를 모으고 미리 정한 대사를 돌려주는 테스트용 음성 입출력.

    :class:`docagent.interfaces.SpeechIO` 를 만족한다. 콘솔에 아무것도 쓰지
    않으므로 pytest 출력이 오염되지 않는다.

    :param script: :meth:`listen` 이 순서대로 돌려줄 사용자 발화 목록.
    :raises ValueError: ``script`` 에 문자열이 아닌 값이 있는 경우.
    """

    def __init__(self, script: Sequence[str] = ()) -> None:
        for index, line in enumerate(script):
            if not isinstance(line, str):
                raise ValueError(
                    f"script[{index}] 는 문자열이어야 합니다: {type(line).__name__}"
                )
        self._script: list[str] = list(script)
        self._cursor = 0
        #: 낭독한 문장 목록(감사·단언용).
        self.spoken: list[str] = []
        #: 실제로 돌려준 사용자 발화 목록.
        self.heard: list[str] = []

    @property
    def remaining(self) -> int:
        """아직 소비하지 않은 스크립트 줄 수."""
        return max(0, len(self._script) - self._cursor)

    @property
    def exhausted(self) -> bool:
        """스크립트를 모두 소비했으면 True."""
        return self.remaining == 0

    def speak(self, text: str) -> None:
        """문장을 기록한다.

        :param text: 낭독할 한국어 문장.
        :returns: ``None``.
        """
        self.spoken.append(str(text))

    def listen(self, timeout_s: float = 10.0) -> str:
        """스크립트의 다음 발화를 돌려준다.

        :param timeout_s: 사용하지 않는다(인터페이스 호환용).
        :returns: 다음 발화. 스크립트가 소진되었으면 빈 문자열.
        """
        del timeout_s
        if self._cursor >= len(self._script):
            self.heard.append("")
            return ""
        line = self._script[self._cursor]
        self._cursor += 1
        self.heard.append(line)
        return line

    def transcript(self) -> tuple[str, ...]:
        """낭독 문장 전체를 튜플로 돌려준다.

        :returns: 낭독 문장 튜플.
        """
        return tuple(self.spoken)

    def __repr__(self) -> str:
        """발화·잔여 스크립트 수만 노출한다(내용은 담지 않는다)."""
        return f"<ScriptedSpeechIO 낭독 {len(self.spoken)}건, 잔여 {self.remaining}줄>"


class ConsoleSpeechIO(ScriptedSpeechIO):
    """콘솔에 발화를 출력하고 stdin 또는 스크립트에서 입력을 받는 음성 입출력.

    데모(:mod:`docagent.demo`)의 기본 구현이다. ``script`` 를 주면 그 목록을
    순서대로 사용자 발화로 쓰고(비대화형·결정론), ``script=None`` 이면
    ``input_fn`` 으로 stdin 에서 읽는다.

    :param script: 사용자 발화 스크립트. ``None`` 이면 stdin 에서 읽는다.
    :param stream: 출력 스트림. ``None`` 이면 :data:`sys.stdout`.
    :param input_fn: 입력 함수. ``None`` 이면 내장 :func:`input`.
        ``script`` 가 주어지면 호출되지 않는다.
    :param echo_user: 스크립트 발화도 콘솔에 ``[사용자]`` 로 찍을지 여부.
    :param wrap_width: 발화 줄바꿈 폭(글자 수). 0 이면 줄바꿈하지 않는다.
    :raises ValueError: ``wrap_width`` 가 음수이거나 ``script`` 형식이 잘못된 경우.
    """

    def __init__(
        self,
        script: Sequence[str] | None = None,
        *,
        stream: TextIO | None = None,
        input_fn: Callable[[], str] | None = None,
        echo_user: bool = True,
        wrap_width: int = 92,
    ) -> None:
        super().__init__(script or ())
        if wrap_width < 0:
            raise ValueError(f"wrap_width 는 0 이상이어야 합니다: {wrap_width}")
        self._interactive = script is None
        self._stream = stream
        self._input_fn = input_fn
        self._echo_user = echo_user
        self._wrap_width = wrap_width

    @property
    def stream(self) -> TextIO:
        """출력 스트림. 생성 시 ``None`` 이었으면 현재 :data:`sys.stdout`."""
        return self._stream if self._stream is not None else sys.stdout

    def write(self, tag: str, text: str) -> None:
        """태그를 붙여 한 문단을 출력한다.

        :param tag: ``[에이전트]`` 같은 태그.
        :param text: 본문.
        :returns: ``None``.
        """
        body = str(text).strip()
        if not body:
            return
        lines = _wrap(body, self._wrap_width)
        pad = " " * (len(tag) + 1)
        print(f"{tag} {lines[0]}", file=self.stream)
        for line in lines[1:]:
            print(f"{pad}{line}", file=self.stream)

    def speak(self, text: str) -> None:
        """문장을 콘솔에 출력하고 기록한다.

        :param text: 낭독할 한국어 문장.
        :returns: ``None``.
        """
        super().speak(text)
        self.write(AGENT_TAG, text)

    def listen(self, timeout_s: float = 10.0) -> str:
        """스크립트 또는 stdin 에서 사용자 발화를 읽는다.

        :param timeout_s: 사용하지 않는다(인터페이스 호환용).
        :returns: 사용자 발화. 스크립트 소진 또는 입력 종료 시 빈 문자열.
        """
        if not self._interactive:
            line = super().listen(timeout_s)
            if line and self._echo_user:
                self.write(USER_TAG, line)
            return line
        reader = self._input_fn if self._input_fn is not None else input
        try:
            line = reader()
        except EOFError:
            line = ""
        except KeyboardInterrupt:  # 사용자가 직접 끊은 것은 정상 종료로 다룬다.
            line = ""
        line = str(line).strip()
        self.heard.append(line)
        return line

    def __repr__(self) -> str:
        """대화형 여부와 발화 수만 노출한다."""
        mode = "대화형" if self._interactive else "스크립트"
        return f"<ConsoleSpeechIO {mode}, 낭독 {len(self.spoken)}건>"


class GoogleSpeechIO:
    """Google Cloud STT/TTS 어댑터(선택적 패키지, 이번 범위에서는 미실행).

    ``google-cloud-speech`` · ``google-cloud-texttospeech`` 는 설치되어 있지
    않다. 이 클래스는 **생성 시점에** 지연 import 를 시도하고, 실패하면
    :class:`~docagent.errors.AdapterUnavailable` 을 던진다. 모듈을 import 하는
    것만으로는 절대 실패하지 않는다.

    :param language_code: 인식·합성 언어(기본 한국어).
    :param voice_name: TTS 음성 이름. ``None`` 이면 기본 음성.
    :param sample_rate_hz: 인식 입력 샘플레이트.
    :param audio_source: 마이크 입력을 바이트로 돌려주는 호출 가능 객체.
        ``None`` 이면 :meth:`listen` 이 :class:`~docagent.errors.ToolExecutionError`
        를 던진다(조용히 빈 문자열을 돌려주지 않는다).
    :param audio_sink: 합성된 오디오 바이트를 재생하는 호출 가능 객체.
        ``None`` 이면 재생하지 않고 :attr:`last_audio` 에만 담아 둔다.
    :raises docagent.errors.AdapterUnavailable: 필요한 패키지가 없는 경우.
    """

    #: 지연 import 대상 패키지 이름(안내 문구용).
    PACKAGES: tuple[str, ...] = ("google-cloud-speech", "google-cloud-texttospeech")

    def __init__(
        self,
        *,
        language_code: str = "ko-KR",
        voice_name: str | None = None,
        sample_rate_hz: int = 16000,
        audio_source: Callable[[float], bytes] | None = None,
        audio_sink: Callable[[bytes], None] | None = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz 는 0 보다 커야 합니다: {sample_rate_hz}")
        self.language_code = language_code
        self.voice_name = voice_name
        self.sample_rate_hz = sample_rate_hz
        self._audio_source = audio_source
        self._audio_sink = audio_sink
        #: 마지막으로 합성한 오디오 바이트(재생기가 없을 때 확인용).
        self.last_audio: bytes = b""
        speech_module, tts_module = self._import_google()
        self._speech = speech_module
        self._tts = tts_module
        self._stt_client = speech_module.SpeechClient()
        self._tts_client = tts_module.TextToSpeechClient()

    @staticmethod
    def _import_google() -> tuple[Any, Any]:
        """Google Cloud 음성 패키지를 지연 import 한다.

        :returns: ``(speech 모듈, texttospeech 모듈)``.
        :raises docagent.errors.AdapterUnavailable: 패키지가 없는 경우.
        """
        try:
            from google.cloud import speech as speech_module  # type: ignore[import-not-found]
            from google.cloud import (  # type: ignore[import-not-found]
                texttospeech as tts_module,
            )
        except ImportError as exc:
            raise AdapterUnavailable(
                package="google-cloud-speech google-cloud-texttospeech",
                feature="Google Cloud 기반 음성 인식·합성",
            ) from exc
        return (speech_module, tts_module)

    def speak(self, text: str) -> None:
        """문장을 합성해 재생한다.

        :param text: 낭독할 한국어 문장.
        :returns: ``None``.
        :raises docagent.errors.ToolExecutionError: 합성 요청이 실패한 경우.
        """
        body = str(text).strip()
        if not body:
            return
        voice_kwargs: dict[str, Any] = {"language_code": self.language_code}
        if self.voice_name:
            voice_kwargs["name"] = self.voice_name
        try:
            response = self._tts_client.synthesize_speech(
                input=self._tts.SynthesisInput(text=body),
                voice=self._tts.VoiceSelectionParams(**voice_kwargs),
                audio_config=self._tts.AudioConfig(
                    audio_encoding=self._tts.AudioEncoding.LINEAR16
                ),
            )
        except Exception as exc:  # noqa: BLE001 — 도메인 예외로 감싸 올린다.
            raise ToolExecutionError(
                f"음성 합성에 실패했습니다: {exc}", tool_name="GoogleSpeechIO.speak"
            ) from exc
        self.last_audio = bytes(response.audio_content)
        if self._audio_sink is not None:
            self._audio_sink(self.last_audio)

    def listen(self, timeout_s: float = 10.0) -> str:
        """마이크 입력을 받아 한국어 문자열로 인식한다.

        :param timeout_s: 최대 녹음 시간(초).
        :returns: 인식된 발화. 인식 결과가 없으면 빈 문자열.
        :raises docagent.errors.ToolExecutionError: 오디오 공급자가 없거나
            인식 요청이 실패한 경우.
        """
        if self._audio_source is None:
            raise ToolExecutionError(
                "마이크 입력 공급자(audio_source)가 주입되지 않아 음성을 받을 수 없습니다.",
                tool_name="GoogleSpeechIO.listen",
            )
        audio_bytes = self._audio_source(float(timeout_s))
        try:
            response = self._stt_client.recognize(
                config=self._speech.RecognitionConfig(
                    encoding=self._speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate_hz,
                    language_code=self.language_code,
                ),
                audio=self._speech.RecognitionAudio(content=audio_bytes),
            )
        except Exception as exc:  # noqa: BLE001 — 도메인 예외로 감싸 올린다.
            raise ToolExecutionError(
                f"음성 인식에 실패했습니다: {exc}", tool_name="GoogleSpeechIO.listen"
            ) from exc
        for result in getattr(response, "results", ()):
            alternatives = getattr(result, "alternatives", ())
            if alternatives:
                return str(alternatives[0].transcript).strip()
        return ""


def _wrap(text: str, width: int) -> list[str]:
    """공백 단위 그리디 줄바꿈(단어를 쪼개지 않는다).

    :param text: 원문 한 줄.
    :param width: 최대 글자 수. 0 이면 줄바꿈하지 않는다.
    :returns: 줄 목록(최소 1줄).
    """
    if width <= 0:
        return [text]
    lines: list[str] = []
    current = ""
    for token in text.split(" "):
        candidate = f"{current} {token}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]
