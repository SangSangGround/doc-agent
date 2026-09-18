"""테스트 지원 모듈 — Mock 구현과 결정론적 픽스처.

음성(STT/TTS)·하드웨어(Arduino)·관리자 대시보드는 이번 범위에서
:mod:`docagent.interfaces` 의 Protocol 과 Mock 구현까지만 둔다.
난수를 쓰는 픽스처는 반드시 seed 를 고정한다.

순환 import 를 막기 위해 이 ``__init__`` 은 재export 를 두지 않는다.
"""
