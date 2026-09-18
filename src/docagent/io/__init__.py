"""IO 모듈 — 이미지 로딩, fields JSON 영속화, 세션 저장·복원.

파일 입출력은 항상 :mod:`pathlib` 경로와 ``encoding="utf-8"`` 을 사용한다.

순환 import 를 막기 위해 이 ``__init__`` 은 재export 를 두지 않는다.
"""
