"""Vision 모듈 — See / Understand / Verify 단계.

문서 이미지 정합, 기입란 탐지, OCR 연동, 구조화(fields JSON) 산출,
기입 전후 잉크 비율 비교 검증을 담당한다.

순환 import 를 막기 위해 이 ``__init__`` 은 재export 를 두지 않는다.
하위 모듈에서 직접 import 하라 (예: ``from docagent.vision.detector import ...``).
"""
