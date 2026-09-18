"""Agent 모듈 — Explain / Ask / Act 단계의 대화·도구 오케스트레이션.

신뢰도 임계값(:data:`docagent.contracts.EXPLAIN_THRESHOLD`,
:data:`docagent.contracts.PARTIAL_THRESHOLD`)에 따라 설명·재확인·사람 지원으로 분기한다.

순환 import 를 막기 위해 이 ``__init__`` 은 재export 를 두지 않는다.
"""
