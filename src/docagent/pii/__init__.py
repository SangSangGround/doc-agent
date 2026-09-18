"""PII 모듈 — 개인정보 탐지·마스킹·유출 차단 게이트.

외부(LLM·네트워크·영구 로그)로 나가는 모든 텍스트가 통과해야 하는 관문이다.
탐지 결과는 :class:`docagent.contracts.PiiSpan` 으로 표현하며 원문 값을 담지 않는다.

순환 import 를 막기 위해 이 ``__init__`` 은 재export 를 두지 않는다.
"""
