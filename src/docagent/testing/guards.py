"""테스트 전용 가드 구현.

운영 경로가 실수로 "아무것도 막지 않는 가드"를 기본값으로 삼는 일을 막기 위해,
통과 전용 가드를 :mod:`docagent.agent.orchestrator` 에서 분리해 여기에 둔다.
:class:`~docagent.agent.orchestrator.DocumentAgent` 의 기본 가드는
:class:`~docagent.agent.guardrails.Guard` (개인정보 게이트를 반드시 확보하는
fail-closed 구현)이며, 검사를 끄고 싶은 단위 테스트만 이 모듈을 **명시적으로**
주입한다.
"""

from __future__ import annotations

from docagent.contracts import Field

__all__ = ["PassThroughGuard"]


class PassThroughGuard:
    """아무것도 막지 않는 가드(**테스트 전용**).

    차단 규칙과 무관한 대화 흐름만 시험할 때 쓴다. 운영 조립 경로
    (:func:`docagent.pipeline.build_session`)는 이 구현을 쓰지 않는다.
    """

    def check(self, user_text: str, field: Field | None = None) -> None:
        """항상 통과한다.

        :param user_text: 사용자 발화.
        :param field: 현재 항목(없을 수 있다).
        :returns: ``None``.
        """
        return None
