"""Fail-closed авторизация статических agent tools."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from mtbank_ai.agent_runtime.contracts import AgentFailureCode, AgentRuntimeError, AgentSpec, ToolSideEffect
from mtbank_ai.agent_runtime.tools import ValidatedToolCall


class ToolAuthorizationError(AgentRuntimeError):
    """Tool call не соответствует immutable AgentSpec."""


class ToolAuthorizer:
    """Разрешает только объявленные read-only tools и один terminal submit."""

    def authorize(
        self,
        *,
        spec: AgentSpec,
        calls: Sequence[ValidatedToolCall],
        completed_retrieval_tools: Collection[str],
        terminal_submitted: bool,
    ) -> int | None:
        if terminal_submitted:
            raise ToolAuthorizationError(AgentFailureCode.POST_TERMINAL_TOOL_CALL)

        terminal_positions = [index for index, call in enumerate(calls) if call.spec.name == spec.terminal_submit_tool]
        if len(terminal_positions) > 1:
            raise ToolAuthorizationError(AgentFailureCode.POST_TERMINAL_TOOL_CALL)
        terminal_index = terminal_positions[0] if terminal_positions else None
        if terminal_index is not None and terminal_index != len(calls) - 1:
            raise ToolAuthorizationError(AgentFailureCode.POST_TERMINAL_TOOL_CALL)

        seen_retrieval = set(completed_retrieval_tools)
        for index, call in enumerate(calls):
            is_terminal_name = call.spec.name == spec.terminal_submit_tool
            if is_terminal_name:
                if call.spec.side_effect is not ToolSideEffect.TERMINAL_SUBMIT:
                    raise ToolAuthorizationError(AgentFailureCode.TERMINAL_SUBMIT_INVALID)
                if index != terminal_index:
                    raise ToolAuthorizationError(AgentFailureCode.POST_TERMINAL_TOOL_CALL)
                if not set(spec.required_retrieval_tools).issubset(seen_retrieval):
                    raise ToolAuthorizationError(AgentFailureCode.REQUIRED_RETRIEVAL_MISSING)
                continue

            if call.spec.side_effect is not ToolSideEffect.READ_ONLY or call.spec.name not in spec.allowed_read_tools:
                raise ToolAuthorizationError(AgentFailureCode.TOOL_NOT_ALLOWED)
            if call.spec.name in spec.required_retrieval_tools:
                seen_retrieval.add(call.spec.name)

        return terminal_index
