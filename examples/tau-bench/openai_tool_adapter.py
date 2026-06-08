import logging
from dataclasses import dataclass, field
from typing import Any

from vllm_tool_parser import parse_tools

logger = logging.getLogger(__name__)


@dataclass
class OpenAIToolCall:
    """OpenAI format tool call structure."""

    id: str
    type: str = "function"
    function: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenAIAssistantMessage:
    """OpenAI format assistant message structure."""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAICompatibleToolCallAdapter:
    """
    Adapter that converts vLLM tool-call parsing results to OpenAI compatible format.

    Wired through ``vllm_tool_parser.parse_tools``.
    """

    def __init__(self, tools_info: list[dict[str, Any]], parser_type: str = "qwen25"):
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse_response_to_openai_format(self, response: str) -> dict[str, Any]:
        """
        Parse a vLLM rollout response to OpenAI compatible format.

        Args:
            response: Raw response text from the vLLM generate endpoint.

        Returns:
            Dictionary containing OpenAI format message and parsing results.
        """
        try:
            parsed = parse_tools(response, self.tools_info, self.parser_type)
            normal_text = parsed["normal_text"]
            calls = parsed["calls"]
            openai_message = self._convert_to_openai_message(normal_text, calls)
            return {"openai_message": openai_message, "parsed_result": parsed, "success": True}
        except Exception as e:
            logger.warning(f"Parsing failed with error: {e}")
            return {"openai_message": None, "parsed_result": None, "success": False, "error": str(e)}

    def _convert_to_openai_message(self, normal_text: str, calls: list[dict[str, Any]]) -> OpenAIAssistantMessage:
        if not calls:
            return OpenAIAssistantMessage(role="assistant", content=normal_text, tool_calls=None)

        openai_tool_calls = []
        for i, call in enumerate(calls):
            openai_tool_calls.append(
                OpenAIToolCall(
                    id=f"call_{i}_{call.get('name', 'unknown')}",
                    type="function",
                    function={"name": call.get("name", ""), "arguments": call.get("parameters", "{}")},
                )
            )

        return OpenAIAssistantMessage(
            role="assistant",
            content=normal_text if normal_text.strip() else None,
            tool_calls=openai_tool_calls,
        )


def create_openai_adapter(
    tools_info: list[dict[str, Any]], parser_type: str = "qwen25"
) -> OpenAICompatibleToolCallAdapter:
    """Factory function to create an OpenAI compatible tool-call adapter."""
    return OpenAICompatibleToolCallAdapter(tools_info, parser_type)
