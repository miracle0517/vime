"""Model-output parsing helpers for agent harnesses."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ParsedModelOutput:
    """Structured view of one decoded model output."""

    reasoning: str
    text: str
    tool_uses: list[dict[str, Any]]


def _empty_chat_request(tools_schema: list[dict] | None = None):
    """Build a minimal vLLM ChatCompletionRequest for the non-streaming parsers.

    vLLM's reasoning / tool-call parsers take the originating request; for
    post-hoc parsing of an already-decoded string only ``tools`` is consulted.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    kwargs: dict[str, Any] = {"messages": []}
    if tools_schema:
        kwargs["tools"] = tools_schema
    return ChatCompletionRequest(**kwargs)


def parse_model_output(
    raw_output: str,
    *,
    tokenizer,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
    reasoning_parser_name: str | None,
) -> ParsedModelOutput:
    """Parse raw model text into reasoning, visible text, and tool uses.

    When ``reasoning_parser_name`` / ``tool_parser_name`` are set the heavy
    format-specific work is delegated to vLLM's reasoning and tool-call parsers
    (``vllm.reasoning`` / ``vllm.tool_parsers``, imported lazily so they are
    only pulled in when explicitly enabled). The coding-agent example leaves
    both unset and relies on the XML fallback, which covers the Anthropic-style
    tool-call text that some coding-agent models still emit occasionally.
    """
    reasoning, body_text = "", raw_output
    if reasoning_parser_name:
        reasoning, body_text = _extract_reasoning(raw_output, tokenizer, reasoning_parser_name)

    body_text, tool_uses = parse_tool_uses(body_text, tools_schema, tool_parser_name, tokenizer)
    return ParsedModelOutput(
        reasoning=reasoning,
        text=(body_text or "").strip(),
        tool_uses=tool_uses,
    )


def _extract_reasoning(raw_output: str, tokenizer, reasoning_parser_name: str) -> tuple[str, str]:
    """Split reasoning from visible text via vLLM's reasoning parser, with a
    ``</think>`` string-split fallback if the parser is unavailable."""
    reasoning, body_text = "", raw_output
    try:
        from vllm.reasoning import ReasoningParserManager

        parser = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)(tokenizer)
        r, b = parser.extract_reasoning(raw_output, _empty_chat_request())
        reasoning = r or ""
        body_text = b if b is not None else raw_output
    except Exception:
        logger.exception(
            "[agent.parsing] vLLM reasoning parsing failed; falling back to </think> split"
        )
        reasoning, body_text = "", raw_output
    if not reasoning and "</think>" in body_text:
        reasoning, body_text = body_text.split("</think>", 1)
    return reasoning, body_text


def parse_tool_uses(
    body_text: str,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
    tokenizer=None,
) -> tuple[str, list[dict[str, Any]]]:
    """Parse tool calls from body text and return visible text plus tool uses."""
    tool_uses: list[dict[str, Any]] = []
    if tool_parser_name and tools_schema:
        try:
            from vllm.tool_parsers import ToolParserManager

            parser = ToolParserManager.get_tool_parser(tool_parser_name)(tokenizer)
            info = parser.extract_tool_calls(body_text, _empty_chat_request(tools_schema))
            if info.tools_called:
                # vLLM returns the text with tool-call markup stripped; ``None``
                # means the whole output was the tool call (no leftover text).
                body_text = info.content or ""
                for call in info.tool_calls:
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw_arguments": call.function.arguments}
                    tool_uses.append({"name": call.function.name or "tool", "input": args})
        except Exception:
            logger.exception("[agent.parsing] vLLM tool-call parsing failed; falling back")

    if not tool_uses and tools_schema:
        body_text, tool_uses = parse_xml_tool_uses(body_text, tools_schema)

    return body_text, tool_uses


def parse_xml_tool_uses(body_text: str, tools_schema: list[dict]) -> tuple[str, list[dict[str, Any]]]:
    """Fallback parser for Anthropic-style XML tool calls."""
    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    tool_uses: list[dict[str, Any]] = []
    cleaned_parts: list[str] = []
    last = 0
    for m in re.finditer(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        body_text,
        flags=re.DOTALL,
    ):
        name, inner = m.group(1), m.group(2)
        if name in valid_tools:
            args = {
                p.group(1): p.group(2).strip()
                for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, flags=re.DOTALL)
            }
            tool_uses.append({"name": name, "input": args})
            cleaned_parts.append(body_text[last : m.start()])
            last = m.end()
    cleaned_parts.append(body_text[last:])
    return "".join(cleaned_parts), tool_uses
