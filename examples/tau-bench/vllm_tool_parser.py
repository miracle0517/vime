"""Local tool-call parser for vLLM rollout (vime counterpart of slime's sglang_tool_parser)."""

import json
import re
from typing import Any

_SPECIAL_TOKENS_RE = re.compile(r"<\|[^>|]*\|>")


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen25") -> dict[str, Any]:
    if parser == "qwen25":
        return _parse_qwen25_tools(response)
    return _parse_qwen25_tools(response)


def _strip_special_tokens(text: str) -> str:
    return _SPECIAL_TOKENS_RE.sub("", text).strip()


def _try_parse_json_tool_call(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "name" in parsed:
            name = parsed.get("name", "")
            parameters = parsed.get("arguments", parsed.get("parameters", {}))
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    pass
            return {"name": name, "parameters": parameters}
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _extract_json_objects(text: str) -> list[str]:
    """Extract complete JSON object substrings via brace matching."""
    objects: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        start = i
        in_string = False
        escape = False
        for j in range(i, len(text)):
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start : j + 1])
                    i = j + 1
                    break
        else:
            i += 1
    return objects


def _parse_tool_calls_from_json_blobs(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for blob in _extract_json_objects(text):
        parsed_call = _try_parse_json_tool_call(blob)
        if parsed_call:
            calls.append(parsed_call)
    return calls


def _parse_tool_call_payload(match: str) -> dict[str, Any] | None:
    match = match.strip()
    parsed_call = _try_parse_json_tool_call(match)
    if parsed_call:
        return parsed_call

    for blob in _extract_json_objects(match):
        parsed_call = _try_parse_json_tool_call(blob)
        if parsed_call:
            return parsed_call

    json_match = re.search(r"\{.*\}", match, re.DOTALL)
    if json_match:
        parsed_call = _try_parse_json_tool_call(json_match.group())
        if parsed_call:
            return parsed_call
    return None


def _parse_qwen25_tools(response: str) -> dict[str, Any]:
    call_open = chr(60) + "tool_call" + chr(62)
    call_close = chr(60) + "/tool_call" + chr(62)
    call_open_alt = chr(60) + "call" + chr(62)
    call_close_alt = chr(60) + "/call" + chr(62)
    pattern = r"(?:" + call_open + "|" + call_open_alt + r")\s*(.*?)\s*(?:" + call_close + "|" + call_close_alt + ")"
    tool_call_pattern = re.compile(pattern, re.DOTALL)
    matches = tool_call_pattern.findall(response)

    if matches:
        parts = tool_call_pattern.split(response)
        normal_text = " ".join(parts[i].strip() for i in range(0, len(parts), 2) if parts[i].strip())
        calls = []
        for match in matches:
            parsed_call = _parse_tool_call_payload(match)
            if parsed_call:
                calls.append(parsed_call)
            else:
                calls.append({"name": match.strip(), "parameters": {}})
        return {
            "normal_text": normal_text,
            "calls": calls,
        }

    cleaned = _strip_special_tokens(response)
    parsed_call = _try_parse_json_tool_call(cleaned)
    if parsed_call:
        return {
            "normal_text": "",
            "calls": [parsed_call],
        }

    calls = _parse_tool_calls_from_json_blobs(cleaned)
    if calls:
        normal_text = cleaned
        for call in calls:
            for blob in _extract_json_objects(cleaned):
                if call["name"] in blob:
                    normal_text = normal_text.replace(blob, "")
        normal_text = _strip_special_tokens(normal_text)
        return {
            "normal_text": normal_text,
            "calls": calls,
        }

    return {
        "normal_text": cleaned,
        "calls": [],
    }
