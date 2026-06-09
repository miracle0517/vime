import importlib.util

spec = importlib.util.spec_from_file_location("vllm_tool_parser", "examples/tau-bench/vllm_tool_parser.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
parse_tools = mod.parse_tools

cases = [
    '<tool_call>\n{"name": "find_user_id_by_email", "arguments": {"email": "a@b.com"}}\n</tool_call>',
    '{"name": "find_user_id_by_email", "arguments": {"email": "a@b.com"}}<|im_end|>',
    '{"name": "get_user_details", "arguments": {"user_id": "u1"}}<|im_end|>',
]
for case in cases:
    result = parse_tools(case, [])
    assert result["calls"], f"failed: {case[:80]}"
    print("OK", result["calls"][0]["name"])
