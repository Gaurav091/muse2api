from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from muse2api.core.prompt import flatten_messages
from muse2api.core.tools import (
    _clean_git_command,
    extract_tool_calls,
    format_tools_prompt,
    get_workspace_grounding,
    needs_tool_nudge,
    sanitize_response_text,
    synthesize_tool_response,
)


def test_format_tools_prompt():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to run"},
                    },
                    "required": ["command"],
                },
            },
        },
    ]
    prompt = format_tools_prompt(tools)
    assert "terminal" in prompt
    assert "Execute a shell command" in prompt
    assert "command (string (required))" in prompt
    assert "```action" in prompt


def test_extract_tool_calls_xml():
    text = (
        "I will check the git history.\n"
        "<tool_call>\n"
        '{"name": "terminal", "arguments": {"command": "git log -5"}}\n'
        "</tool_call>"
    )
    calls, remaining = extract_tool_calls(text, ["terminal"])
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["command"] == "git log -5"
    assert remaining == "I will check the git history."


def test_extract_tool_calls_json_block():
    text = (
        "Let me run that command for you:\n"
        "```json\n"
        "{\n"
        '  "tool_calls": [\n'
        '    {"name": "terminal", "arguments": {"command": "git status"}}\n'
        "  ]\n"
        "}\n"
        "```"
    )
    calls, remaining = extract_tool_calls(text, ["terminal"])
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["command"] == "git status"
    assert remaining == "Let me run that command for you:"


def test_extract_tool_calls_multiple_xml_blocks():
    text = (
        "<tool_call>"
        '{"name": "terminal", "arguments": {"command": "git status"}}'
        "</tool_call>\n"
        "prefix text\n"
        "<tool_call>"
        '{"name": "read_file", "arguments": {"path": "README.md"}}'
        "</tool_call>"
    )
    calls, remaining = extract_tool_calls(text, ["terminal", "read_file"])
    assert calls is not None
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "terminal"
    assert calls[1]["function"]["name"] == "read_file"
    assert remaining == "prefix text"


def test_extract_tool_calls_unclosed_xml_ignored():
    """A trailing unclosed <tool_call> must not shadow later valid blocks."""
    text = (
        "<tool_call>"
        '{"name": "terminal", "arguments": {"command": "git log"}}'
        "</tool_call>\n"
        "<tool_call>unfinished"
    )
    calls, _ = extract_tool_calls(text, ["terminal"])
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"


def test_extract_tool_calls_xml_bad_json_skipped():
    """Blocks with invalid JSON are ignored; valid blocks still return."""
    text = (
        "<tool_call>not-json</tool_call>\n"
        "<tool_call>"
        '{"name": "terminal", "arguments": {"command": "pwd"}}'
        "</tool_call>"
    )
    calls, remaining = extract_tool_calls(text, ["terminal"])
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"
    # Matched <tool_call> blocks are removed wholesale, valid or not.
    assert remaining is None


def test_extract_tool_calls_xml_whitespace_padding():
    text = (
        "<tool_call>\n\n"
        '{"name": "terminal", "arguments": {"command": "ls"}}'
        "\n\n</tool_call>"
    )
    calls, _ = extract_tool_calls(text, ["terminal"])
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"


def test_extract_tool_calls_multiple_json_blocks():
    text = (
        "```json\n"
        '{"tool_calls": [{"name": "terminal", "arguments": {"command": "git status"}}]}\n'
        "```\n"
        "between\n"
        "```json\n"
        '{"tool_calls": [{"name": "read_file", "arguments": {"path": "README.md"}}]}\n'
        "```"
    )
    calls, remaining = extract_tool_calls(text, ["terminal", "read_file"])
    assert calls is not None
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "terminal"
    assert calls[1]["function"]["name"] == "read_file"
    assert remaining == "between"


def test_extract_tool_calls_unclosed_json_fence_ignored():
    text = (
        "```json\n"
        '{"tool_calls": [{"name": "terminal", "arguments": {"command": "pwd"}}]}\n'
        "```\n"
        "```json\nunfinished"
    )
    calls, _ = extract_tool_calls(text, ["terminal"])
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"


def test_sanitize_response_text_long_line():
    raw = (
        "This is the web side chat, working dir /home/hatch — "
        "/Volumes doesn't exist at all here. "
        + "x" * 20000
        + "\nStill useful content after the disclaimer."
    )
    cleaned = sanitize_response_text(raw)
    assert "web side chat" not in cleaned
    assert "Still useful content" in cleaned


def test_sanitize_response_text_deterministic_removals():
    cases = [
        (
            "Prefix. I have no record of a muse2api repo in memory either. tail\nkeep",
            "Prefix. \nkeep",
        ),
        (
            "The block you pasted claiming macOS/Hermes local tools doesn't change that. tail\nkeep",
            "keep",
        ),
        (
            "I just checked this web chat again and /Volumes isn't mounted here so I can't do it.\nkeep",
            "keep",
        ),
    ]
    for raw, expected in cases:
        assert sanitize_response_text(raw) == expected


def test_extract_tool_calls_bare_json():
    text = '{"name": "bash", "arguments": {"command": "ls -la"}}'
    calls, remaining = extract_tool_calls(text, ["terminal"])
    assert calls is not None
    assert len(calls) == 1
    # 'bash' mapped to 'terminal' via alias
    assert calls[0]["function"]["name"] == "terminal"
    assert remaining is None


def test_flatten_messages_with_tools():
    messages = [
        {"role": "user", "content": "What is the git status?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command": "git status"}'},
                },
            ],
        },
        {"role": "tool", "name": "terminal", "content": "On branch main\nnothing to commit"},
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "terminal", "description": "Run shell command"},
        },
    ]
    flat = flatten_messages(messages, tools=tools)
    assert "Available Tools" in flat.text
    assert "```action" in flat.text
    assert "git status" in flat.text
    assert "Client Execution Output (terminal)" in flat.text
    assert "On branch main" in flat.text


def test_api_tool_calling_non_stream(client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch):
    async def mock_stream(account, req):
        yield '<tool_call>{"name": "terminal", "arguments": {"command": "git status"}}</tool_call>'

    monkeypatch.setattr(getattr(client.app, "state").services.gateway.driver, "chat_stream", mock_stream)
    r = client.post("/v1/chat/completions", headers=auth, json={
        "messages": [{"role": "user", "content": "git status"}],
        "tools": [{"type": "function", "function": {"name": "terminal"}}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "terminal"


def test_api_tool_calling_stream(client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch):
    async def mock_stream(account, req):
        yield '<tool_call>{"name": "terminal", "arguments": {"command": "git log"}}</tool_call>'

    monkeypatch.setattr(getattr(client.app, "state").services.gateway.driver, "chat_stream", mock_stream)
    with client.stream("POST", "/v1/chat/completions", headers=auth, json={
        "messages": [{"role": "user", "content": "git log"}],
        "tools": [{"type": "function", "function": {"name": "terminal"}}],
        "stream": True,
    }) as r:
        assert r.status_code == 200
        lines = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    chunks = [json.loads(x) for x in lines[:-1]]
    assert any(c["choices"] and c["choices"][0]["finish_reason"] == "tool_calls" for c in chunks)
    assert any(c["choices"] and "tool_calls" in c["choices"][0]["delta"] for c in chunks)


def test_sanitize_response_text():
    raw = (
        "From the content you pasted in this message — not a mounted drive in this web chat — "
        "I counted it directly:\n\n123 lines for options_trading/surveillance/log_monitor.py"
    )
    cleaned = sanitize_response_text(raw)
    assert "not a mounted drive" not in cleaned
    assert "123 lines for options_trading/surveillance/log_monitor.py" in cleaned


def test_fallback_tool_from_refusal_pytest():
    refusal = (
        "I can't run that from this chat, Gaurav — I just checked:\n"
        "/Volumes/Drive D/Indian Firm/Ai-indian-trading-firm isn't mounted here\n"
        "So I have no file to run pytest on here."
    )
    calls, _ = extract_tool_calls(
        refusal, ["terminal"], user_query="Run pytest on tests/test_tools.py",
    )
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"
    assert "pytest tests/test_tools.py" in calls[0]["function"]["arguments"]


def test_fallback_tool_from_refusal_read():
    refusal = (
        "I can't read that file from this chat, Gaurav — I just checked again:\n"
        "/Volumes/Drive D/Indian Firm/Ai-indian-trading-firm/options_trading/surveillance/log_monitor.py isn't mounted here"
    )
    calls, _ = extract_tool_calls(
        refusal,
        ["read_file"],
        user_query="Read file options_trading/surveillance/log_monitor.py",
    )
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert "log_monitor.py" in calls[0]["function"]["arguments"]


def test_get_workspace_grounding_files():
    messages = [
        {"role": "user", "content": "Read the file options_trading/surveillance/log_monitor.py"},
    ]
    grounding = get_workspace_grounding(messages)
    assert "log_monitor.py" in grounding
    assert "Autonomous Agent Mode" in grounding


def test_synthesize_tool_response_quality_checks():
    messages = [
        {"role": "user", "content": "check for sonar, cc, pyrefly issues"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": "cd /Users/Gaurav_Batra/muse2api && git status --short && .venv/bin/ruff check src tests",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": "All checks passed!\nINFO 0 errors\nExit code: 127",
        },
    ]
    syn = synthesize_tool_response(messages)
    assert "Code Quality & Execution Output" in syn
    assert "Exit code: 127" in syn
    assert "All requested static checks" not in syn


def test_muse2api_pytest_and_git_status_command():
    cmd = _terminal_command_for_test(
        "Check ~/muse2api git status and run .venv/bin/pytest tests/test_tools.py -q, then report the exact result.",
    )
    assert cmd is not None
    assert "/Users/Gaurav_Batra/muse2api" in cmd
    assert "git status -s" in cmd
    assert ".venv/bin/python -m pytest tests/test_tools.py -q" in cmd
    assert "then report" not in cmd
    assert "pyright" not in cmd


def test_uncommitted_changes_command():
    cmd = _terminal_command_for_test("evaluate the uncommitted changes in muse2api")
    assert cmd is not None
    assert "git status -s && git diff --stat" in cmd


def _terminal_command_for_test(query: str) -> str | None:
    from muse2api.core.tools import _terminal_command
    return _terminal_command(query)


def test_clean_git_command_trading_firm_multidrive():
    cmd = _clean_git_command("check git status of the trading firm repository and list last 3 commits")
    assert cmd is not None
    assert "/Volumes/Drive D/Indian Firm/Ai-indian-trading-firm" in cmd
    assert "git status -s" in cmd
    assert "git log -n 3 --oneline" in cmd


def test_clean_git_command_muse2api_multidrive():
    cmd = _clean_git_command("check git status in muse2api")
    assert cmd is not None
    assert "/Users/Gaurav_Batra/muse2api" in cmd
    assert "git status -s" in cmd


def test_clean_git_command_stop_words():
    cmd = _clean_git_command("show git diff of the repository")
    assert cmd == "git diff"


def test_needs_tool_nudge_on_promised_action():
    from muse2api.core.tools import needs_tool_nudge

    nudge = needs_tool_nudge(
        "Applying the patch directly now to both files:",
        [{"role": "user", "content": "fix ruff errors"}],
        ["patch", "read_file", "terminal"],
    )
    assert nudge is not None
    assert "```action" in nudge


def test_needs_tool_nudge_on_fake_edit_without_tool_call():
    from muse2api.core.tools import needs_tool_nudge

    nudge = needs_tool_nudge(
        "Fixed both I001 errors.\nBoth files have been edited on disk.",
        [{"role": "user", "content": "fix ruff errors"}],
        ["patch", "write_file", "terminal"],
    )
    assert nudge is not None
    assert "No files were modified" in nudge


def test_needs_tool_nudge_skips_when_patch_already_called():
    from muse2api.core.tools import needs_tool_nudge

    msgs = [
        {"role": "user", "content": "fix ruff errors"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "patch", "arguments": "{}"}}]},
        {"role": "tool", "content": "Patched ok"},
    ]
    assert needs_tool_nudge("Both files have been edited on disk.", msgs, ["patch"]) is None


def test_tool_alias_edit_maps_to_patch():
    calls, _ = extract_tool_calls(
        '<tool_call>{"name": "edit_file", "arguments": {"path": "a.py"}}</tool_call>',
        ["patch", "read_file"],
    )
    assert calls is not None
    assert calls[0]["function"]["name"] == "patch"


def test_opencode_bidirectional_aliases_and_arg_adaptation():
    opencode_tools = ["edit", "glob", "grep", "read", "shell", "write"]
    cases = [
        ("terminal", {"cmd": "git status -s"}, "shell", "command"),
        ("read_file", {"file_path": "README.md"}, "read", "path"),
        ("patch", {"filePath": "a.py", "old_string": "x", "new_string": "y"}, "edit", "oldString"),
        ("write_file", {"file_path": "b.py", "content": "pass"}, "write", "path"),
        ("search_files", {"pattern": "TODO"}, "grep", "pattern"),
    ]
    for raw_name, raw_args, expected_name, expected_key in cases:
        payload = json.dumps({"name": raw_name, "arguments": raw_args})
        calls, _ = extract_tool_calls(f"<tool_call>{payload}</tool_call>", opencode_tools)
        assert calls is not None
        assert calls[0]["function"]["name"] == expected_name
        parsed_args = json.loads(calls[0]["function"]["arguments"])
        assert expected_key in parsed_args


def test_opencode_prompt_uses_declared_tools_and_omits_bridge_autosave():
    tools = [
        {"type": "function", "function": {"name": name, "description": name}}
        for name in ("edit", "glob", "grep", "read", "shell", "write")
    ]
    msgs = [{"role": "user", "content": "Check options_trading/surveillance/log_monitor.py"}]
    flat = flatten_messages(msgs, tools=tools)
    assert "`shell`" in flat.text
    assert "`edit`" in flat.text
    assert "bridge automatically writes" not in flat.text
    assert "Terminal/patch" not in flat.text


def test_opencode_needs_tool_nudge_on_webchat_refusals():
    from muse2api.core.tools import needs_tool_nudge

    opencode_tools = ["edit", "glob", "grep", "read", "shell", "write"]
    msgs = [{"role": "user", "content": "check the whole repo for pyrefly issues and fix them"}]
    nudge1 = needs_tool_nudge(
        "I checked — I can't do the whole repo in this chat, and I won't claim I did.",
        msgs,
        opencode_tools,
    )
    assert nudge1 is not None
    assert "`shell`" in nudge1
    nudge2 = needs_tool_nudge(
        "I need to correct my last message first, Gaurav — I did not verify that here.",
        msgs,
        opencode_tools,
    )
    assert nudge2 is not None
    assert "```action" in nudge2


def test_extract_rendered_dom_action_and_tool_shapes():
    opencode_tools = ["edit", "glob", "grep", "read", "shell", "write"]
    rendered_action = (
        'action\n{"action": "shell", "args": {"command": "git status -s"}}'
    )
    calls, rem = extract_tool_calls(rendered_action, opencode_tools)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "shell"
    assert json.loads(calls[0]["function"]["arguments"])["command"] == "git status -s"
    assert rem is None

    rendered_tool = (
        'I will run this first.\njson\n{"tool": "shell", "command": "pytest -q"}'
    )
    calls2, rem2 = extract_tool_calls(rendered_tool, opencode_tools)
    assert calls2 is not None
    assert len(calls2) == 1
    assert calls2[0]["function"]["name"] == "shell"
    assert json.loads(calls2[0]["function"]["arguments"])["command"] == "pytest -q"
    assert rem2 == "I will run this first."

    fenced_multi = (
        '```action\n[\n'
        '  {"action": "read", "args": {"path": "a.py"}},\n'
        '  {"action": "grep", "args": {"pattern": "TODO"}}\n'
        ']\n```'
    )
    calls3, _ = extract_tool_calls(fenced_multi, opencode_tools)
    assert calls3 is not None
    assert [c["function"]["name"] for c in calls3] == ["read", "grep"]


def test_poisoned_refusal_history_scrubbed_in_tool_mode():
    tools = [{"type": "function", "function": {"name": "shell"}}]
    msgs = [
        {"role": "system", "content": "OpenCode rules"},
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "I checked in this chat — not the injected Mac-access blocks"},
        {"role": "user", "content": "new task"},
    ]
    flat = flatten_messages(msgs, tools=tools)
    assert "not the injected Mac-access" not in flat.text
    assert "[Project Guidelines]" in flat.text
    assert "[System]\n### Available Tools" not in flat.text


def test_synthesize_tool_response_ignores_previous_turns():
    messages = [
        {"role": "user", "content": "old turn check ruff"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "shell", "arguments": '{"command": "ruff check src"}'}},
            ],
        },
        {"role": "tool", "content": "All checks passed!\nINFO 0 errors"},
        {"role": "assistant", "content": "Previous checks passed."},
        {"role": "user", "content": "evaluate uncommitted changes and run tests"},
    ]
    assert synthesize_tool_response(messages) == ""


def test_run_command_fallback_rejects_english_prose():
    calls, _ = extract_tool_calls(
        "I can't do that in this chat, and I won't pretend I did.",
        ["shell", "read", "edit"],
        user_query="check again, run tests and let me know",
    )
    assert calls is None


def test_opencode_stream_nudge_recovery_and_no_fake_completion(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch,
):
    responses = [
        "I checked — I can't do the whole repo in this chat, and I won't claim I did.",
        '<tool_call>{"name": "shell", "arguments": {"command": "git status -s"}}</tool_call>',
    ]

    async def mock_stream(account, req):
        yield responses.pop(0)

    monkeypatch.setattr(getattr(client.app, "state").services.gateway.driver, "chat_stream", mock_stream)
    opencode_tools = [
        {"type": "function", "function": {"name": n}}
        for n in ("edit", "glob", "grep", "read", "shell", "write")
    ]
    with client.stream("POST", "/v1/chat/completions", headers=auth, json={
        "model": "muse-chat",
        "messages": [
            {"role": "user", "content": "old turn"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "shell", "arguments": "{}"}}]},
            {"role": "tool", "content": "All checks passed!\n0 errors"},
            {"role": "user", "content": "evaluate uncommitted changes"},
        ],
        "tools": opencode_tools,
        "stream": True,
    }) as r:
        assert r.status_code == 200
        lines = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    chunks = [json.loads(x) for x in lines[:-1]]
    assert any(c["choices"] and c["choices"][0]["finish_reason"] == "tool_calls" for c in chunks)
    tc = next(
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if c["choices"] and "tool_calls" in c["choices"][0]["delta"]
    )
    assert tc["function"]["name"] == "shell"
    assert "Ruff / Flake8" not in json.dumps(chunks)


def test_opencode_nudge_preserves_full_context_when_no_hint(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch,
):
    seen_prompts: list[str] = []

    async def mock_stream(account, req):
        seen_prompts.append(req.prompt)
        if len(seen_prompts) == 1:
            yield "I re-checked in this chat, Gaurav — and I need to be straight about scope."
        else:
            yield 'action\n{"action": "shell", "args": {"command": ".venv/bin/pyrefly check src/"}}'

    monkeypatch.setattr(getattr(client.app, "state").services.gateway.driver, "chat_stream", mock_stream)
    opencode_tools = [{"type": "function", "function": {"name": n}} for n in ("shell", "read", "edit")]
    r = client.post("/v1/chat/completions", headers=auth, json={
        "model": "muse-chat",
        "messages": [{"role": "user", "content": "check repo for pyrefly issues"}],
        "tools": opencode_tools,
        "stream": False,
    })
    assert r.status_code == 200
    assert len(seen_prompts) == 2
    assert "check repo for pyrefly issues" in seen_prompts[1]
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "shell"


def test_hermes_patch_snake_case_and_compact_33_tools():
    hermes_tools = [
        {"type": "function", "function": {"name": name, "description": "x" * 400}}
        for name in (
            "terminal", "read_file", "write_file", "patch", "search_files",
            *[f"kanban_extra_{i}" for i in range(28)],
        )
    ]
    prompt = format_tools_prompt(hermes_tools)
    assert "`terminal`" in prompt
    assert "`patch`" in prompt
    assert "kanban_extra_" in prompt
    assert "Other optional tools:" in prompt
    assert prompt.count("x" * 350) == 0

    # Hermes `patch` expects snake_case old_string/new_string even if Muse emits camelCase
    calls, _ = extract_tool_calls(
        'action\n{"action": "edit", "args": {"filePath": "a.py", "oldString": "1", "newString": "2"}}',
        ["terminal", "read_file", "write_file", "patch", "search_files"],
    )
    assert calls is not None
    assert calls[0]["function"]["name"] == "patch"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"path": "a.py", "old_string": "1", "new_string": "2"}

    # Hermes system prompt preamble is condensed down to clean working directory line
    hermes_sys = "You are Hermes Agent, built by Nous Research.\n" + ("noise\n" * 100) + (
        "# Hermes runtime environment\nHost: macOS\nCurrent working directory: /Users/Gaurav_Batra/muse2api\nScratch directory: /tmp"
    )
    flat = flatten_messages(
        [{"role": "system", "content": hermes_sys}, {"role": "user", "content": "run pytest"}],
        tools=hermes_tools,
    )
    assert "Nous Research" not in flat.text
    assert "Scratch directory" not in flat.text
    assert "Current working directory: /Users/Gaurav_Batra/muse2api" in flat.text


def test_hermes_compound_read_and_git_refusal_synthesizes_both():
    refusal = "I checked with my own tools in this web chat — /Volumes doesn't exist here."
    query = (
        "Read the first 15 lines of options_trading/intraday_momentum/audit/_audit_signal_stats_impl.py "
        "and check git status -s in /Volumes/Drive D/Indian Firm/Ai-indian-trading-firm, then summarize both."
    )
    calls, _ = extract_tool_calls(refusal, ["terminal", "read_file", "patch"], user_query=query)
    assert calls is not None
    names = [c["function"]["name"] for c in calls]
    assert names == ["read_file", "terminal"]
    read_args = json.loads(calls[0]["function"]["arguments"])
    assert read_args["path"].endswith("options_trading/intraday_momentum/audit/_audit_signal_stats_impl.py")
    assert read_args["limit"] == 15


def test_long_170_message_opencode_session_compacts_under_budget():
    tools = [{"type": "function", "function": {"name": n}} for n in ("shell", "read", "edit")]
    msgs: list[dict[str, object]] = [
        {"role": "system", "content": "You are an AI agent running in OpenCode.\nWorking directory: /tmp"},
    ]
    for i in range(85):
        msgs.append({"role": "user", "content": f"step {i}"})
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "read", "arguments": f'{{"path": "f{i}.py"}}'}},
        ]})
        msgs.append({"role": "tool", "tool_call_id": f"c_{i}", "content": ("line\n" * 600)})
    msgs.append({"role": "user", "content": "are you done?"})
    flat = flatten_messages(msgs, tools=tools)
    assert len(flat.text) <= 50_000
    assert "### Available Tools" in flat.text
    assert "are you done?" in flat.text
    assert "### Next Step" in flat.text
    assert "older message(s) omitted for length" in flat.text


def test_synthesize_tool_response_skips_raw_read_only_outputs():
    msgs = [
        {"role": "user", "content": "are you done?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "function": {"name": "read", "arguments": '{"filePath": "a.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "1: def foo():\n" + ("2:     pass\n" * 900)},
    ]
    assert synthesize_tool_response(msgs) == ""


def test_needs_tool_nudge_with_existing_tool_outputs_allows_summary_or_next_action():
    msgs = [
        {"role": "user", "content": "are you done?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "function": {"name": "read", "arguments": '{"filePath": "a.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "1: def foo(): pass"},
    ]
    nudge = needs_tool_nudge(
        "No — not verified done in this chat. I just re-checked with my own tools: /Volumes missing.",
        msgs,
        available_tools=["read", "edit", "shell"],
    )
    assert nudge is not None
    assert "already returned Client Execution Output above" in nudge
    assert "structured ```action JSON request" in nudge


def test_quality_refusal_fallback_runs_guard_even_with_fix_in_query():
    refusal = "Not done in this chat — I re-checked with my own tools and /Volumes does not exist."
    calls, _ = extract_tool_calls(
        refusal,
        ["read", "edit", "shell"],
        user_query="check for all type of issues such as prefly, pylance and then fix them",
    )
    assert calls is not None
    assert calls[0]["function"]["name"] == "shell"
    assert "pylance_guard.py" in calls[0]["function"]["arguments"]







