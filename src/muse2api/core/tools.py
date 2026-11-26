"""Emulated tool calling and workspace grounding for muse2api (milestone C3).

muse.ai has no native function calling. This module:
1. Formats OpenAPI / OpenAI tool definitions into clear system prompts for Muse.
2. Formats multi-turn tool history (assistant tool calls and tool responses).
3. Parses tool calls from Muse output (XML <tool_call>, markdown ```json, or bare JSON).
4. Automatically attaches local git and workspace grounding when relevant.
5. Synthesizes tool calls if Muse inadvertently outputs a web-chat refusal for an executable task.
6. Sanitizes web-chat disclaimers from responses.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

TRADING_FIRM_ROOT = "/Volumes/Drive D/Indian Firm/Ai-indian-trading-firm"
MUSE2API_ROOT = "/Users/Gaurav_Batra/muse2api"
GIT_STATUS_SHORT = "git status -s"
_GIT_STATUS = "git status"

_DICT_STR_ANY: TypeAlias = dict[str, Any]
_LIST_ANY: TypeAlias = list[Any]

_STOP_WORDS = frozenset({
    "of", "the", "in", "to", "for", "with", "from", "on", "and", "or",
    "repo", "repository",
})

_QUALITY_WORDS = ("sonar", "cc", "prefly", "pyrefly", "pylance")
_GIT_WORDS = ("git", "commit", "commits", "diff", "branch", "repo", "repository",
              "status", "log", "head")
_CODE_WORDS = ("test", "tests", "pytest", "refactor", "function", "class",
               "module", "code", "file", "files")

_TOOL_GROUPS: tuple[tuple[str, ...], ...] = (
    ("terminal", "shell", "bash", "sh", "exec", "execute", "execute_command",
     "run_command", "cmd", "run"),
    ("read_file", "read", "cat", "view_file"),
    ("write_file", "write", "create_file", "save_file"),
    ("patch", "edit", "edit_file", "apply_patch", "str_replace", "replace"),
    ("search_files", "grep", "ripgrep", "rg"),
    ("glob", "find", "find_files", "list_files"),
)

_TOOL_GROUP_BY_NAME: dict[str, tuple[str, ...]] = {
    alias: group
    for group in _TOOL_GROUPS
    for alias in group
}

_REFUSAL_SIGNALS = (
    "web side chat",
    "in this web chat",
    "not in this chat",
    "not done in this chat",
    "from this chat",
    "/home/hatch",
    "~/workspace",
    "isn't mounted",
    "is not mounted",
    "no file to run",
    "won't pretend",
    "no record of a",
    "no muse2api repo",
    "no clone in my workspace",
    "no clone in ~/workspace",
    "tools aren't available to me here",
    "tools are not available to me here",
    "autonomous agent mode",
    "weren't from a tool call in this chat",
    "not a tool call in this chat",
    "this vm is",
    "this chat, gaurav",
    "can't do the whole repo",
    "cannot do the whole repo",
    "in this chat, and i won't",
    "did not verify that here",
    "didn't verify that here",
    "cannot verify that here",
    "cannot run commands in this chat",
    "can't run commands in this chat",
    "no local file access",
    "no shell access",
    "no terminal access",
    "can't inspect local",
    "cannot inspect local",
    "i need to correct my last",
    "i need to correct the",
    "not the injected",
    "whole repo: not checked",
    "with my own tool runs",
    "my own tools",
    "i re-checked in this chat",
    "i won't emit that",
    "not outputting an action block",
    "don't have a client-executed",
    "ran into a problem while responding",
)

_XML_OPEN = "<tool_call>"
_XML_CLOSE = "</tool_call>"
_FENCE = "```"

_FILE_PATH_RE = re.compile(
    r"((?:options_trading|tests|scripts|docs|src|data)/[a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+|pyproject\.toml|README\.md)",
)
_BASE_GIT_RE = re.compile(
    r"\b(git\s+(?:diff|show|branch|checkout|add|commit|blame|rev-parse))\b",
    re.IGNORECASE,
)
_WORKSPACE_FILE_RE = re.compile(r"\b[a-zA-Z0-9_\-./]+\.(?:py|json|toml|md|sh|yaml|yml)\b")


_CORE_CODING_TOOLS = frozenset({
    "terminal", "shell", "bash", "read_file", "read", "write_file", "write",
    "patch", "edit", "edit_file", "search_files", "grep", "glob",
    "execute_code", "execute", "delegate_task", "clarify", "web_search", "web_extract",
})


def _format_param_line(p_name: str, p_spec: dict[str, Any], req_list: list[Any], compact: bool) -> str:
    """Render a single parameter line for a tool definition."""
    spec = p_spec if isinstance(p_spec, dict) else {}
    ptype = spec.get("type", "any")
    pdesc = str(spec.get("description", "") or "")
    if compact and len(pdesc) > 140:
        pdesc = pdesc[:140].rstrip() + "..."
    req_tag = " (required)" if p_name in req_list else ""
    return f"    * {p_name} ({ptype}{req_tag}): {pdesc}"


def _format_tool_lines(tool: dict[str, Any], *, compact: bool = False) -> list[str]:
    """Render the prompt lines for a single tool definition."""
    fn = cast(_DICT_STR_ANY, tool.get("function", tool))
    name = str(fn.get("name", ""))
    params = cast(_DICT_STR_ANY, fn.get("parameters", {}))
    props = cast(_DICT_STR_ANY, params.get("properties", {}))
    req_raw: Any = params.get("required", [])
    req_list: list[Any] = []
    if isinstance(req_raw, list):
        req_list = cast(_LIST_ANY, req_raw)

    if compact and name.lower() not in _CORE_CODING_TOOLS:
        p_keys = ", ".join(props.keys())
        return [f"- Tool: `{name}`" + (f" (params: {p_keys})" if p_keys else "")]

    lines = [f"- Tool: `{name}`"]
    desc = str(fn.get("description", "") or "")
    if compact and len(desc) > 220:
        desc = desc[:220].rstrip() + "..."
    if desc:
        lines.append(f"  Description: {desc}")
    if not props:
        return lines

    lines.append("  Parameters:")
    for p_name, p_spec in props.items():
        lines.append(_format_param_line(p_name, p_spec, req_list, compact))
    return lines


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    """Return non-empty declared tool names in declaration order."""
    out: list[str] = []
    for t in tools:
        fn = cast(_DICT_STR_ANY, t.get("function", t))
        name = str(fn.get("name", "")).strip()
        if name:
            out.append(name)
    return out


def _pick_tool(available_tools: list[str] | None, canonical: str) -> str | None:
    """Pick the client tool name matching ``canonical`` (or one of its aliases)."""
    if not available_tools:
        return None
    names_map = {t.lower(): t for t in available_tools}
    low = canonical.lower()
    if low in names_map:
        return names_map[low]
    group = _TOOL_GROUP_BY_NAME.get(low)
    if not group:
        return None
    for candidate in group:
        if candidate in names_map:
            return names_map[candidate]
    return None


def _preferred_action_tools(tool_names: list[str]) -> list[str]:
    """Ordered subset of client action tools (shell/read/search/edit/write)."""
    picks: list[str] = []
    for canonical in ("terminal", "read_file", "glob", "search_files", "patch", "write_file"):
        t = _pick_tool(tool_names, canonical)
        if t and t not in picks:
            picks.append(t)
    return picks or tool_names[:6]


def format_tool_reminder(tools: list[dict[str, Any]]) -> str:
    """Short end-of-prompt reminder naming active tools and the ```action JSON syntax."""
    names = _tool_names(tools)
    if not names:
        return ""
    action = _preferred_action_tools(names)
    example_tool = _pick_tool(names, "terminal") or action[0]
    example_args = '{"command": "<cmd>"}' if example_tool in ("terminal", "shell", "bash") else "{}"
    edit_tools = [t for c in ("patch", "write_file") if (t := _pick_tool(names, c))]
    edit_str = " or ".join(f"`{t}`" for t in edit_tools) if edit_tools else _EDIT_WRITE_TOOL_PHRASE
    return (
        f"Available tools: {', '.join(f'`{t}`' for t in action)}. "
        "CRITICAL: NEVER run your own built-in/cloud sandbox tools to check local Mac paths "
        "(they are not mounted in your container). "
        "Use ONLY ```action JSON blocks for client execution, or answer directly from the "
        "Client Execution Output already returned above. "
        "When you need to inspect, search, edit, or test anything, reply ONLY with a JSON block "
        "in ```action ... ``` (for example: "
        f'{{"action": "{example_tool}", "args": {example_args}}}). '
        f"Only report files edited after {edit_str} returns a success result, "
        "and keep chaining steps until the entire task is verified complete."
    )


def format_tools_prompt(tools: list[dict[str, Any]]) -> str:
    """Format OpenAI tool definitions into a collaborative client-runner workflow prompt."""
    if not tools:
        return ""

    names = _tool_names(tools)
    action = _preferred_action_tools(names)
    edit_tools = [t for c in ("patch", "write_file") if (t := _pick_tool(names, c))]
    edit_examples = " or ".join(f"`{t}`" for t in edit_tools) or _EDIT_WRITE_TOOL_PHRASE
    example_tool = _pick_tool(names, "terminal") or (action[0] if action else "shell")
    example_args = (
        '{"command": "git status -s"}'
        if example_tool in ("terminal", "shell", "bash")
        else '{"path": "README.md"}'
    )

    lines = [
        "### Available Tools (Client-Executed Local Workflow)",
        ("We are collaborating in a human-in-the-loop coding workflow. You do NOT have direct "
         "local access and should NOT run bash in your own remote sandbox or pretend to execute "
         "anything yourself."),
        ("Instead, the client executes any ```action JSON block you output locally on the "
         "repository and returns the real output on the next turn."),
        "",
        "Workspace repositories:",
        f"- {MUSE2API_ROOT} (Muse2API gateway repo)",
        f"- {TRADING_FIRM_ROOT} (Trading firm repo)",
        "",
        "Declared tools & parameters:",
        "",
    ]

    compact = len(tools) > 12
    extra_names: list[str] = []
    for t in tools:
        fn = cast(_DICT_STR_ANY, t.get("function", t))
        t_name = str(fn.get("name", ""))
        if compact and t_name.lower() not in _CORE_CODING_TOOLS:
            if t_name:
                extra_names.append(f"`{t_name}`")
            continue
        lines.extend(_format_tool_lines(t, compact=compact))
        if not compact:
            lines.append("")
    if extra_names:
        lines.append(f"- Other optional tools: {', '.join(extra_names)}")
        lines.append("")

    lines.extend([
        "### Workflow Rules",
        "1. Whenever you need to inspect files, search, run commands/tests/linters, or edit/write "
        "files on the user's Mac, output ONLY a JSON block in ```action ... ```:",
        "```action",
        f'{{"action": "{example_tool}", "args": {example_args}}}',
        "```",
        "2. You may also output multiple actions as a JSON array when inspecting multiple items:",
        "```action",
        f'[{{"action": "{example_tool}", "args": {example_args}}}]',
        "```",
        "3. After each `Client Execution Output` arrives, review it and emit the next "
        "```action block(s) to continue (inspect -> read -> edit -> run tests/checks -> verify).",
        f"4. Never claim a file was edited unless {edit_examples} succeeded in `Client Execution "
        "Output`, and never paste full-file markdown dumps instead of calling edit/write.",
        "5. Do not stop after 1 read/status check when fixes or full-repo checks were requested — "
        "keep emitting ```action blocks across turns until all fixes and checks are verified.",
        "6. Only after everything requested is verified complete, write the final response in "
        "plain markdown with NO ```action block.",
    ])
    return "\n".join(lines)


def _normalize_tool_name(name: str, available_tools: list[str] | None) -> str | None:
    """Map tool name aliases bidirectionally (e.g. bash<->shell<->terminal) or check validity."""
    if not name:
        return None
    if not available_tools:
        return name
    return _pick_tool(available_tools, name)


def _adapt_cmd_args(norm_name: str, adapted: dict[str, Any]) -> dict[str, Any]:
    """Normalize shell/terminal argument names to `command`."""
    low = norm_name.lower()
    if low not in ("shell", "terminal", "bash", "sh") or "command" in adapted:
        return adapted
    for alt in ("cmd", "script", "code"):
        if isinstance(adapted.get(alt), str):
            adapted["command"] = adapted.pop(alt)
            break
    return adapted


def _adapt_path_args(norm_name: str, adapted: dict[str, Any]) -> dict[str, Any]:
    """Normalize read/write/edit argument names to `path`."""
    low = norm_name.lower()
    if low not in ("read", "read_file", "write", "write_file", "edit", "edit_file", "patch") or "path" in adapted:
        return adapted
    for alt in ("file_path", "filePath", "filepath", "file"):
        if isinstance(adapted.get(alt), str):
            adapted["path"] = adapted.pop(alt)
            break
    return adapted


def _adapt_edit_args(adapted: dict[str, Any]) -> dict[str, Any]:
    """Rename snake_case edit keys to camelCase."""
    for src, dst in (("old_string", "oldString"), ("new_string", "newString"),
                     ("replace_all", "replaceAll")):
        if dst not in adapted and src in adapted:
            adapted[dst] = adapted.pop(src)
    return adapted


def _adapt_patch_args(adapted: dict[str, Any]) -> dict[str, Any]:
    """Rename camelCase patch keys to snake_case."""
    for src, dst in (("oldString", "old_string"), ("newString", "new_string"),
                     ("replaceAll", "replace_all")):
        if dst not in adapted and src in adapted:
            adapted[dst] = adapted.pop(src)
    return adapted


def _adapt_search_files_args(adapted: dict[str, Any]) -> dict[str, Any]:
    """Rename `include` to `file_glob` for search_files."""
    if "file_glob" not in adapted and isinstance(adapted.get("include"), str):
        adapted["file_glob"] = adapted.pop("include")
    return adapted


def _adapt_grep_args(adapted: dict[str, Any]) -> dict[str, Any]:
    """Rename `file_glob` to `include` for grep."""
    if "include" not in adapted and isinstance(adapted.get("file_glob"), str):
        adapted["include"] = adapted.pop("file_glob")
    return adapted


def _adapt_string_args(norm_name: str, adapted: dict[str, Any]) -> dict[str, Any]:
    """Normalize edit (camelCase) vs patch (snake_case) and search_files/grep parameters."""
    low = norm_name.lower()
    if low in ("edit", "edit_file"):
        return _adapt_edit_args(adapted)
    if low == "patch":
        return _adapt_patch_args(adapted)
    if low == "search_files":
        return _adapt_search_files_args(adapted)
    if low == "grep":
        return _adapt_grep_args(adapted)
    return adapted


def _adapt_tool_args(norm_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize common cross-harness argument names (e.g. file_path -> path, cmd -> command)."""
    adapted = dict(args)
    adapted = _adapt_cmd_args(norm_name, adapted)
    adapted = _adapt_path_args(norm_name, adapted)
    adapted = _adapt_string_args(norm_name, adapted)
    return adapted


def _safe_json_loads(text: str) -> Any | None:
    """Parse JSON without raising; returns None on malformed input."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _build_call_entry(
    name: str, args: Any, available_tools: list[str] | None,
) -> dict[str, Any] | None:
    """Build an OpenAI-style tool-call entry, or None when the tool is unknown."""
    norm_name = _normalize_tool_name(name, available_tools)
    if not norm_name:
        return None

    if isinstance(args, dict):
        adapted = _adapt_tool_args(norm_name, cast(_DICT_STR_ANY, args))
        args_str = json.dumps(adapted, ensure_ascii=False)
    elif isinstance(args, str):
        args_str = args
    else:
        args_str = json.dumps(args or {})

    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {
            "name": norm_name,
            "arguments": args_str,
        },
    }


_ACTION_META_KEYS = frozenset({"name", "action", "tool"})
_ACTION_ARGS_KEYS = ("arguments", "args", "input", "parameters")


def _extract_dict_entry(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a single action/tool dict ({name|action|tool, arguments|args|input|...})."""
    tool_name = next(
        (payload[k] for k in ("name", "action", "tool") if isinstance(payload.get(k), str)),
        None,
    )
    if not tool_name:
        return None
    for arg_k in _ACTION_ARGS_KEYS:
        if arg_k in payload:
            return {"name": tool_name, "arguments": payload[arg_k]}
    extra_args = {k: v for k, v in payload.items() if k not in _ACTION_META_KEYS}
    return {"name": tool_name, "arguments": extra_args}


def _single_tool_entry(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """A single {name|action|tool, ...} payload, or None when it carries a batch list."""
    if isinstance(payload.get("tool_calls"), list) or isinstance(payload.get("actions"), list):
        return None
    entry = _extract_dict_entry(payload)
    return [entry] if entry is not None else None


def _entry_from_item(tc: Any) -> dict[str, Any] | None:
    """Normalize one tool-call item; None when it is not usable."""
    if not isinstance(tc, dict):
        return None
    return _extract_dict_entry(cast(_DICT_STR_ANY, tc))


def _tool_list_items(data: Any) -> list[Any]:
    """Coerce a payload into a list of tool-call items."""
    if isinstance(data, list):
        return cast(_LIST_ANY, data)
    if not isinstance(data, dict):
        return []
    payload = cast(_DICT_STR_ANY, data)
    for key in ("tool_calls", "actions"):
        raw_calls = payload.get(key)
        if isinstance(raw_calls, list):
            return cast(_LIST_ANY, raw_calls)
    return []


def _tool_entries(data: Any) -> list[dict[str, Any]]:
    """Normalize a parsed JSON payload into a list of {name, arguments} entries."""
    if isinstance(data, dict):
        single = _single_tool_entry(cast(_DICT_STR_ANY, data))
        if single is not None:
            return single
    return [
        entry
        for entry in (_entry_from_item(tc) for tc in _tool_list_items(data))
        if entry is not None
    ]


def _calls_from_payload(
    data: Any, available_tools: list[str] | None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for tc in _tool_entries(data):
        c = _build_call_entry(tc.get("name", ""), tc.get("arguments", {}), available_tools)
        if c:
            calls.append(c)
    return calls


def _xml_blocks(text: str) -> list[tuple[int, int]]:
    """Non-overlapping spans of ``<tool_call>...</tool_call>`` blocks.

    Deterministic linear scan (``str.find``) instead of a lazy-DOTALL regex,
    which is O(n^2) on input with many unclosed tags (backtracking).
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        start = text.find(_XML_OPEN, pos)
        if start < 0:
            break
        end = text.find(_XML_CLOSE, start + len(_XML_OPEN))
        if end < 0:
            break
        spans.append((start, end + len(_XML_CLOSE)))
        pos = spans[-1][1]
    return spans


def _fence_lang_end(text: str, body_start: int) -> int:
    """Advance past an optional ```json / ```action language tag."""
    for lang in ("action", "json"):
        if text.startswith(lang, body_start):
            tag_end = body_start + len(lang)
            if tag_end >= len(text) or text[tag_end].isspace():
                return tag_end
    return body_start


def _json_code_blocks(text: str) -> list[tuple[int, int]]:
    """Non-overlapping spans of ```json / ```action blocks whose body is JSON."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        start = text.find(_FENCE, pos)
        if start < 0:
            break
        body_start = _fence_lang_end(text, start + len(_FENCE))
        end = text.find(_FENCE, body_start)
        if end < 0:
            break
        body = text[body_start:end].strip()
        if body.startswith(("{", "[")):
            spans.append((start, end + len(_FENCE)))
        pos = end + len(_FENCE)
    return spans


def _xml_content(text: str, start: int, end: int) -> str:
    return text[start + len(_XML_OPEN):end - len(_XML_CLOSE)].strip()


def _fence_content(text: str, start: int, end: int) -> str:
    inner = text[start + len(_FENCE):end - len(_FENCE)].strip()
    for lang in ("action", "json"):
        if inner.startswith(lang) and (len(inner) == len(lang) or inner[len(lang)].isspace()):
            return inner[len(lang):].lstrip()
    return inner


def _extract_blocks(
    locate: Callable[[str], list[tuple[int, int]]],
    content_at: Callable[[str, int, int], str],
    text: str,
    available_tools: list[str] | None,
) -> tuple[list[dict[str, Any]], str] | None:
    """Parse every located block; return (calls, remaining) or None."""
    spans = locate(text)
    if not spans:
        return None
    calls: list[dict[str, Any]] = []
    remaining_parts: list[str] = []
    prev = 0
    for start, end in spans:
        payload = _safe_json_loads(content_at(text, start, end))
        if payload is not None:
            calls.extend(_calls_from_payload(payload, available_tools))
        remaining_parts.append(text[prev:start])
        prev = end
    remaining_parts.append(text[prev:])
    if not calls:
        return None
    return calls, "".join(remaining_parts).strip()


def _extract_from_blocks(
    text: str, available_tools: list[str] | None,
) -> tuple[list[dict[str, Any]] | None, str | None] | None:
    """Parse XML <tool_call> and markdown JSON blocks; return (calls, remaining) or None."""
    for locate, content_at in (
        (_xml_blocks, _xml_content),
        (_json_code_blocks, _fence_content),
    ):
        found = _extract_blocks(locate, content_at, text, available_tools)
        if found is not None:
            calls, remaining = found
            return calls, remaining or None
    return None


def is_refusal(text: str) -> bool:
    """True when the reply reads like a web-chat refusal instead of a tool call."""
    low = text.lower()
    return any(sig in low for sig in _REFUSAL_SIGNALS)


def _mentions(low: str, words: tuple[str, ...]) -> bool:
    """True when any ``words`` occurs in ``low``."""
    return any(w in low for w in words)


def _git_workspace_prefix(low: str) -> str:
    """Target workspace prefix if the query specifies a repo or drive."""
    if _mentions(low, ("trading", "firm", "options", "drive d", "indian firm")):
        return f'cd "{TRADING_FIRM_ROOT}" && '
    if "muse2api" in low or "/users/gaurav_batra/muse2api" in low:
        return f"cd {MUSE2API_ROOT} && "
    return ""


def _recent_commit_count(low: str) -> str:
    """Number of recent commits requested in the query (or a default)."""
    m = re.search(r"last\s+(\d+)\s+commits?", low)
    return m.group(1) if m else "3"


def _git_request_kind(low: str) -> str | None:
    """Classify a git request as status / log / diff / both, else None."""
    if _mentions(low, ("uncommitted", "uncommited")):
        return "uncommitted"
    wants_status = _mentions(low, ("status", _GIT_STATUS))
    wants_log = _mentions(low, ("commit", "git log", "log"))
    if wants_status and wants_log:
        return "status_log"
    if wants_status:
        return "status"
    if wants_log:
        return "log"
    return None


def _clean_git_args(args_str: str) -> list[str]:
    """Clean git arguments by removing stop words."""
    clean_args: list[str] = []
    for a in args_str.split():
        if a.lower() in _STOP_WORDS:
            break
        clean_args.append(a)
    return clean_args


def _parse_git_subcommand(user_query: str) -> str | None:
    """Clean git subcommand + args (stopping at natural-language filler words)."""
    m = _BASE_GIT_RE.search(user_query)
    if not m:
        return None
    base_cmd = m.group(1).strip()
    clean_args = _clean_git_args(user_query[m.end():])
    return f"{base_cmd} {' '.join(clean_args)}".strip()


def _clean_git_command(user_query: str) -> str | None:
    """A git command matching the user's query, with a workspace prefix."""
    low = user_query.lower()
    prefix = _git_workspace_prefix(low)
    kind = _git_request_kind(low)
    if kind == "uncommitted":
        return f"{prefix}{GIT_STATUS_SHORT} && git diff --stat"
    if kind == "status_log":
        return f"{prefix}{GIT_STATUS_SHORT} && git log -n {_recent_commit_count(low)} --oneline"
    if kind == "status":
        return f"{prefix}{GIT_STATUS_SHORT}"
    if kind == "log":
        return f"{prefix}git log -n {_recent_commit_count(low)} --oneline"
    sub = _parse_git_subcommand(user_query)
    return f"{prefix}{sub}" if sub else None


def _repo_check_command(low: str) -> str | None:
    """A combined repo health command when the user asks about muse2api checks."""
    if "muse2api" not in low:
        return None
    if _mentions(
        low,
        (
            "sonar", "cc", "prefly", "pyrefly", "pylance",
            "lint", "issue", "issues",
        ),
    ) or ("check" in low and _GIT_STATUS not in low and "pytest" not in low
          and "uncommitted" not in low and "uncommited" not in low):
        return (
            f"cd {MUSE2API_ROOT} && {GIT_STATUS_SHORT} && "
            ".venv/bin/ruff check src tests && .venv/bin/pyrefly check src/"
        )
    return None


def _pytest_command(user_query: str) -> str | None:
    """Normalize a pytest invocation mentioned in the query."""
    m = re.search(r"\bpytest\b", user_query, re.IGNORECASE)
    if not m:
        return None
    rest = user_query[m.end():].split("\n", 1)[0]
    tail = re.sub(r"^\s+on\s+", " ", rest, flags=re.IGNORECASE).strip()
    # Truncate at the first trailing keyword separator (linear scan, no backtracking).
    _cut = len(tail)
    _low_tail = tail.lower()
    for _sep in (",", ";", " and", " then", " to", " for", " with"):
        _idx = _low_tail.find(_sep)
        if _idx != -1 and _idx < _cut:
            _cut = _idx
    tail = tail[:_cut].strip()
    cmd = f".venv/bin/python -m pytest {tail}".strip()
    low = user_query.lower()
    prefix = _git_workspace_prefix(low)
    if _GIT_STATUS in low:
        return f"{prefix}{GIT_STATUS_SHORT} && {cmd}"
    return f"{prefix}{cmd}"


_SAFE_CMD_HEAD_RE = re.compile(
    r"^(?:git|pytest|python(?:3)?|ruff|pyright|pyrefly|flake8|mypy|radon|uv|"
    r"npm|npx|node|ls|cat|head|tail|grep|rg|find|\.?/|\.venv/)",
    re.IGNORECASE,
)


def _run_command(user_query: str) -> str | None:
    """A bare run/execute command mentioned in the query (only real shell commands)."""
    m = re.search(r"\b(?:run|execute)\b", user_query, re.IGNORECASE)
    if m is None:
        return None
    rest = user_query[m.end():].split("\n", 1)[0].lstrip()
    if not rest:
        return None
    if rest.startswith("`") and "`" in rest[1:]:
        cmd = rest.split("`")[1].strip()
        return cmd or None
    if _SAFE_CMD_HEAD_RE.match(rest):
        return rest
    return None


def _quality_repo_query(low: str) -> bool:
    """True for queries about quality-tool checks on a repository."""
    return _mentions(low, _QUALITY_WORDS) and (
        "repo" in low or _mentions(low, ("check", "issue", "issues", "pylance", "pyrefly", "prefly"))
    )


def _terminal_command(user_query: str) -> str | None:
    """A shell command the user asked to run (pytest, git, repo checks, run/execute)."""
    low = user_query.lower()
    pytest_cmd = _pytest_command(user_query)
    if pytest_cmd:
        return pytest_cmd
    repo_cmd = _repo_check_command(low)
    if repo_cmd:
        return repo_cmd
    if _quality_repo_query(low) or "pylance_guard.py" in low:
        if "muse2api" in low and not _mentions(low, ("pylance", "trading", "firm")):
            cmd = ".venv/bin/python -m pytest -q && .venv/bin/ruff check src tests"
        else:
            cmd = ".venv/bin/python scripts/ops/pylance_guard.py"
        prefix = _git_workspace_prefix(low)
        if "git status" in low:
            return f"{prefix}{GIT_STATUS_SHORT} && {cmd}"
        return f"{prefix}{cmd}"
    cmd_git = _clean_git_command(user_query)
    if cmd_git:
        return cmd_git
    return _run_command(user_query)


def _refusal_file_path(text: str, user_query: str) -> str | None:
    """A repo file path mentioned in the user query or refusal."""
    m_path = _FILE_PATH_RE.search(user_query) or _FILE_PATH_RE.search(text)
    if not m_path:
        return None
    path = m_path.group(1).strip()
    if not path.startswith("/"):
        low = user_query.lower()
        if TRADING_FIRM_ROOT.lower() in low or _mentions(low, ("trading", "firm", "options")):
            return f"{TRADING_FIRM_ROOT}/{path}"
        if "muse2api" in low:
            return f"{MUSE2API_ROOT}/{path}"
    return path


def _read_line_limit(user_query: str) -> int | None:
    """Extract requested line limit such as 'first 15 lines' from query."""
    m = re.search(r"\b(?:first|top|lines?)\s+(\d+)\s*(?:lines?)?\b", user_query, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _terminal_fallback(user_query: str,
                       available_tools: list[str]) -> list[dict[str, Any]] | None:
    """Synthesize a terminal/shell tool call when the user asked to run a command."""
    terminal_tool = _pick_tool(available_tools, "terminal")
    if not terminal_tool:
        return None
    cmd = _terminal_command(user_query)
    if not cmd:
        return None
    c = _build_call_entry(terminal_tool, {"command": cmd}, available_tools)
    return [c] if c else None


def _read_file_fallback(text: str, user_query: str,
                        available_tools: list[str]) -> list[dict[str, Any]] | None:
    """Synthesize a read/read_file tool call for a repo path mentioned in the reply."""
    read_tool = _pick_tool(available_tools, "read_file")
    if not read_tool:
        return None
    path = _refusal_file_path(text, user_query)
    if not path:
        return None
    args: dict[str, Any] = {"path": path}
    limit = _read_line_limit(user_query)
    if limit is not None:
        args["limit"] = limit
    c = _build_call_entry(read_tool, args, available_tools)
    return [c] if c else None


def _fallback_tool_from_refusal(
    text: str,
    user_query: str,
    available_tools: list[str] | None,
) -> list[dict[str, Any]] | None:
    """When Muse outputs a web-chat refusal instead of calling a tool, synthesize calls."""
    if not available_tools or not text:
        return None
    if not is_refusal(text):
        return None

    term_calls = _terminal_fallback(user_query, available_tools) or []
    wants_explicit_read = bool(
        _FILE_PATH_RE.search(user_query)
        and _mentions(user_query.lower(), ("read", "line", "lines", "inspect", "show", "head"))
    )
    if not term_calls or wants_explicit_read:
        read_calls = _read_file_fallback(text, user_query, available_tools) or []
    else:
        read_calls = []
    combined = read_calls + term_calls
    return combined or None


def _bare_json_calls(
    text: str, available_tools: list[str] | None,
) -> list[dict[str, Any]] | None:
    """Tool calls from a top-level JSON object or array, or None when not one."""
    stripped = text.strip()
    if not ((stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))):
        return None
    payload = _safe_json_loads(stripped)
    if payload is None:
        return None
    calls = _calls_from_payload(payload, available_tools)
    return calls or None


def _embedded_json_calls(
    text: str, available_tools: list[str] | None,
) -> tuple[list[dict[str, Any]], str | None] | None:
    """Extract JSON action objects/arrays rendered without ``` fences (e.g. DOM innerText)."""
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    kept: list[str] = []
    pos = 0
    length = len(text)
    while pos < length:
        brace = text.find("{", pos)
        bracket = text.find("[", pos)
        candidates = [i for i in (brace, bracket) if i >= 0]
        if not candidates:
            kept.append(text[pos:])
            break
        idx = min(candidates)
        try:
            payload, end_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            kept.append(text[pos:idx + 1])
            pos = idx + 1
            continue
        found = _calls_from_payload(payload, available_tools)
        if not found:
            kept.append(text[pos:end_idx])
            pos = end_idx
            continue
        prefix = text[pos:idx]
        # Strip rendered code-block label lines like "action\n" or "json\n" or "Copy\n"
        prefix = re.sub(r"(?:^|\n)\s*(?:Copy\s+)?(?:action|json)\s*$", "", prefix, flags=re.IGNORECASE)
        kept.append(prefix)
        calls.extend(found)
        pos = end_idx

    if not calls:
        return None
    remaining = "".join(kept).strip() or None
    return calls, remaining


def extract_tool_calls(
    text: str,
    available_tools: list[str] | None = None,
    user_query: str = "",
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Parse tool calls from model output, with optional fallback synthesis on refusal."""
    if not text:
        return None, text

    found = _extract_from_blocks(text, available_tools)
    if found is not None:
        return found

    calls = _bare_json_calls(text, available_tools)
    if calls:
        return calls, None

    embedded = _embedded_json_calls(text, available_tools)
    if embedded is not None:
        return embedded

    # Fallback synthesis if caller passed user_query and model refused an actionable query
    if user_query and available_tools:
        fallback = _fallback_tool_from_refusal(text, user_query, available_tools)
        if fallback:
            return fallback, None

    return None, text


def _remove_phrase_to_eol(text: str, phrase: str, anchor: str = "", min_gap: int = 0) -> str:
    """Remove from the first occurrence of ``phrase`` to its end-of-line.

    Deterministic O(n) replacement for ``phrase[^\n]*`` / ``phrase[^\n]*?anchor[^\n]*``
    regexes — no backtracking, guaranteed linear runtime. ``anchor`` (when given) must
    appear after ``phrase`` with at least ``min_gap`` characters between them.
    """
    low = text.lower()
    start = low.find(phrase)
    while start != -1:
        if not anchor or low.find(anchor, start + len(phrase) + min_gap) != -1:
            eol = text.find("\n", start)
            return text[:start] if eol == -1 else text[:start] + text[eol:]
        start = low.find(phrase, start + 1)
    return text


def sanitize_response_text(text: str) -> str:
    """Clean out boilerplate web-chat disclaimers and sandbox caveats."""
    if not text:
        return text

    # Remove zero-width spaces that precede repeated chunks
    text = text.replace("-\u200b", "")

    disclaimer_patterns = [
        # "No — not in this chat, Gaurav. I just checked here directly: ..."
        r"No\s*—\s*not in this chat[^\n]*",
        # "From the content you pasted ... — not a mounted drive in this web chat — ..."
        (r"From the (?:content|git context) you pasted in this message\s*—\s*"
         r"not a mounted drive in this web chat\s*—\s*"),
        # "— not a mounted drive in this web chat —"
        r"—\s*not a mounted drive in this web chat\s*—\s*",
        # "This is the web side chat, working dir /home/hatch — /Volumes doesn't exist..."
        r"This is the web side chat[^\n]*",
        # "No muse2api repo/folder anywhere in ~/workspace..."
        r"No muse2api repo/folder anywhere in ~/workspace[^\n]*",
        # "Which route do you ... want?"
        r"Which route do you[^\n]*want\??",
        # "and I won't pretend a terminal tool on your Mac is available in this web chat."
        (r"and I won't pretend a (?:Hermes/)?terminal tool on your Mac "
         r"is available in this web chat\.?"),
        # "in this web chat"
        r"\bin this web chat\b",
    ]
    for pat in disclaimer_patterns:
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Deterministic equivalents of the double-quantifier disclaimer regexes.
    text = _remove_phrase_to_eol(
        text, "i have no record of a ", " repo in memory either", min_gap=1,
    )
    text = _remove_phrase_to_eol(
        text, "the block you pasted claiming ", " tools doesn't change that", min_gap=1,
    )
    text = _remove_phrase_to_eol(text, "i just checked", " isn't mounted here")

    # Clean leading/trailing empty lines
    text = re.sub(r"^\s*\n+", "", text)
    return text.strip()


def _collect_from_turn_messages(turn_messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Collect tool outputs and call summaries from post-user messages."""
    tool_outputs: list[str] = []
    tool_calls: list[str] = []
    for m in turn_messages:
        if m.get("role") == "tool":
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                tool_outputs.append(c.strip())
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                tool_calls.append(f"{fn.get('name', 'tool')}: {fn.get('arguments', '')}")
    return tool_outputs, tool_calls


def _collect_tool_history(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Split current-turn tool outputs and tool-call summaries (after the last user message)."""
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            last_user_idx = idx
            break
    turn_messages = messages[last_user_idx + 1:] if last_user_idx >= 0 else messages
    return _collect_from_turn_messages(turn_messages)


def _quality_report(tool_calls_str: str, latest_output: str) -> str | None:
    """Summarize static-analysis output accurately without fabricating results."""
    mentions_toolchain = any(
        k in tool_calls_str for k in ("ruff", "pyrefly", "pyright", "flake8", "radon")
    )
    if not mentions_toolchain and "all checks passed" not in latest_output.lower():
        return None
    return _execution_section("Code Quality & Execution Output", latest_output)


def _execution_section(title: str, latest_output: str) -> str:
    """A fenced output section for pytest / file reads / general command runs."""
    return f"### {title}:\n```\n{latest_output[:8000]}\n```"


_SECTION_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("pytest", "passed in", "failed in"), "Test Execution Results"),
    (("git ", "on branch", "commit ", "\n?? ", "\nm "), "Git Status & Commit History"),
    (("read_file", "cat ", "\"total_lines\":"), "File Contents"),
)


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)


def _tool_section_title(tool_calls_str: str, latest_output: str) -> str:
    """Best heading for a tool output based on what was run."""
    low = latest_output.lower()
    for needles, title in _SECTION_RULES:
        if _contains_any(tool_calls_str, needles) or _contains_any(low, needles):
            return title
    return "Execution Output"


_INSPECTION_TOOL_NAMES = frozenset({
    "read", "read_file", "view_file", "cat",
    "glob", "grep", "search_files", "find_files", "list_files",
})


def _has_only_inspection_calls(tool_calls: list[str]) -> bool:
    """Return True when every recorded tool call is a read/search/list inspection."""
    if not tool_calls:
        return False
    names = [tc.split(":", 1)[0].strip().lower() for tc in tool_calls]
    return bool(names) and all(name in _INSPECTION_TOOL_NAMES for name in names)


def synthesize_tool_response(messages: list[dict[str, Any]]) -> str:
    """Synthesize an answer when the model refused despite having tool execution results."""
    tool_outputs, tool_calls = _collect_tool_history(messages)
    if not tool_outputs or _has_only_inspection_calls(tool_calls):
        return ""

    latest_output = "\n\n".join(tool_outputs)
    tool_calls_str = " ".join(tool_calls).lower()

    report = _quality_report(tool_calls_str, latest_output)
    if report is not None:
        return report
    return _execution_section(_tool_section_title(tool_calls_str, latest_output), latest_output)


try:
    import workspace_helper  # pyright: ignore[reportMissingImports]
except ImportError:
    _repo_root = Path(__file__).resolve().parents[3]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    try:
        import workspace_helper  # pyright: ignore[reportMissingImports]
    except ImportError:
        workspace_helper = None  # type: ignore[assignment]


def _load_workspace_helper() -> Any:
    """Workspace helper module (resolved once at import time), or None."""
    return workspace_helper


def _last_user_query(messages: list[dict[str, Any]]) -> str:
    """Return the most recent user message rendered as plain text."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            texts = [
                cast(_DICT_STR_ANY, part).get("text", "")
                for part in cast(_LIST_ANY, c)
                if isinstance(part, dict)
            ]
            return " ".join(texts)
    return ""


def _is_workspace_query(query: str) -> bool:
    """Return True if query refers to repository files, git, tests, or code."""
    q = query.lower()
    if _WORKSPACE_FILE_RE.search(q):
        return True
    if "/" in q or "\\" in q:
        return True
    return _mentions(q, _GIT_WORDS) or _mentions(q, _CODE_WORDS)


_PROMISE_SUFFIX_RE = re.compile(
    r"(?:applying|patching|editing|updating|writing|running|checking|fixing|inspecting)\b"
    r"[^\n]{0,100}(?::|\bnow\b|\bnext\b|\bfirst\b)[^\n]{0,40}$",
    re.IGNORECASE,
)
_MIDTASK_PROGRESS_RE = re.compile(
    r"^(?:doing|starting|continuing|working on|applying|making|inspecting|checking|running)\b"
    r"[^\n]{0,200}[.:]\s*$",
    re.IGNORECASE,
)
_FAKE_EDIT_PHRASES = (
    "have been edited on disk",
    "edited on disk",
    "fixed both",
    "patched both",
    "updated both files",
    "both files have been",
    "changes have been applied",
    "edit has been applied",
)


_ACTION_TOOL_NAMES = frozenset({
    "patch", "edit", "edit_file", "apply_patch",
    "write_file", "write", "create_file",
    "terminal", "shell", "bash", "execute",
    "read_file", "read", "glob", "grep", "search_files",
})

_EDIT_TOOL_NAMES = frozenset({"patch", "write_file", "write", "edit", "edit_file"})

_EDIT_WRITE_TOOL_PHRASE = "an edit/write tool"


def _current_turn_used_edit_tool(messages: list[dict[str, Any]]) -> bool:
    for m in reversed(messages):
        if m.get("role") == "user":
            break
        for tc in m.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name", "").lower()
            if name in _EDIT_TOOL_NAMES:
                return True
    return False


def _compute_tool_context(
    available_tools: list[str],
    stripped: str,
    messages: list[dict[str, Any]],
) -> tuple[str, str, str, str, list[str] | None]:
    """Compute tool context for needs_tool_nudge.

    Returns: (action_list, edit_list, example_tool, example_args, tool_outputs_or_None)
    """
    action = _preferred_action_tools(available_tools)
    action_list = ", ".join(f"`{t}`" for t in action)
    edit_tools = [t for c in ("patch", "write_file") if (t := _pick_tool(available_tools, c))]
    edit_list = " or ".join(f"`{t}`" for t in edit_tools) or _EDIT_WRITE_TOOL_PHRASE
    example_tool = _pick_tool(available_tools, "terminal") or action[0]
    example_args = '{"command": "..."}' if example_tool in ("terminal", "shell", "bash") else "{}"

    tool_outputs, _ = _collect_tool_history(messages) if is_refusal(stripped) else (None, None)

    return action_list, edit_list, example_tool, example_args, tool_outputs


def needs_tool_nudge(
    text: str,
    messages: list[dict[str, Any]],
    available_tools: list[str] | None = None,
) -> str | None:
    """Return a correction prompt if Muse promised an action, refused mid-task, or faked an edit."""
    if not text or not available_tools:
        return None
    norm = {t.lower() for t in available_tools}
    if not (norm & _ACTION_TOOL_NAMES):
        return None
    stripped = text.strip()
    low = stripped.lower()
    if _PROMISE_SUFFIX_RE.search(stripped) or _MIDTASK_PROGRESS_RE.match(stripped):
        return (
            "You stopped after promising a next step without outputting the action JSON. "
            "Output ONLY a ```action JSON block with the next action right now."
        )
    if _current_turn_used_edit_tool(messages):
        return None

    action_list, edit_list, example_tool, example_args, tool_outputs = _compute_tool_context(
        available_tools, stripped, messages,
    )

    if "ran into a problem while responding" in low:
        return (
            "Please continue the user's request above: emit a ```action JSON block if a tool call "
            f"({action_list}) is needed, or answer the user's question directly."
        )
    if is_refusal(stripped):
        context_note = (
            "The user's local Mac runner already returned Client Execution Output above. "
            if tool_outputs
            else ""
        )
        return (
            "You are NOT pretending to have Mac access — `/Volumes` is only on the user's Mac, "
            "not in your container. Your role here is to emit a structured ```action JSON request "
            "so the user's local Mac client can execute it on their Mac and send you the real "
            f"output. {context_note}"
            f"Please output the next ```action JSON block using ({action_list}), for example:\n"
            "```action\n"
            f'{{"action": "{example_tool}", "args": {example_args}}}\n'
            "```"
        )
    has_fake_claim = any(p in low for p in _FAKE_EDIT_PHRASES)
    has_prose_file_dump = bool(re.search(r"(?:^|\n)File:\s*\S+", stripped) and "```" in stripped)
    if has_fake_claim or has_prose_file_dump:
        return (
            f"No files were modified yet because no {edit_list} ```action block was emitted. "
            f"Output ONLY a ```action JSON block for {edit_list} right now instead of pasting code."
        )
    return None


def get_workspace_grounding(
    messages: list[dict[str, Any]],
    *,
    tool_mode: bool = False,
) -> str:
    """Fetch relevant local git & workspace context (files + commits) for the user query."""
    user_query = _last_user_query(messages)
    if not user_query or not _is_workspace_query(user_query):
        return ""

    helper = _load_workspace_helper()
    if helper is None:
        return ""

    try:
        base = helper.get_default_workspace(user_query)
        paths = helper.find_local_paths(user_query, base_dir=base)
        ctx = helper.build_context_for_paths(paths, prompt=user_query, tool_mode=tool_mode)
    except (AttributeError, TypeError, ValueError, OSError):
        log.debug("Workspace grounding fetch failed")
    else:
        return ctx or ""
    return ""
