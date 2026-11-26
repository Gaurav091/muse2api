"""Convert OpenAI-style message lists into a single prompt for the muse web chat.

muse.ai's web UI is a single text box, so multi-turn context and system prompts
are serialised into one message. Image parts are extracted separately so that
drivers can attach them as files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

from .tools import (
    format_tool_reminder,
    format_tools_prompt,
    get_workspace_grounding,
    is_refusal,
)

ROLE_LABELS = {
    "system": "System",
    "developer": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool Result",
}


@dataclass
class FlatPrompt:
    text: str
    images: list[str] = field(default_factory=list[str])
    """Image references (data URLs or http(s) URLs) found in the messages, in order."""


def _part_text(part: dict[str, Any]) -> str:
    """Plain text of a single content part (empty for non-text parts)."""
    if part.get("type") in ("text", "input_text", "output_text"):
        return part.get("text", "")
    return ""


def _part_image_ref(part: dict[str, Any]) -> list[str]:
    """Image reference URLs carried by a content part (empty when none)."""
    if part.get("type") not in ("image_url", "input_image"):
        return []
    ref = part.get("image_url")
    if isinstance(ref, dict):
        ref = cast(dict[str, Any], ref).get("url")
    return [ref] if ref else []


def _content_to_text(content: Any, images: list[str]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append(str(part))
            continue
        p = cast(dict[str, Any], part)
        images.extend(_part_image_ref(p))
        parts.append(_part_text(p))
    return "\n".join(p for p in parts if p)


def _tool_call_text(tc: Any, *, action_style: bool = False) -> str | None:
    """Render a single tool_call entry as an action block or XML tool-call line."""
    if not isinstance(tc, dict):
        return None
    fn = cast(dict[str, Any], tc).get("function", {})
    fn_dict = cast(dict[str, Any], fn) if isinstance(fn, dict) else None
    if fn_dict is None:
        return None
    name = fn_dict.get("name", "")
    args = fn_dict.get("arguments", "{}")
    if isinstance(args, dict):
        args = json.dumps(args, ensure_ascii=False)
    if action_style:
        return f'```action\n{{"action": "{name}", "args": {args}}}\n```'
    return f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'


def _message_to_text(
    msg: dict[str, Any], images: list[str], *, action_style: bool = False,
) -> str:
    content_text = _content_to_text(msg.get("content"), images).strip()
    role = msg.get("role", "user")
    tool_calls = msg.get("tool_calls")

    parts: list[str] = []
    if content_text:
        parts.append(content_text)

    if isinstance(tool_calls, list):
        for tc in cast(list[Any], tool_calls):
            line = _tool_call_text(tc, action_style=action_style)
            if line is not None:
                parts.append(line)

    if role == "tool":
        name = msg.get("name") or msg.get("tool_call_id") or "tool"
        prefix = (
            f"Client Execution Output ({name}):"
            if action_style
            else f"[Tool Output ({name})]"
        )
        return f"{prefix}\n" + "\n".join(parts)

    return "\n\n".join(parts)


def message_turns(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Role/text pairs, using the same text extraction as ``flatten_messages``."""
    turns: list[tuple[str, str]] = []
    for msg in messages:
        text = _message_to_text(msg, []).strip()
        if text:
            turns.append((msg.get("role", "user"), text))
    return turns


def followup_text(previous: list[tuple[str, str]], incoming: list[tuple[str, str]]) -> str | None:
    """Text to send when ``incoming`` continues a conversation already on the page.

    Returns only the new user text. ``None`` means the histories do not line up, so the
    caller must open a fresh thread and send the full prompt instead.
    """
    if not previous or len(incoming) <= len(previous):
        return None
    if incoming[: len(previous)] != list(previous):
        return None
    extra = list(incoming[len(previous) :])
    while extra and extra[0][0] in ("assistant", "tool"):
        extra = extra[1:]
    user_text: list[str] = []
    for role, text in extra:
        if role != "user":
            return None
        user_text.append(text)
    return "\n\n".join(user_text) or None


def _is_verbatim_single_turn(turns: list[tuple[str, str]], tool_section: str,
                              grounding_section: str) -> bool:
    """A lone user turn without tools or workspace context is sent verbatim."""
    return (
        len(turns) == 1
        and turns[0][0] == "user"
        and not tool_section
        and not grounding_section
    )


def _is_poisoned_assistant_reply(text: str) -> bool:
    """True for stale refusal/apology/canned-report turns that poison subsequent turns."""
    return (
        is_refusal(text)
        or text.startswith("I have accessed the repository on your machine and verified the code quality:")
    )


def _condense_harness_system(text: str) -> str:
    """Keep concise cwd/project guidance while dropping conflicting agent preambles."""
    low = text.lower()
    if "running in opencode" not in low and "you are hermes agent" not in low:
        return text
    if "you are hermes agent" in low:
        for line in text.splitlines():
            if line.strip().lower().startswith("current working directory:"):
                return line.strip()
        return ""
    for marker in ("<env>", "Working directory:", "# Instructions", "# AGENTS.md", "Project Rules"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:idx + 4000].strip()
    return ""


def _render_single_block(role: str, text: str, tool_section: str) -> str | None:
    """Render one turn block; return None when the turn should be skipped."""
    if tool_section and role == "assistant" and _is_poisoned_assistant_reply(text):
        return None
    if tool_section and role in ("system", "developer"):
        condensed = _condense_harness_system(text)
        if not condensed:
            return None
        return f"[Project Guidelines]\n{condensed}"
    label = ROLE_LABELS.get(role, role.title())
    return f"[{label}]\n{text}"


def _final_instruction(tool_section: str, tool_reminder: str = "") -> str:
    """Build the closing instruction block."""
    if tool_section:
        instruction = (
            "### Next Step\n"
            "Continue the workflow above. Use any `Client Execution Output` results pasted above "
            "as ground truth, emit the next ```action JSON block(s) if further inspection, edits, "
            "or tests are needed, or write the final summary once everything requested is verified."
        )
        if tool_reminder:
            instruction = f"{instruction}\n{tool_reminder}"
        return instruction
    return "[Instruction]\nContinue the conversation above as the Assistant."


_MAX_TOOL_PROMPT_CHARS = 48_000
_MAX_RECENT_TOOL_CHARS = 12_000
_MAX_OLD_TOOL_CHARS = 600


def _clip_middle(text: str, max_chars: int) -> str:
    """Keep head and tail of an oversized block within ``max_chars``."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head - 40
    omitted = len(text) - head - max(0, tail)
    return f"{text[:head]}\n... [truncated {omitted} chars] ...\n{text[-tail:]}" if tail > 0 else text[:max_chars]


def _collect_head_and_tail(middle: list[str]) -> tuple[list[str], list[str], str]:
    """Collect header blocks and tail instruction from middle."""
    head: list[str] = []
    if middle and middle[0].startswith("### Available Tools"):
        head.append(middle.pop(0))
    while middle and middle[0].startswith("[Project Guidelines]"):
        head.append(middle.pop(0))
    tail_instr = middle.pop() if middle and middle[-1].startswith("### Next Step") else ""
    return head, middle, tail_instr


def _keep_prior_within_budget(
    prior: list[str], used: int, budget: int
) -> tuple[list[str], int]:
    """Keep prior turns within remaining budget, returning kept blocks and final used count."""
    kept_prior: list[str] = []
    omitted = 0
    for b in reversed(prior):
        clipped = _clip_middle(b, _MAX_OLD_TOOL_CHARS) if b.startswith("Client Execution Output") else b
        if used + len(clipped) + 2 <= budget:
            kept_prior.append(clipped)
            used += len(clipped) + 2
        else:
            omitted += 1
    kept_prior.reverse()
    if omitted:
        kept_prior.insert(0, f"[Earlier conversation: {omitted} older message(s) omitted for length]")
    return kept_prior, used


def _compact_tool_blocks(blocks: list[str], budget: int) -> list[str]:
    """Keep tool header, project guidelines, current turn, and newest prior turns within budget."""
    if sum(len(b) for b in blocks) <= budget or len(blocks) <= 3:
        return blocks
    middle = list(blocks)
    head, middle, tail_instr = _collect_head_and_tail(middle)

    last_user_idx = max((i for i, b in enumerate(middle) if b.startswith("[User]\n")), default=-1)
    prior = middle[:last_user_idx] if last_user_idx > 0 else []
    current = middle[last_user_idx:] if last_user_idx >= 0 else middle

    current = [
        _clip_middle(b, _MAX_RECENT_TOOL_CHARS) if b.startswith("Client Execution Output") else b
        for b in current
    ]
    used = sum(len(b) for b in head) + sum(len(b) for b in current) + len(tail_instr) + 200

    kept_prior, used = _keep_prior_within_budget(prior, used, budget)

    out = head + kept_prior + current
    if tail_instr:
        out.append(tail_instr)
    return out


def _render_blocks(turns: list[tuple[str, str]], tool_section: str,
                   grounding_section: str,
                   tool_reminder: str = "") -> list[str]:
    """Render the full multi-turn prompt block list."""
    blocks: list[str] = []
    if tool_section:
        blocks.append(tool_section)
    if grounding_section:
        blocks.append(f"### Local Workspace Context\n{grounding_section}" if tool_section
                      else f"[System: Local Workspace Context]\n{grounding_section}")

    for role, text in turns:
        block = _render_single_block(role, text, tool_section)
        if block is not None:
            blocks.append(block)

    blocks.append(_final_instruction(tool_section, tool_reminder))
    if tool_section:
        return _compact_tool_blocks(blocks, _MAX_TOOL_PROMPT_CHARS)
    return blocks


def flatten_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> FlatPrompt:
    images: list[str] = []
    action_style = bool(tools)
    turns: list[tuple[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = _message_to_text(msg, images, action_style=action_style).strip()
        if text:
            turns.append((role, text))

    if not turns:
        return FlatPrompt(text="", images=images)

    tool_section = format_tools_prompt(tools) if tools else ""
    tool_reminder = format_tool_reminder(tools) if tools else ""
    grounding_section = "" if tools else get_workspace_grounding(messages)

    if _is_verbatim_single_turn(turns, tool_section, grounding_section):
        return FlatPrompt(text=turns[0][1], images=images)

    return FlatPrompt(
        text="\n\n".join(_render_blocks(turns, tool_section, grounding_section, tool_reminder)),
        images=images,
    )
