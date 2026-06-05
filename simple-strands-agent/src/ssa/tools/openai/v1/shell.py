"""Shell tool variant that accepts the OpenAI ``local_shell``-style ``cmd``
argv array (e.g. ``["bash", "-lc", "ls -R"]``).

Models like ``gpt-oss-120b`` and Codex are trained against OpenAI's built-in
``shell`` tool whose input is ``{"cmd": [...]}``. They send that shape even
when our schema advertises something else, so this tool exposes the native
shape and reuses the rest of ``ssa.tools.openai.shell`` for execution,
formatting, and apply_patch heredoc emulation.
"""

import logging
import os
import shlex
import numpy as np
import traceback
from typing import Any, Dict, List, Literal

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment
from ssa.tools.openai.shell import (
    _emulate_apply_patch_in_command,
    _is_apply_patch_command,
    format_command_preview,
    print_execution_result,
)
from ssa.tools.utils import truncate_command

logger = logging.getLogger(__name__)
console = Console()

MAX_LINES_LIMIT: int = 250
MAX_CHARS_LIMIT: int = 20_000


def _clip_middle_output(text: str, max_lines: int, max_chars: int) -> str:
    """Clip the middle of ``text`` so both line and char limits hold.

    Always keeps a head + tail of lines separated by a
    ``< ... N lines clipped ... >`` marker.
    """
    lines = text.splitlines()
    n = len(lines)
    if n <= max_lines and len(text) <= max_chars:
        return text

    li = np.char.str_len(lines)
    li_cumsum = np.cumsum(li)

    def len_full_line_chunk(start: int, end: int) -> int:
        # get chunk full lines included len with start and end inclusive
        k = end - start
        if k < 0:
            return 0
        if k == 0:
            return li_cumsum[start] + 1 # add 1 for newline char
        # k-1 internal "\n" separators between joined lines.
        return (li_cumsum[end] - li_cumsum[start]) + (k - 1)
    
    lines_keep = min(n, max_lines)
    # skip binary search for now
    while lines_keep > 0:
        head_lines = lines_keep // 2
        tail_lines = lines_keep - head_lines
        clipped_lines = n - head_lines - tail_lines
        marker_char_len = len(f"\n\n< ... {clipped_lines} lines clipped ... >\n\n")
        total = len_full_line_chunk(0, head_lines-1) + marker_char_len + len_full_line_chunk(n-tail_lines, n-1)
        if total <= max_chars:
            break
        lines_keep -= 1

    head_count = lines_keep // 2
    tail_count = lines_keep - head_count
    clipped = n - lines_keep
    
    # prefix = [0]
    # for line in lines:
    #     prefix.append(prefix[-1] + len(line))

    # def joined_len(start: int, end: int) -> int:
    #     k = end - start
    #     if k <= 0:
    #         return 0
    #     # k-1 internal "\n" separators between joined lines.
    #     return (prefix[end] - prefix[start]) + (k - 1)

    # target_keep = min(n, max_lines)
    # while target_keep > 0:
    #     head_count = target_keep // 2
    #     tail_count = target_keep - head_count
    #     clipped = n - target_keep
    #     marker_len = len(f"\n\n< ... {clipped} lines clipped ... >\n\n")
    #     total = joined_len(0, head_count) + marker_len + joined_len(n - tail_count, n)
    #     if total <= max_chars:
    #         break
    #     target_keep -= 1

    # head_count = target_keep // 2
    # tail_count = target_keep - head_count
    # clipped = n - target_keep
    head = "\n".join(lines[:head_count]) if head_count > 0 else ""
    tail = "\n".join(lines[n - tail_count:]) if tail_count > 0 else ""
    return head + f"\n\n< ... {clipped} lines clipped ... >\n\n" + tail


TOOL_DESCRIPTION = """Run commands in a bash shell.
* The "cmd" field is an argv array. The conventional invocation is ["bash", "-lc", "<your command>"].
* When invoking this tool, the contents of the command do NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* State is persistent across command calls.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
"""


TOOL_SPEC = {
    "name": "shell",
    "description": TOOL_DESCRIPTION,
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Argv array for the shell command. Typically "
                        "[\"bash\", \"-lc\", \"<command>\"]."
                    ),
                },
            },
            "required": ["cmd"],
        }
    },
}


# argv[0] values we recognize as a shell launcher whose ``-c`` / ``-lc`` /
# ``-ic`` argument carries the actual command body. Detecting this lets us
# pass the body straight to ``execute_bash`` (which already wraps in bash)
# instead of double-wrapping via ``shlex.join``.
_SHELL_LAUNCHERS = {"bash", "sh", "/bin/bash", "/bin/sh"}
_SHELL_C_FLAGS = {"-c", "-lc", "-ic", "-cl", "-ci"}


def _cmd_array_to_command_string(cmd: List[str]) -> str:
    """Collapse an argv array into a single bash command string.

    For the common ``["bash", "-lc", "<body>"]`` shape we return ``<body>``
    verbatim — the env's bash invocation will wrap it again. Otherwise we
    ``shlex.join`` the argv so it can be re-parsed by bash.
    """
    if not cmd:
        return ""
    if (
        len(cmd) >= 3
        and cmd[0] in _SHELL_LAUNCHERS
        and cmd[1] in _SHELL_C_FLAGS
    ):
        # Anything after the body becomes positional args to the inner shell;
        # round-trip them via shlex so they survive re-parsing.
        body = cmd[2]
        if len(cmd) > 3:
            body = body + " " + shlex.join(cmd[3:])
        return body
    return shlex.join(cmd)


def shell(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    verbose = kwargs.get("verbose", True)
    environment: Environment = kwargs["environment"]
    show_panel: bool = kwargs.pop("show_panel", True)

    cmd = tool_input.get("cmd")
    if cmd is None:
        # Tolerate models that occasionally fall back to {"input": "..."} —
        # treat it as ``["bash", "-lc", input]`` so we still execute.
        legacy_input = tool_input.get("input")
        if isinstance(legacy_input, str) and legacy_input:
            cmd = ["bash", "-lc", legacy_input]
        else:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "cmd is required (argv array)"}],
            }

    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "cmd must be an array of strings"}],
        }

    command = _cmd_array_to_command_string(cmd)
    if not command.strip():
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "cmd resolved to empty command"}],
        }

    _timeout = int(tool_input.get("timeout", 120))
    timeout = min(_timeout, 600)
    ignore_errors = bool(tool_input.get("ignore_errors", False))
    workdir = tool_input.get("workdir") or environment.workdir or os.getcwd()

    _tool_cfg = kwargs.get("tool_params", {}).get("openai.v1.shell", {})
    publish_partial_output: bool = _tool_cfg.get("publish_partial_output", True)
    max_chars_limit: int = _tool_cfg.get("max_chars_limit", MAX_CHARS_LIMIT)
    max_lines_limit: int = _tool_cfg.get("max_lines_limit", MAX_LINES_LIMIT)

    original_command = command
    emulated_apply_patch = False
    if _is_apply_patch_command(command):
        _kwargs = {**kwargs}
        # Promote our own per-tool config to apply_patch's expected key so
        # the heredoc emulation picks up the same settings.
        _tool_params = kwargs.get("tool_params", {})
        if not _tool_params.get("openai.apply_patch"):
            _kwargs["tool_params"] = {
                **_tool_params,
                "openai.apply_patch": _tool_cfg,
            }
        logger.info("Emulating apply_patch heredoc inline in shell command")
        rewritten, patch_error = _emulate_apply_patch_in_command(
            command,
            tool_use_id,
            tool_input.get("description", ""),
            _kwargs,
        )
        if patch_error is not None:
            logger.info("apply_patch failed; aborting shell command without execution")
            error_output = (
                f"Input: {truncate_command(original_command)}\n"
                f"Status: error\n"
                f"Exit Code: 1\n"
                f"Output: \n"
                f"Error: {patch_error}"
            )
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": error_output}],
            }
        if rewritten != command:
            command = rewritten
            emulated_apply_patch = True

    if tool_input.get("timeout"):
        logger.info(f"Received manual timeout of {_timeout} sec, overriding default")
        if _timeout > 600:
            logger.info("model suggested timeout exceed the permissible limit of 600s, capping the timeout")

    if show_panel:
        console.print("\n[bold blue]Command Execution Plan[/bold blue]\n")
        console.print(format_command_preview(command))

    try:
        if not show_panel:
            console.print("\n[bold green]⏳ Starting Command Execution...[/bold green]\n")

        result: Dict[str, Any] = environment.execute_bash(command, workdir, timeout, verbose)
        if emulated_apply_patch:
            result["command"] = original_command
        if show_panel:
            print("\n✅ Command Execution Complete\n")
            print_execution_result(result)

        is_success = result.get("status", "") == "success"

        if not publish_partial_output and result["exit_code"] == 124:
            result["output"] = ""
        execution_output = (
            (f"Input command timed-out with limit {timeout} sec\n" if result["exit_code"] == 124 else "")
            + f"Input: {truncate_command(result['command'])}\n"
            f"Status: {result['status']}\n"
            f"Exit Code: {result['exit_code']}\n"
            f"Output: {result['output']}\n"
            "Error: " + (f'{result["error"]}' if result["error"] else "None")
        )
        lines_count = len(execution_output.splitlines())
        chars_count = len(execution_output)
        if lines_count > max_lines_limit or chars_count > max_chars_limit:
            logger.info(
                f"Clipping shell output for command: {command} "
                f"(lines: {lines_count}/{max_lines_limit}, "
                f"chars: {chars_count}/{max_chars_limit})"
            )
            execution_output = _clip_middle_output(
                execution_output, max_lines_limit, max_chars_limit
            )

        status: Literal["success", "error"] = (
            "success" if is_success or ignore_errors else "error"
        )

        return {
            "toolUseId": tool_use_id,
            "status": status,
            "content": [{"text": execution_output}],
        }

    except Exception as e:
        if show_panel:
            console.print(
                Panel(
                    f"[bold red]Error: {str(e)}[/bold red]",
                    title="[bold red]❌ Execution Failed",
                    border_style="red",
                    box=ROUNDED,
                )
            )
        logger.warning(traceback.format_exc())
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Bash shell error: {str(e)}"}],
        }
