import logging
import os
import traceback
from typing import Any, Dict, List, Literal

from rich import box
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment
from ssa.tools.openai.shell import (
    _is_apply_patch_command,
    _emulate_apply_patch_in_command,
    print_execution_result,
)
from ssa.tools.utils import truncate_command

logger = logging.getLogger(__name__)
console = Console()

MAX_LINES_LIMIT: int = 250
MAX_CHARS_LIMIT: int = 20_000


TOOL_SPEC = {
    "name": "batch_shell",
    "description": (
        "Run one or more bash commands in a single tool call.\n"
        "\n"
        "When to pack multiple inputs in one call (PREFERRED whenever possible):\n"
        "* Independent exploration steps whose next command does not depend on the previous output, e.g. layout discovery: `ls -la`, `cat README.md`, `find . -name pyproject.toml`, `git log --oneline -20`.\n"
        "* Running several unrelated test files or check commands together, e.g. `pytest tests/test_a.py`, `pytest tests/test_b.py`, `ruff check src/`.\n"
        "* Reading multiple files or inspecting multiple symbols at once, e.g. `sed -n '1,80p' a.py`, `sed -n '1,80p' b.py`, `grep -n foo src/`.\n"
        "\n"
        "When to use a SINGLE input:\n"
        "* The next command genuinely depends on the previous output (e.g. you need to see a file before deciding what to edit).\n"
        "* You are running a single long command or a script.\n"
        "\n"
        "Example of batched call (one tool call, three inputs):\n"
        "  inputs = [\n"
        "    {\"input\": \"ls -la /app\",                 \"description\": \"top-level layout\"},\n"
        "    {\"input\": \"cat /app/README.md\",          \"description\": \"project overview\"},\n"
        "    {\"input\": \"find /app -name pyproject.toml\", \"description\": \"locate package config\"},\n"
        "  ]\n"
        "\n"
        "Other notes:\n"
        "* Commands are executed sequentially in the order provided.\n"
        "* Each command runs in the same shell environment and shares state (working directory, environment variables).\n"
        "* Set ignore_errors=false on a command only when a failure should stop the remaining commands.\n"
        "* The contents of each command do NOT need to be XML-escaped.\n"
        "* State is persistent across command calls and discussions with the user.\n"
        "* Avoid commands that may produce a very large amount of output.\n"
        "* Run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "array",
                    "description": "List of shell commands to execute. Each entry is an object with 'input', 'description', and optional 'timeout' and 'ignore_errors' fields.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "The shell command to run.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Why this command is being run.",
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Optional timeout in seconds for the command (default: 10, max: 600).",
                            },
                            "ignore_errors": {
                                "type": "boolean",
                                "description": "Set to false only when commands must run sequentially and a failure should stop execution. Default true.",
                            },
                        },
                        "required": ["description", "input"],
                    },
                },
            },
            "required": ["inputs"],
        }
    },
}


def format_batch_preview(inputs: List[Dict]) -> Panel:
    """Create rich preview panel for batch command execution."""
    details = Table(show_header=True, box=box.SIMPLE)
    details.add_column("#", style="cyan", justify="right", width=3)
    details.add_column("Command", style="green")
    details.add_column("Timeout", style="yellow", justify="right", width=8)

    for i, cmd_spec in enumerate(inputs):
        cmd = cmd_spec.get("input", "")
        timeout = cmd_spec.get("timeout", 10)
        details.add_row(str(i + 1), Syntax(cmd, "bash", theme="monokai", line_numbers=False), f"{timeout}s")

    return Panel(
        details,
        title=f"[bold blue]🚀 Batch Execution Plan ({len(inputs)} commands)",
        border_style="blue",
        box=ROUNDED,
    )


def format_batch_summary(results: List[Dict]) -> Panel:
    """Format batch execution summary as a rich panel."""
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = len(results) - success_count

    summary_table = Table(show_header=False, box=box.SIMPLE)
    summary_table.add_column("Property", style="cyan", justify="right")
    summary_table.add_column("Value")

    summary_table.add_row("Total Commands", f"{len(results)}")
    summary_table.add_row("Successful", f"[green]{success_count}[/green]")
    summary_table.add_row("Failed", f"[red]{error_count}[/red]")

    status = "success" if error_count == 0 else "warning" if error_count < len(results) else "error"
    icons = {"success": "✅", "warning": "⚠️", "error": "❌"}
    colors = {"success": "green", "warning": "yellow", "error": "red"}

    return Panel(
        summary_table,
        title=f"[bold {colors[status]}]{icons[status]} Batch Execution Summary",
        border_style=colors[status],
        box=ROUNDED,
    )


def _clip_output(output: str, command: str, max_lines: int = MAX_LINES_LIMIT, max_chars: int = MAX_CHARS_LIMIT) -> str:
    """Clip output that exceeds line or character limits."""
    lines = output.splitlines()
    if len(lines) > max_lines:
        logger.info(f"Clipping shell output for command: {command} due to lines len ({len(lines)}) exceeding limit ({max_lines})")
        clipped_lines = len(lines) - max_lines
        output = (
            "\n".join(lines[:max_lines // 2])
            + f"\n\n< ... {clipped_lines} lines clipped ... >\n\n"
            + "\n".join(lines[-max_lines // 2:])
        )
    if len(output) > max_chars:
        logger.info(f"Clipping shell output for command: {command} due to chars len ({len(output)}) exceeding limit ({max_chars})")
        output = output[:max_chars] + "\n<response clipped>"
    return output


def batch_shell(tool: ToolUse, **kwargs: Any) -> ToolResult:

    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    environment: Environment = kwargs["environment"]
    verbose = kwargs.get("verbose", True)
    show_panel: bool = kwargs.pop("show_panel", True)

    _tool_cfg = kwargs.get("tool_params", {}).get("openai.batch_shell", {})
    _default_timeout = _tool_cfg.get("timeout", None) or environment.timeout or 10
    publish_partial_output: bool = _tool_cfg.get("publish_partial_output", True)
    max_chars_limit: int = _tool_cfg.get("max_chars_limit", MAX_CHARS_LIMIT)
    max_lines_limit: int = _tool_cfg.get("max_lines_limit", MAX_LINES_LIMIT)

    inputs = tool_input.get("inputs")
    if not inputs or not isinstance(inputs, list):
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "inputs is required and must be a non-empty array"}],
        }

    workdir = environment.workdir or os.getcwd()
    all_outputs: List[str] = []
    all_results: List[Dict] = []
    overall_success = True

    if show_panel:
        console.print(format_batch_preview(inputs))

    for i, cmd_spec in enumerate(inputs):
        cmd = cmd_spec.get("input")
        desc = cmd_spec.get("description", "")
        timeout = min(
            float(cmd_spec.get("timeout", _default_timeout)),
            600.,
        )
        ignore_errors = bool(cmd_spec.get("ignore_errors", True))

        if not cmd:
            all_outputs.append(f"--- Command {i + 1}: (empty, skipped) ---")
            continue

        try:
            original_cmd = cmd
            patch_error: str | None = None
            if _is_apply_patch_command(cmd):
                logger.info("Emulating apply_patch heredoc inline in batch_shell command")
                _kwargs = {**kwargs}
                if not kwargs.get("tool_params", {}.get("openai.apply_patch", {})):
                    _tool_params = kwargs.get("tool_params", {})
                    tool_params = {
                        **_tool_params,
                        **{
                            "openai.apply_patch": _tool_cfg,
                        }
                    }
                    _kwargs["tool_params"] = tool_params
                cmd, patch_error = _emulate_apply_patch_in_command(
                    cmd, tool_use_id, desc, _kwargs,
                )
            if patch_error is not None:
                logger.info("apply_patch failed; aborting batch_shell command without execution")
                result = {
                    "status": "error",
                    "exit_code": 1,
                    "command": original_cmd,
                    "output": "",
                    "error": patch_error,
                }
            else:
                result = environment.execute_bash(cmd, workdir, timeout, verbose)
                if cmd != original_cmd:
                    result["command"] = original_cmd
            all_results.append(result)

            is_success = result.get("status", "") == "success"
            if not publish_partial_output and result['exit_code'] == 124:
                result['output'] = ""
            execution_output = (
                (f"Command timed-out with limit {timeout} sec\n" if result['exit_code'] == 124 else "")
                + f"Command: {truncate_command(result['command'])}\n"
                f"Status: {result['status']}\n"
                f"Exit Code: {result['exit_code']}\n"
                f"Output: {result['output']}\n"
                "Error: " + (f'{result["error"]}' if result["error"] else 'None')
            )
            execution_output = _clip_output(execution_output, cmd, max_lines_limit, max_chars_limit)
            all_outputs.append(execution_output)

            if show_panel:
                print(f"\n  [{i + 1}/{len(inputs)}]")
                print_execution_result(result)

            if not is_success and not ignore_errors:
                overall_success = False
                skipped = len(inputs) - i - 1
                if skipped > 0:
                    all_outputs.append(f"\n--- {skipped} remaining command(s) skipped due to failure ---")
                break

        except Exception as e:
            logger.warning(traceback.format_exc())
            all_outputs.append(f"--- Command {i + 1}: {desc} ---\nShell error: {str(e)}")
            if show_panel:
                print(f"\n  [{i + 1}/{len(inputs)}] ✗ {cmd}")
                print(f"  Error: {str(e)}\n")
            if not ignore_errors:
                overall_success = False
                skipped = len(inputs) - i - 1
                if skipped > 0:
                    all_outputs.append(f"\n--- {skipped} remaining command(s) skipped due to error ---")
                break

    if show_panel and all_results:
        console.print(format_batch_summary(all_results))

    combined_output = "\n\n".join(all_outputs)
    status: Literal["success", "error"] = "success" if overall_success else "error"

    return {"toolUseId": tool_use_id, "status": status, "content": [{"text": combined_output}]}
