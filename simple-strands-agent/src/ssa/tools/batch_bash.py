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
from ssa.tools.bash import print_execution_result
from ssa.tools.utils import truncate_command

logger = logging.getLogger(__name__)
console = Console()

MAX_LINES_LIMIT: int = 250
MAX_CHARS_LIMIT: int = 20_000


TOOL_SPEC = {
    "name": "batch_bash",
    "description": (
        "* Run multiple bash commands in a single tool call. "
        "* Use this tool to execute several independent commands at once instead of calling bash multiple times. "
        "* Commands are executed sequentially in the order provided. "
        "* Each command runs in the same shell environment and shares state (working directory, environment variables). "
        " - If you need strict sequential execution where a failure stops remaining commands, set ignore_errors=false.\n"
        "* When invoking this tool, the contents of each command does NOT need to be XML-escaped.\n"
        "* You don't have access to the internet via this tool.\n"
        "* You do have access to a mirror of common linux and python packages via apt and pip.\n"
        "* State is persistent across command calls and discussions with the user.\n"
        "* Please avoid commands that may produce a very large amount of output.\n"
        "* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "description": "List of bash commands to execute. Each command is an object with 'command', 'description', and optional 'timeout' and 'ignore_errors' fields.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to run.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Why this command is being run.",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Optional timeout in seconds for the command. Defaults to 120 seconds.",
                            },
                            "ignore_errors": {
                                "type": "boolean",
                                "description": "Set to false only when commands must run sequentially and a failure should stop execution. Default true. ",
                            },
                        },
                        "required": ["description", "command"],
                    },
                },
            },
            "required": ["commands"],
        }
    },
}


def format_batch_preview(commands: List[Dict]) -> Panel:
    """Create rich preview panel for batch command execution."""
    details = Table(show_header=True, box=box.SIMPLE)
    details.add_column("#", style="cyan", justify="right", width=3)
    details.add_column("Command", style="green")
    details.add_column("Timeout", style="yellow", justify="right", width=8)

    for i, cmd_spec in enumerate(commands):
        cmd = cmd_spec.get("command", "")
        timeout = cmd_spec.get("timeout", 120)
        details.add_row(str(i + 1), Syntax(cmd, "bash", theme="monokai", line_numbers=False), f"{timeout}s")

    return Panel(
        details,
        title=f"[bold blue]🚀 Batch Execution Plan ({len(commands)} commands)",
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


def batch_bash(tool: ToolUse, **kwargs: Any) -> ToolResult:

    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    environment: Environment = kwargs["environment"]
    verbose = kwargs.get("verbose", True)
    show_panel: bool = kwargs.pop("show_panel", True)

    # Per-tool config from YAML (agent.tools.batch_bash)
    _tool_cfg = kwargs.get("tool_params", {}).get("batch_bash", {})
    _default_timeout = _tool_cfg.get("timeout", None) or environment.timeout or 120
    publish_partial_output: bool = _tool_cfg.get("publish_partial_output", True)
    max_chars_limit: int = _tool_cfg.get("max_chars_limit", MAX_CHARS_LIMIT)
    max_lines_limit: int = _tool_cfg.get("max_lines_limit", MAX_LINES_LIMIT)

    commands = tool_input.get("commands")
    if not commands or not isinstance(commands, list):
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "commands is required and must be a non-empty array"}],
        }

    workdir = environment.workdir or os.getcwd()
    all_outputs: List[str] = []
    all_results: List[Dict] = []
    overall_success = True

    if show_panel:
        console.print(format_batch_preview(commands))

    for i, cmd_spec in enumerate(commands):
        cmd = cmd_spec.get("command")
        desc = cmd_spec.get("description", "")
        timeout = int(cmd_spec.get("timeout", _default_timeout))
        ignore_errors = bool(cmd_spec.get("ignore_errors", True))

        if not cmd:
            all_outputs.append(f"--- Command {i + 1}: (empty, skipped) ---")
            continue

        try:
            result = environment.execute_bash(cmd, workdir, timeout, verbose)
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
                print(f"\n  [{i + 1}/{len(commands)}]")
                print_execution_result(result)

            if not is_success and not ignore_errors:
                overall_success = False
                skipped = len(commands) - i - 1
                if skipped > 0:
                    all_outputs.append(f"\n--- {skipped} remaining command(s) skipped due to failure ---")
                break

        except Exception as e:
            logger.warning(traceback.format_exc())
            all_outputs.append(f"--- Command {i + 1}: {desc} ---\nBash shell error: {str(e)}")
            if show_panel:
                print(f"\n  [{i + 1}/{len(commands)}] ✗ {cmd}")
                print(f"  Error: {str(e)}\n")
            if not ignore_errors:
                overall_success = False
                skipped = len(commands) - i - 1
                if skipped > 0:
                    all_outputs.append(f"\n--- {skipped} remaining command(s) skipped due to error ---")
                break

    if show_panel and all_results:
        console.print(format_batch_summary(all_results))

    combined_output = "\n\n".join(all_outputs)
    status: Literal["success", "error"] = "success" if overall_success else "error"

    return {"toolUseId": tool_use_id, "status": status, "content": [{"text": combined_output}]}
