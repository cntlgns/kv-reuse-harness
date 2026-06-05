import logging
import os
import re
import traceback
from typing import Any, Dict, List, Literal, Union

from rich import box
from rich.box import ROUNDED
from rich.panel import Panel
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.markup import escape as rich_escape
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment
from ssa.tools.openai.apply_patch import apply_patch as _apply_patch_handler

from ssa.tools.utils import truncate_command

# Initialize logging and set paths
logger = logging.getLogger(__name__)
console = Console()

# Regex to strip ANSI escape sequences (colors, bold, etc.) that cause Rich rendering hangs
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

MAX_LINES_LIMIT: int = 250
MAX_CHARS_LIMIT: int = 20_000


TOOL_DESCRIPTION="""Run commands in a bash shell
* When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* State is persistent across command calls and discussions with the user.
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
                "description": {
                    "type": "string",
                    "description": "Why I am running this bash command",
                },
                "input": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in seconds for the command. "
                        "Must be set conservatively based on the expected runtime of the command. "
                        "A premature timeout kills in-progress work, wastes prior compute, and forces a retry. "
                        "If the command runs a script, inspect or reason about its internal steps (waits, retries, nested timeouts) and set the outer timeout to comfortably exceed the total expected duration. "
                        "Reference ranges: "
                        "5-10s for instant commands (ls, cat, chmod, echo, which). "
                        "30-60s for installs, short builds, single network requests. "
                        "120-180s for test suites, multi-step builds, moderate network operations. "
                        "300-600"
                },
            },
            "required": ["description", "input"],
        }
    },
}


def format_command_preview(command: Union[str, Dict]) -> Panel:
    """Create rich preview panel for command execution."""
    details = Table(show_header=False, box=box.SIMPLE)
    details.add_column("Property", style="cyan", justify="right")
    details.add_column("Value", style="green")

    # Format command info
    cmd_str = command if isinstance(command, str) else command.get("command", "")
    details.add_row("🔷 Command", Syntax(cmd_str, "bash", theme="monokai", line_numbers=False))

    return Panel(
        details,
        title="[bold blue]🚀 Command Execution Preview",
        border_style="blue",
        box=ROUNDED,
    )

def _sanitize_for_rich(text: str) -> str:
    """Strip ANSI escape sequences and escape Rich markup to prevent rendering hangs."""
    text = _ANSI_ESCAPE_RE.sub('', text)
    return rich_escape(text)

def format_execution_result(result: Dict[str, Any]) -> Panel:
    """Format command execution result as a rich panel."""
    result_table = Table(show_header=False, box=box.SIMPLE)
    result_table.add_column("Property", style="cyan", justify="right")
    result_table.add_column("Value")

    # Status with appropriate styling
    status_style = "green" if result["status"] == "success" else "red"
    status_icon = "✓" if result["status"] == "success" else "✗"

    result_table.add_row(
        "Status",
        f"[{status_style}]{status_icon} {result['status'].capitalize()}[/{status_style}]",
    )
    result_table.add_row("Exit Code", f"{result['exit_code']}")

    # Sanitize command for Rich (commands can also contain brackets)
    result_table.add_row("Command", _sanitize_for_rich(result["command"]))

    # Output (truncate if too long)
    output = _sanitize_for_rich(result["output"])
    if len(output.splitlines()) > 20:
        output = "\n".join(output.splitlines()[:20]) + "...\n[dim](output truncated)[/dim]"
    result_table.add_row("Output", output)

    # Error (if any)
    if result["error"]:
        result_table.add_row("Error", f"[red]{_sanitize_for_rich(result['error'])}[/red]")

    border_style = "green" if result["status"] == "success" else "red"
    icon = "🟢" if result["status"] == "success" else "🔴"

    return Panel(
        result_table,
        title=f"[bold {border_style}]{icon} Command Result",
        border_style=border_style,
        box=ROUNDED,
    )

def print_execution_result(result: Dict[str, Any]) -> None:
    """Print command execution result using plain print(). Cannot hang on any input."""
    status_icon = "✓" if result["status"] == "success" else "✗"
    print(f"  Status: {status_icon} {result['status']}  |  Exit Code: {result['exit_code']}")
    print(f"  Command: {result['command']}")
    output_lines = result["output"].splitlines()
    if len(output_lines) > 20:
        preview = "\n    ".join(output_lines[:10] + ["...", f"({len(output_lines)} lines total)"] + output_lines[-5:])
    else:
        preview = "\n    ".join(output_lines) if output_lines else "(empty)"
    print(f"  Output:\n    {preview}")
    if result["error"]:
        print(f"  Error: {result['error']}")
    print()

def format_summary(results: List[Dict[str, Any]], parallel: bool) -> Panel:
    """Format execution summary as a rich panel."""
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count

    summary_table = Table(show_header=False, box=box.SIMPLE)
    summary_table.add_column("Property", style="cyan", justify="right")
    summary_table.add_column("Value")

    summary_table.add_row("Total Commands", f"{len(results)}")
    summary_table.add_row("Successful", f"[green]{success_count}[/green]")
    summary_table.add_row("Failed", f"[red]{error_count}[/red]")
    summary_table.add_row("Execution Mode", "Parallel" if parallel else "Sequential")

    status = "success" if error_count == 0 else "warning" if error_count < len(results) else "error"
    icons = {"success": "✅", "warning": "⚠️", "error": "❌"}
    colors = {"success": "green", "warning": "yellow", "error": "red"}

    return Panel(
        summary_table,
        title=f"[bold {colors[status]}]{icons[status]} Execution Summary",
        border_style=colors[status],
        box=ROUNDED,
    )

def _find_apply_patch_heredoc(command: str):
    """Find apply_patch heredoc in a command string.
    Returns a regex match object if found, None otherwise.
    """
    return re.search(r'apply_patch\s*<<-?\s*[\'"]?(\w+)[\'"]?', command)

def _is_apply_patch_command(command: str) -> bool:
    """Check if a bash command is an apply_patch heredoc invocation."""
    return _find_apply_patch_heredoc(command) is not None and "*** Begin Patch" in command

def _find_apply_patch_block(command: str):
    """Locate an apply_patch heredoc block within ``command``.

    Returns ``(block_start, block_end, patch_body, same_line_suffix)`` where:
      * ``block_start`` is the offset of the ``apply_patch`` keyword.
      * ``block_end`` is the offset just past the closing delimiter line
        (including its trailing newline if any).
      * ``patch_body`` is the patch text between the two delimiter lines.
      * ``same_line_suffix`` is whatever follows the heredoc marker on the same
        line as ``apply_patch <<DELIM`` (e.g. ``" && echo ssa"`` when the model
        writes ``apply_patch <<'PATCH' && echo ssa``).

    Returns ``None`` if no well-formed heredoc block is present.
    """
    match = _find_apply_patch_heredoc(command)
    if not match:
        return None
    delimiter = match.group(1)
    heredoc_line_end = command.find("\n", match.start())
    if heredoc_line_end == -1:
        return None

    same_line_suffix = command[match.end():heredoc_line_end]
    remainder_start = heredoc_line_end + 1
    lines = command[remainder_start:].split("\n")
    delim_idx: int | None = None
    for i, line in enumerate(lines):
        if line == delimiter or line.strip() == delimiter:
            delim_idx = i
            break
    if delim_idx is None:
        return None

    patch_body = "\n".join(lines[:delim_idx]).strip()
    delim_line_start = remainder_start + sum(len(line) + 1 for line in lines[:delim_idx])
    delim_line_end = delim_line_start + len(lines[delim_idx])
    if delim_line_end < len(command) and command[delim_line_end] == "\n":
        block_end = delim_line_end + 1
    else:
        block_end = delim_line_end
    return match.start(), block_end, patch_body, same_line_suffix

def _build_apply_patch_emulation(output: str) -> str:
    """Build a shell snippet that emits ``output`` on stdout and exits 0.

    A subshell is used so the block can chain with ``&&`` / ``||`` on the
    surrounding line (the subshell's exit status becomes the block's exit
    status; ``cat`` on a heredoc always exits 0).
    """
    if not output:
        return "(:)"
    sentinel = "APPLY_PATCH_EMU_EOF"
    i = 0
    output_lines = output.split("\n")
    while any(line == sentinel for line in output_lines):
        i += 1
        sentinel = f"APPLY_PATCH_EMU_EOF_{i}"
    return (
        f"( cat <<'{sentinel}'\n"
        f"{output}\n"
        f"{sentinel}\n"
        f")"
    )

def _emulate_apply_patch_in_command(
    command: str,
    tool_use_id: str,
    description: str,
    kwargs: Dict[str, Any],
) -> tuple[str, str | None]:
    """Apply every apply_patch heredoc in ``command`` via the apply_patch tool
    and replace each block in-place with a shell snippet that reproduces its
    output on stdout.

    Returns ``(rewritten_command, None)`` when all patches apply successfully.
    On the first patch failure, returns ``(command_at_that_point, error_msg)``
    without further rewriting — callers should short-circuit and skip shell
    execution so that commands following a failed patch never run.
    """
    max_iters = 10  # guard against pathological inputs
    for _ in range(max_iters):
        if not _is_apply_patch_command(command):
            return command, None
        block = _find_apply_patch_block(command)
        if block is None:
            return command, None
        start, end, patch_body, same_line_suffix = block
        if not patch_body:
            return command, None

        patch_tool: ToolUse = {
            "toolUseId": tool_use_id,
            "name": "apply_patch",
            "input": {
                "patch": patch_body,
                "description": description or "apply_patch via shell redirect",
            },
        }
        patch_result = _apply_patch_handler(patch_tool, **kwargs)
        output_text = "".join(
            block_item.get("text", "")
            for block_item in (patch_result.get("content") or [])
            if isinstance(block_item, dict)
        )
        if patch_result.get("status") != "success":
            return command, output_text or "apply_patch failed"

        snippet = _build_apply_patch_emulation(output_text)
        replacement = snippet + same_line_suffix
        after = command[end:]
        if after and not replacement.endswith("\n"):
            replacement += "\n"
        command = command[:start] + replacement + after
    return command, None

def shell(tool: ToolUse, **kwargs: Any) -> ToolResult:

    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    verbose = kwargs.get("verbose", True)
    environment: Environment = kwargs["environment"]
    show_panel: bool = kwargs.pop("show_panel", True)

    # Extract and validate parameters
    command = tool_input.get("input")
    if command is None:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "input is required"}],
        }

    _timeout = int(tool_input.get("timeout", 120))
    timeout = min(_timeout, 600)
    ignore_errors = bool(tool_input.get("ignore_errors", False))
    workdir = environment.workdir or os.getcwd()

    # Per-tool config from YAML (agent.tools.openai.shell)
    _tool_cfg = kwargs.get("tool_params", {}).get("openai.shell", {})
    publish_partial_output: bool = _tool_cfg.get("publish_partial_output", True)
    max_chars_limit: int = _tool_cfg.get("max_chars_limit", MAX_CHARS_LIMIT)
    max_lines_limit: int = _tool_cfg.get("max_lines_limit", MAX_LINES_LIMIT)

    # Emulate apply_patch as an installed CLI by rewriting each heredoc block
    # into a snippet that echoes the captured output on stdout.
    #TODO: install tools.openai.apply_patch as cli in docker/local before runs begin
    # Will inject agent specific code in user-env
    original_command = command
    emulated_apply_patch = False
    if _is_apply_patch_command(command):
        _kwargs = {**kwargs}
        if not kwargs.get("tool_params", {}.get("openai.apply_patch", {})):
            _tool_params = kwargs.get("tool_params", {})
            _tool_cfg = kwargs.get("tool_params", {}).get("openai.shell", {})
            tool_params = {
                **_tool_params,
                **{
                    "openai.apply_patch": _tool_cfg,
                }
            }
            _kwargs["tool_params"] = tool_params
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

    # Only show UI elements in interactive mode
    if show_panel:
        # Show command previews
        console.print("\n[bold blue]Command Execution Plan[/bold blue]\n")

        # Show preview for each command
        console.print(format_command_preview(command))

    try:
        if not show_panel:
            console.print("\n[bold green]⏳ Starting Command Execution...[/bold green]\n")
        
        result = environment.execute_bash(command, workdir, timeout, verbose)
        if emulated_apply_patch:
            result["command"] = original_command
        if show_panel:
            print("\n✅ Command Execution Complete\n")
            print_execution_result(result)

        # Process results for tool output
        is_success = result.get("status", "") == "success"

        if not publish_partial_output and result['exit_code'] == 124:
            result['output'] = ""
        execution_output = (
            (f"Input command timed-out with limit {timeout} sec\n" if result['exit_code']==124 else "")
            +
            f"Input: {truncate_command(result['command'])}\n"
            f"Status: {result['status']}\n"
            f"Exit Code: {result['exit_code']}\n"
            f"Output: {result['output']}\n"
            "Error: " + (f'{result["error"]}' if result["error"] else 'None')
        )
        lines = execution_output.splitlines()
        if len(lines) > max_lines_limit:
            logger.info(f"Clipping shell output for command: {command} due to lines len ({len(lines)}) exceeding limit ({max_lines_limit})")
            clipped_lines = len(lines) - max_lines_limit
            execution_output = (
                "\n".join(lines[:max_lines_limit // 2])
                + f"\n\n< ... {clipped_lines} lines clipped ... >\n\n"
                + "\n".join(lines[-max_lines_limit//2:])
            )
        if len(execution_output) > max_chars_limit:
            logger.info(f"Clipping shell output for command: {command} due to chars len ({len(execution_output)}) exceeding limit ({max_chars_limit})")
            execution_output = execution_output[:max_chars_limit] + "\n<response clipped>"

        status: Literal["success", "error"] = (
            "success" if is_success or ignore_errors else "error"
        )

        return {"toolUseId": tool_use_id, "status": status, "content": [{"text": execution_output}]}

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
