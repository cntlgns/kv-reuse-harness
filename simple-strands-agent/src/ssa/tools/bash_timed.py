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
from ssa.tools.utils import truncate_command

# Initialize logging and set paths
logger = logging.getLogger(__name__)
console = Console()

# Regex to strip ANSI escape sequences (colors, bold, etc.) that cause Rich rendering hangs
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

MAX_LINES_LIMIT: int = 250
MAX_CHARS_LIMIT: int = 20_000
DEFAULT_BASH_TIMEOUT: float = 120 # used only when environment/model does not provide any timeout


TOOL_SPEC = {
    "name": "bash_timed",
    "description": "Run commands in a bash shell\n"
    "* When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.\n"
    "* You don't have access to the internet via this tool.\n"
    "* You do have access to a mirror of common linux and python packages via apt and pip.\n"
    "* State is persistent across command calls and discussions with the user.\n"
    "* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n"
    "* Please avoid commands that may produce a very large amount of output.\n"
    "* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                 "description": {
                    "type": "string",
                    "description": "Why I am running this bash command",
                },
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds for the command. Set this based on the expected "
                        "duration of the command. For instant commands like `ls`, `cat`, `echo`, `grep`, use 5. "
                        "For short commands like `find`, `git diff`, `git log`, use 15. "
                        "For moderate commands like `pip install`, `git clone`, use 60. "
                        "For long-running commands like test suites, builds, or large downloads, use 200 or more. "
                        "If unsure, default to 120."
                    )
                }
            },
            "required": ["description","command", "timeout"],
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

def bash_timed(tool: ToolUse, **kwargs: Any) -> ToolResult:

    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    verbose = kwargs.get("verbose", True)
    environment: Environment = kwargs["environment"]
    show_panel: bool = kwargs.pop("show_panel", True)

    # Per-tool config from YAML (agent.tools.bash_timed)
    _tool_cfg = kwargs.get("tool_params", {}).get("bash_timed", {})
    _default_timeout = _tool_cfg.get("timeout", None) or environment.timeout or DEFAULT_BASH_TIMEOUT
    publish_partial_output: bool = _tool_cfg.get("publish_partial_output", True)
    max_chars_limit: int = _tool_cfg.get("max_chars_limit", MAX_CHARS_LIMIT)
    max_lines_limit: int = _tool_cfg.get("max_lines_limit", MAX_LINES_LIMIT)

    # Extract and validate parameters
    command = tool_input.get("command")
    if command is None:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Command is required"}],
        }

    _timeout = float(tool_input.get("timeout", _default_timeout))
    timeout = min(_timeout, 600)
    if tool_input.get("timeout"):
        logger.info(f"Received manual timeout of {_timeout} sec, overriding default")
        if _timeout > 600:
            logger.info("model suggested timeout exceed the permissible limit of 600s, capping the timeout")
    ignore_errors = bool(tool_input.get("ignore_errors", False))
    workdir = environment.workdir or os.getcwd()

    # Only show UI elements in interactive mode
    if show_panel:
        # Show command previews
        console.print("\n[bold blue]Command Execution Plan[/bold blue]\n")

        # Show preview for each command
        console.print(format_command_preview(command))

    try:
        if show_panel:
            console.print("\n[bold green]⏳ Starting Command Execution...[/bold green]\n")
        
        result = environment.execute_bash(command, workdir, timeout, verbose)
        if show_panel:
            print("\n✅ Command Execution Complete\n")
            print_execution_result(result)

        # Process results for tool output
        is_success = result.get("status", "") == "success"

        if not publish_partial_output and result['exit_code'] == 124:
            result['output'] = ""
        execution_output = (
            (f"Command timed-out with limit {timeout} sec\n" if result['exit_code']==124 else "")
            +
            f"Command: {truncate_command(result['command'])}\n"
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
