import os
import logging
import shlex
import traceback
from typing import Any

from rich import box
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment

MAX_CHARS_LIMIT: int = 20_000
DEFAULT_MAX_RESULTS: int = 50
RG_TIMEOUT_SEC: float = 30.0

LOG = logging.getLogger(__name__)
console = Console()


TOOL_SPEC = {
    "name": "search",
    "description": (
        "A powerful search tool built on ripgrep.\n"
        "* ALWAYS use this tool for search tasks. NEVER invoke grep or rg as a Bash command.\n"
        "* `query` supports full regex syntax (ripgrep flavor), e.g. \"log.*Error\", \"function\\s+\\w+\".\n"
        "* `path` restricts the search to a specific file or directory; defaults to the working directory.\n"
        "* Returns matching lines with file path and line number, capped at `max_results` entries."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Regex pattern to search for (ripgrep syntax).",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path to a file or directory to search in, e.g. `/repo/file.py` or `/repo`. "
                        "Defaults to the current working directory."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of matching lines to return. Defaults to {DEFAULT_MAX_RESULTS}."
                    ),
                },
            },
            "required": ["query"],
        }
    },
}


def format_basic_execution_result(command: str, result: str) -> Panel:
    """Format command execution result as a rich panel."""
    result_table = Table(show_header=False, box=box.SIMPLE)
    result_table.add_column("Property", style="cyan", justify="right")
    result_table.add_column("Value")

    result_table.add_row(
        "Command",
        Syntax(command, "bash", theme="monokai", line_numbers=False),
    )

    output = result
    if len(output.splitlines()) > 20:
        output = "\n".join(output.splitlines()[:20]) + "...\n[dim](output truncated)[/dim]"
    result_table.add_row("Output", output)

    return Panel(
        result_table,
        title="[bold green]🟢 Ripgrep Command Result",
        border_style="green",
        box=ROUNDED,
    )


def search(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    show_panel: bool = kwargs.pop("show_panel", True)
    environment: Environment = kwargs["environment"]
    _tool_cfg = kwargs.get("tool_params", {}).get("xai.search", {})
    verbose: bool = _tool_cfg.get("verbose", kwargs.get("verbose", False))
    max_chars_limit: int = _tool_cfg.get("max_chars_limit", MAX_CHARS_LIMIT)

    try:
        query = tool_input.get("query")
        if not query:
            raise ValueError("query parameter is required")

        max_results = tool_input.get("max_results", DEFAULT_MAX_RESULTS)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            raise ValueError(f"max_results must be an integer, got: {max_results!r}")
        if max_results <= 0:
            raise ValueError(f"max_results must be > 0, got: {max_results}")

        env_workdir = environment.workdir or os.getcwd()
        raw_path = tool_input.get("path") or env_workdir
        # Docker exec requires an absolute cwd; resolve relative paths against the env workdir.
        path = raw_path if os.path.isabs(raw_path) else os.path.normpath(os.path.join(env_workdir, raw_path))

        if environment.dir_exists(path):
            workdir = path
        elif environment.file_exists(path):
            parent = os.path.dirname(path) or env_workdir
            workdir = parent if environment.dir_exists(parent) else env_workdir
        else:
            raise ValueError(f"Provided path: {raw_path} does not exist")

        def _build_command(literal: bool) -> str:
            return (
                f"rg {'-F ' if literal else ''}-n --no-heading -m {max_results} "
                f"-e {shlex.quote(query)} {shlex.quote(path)}"
            )

        _command = _build_command(literal=False)
        LOG.info(f"search: executing {_command!r} in workdir={workdir!r}")

        bash_result = environment.execute_bash(
            command=_command,
            workdir=workdir,
            timeout=RG_TIMEOUT_SEC,
            verbose=verbose,
        )
        raw_output = bash_result.get("output", "") or ""

        # Fall back to fixed-string (-F) mode in two cases:
        #   1. rg exited with a regex parse error (exit code 2).
        #   2. the regex parsed cleanly but returned no output — often the model
        #      passed code as a query  which requires brackets escaping
        retry_reason = None
        if bash_result.get("exit_code") == 2 and "regex parse error" in raw_output:
            retry_reason = "regex parse error"
        elif not raw_output.strip():
            retry_reason = "no matches in regex mode"

        if retry_reason is not None:
            LOG.info(
                f"search: {retry_reason} on {query!r}; retrying with -F (fixed string)"
            )
            _command = _build_command(literal=True)
            LOG.info(f"search: executing {_command!r} in workdir={workdir!r}")
            bash_result = environment.execute_bash(
                command=_command,
                workdir=workdir,
                timeout=RG_TIMEOUT_SEC,
                verbose=verbose,
            )
            raw_output = bash_result.get("output", "") or ""

        if show_panel:
            console.print("\n[bold green]✅ Ripgrep Command Execution Complete[/bold green]\n")
            try:
                console.print(format_basic_execution_result(_command, raw_output))
            except Exception as e:
                LOG.warning(f"Failed to render ripgrep panel. Details: {e}")

        lines = raw_output.splitlines()[:max_results]
        result = "\n".join(lines)
        if len(result) > max_chars_limit:
            LOG.info(
                f"Clipping search output due to chars len ({len(result)}) "
                f"exceeding limit ({max_chars_limit})"
            )
            result = result[:max_chars_limit] + "\n<response clipped>"

        if not result:
            if bash_result.get("exit_code") == 124:
                msg = (
                    f"Error: ripgrep timed out after {RG_TIMEOUT_SEC:g}s with no output. "
                    f"Narrow the path or tighten the query and retry."
                )
            else:
                msg = "Error: No matches found"
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": msg}],
            }
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result}],
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if show_panel:
            try:
                console.print(Panel(error_msg, title="[bold red]Error", border_style="red"))
            except Exception as render_err:
                LOG.warning(f"Error message failed to render. Msg={error_msg}. Details: {render_err}")
        LOG.error(f"{traceback.format_exc()}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_msg}],
        }
