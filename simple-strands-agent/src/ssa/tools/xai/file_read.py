import os
import logging
import re
import shlex
from typing import Any

from rich.console import Console
from rich.panel import Panel
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment
from ssa.tools.utils import create_rich_panel, get_tree_from_files

LOG = logging.getLogger(__name__)
console = Console()
MAX_CHARS_LIMIT: int = 20_000


TOOL_SPEC = {
    "name": "file_read",
    "description": (
        "Tool for viewing files and directories.\n"
        "* If `path` is a file, `view` displays the result of applying `cat -n`.\n"
        "  If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep.\n"
        "* Use `view_range` to display specific line ranges of a file, e.g. [11, 20] shows lines 11-20.\n"
        "* If output is too long, it will be truncated and marked with `<response clipped>`."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                    "type": "string"
                },
                "view_range": {
                    "description": (
                        "Optional parameter when `path` points to a file. If none is given, "
                        "the full file is shown. If provided, the file will be shown in the indicated line number range, "
                        "e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows "
                        "all lines from `start_line` to the end of the file."
                    ),
                    "items": {"type": "integer"},
                    "type": "array"
                },
            },
            "required": ["path"]
        }
    }
}


def _execute_bash(command: str, workdir: str, environment: Environment, verbose: bool = False) -> str:
    try:
        bash_result = environment.execute_bash(command=command, workdir=workdir, verbose=verbose)
        return bash_result.get("output", "")
    except Exception as e:
        LOG.warning(f"Bash execution failed for command: {command} in workdir: {workdir}. Details: {e}")
        return ""


def _list_files(path: str, environment: Environment, max_depth: int = 2) -> list[str]:
    command = f"find {shlex.quote(path)} -mindepth 1 -maxdepth {max_depth} -not -path '*/.*'"
    result = _execute_bash(command=command, workdir=path, environment=environment, verbose=False)
    return result.splitlines()


_LINE_TOKEN_RE = re.compile(r"line(?:no|number)?", re.IGNORECASE)


def _looks_like_line_endpoint(key: str, endpoint: str) -> bool:
    """Return True if `key` looks like a line-range endpoint of the given kind.
    """
    # Split on non-alpha AND on camelCase transitions so "endLine" → ["end", "Line"].
    parts = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", key)
    tokens = [t for t in re.split(r"[^a-zA-Z]+", parts.lower()) if t]
    if endpoint not in tokens:
        return False
    other = "end" if endpoint == "start" else "start"
    if other in tokens:  # ambiguous (e.g. "start_end")
        return False
    # bare endpoint, or paired with a line/lineno/linenumber token
    return len(tokens) == 1 or any(_LINE_TOKEN_RE.fullmatch(t) for t in tokens if t != endpoint)


def _infer_line_range(tool_input: dict) -> tuple[Any, Any, list[str]]:
    """Scan `tool_input` for keys that look like start/end line endpoints.

    Returns ``(start_value, end_value, matched_keys)``. Either value may be
    ``None`` if no matching key was found.
    """
    start_val: Any = None
    end_val: Any = None
    used: list[str] = []
    for key, val in tool_input.items():
        if val is None or key in ("path", "view_range"):
            continue
        if start_val is None and _looks_like_line_endpoint(key, "start"):
            start_val = val
            used.append(key)
        elif end_val is None and _looks_like_line_endpoint(key, "end"):
            end_val = val
            used.append(key)
    return start_val, end_val, used


def file_read(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Read/view files and directories."""
    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    include_summary: bool = kwargs.pop("include_summary", True)
    show_panel: bool = kwargs.pop("show_panel", True)
    _tool_cfg = kwargs.get("tool_params", {}).get("xai.file_read", {})
    show_numbered: bool = _tool_cfg.get("show_numbered", kwargs.get("show_numbered", True))
    max_chars_limit: int = _tool_cfg.get("max_chars_limit", MAX_CHARS_LIMIT)
    environment: Environment = kwargs["environment"]
    workdir = environment.workdir or os.getcwd()

    try:
        path: str = tool_input.get("path", "").strip()
        if not path:
            raise ValueError("path parameter is required")

        # ── Directory view ──
        if environment.dir_exists(path):
            all_files = _list_files(path, environment, max_depth=2)
            tree, tree_str = get_tree_from_files(all_files)
            result = (
                f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n"
                + tree_str
            )
            if show_panel:
                try:
                    console.print(Panel(result, title="[bold green]File Tree", border_style="blue"))
                except Exception as e:
                    LOG.warning(f"Failed to render file tree. Details: {e}")

        # ── File view ──
        elif environment.file_exists(path):
            cat_command = "cat " + ("-n " if show_numbered else "") + f"{shlex.quote(path)}"
            numbered_content = _execute_bash(cat_command, workdir=workdir, environment=environment)
            len_lines = len(numbered_content.splitlines())
            view_range = tool_input.get("view_range", None)
            if view_range is None:
                _start, _end, _used = _infer_line_range(tool_input)
                if _start is not None or _end is not None:
                    view_range = [
                        int(_start) if _start is not None else 1,
                        int(_end) if _end is not None else -1,
                    ]
                    LOG.info(
                        f"file_read: inferred view_range={view_range} from keys={_used}"
                    )

            if view_range is not None:
                start_line = view_range[0]
                end_line = view_range[1]
                if end_line == -1:
                    end_line = None

                if start_line > 0:
                    start_line -= 1  # normalize to 0-idx
                if end_line is not None:
                    end_line -= 1
                else:
                    end_line = len_lines
                if end_line is not None and start_line > end_line:
                    raise ValueError(
                        f"Incorrect `view_range`={view_range}, `start_line` must be less than `end_line`"
                    )

                lines_range = numbered_content.splitlines()[start_line : end_line + 1]
                if len(lines_range) > 0:
                    result = (
                        f"Here's the content of {path} (which has {len_lines} total lines) "
                        f"with view_range=({start_line + 1}, {end_line + 1}):\n"
                        + "\n".join(lines_range)
                    )
                elif len_lines > 0:
                    raise ValueError(
                        f"view_range is outside the file contents. The file has {len_lines} lines."
                    )
                else:
                    result = f"{path} is empty file contents"
            else:
                if show_panel:
                    view_panel = create_rich_panel(
                        numbered_content,
                        f"📄 {os.path.basename(path)}",
                        path,
                    )
                    try:
                        console.print(f"Valid file workdir: {workdir}")
                        console.print(view_panel)
                    except Exception as e:
                        LOG.warning(f"Failed to render file-contents\n{numbered_content}\nDetails: {e}")
                result = (
                    (f"Here's the content of {path} (which has {len_lines} lines):\n" if include_summary else "")
                    + numbered_content
                )
        else:
            raise ValueError(f"Provided path: {path} does not exist, make sure to use absolute path only (e.g. /repo/file.py). Current workdir: {workdir}")

        if len(result) > max_chars_limit:
            LOG.info(
                f"Clipping file_read output due to chars len ({len(result)}) exceeding limit ({max_chars_limit})"
            )
            result = result[:max_chars_limit] + "\n<response clipped>"

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
            except Exception:
                pass
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_msg}],
        }
