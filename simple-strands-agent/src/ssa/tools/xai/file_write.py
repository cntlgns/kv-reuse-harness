import os
import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel
from strands.types.tools import ToolResult, ToolUse

from ssa.environments.environment import Environment
from ssa.tools.utils import create_rich_panel

LOG = logging.getLogger(__name__)
console = Console()
MAX_CHARS_LIMIT: int = 20_000


TOOL_SPEC = {
    "name": "file_write",
    "description": (
        "Tool for creating new files.\n"
        "* Creates a new file at the given path with the provided content.\n"
        "* Fails if the file already exists — use `file_edit` to modify existing files.\n"
        "* The parent directory must already exist."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "path": {
                    "description": "Absolute path for the new file, e.g. `/repo/new_file.py`.",
                    "type": "string"
                },
                "file_text": {
                    "description": "The full content of the file to be created.",
                    "type": "string"
                },
                "description": {
                    "description": "Why I'm creating this file",
                    "type": "string"
                },
            },
            "required": ["path", "file_text"]
        }
    }
}


def file_write(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Create a new file with the given content."""
    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})

    show_panel: bool = kwargs.pop("show_panel", True)
    environment: Environment = kwargs["environment"]
    workdir = environment.workdir or os.getcwd()

    _tool_cfg = kwargs.get("tool_params", {}).get("xai.file_write", {})
    restrict_workspace_create: bool = _tool_cfg.get(
        "restrict_workspace_create", kwargs.get("restrict_workspace_create", True)
    )
    allow_overwrite: bool = _tool_cfg.get(
        "allow_overwrite", kwargs.get("allow_overwrite", False)
    )

    try:
        path: str = tool_input.get("path", "")
        if not path:
            raise ValueError("path parameter is required")

        file_text = tool_input.get("file_text", None)
        if file_text is None:
            raise ValueError("file_text is required")

        if not allow_overwrite and environment.file_exists(path):
            raise ValueError(
                f"Path {path} already exists. Use `file_edit` to modify existing files."
            )
        
        if restrict_workspace_create and workdir is not None and not path.startswith(workdir):
            raise ValueError(
                f"Path `{path}` is not within current workspace: {workdir}. "
                "Only create new files within workspace directory by providing full absolute paths."
            )

        if not os.path.isabs(path):
            raise ValueError(
                f"Path `{path}` is not an absolute path. Provide the full absolute path, e.g. `/repo/src/file.py`."
            )

        if not environment.dir_exists(os.path.dirname(path)):
            raise ValueError(f"Directory containing provided path: {path} does not exist.")

        if show_panel:
            view_panel = create_rich_panel(file_text, f"📄 {os.path.basename(path)}", path)
            console.print(f"Valid file workdir: {workdir}")
            try:
                console.print(view_panel)
            except Exception as e:
                LOG.warning(f"Failed to render file contents:\n{file_text}\nDetails: {e}")

        environment.write_file(file_text, path)
        result = f"New file created with contents from file_text.\nFile: {path}"

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
