
import os
import logging
from typing import Any, List, Tuple, Set
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree


CODE_EXTENSIONS: Set = {
    # Configuration files
    ".xml",
    ".config",
    ".cfg",
    ".toml",
    ".ini",
    "Config",
    # Java ecosystem
    ".java",
    ".kt",  # Kotlin
    ".scala",
    ".groovy",
    # .NET ecosystem
    ".cs",  # C#
    ".vb",  # VB.NET
    ".fs",  # F#
    ".csproj",
    ".sln",
    # Python
    ".py",
    ".pyx",  # Cython
    ".pyi",  # Type stubs
    # JavaScript/TypeScript
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".mjs",
    ".cjs",
    # C/C++
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    # Other popular languages
    ".go",
    ".rs",  # Rust
    ".rb",  # Ruby
    ".php",
    ".swift",
    ".dart",
    ".r",
    ".sql",
    ".sh",
    ".bash",
    ".ps1",  # PowerShell
}


LOG = logging.getLogger(__file__)

MAX_CMD_ECHO_LEN: int = 200


def truncate_command(cmd: str, max_len: int = MAX_CMD_ECHO_LEN) -> str:
    """Truncate long commands in tool feedback to save tokens."""
    if len(cmd) <= max_len:
        return cmd
    return cmd[:max_len] + f"... ({len(cmd)} chars total)"


def detect_language(file_path: str) -> str:
    """Detect language for syntax highlighting based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".java": "java",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sh": "bash",
        ".tsx": "typescript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".php": "php",
        ".rb": "ruby",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".xml": "xml",
        ".sql": "sql",
        ".r": "r",
        ".swift": "swift",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".lua": "lua",
        ".pl": "perl",
    }
    return lang_map.get(ext, "text")


def format_output(title: str, content: Any, style: str = "default") -> Panel:
    """Format output with Rich panel."""
    panel = Panel(
        content,
        title=title,
        border_style=style,
        box=box.ROUNDED,
        expand=False,
        padding=(1, 2),
    )
    return panel

def get_file_extension(filepath: str) -> str:
    path = Path(filepath)
    return path.suffix if path.suffix else os.path.basename(filepath)

def create_rich_panel(content: str, title: str = None, file_path: str = None) -> Panel:
    """Create a Rich panel with optional syntax highlighting.

    Args:
        content (str): Content to display in panel
        title (str, optional): Panel title
        file_path (str, optional): Path to file for language detection

    Returns:
        Panel: Rich panel object
    """
    if file_path:
        language = detect_language(file_path)
        syntax = Syntax(content, language, theme="monokai", line_numbers=True)
        content_for_panel = syntax
    else:
        content_for_panel = Text(content)

    return Panel(
        content_for_panel,
        title=f"[bold green]{title}" if title else None,
        border_style="blue",
        box=box.DOUBLE,
        expand=False,
        padding=(1, 2),
    )

def create_tree_string(files_by_dir, root_label="Project Files"):
    """Convert files grouped by directory into a tree string (ASCII only)"""
    lines = [root_label]
    
    sorted_dirs = sorted(files_by_dir.items())
    
    for i, (dir_path, files) in enumerate(sorted_dirs):
        is_last_dir = i == len(sorted_dirs) - 1
        dir_connector = "`-- " if is_last_dir else "|-- "
        
        lines.append(f"{dir_connector}[DIR] {dir_path}")
        
        sorted_files = sorted(files)
        for j, file_name in enumerate(sorted_files):
            is_last_file = j == len(sorted_files) - 1
            
            if is_last_dir:
                file_prefix = "    "
            else:
                file_prefix = "|   "
            
            file_connector = "`-- " if is_last_file else "|-- "
            lines.append(f"{file_prefix}{file_connector}{file_name}")
    
    return "\n".join(lines)

def get_tree_from_files(matching_files: List[str]) -> Tuple[Tree, str]:
    tree = Tree("🔍 Found Files")
    files_by_dir = {}

    # Group files by directory
    for file_path in matching_files:
        dir_path = os.path.dirname(file_path) or "."
        if dir_path not in files_by_dir:
            files_by_dir[dir_path] = []
        files_by_dir[dir_path].append(os.path.basename(file_path))

    # Create tree structure
    for dir_path, files in sorted(files_by_dir.items()):
        dir_node = tree.add(f"📁 {dir_path}")
        for file_name in sorted(files):
            dir_node.add(f"📄 {file_name}")
    return tree, create_tree_string(files_by_dir)
