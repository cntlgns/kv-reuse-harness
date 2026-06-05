"""
Tool loader: reads tool names from config and dynamically imports them.

Each tool module at ``ssa.tools.<name>`` must expose a callable with the same
name as the module (e.g. ``ssa.tools.bash`` exports ``bash``).

Tool entries in config are a dict keyed by tool name, with optional
per-tool parameters::

    agent:
      tools:
        bash:
          timeout: 300
          max_lines_limit: 500
        str_replace_editor:
          max_chars_limit: 30000
        submit:
"""

import importlib
import logging
from typing import Any, Callable, Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf

LOG = logging.getLogger(__name__)


def _flatten_tool_entries(
    raw: Dict[str, Any], prefix: str = ""
) -> List[Tuple[str, Dict[str, Any]]]:
    """Flatten nested tool dicts into dotted tool names.

    Hydra interprets dotted YAML keys (e.g. ``openai.shell:``) as nested dicts

    A key whose value is a plain dict *without* any tool-config keys is treated
    as a namespace (flattened).  A key whose value is ``None`` or a dict with
    actual config values is treated as a leaf tool entry.
    """
    results: List[Tuple[str, Dict[str, Any]]] = []
    for key, value in raw.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and all(isinstance(v, (dict, type(None))) for v in value.values()):
            # Could be a namespace OR a tool with only dict-typed config params.
            # Heuristic: try importing as a tool first; if it fails, treat as namespace.
            try:
                importlib.import_module(f"ssa.tools.{full_key}")
                results.append((full_key, value or {}))
            except ModuleNotFoundError:
                results.extend(_flatten_tool_entries(value, prefix=full_key))
        else:
            results.append((full_key, value if isinstance(value, dict) else {}))
    return results


def load_tools(cfg: DictConfig) -> Tuple[List[Callable], Dict[str, Dict[str, Any]]]:
    """Load tool callables listed in ``agent.tools``.

    ``agent.tools`` is a dict mapping tool names to their optional config.
    Dotted tool names (e.g. ``openai.shell``) are supported — Hydra expands
    them to nested dicts which this loader flattens automatically.

    Args:
        cfg: Hydra DictConfig with ``agent.tools``.

    Returns:
        Tuple of (tool callables list, tool_params dict keyed by tool name).
    """
    raw_tools = OmegaConf.to_container(cfg.agent.tools, resolve=True)
    tools: List[Callable] = []
    tool_params: Dict[str, Dict[str, Any]] = {}
    tool_names: List[str] = []

    for name, params in _flatten_tool_entries(raw_tools):
        module = importlib.import_module(f"ssa.tools.{name}")
        tools.append(module)
        tool_names.append(name)
        if params:
            tool_params[name] = params
        LOG.info(f"Loaded tool: {name}" + (f" with config: {tool_params[name]}" if name in tool_params else ""))

    LOG.info(f"Loaded {len(tools)} tools: {tool_names}")
    return tools, tool_params
