import base64
import os
import json
import logging
import shlex
from typing import TYPE_CHECKING, List


LOG = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ..environments import DockerEnvironment

def put_git_marker(env: "DockerEnvironment") -> str:
    command = f"git -C {env.workdir} add ."
    _ = env.execute_bash(command=command, verbose=True)
    commit_message = "SRA begin changes"
    command = f'git -C {env.workdir} commit -m {shlex.quote(commit_message)}'
    _ = env.execute_bash(command=command, verbose=True)
    # Get commit_id
    command = f"git -C {env.workdir} rev-parse HEAD"
    output = env.execute_bash(command=command, verbose=True)
    commit_id = output.get("output").strip()
    LOG.info(f"Recording SRA commit marker: {commit_id}")
    return commit_id

def _get_text_paths(env: "DockerEnvironment", diff_args: str) -> list[str]:
    """Return only text (non-binary) changed file paths from a git diff."""
    command = f"git -C {env.workdir} diff --numstat {diff_args}"
    result = env.execute_bash(command=command)
    output = result.get("output", "")
    text_files = []
    binary_files = []
    for line in output.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        path = parts[2]
        if " => " in path:
            path = path.split(" => ")[-1]
        if parts[0] == "-" and parts[1] == "-":
            binary_files.append(path)
        else:
            text_files.append(path)
    if binary_files:
        LOG.info(f"Skipping binary files from patch: {binary_files}")
    return text_files

MAX_PATCH_SIZE = 1 * 1024 * 1024  # 1 MB

# Directories commonly created at runtime that should never appear in a patch
_JUNK_DIRS = [
    "venv", "env",
    "node_modules",
    "__pycache__",
    "build", "_build", "dist", "*.egg-info",
]


def get_git_patch(env: "DockerEnvironment", paths: List[str] | None = None, cleanup: bool = False) -> str:
    """Stage changes and return the git diff (vs init_commit) as a patch string.

    If ``cleanup`` is True, the intermediate patch file written inside the
    container is removed before returning.
    """
    # stage all changes first so new files appear in both binary detection and diff
    _ = env.execute_bash(command=f"git -C {env.workdir} add .")
    paths = paths or []
    if paths:
        # Convert absolute paths to relative paths (relative to workdir) for git diff
        text_files = [os.path.relpath(p, env.workdir) if os.path.isabs(p) else p for p in paths]
        LOG.info(f"Using provided paths for extracting git changes: {text_files}")
    else:
        ignore_pattern = [
            "':(glob,exclude)**/test_*'",
            "':(glob,exclude)**/test/**'",
            "':(glob,exclude)**/tests/**'",
        ]
        # exclude common non-source directories that agents may create
        for d in _JUNK_DIRS:
            ignore_pattern.append(f"':(glob,exclude)**/{d}/**'")
        # exclude all hidden directories and hidden files
        ignore_pattern.append("':(glob,exclude)**/.*/**'")
        ignore_pattern.append("':(glob,exclude)**/.*'")
        extensions = []
        # extensions += ["':(glob)**/*.py'",]
        extensions += ["':(exclude)*.bak'","':(exclude)*.md'"]
        pathspec = " ".join(extensions) if extensions else "."
        ignore_str = " ".join(ignore_pattern)
        diff_args = f"{env.init_commit} -- {pathspec} {ignore_str}".strip()
        # get only text file paths (binary files are excluded)
        text_files = _get_text_paths(env, diff_args)
    filename = "sra_patch.patch"
    if text_files:
        file_args = " ".join(shlex.quote(f) for f in text_files)
        command = f"git -C {env.workdir} diff {env.init_commit} -- {file_args} > {filename}"
    else:
        command = f"echo -n > {os.path.join(env.workdir, filename)}"
    _ = env.execute_bash(command=command)
    patch_file = os.path.join(env.workdir, filename)
    command = f"base64 {patch_file}"
    result = env.execute_bash(command)
    patch_bytes = base64.b64decode(result.get("output", ""))
    if cleanup:
        _ = env.execute_bash(command=f"rm -f {shlex.quote(patch_file)}")
        _ = env.execute_bash(command=f"git -C {env.workdir} reset")
    return patch_bytes.decode("utf-8", errors="replace")

def extract_git_changes(env: "DockerEnvironment", output_dir: str, paths: List[str] | None = None) -> None:
    paths = paths or []
    patch = get_git_patch(env, paths=paths)
    filename = os.path.join(output_dir, "sra_patch.patch")
    LOG.info(f"Saving the patch to file: {filename}")
    with open(filename, "w") as f:
        f.write(patch)
    patch_size = len(patch.encode("utf-8"))
    if patch_size > MAX_PATCH_SIZE:
        LOG.warning(
            f"Patch is suspiciously large ({patch_size / 1024 / 1024:.1f} MB). "
            "Likely includes unintended files. Review the diff paths."
        )
    LOG.info(f"Final SRA patch submission:\n{patch}")
    pred_filename = filename.replace(".patch", ".json")
    pred = {
        env._cfg.dataset.identifier: {
            "instance_id": env._cfg.dataset.identifier,
            "model_name_or_path": "sra",
            "model_patch": patch,
        }
    }
    LOG.info(f"Saving the prediction to file: {pred_filename}")
    with open(pred_filename, "w") as f:
        json.dump(pred, f)
