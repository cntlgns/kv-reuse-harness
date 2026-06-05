"""Tests for ssa.utils.git_ops.get_git_patch."""

from __future__ import annotations

import subprocess

import pytest

from ssa.utils.git_ops import get_git_patch


@pytest.fixture
def git_repo_with_change(git_repo):
    """Initial commit, then modify the tracked file and add a new file (unstaged)."""
    (git_repo / "hello.py").write_text("print('goodbye')\n")
    (git_repo / "new_file.py").write_text("print('new')\n")
    # Return the initial commit SHA — get_git_patch diffs against init_commit
    result = subprocess.run(
        ["git", "-C", str(git_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    init_commit = result.stdout.strip()
    return git_repo, init_commit


def test_get_git_patch_captures_modified_and_new_files(
    git_repo_with_change, shell_env_factory
):
    """get_git_patch returns a unified diff containing modified and new files,
    and the ``paths`` filter restricts output to the listed paths."""
    repo, init_commit = git_repo_with_change
    env = shell_env_factory(str(repo), init_commit=init_commit)

    # Full patch: both files appear
    patch_all = get_git_patch(env)
    assert "hello.py" in patch_all
    assert "new_file.py" in patch_all
    assert "goodbye" in patch_all
    assert "new" in patch_all

    # Restricted patch: only hello.py
    env2 = shell_env_factory(str(repo), init_commit=init_commit)
    patch_filtered = get_git_patch(env2, paths=["hello.py"])
    assert "hello.py" in patch_filtered
    assert "new_file.py" not in patch_filtered
