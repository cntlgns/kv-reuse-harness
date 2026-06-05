"""Tests for ssa.environments._streaming_exec."""

from __future__ import annotations

from ssa.environments._streaming_exec import (
    HeadTailBuffer,
    run_with_streaming_capture,
)


def test_streaming_exec_basic():
    """A fast command exits 0 and its stdout is captured."""
    rc, out = run_with_streaming_capture(["bash", "-c", "echo hi"])
    assert rc == 0
    assert "hi" in out


def test_streaming_exec_timeout_returns_124_with_partial():
    """A command that prints before sleeping past the deadline returns 124
    with the pre-sleep output preserved."""
    rc, out = run_with_streaming_capture(
        ["bash", "-c", "echo START; sleep 10"],
        timeout_sec=1.0,
    )
    assert rc == 124
    assert "START" in out


def test_headtail_buffer_drops_middle():
    """Writing past head_cap + tail_cap yields a dropped-bytes marker."""
    buf = HeadTailBuffer(head_cap=10, tail_cap=10)
    buf.append(b"A" * 10)  # fills head
    buf.append(b"B" * 20)  # fills tail, evicts some
    buf.append(b"C" * 10)  # pushes more through tail

    out = buf.materialize()

    assert out.startswith("A" * 10)
    assert out.endswith("C" * 10)
    assert "bytes dropped" in out
