"""
Shared streaming-exec helpers used by both the Popen-based environments
(LocalEnvironment, SSADockerEnvironment) and the docker-SDK-based
DockerConnector. Centralizes the head+tail output buffer and the
non-blocking read loop so partial-output-on-timeout behaves identically
across all three.
"""
from __future__ import annotations

import collections
import os
import select
import subprocess
import time
from typing import Optional

# Head+tail cap on streamed stdout.
STDOUT_HEAD_CAP = 1 * 1024 * 1024 # 1MB
STDOUT_TAIL_CAP = 1 * 1024 * 1024


class HeadTailBuffer:
    """Byte accumulator with head+tail retention; middle is evicted."""

    def __init__(
        self,
        head_cap: int = STDOUT_HEAD_CAP,
        tail_cap: int = STDOUT_TAIL_CAP,
    ):
        self._head_cap = head_cap
        self._tail_cap = tail_cap
        self._head: list[bytes] = []
        self._head_size = 0
        self._tail: collections.deque[bytes] = collections.deque()
        self._tail_size = 0
        self._dropped = 0

    def append(self, chunk: bytes) -> None:
        if self._head_size < self._head_cap:
            take = min(len(chunk), self._head_cap - self._head_size)
            if take:
                self._head.append(chunk[:take])
                self._head_size += take
            chunk = chunk[take:]
            if not chunk:
                # no leftover for tail
                return
        self._tail.append(chunk)
        self._tail_size += len(chunk)
        while self._tail_size > self._tail_cap and len(self._tail) > 1:
            evicted = self._tail.popleft()
            self._tail_size -= len(evicted)
            self._dropped += len(evicted)

    def materialize(self) -> str:
        if not self._head and not self._tail:
            return ""
        head_bytes = b"".join(self._head)
        tail_bytes = b"".join(self._tail)
        if self._dropped:
            marker = f"\n\n< ... {self._dropped} bytes dropped ... >\n\n".encode()
            return (head_bytes + marker + tail_bytes).decode(errors="replace")
        return (head_bytes + tail_bytes).decode(errors="replace")


def run_with_streaming_capture(
    argv: list[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout_sec: Optional[float] = None,
) -> tuple[int, str]:
    """Run argv via Popen, streaming merged stdout+stderr into a head+tail buffer.

    Returns (returncode, stdout) where stdout is a decoded string (never None).
    """
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    os.set_blocking(fd, False)

    buf = HeadTailBuffer()
    deadline = (time.monotonic() + timeout_sec) if timeout_sec else None
    timed_out = False

    try:
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                wait = min(remaining, 1.0)
            else:
                wait = 1.0

            r, _, _ = select.select([fd], [], [], wait)
            if fd in r:
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    chunk = b""
                if not chunk:
                    break  # EOF
                buf.append(chunk)
            elif proc.poll() is not None:
                try:
                    while True:
                        chunk = os.read(fd, 65536)
                        if not chunk:
                            break
                        buf.append(chunk)
                except BlockingIOError:
                    pass
                break

        if timed_out:
            proc.kill()
        return_code = proc.wait(timeout=5)
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        raise
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass

    stdout = buf.materialize()
    if timed_out:
        return_code = 124

    return return_code, stdout
