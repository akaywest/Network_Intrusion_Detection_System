"""Data-path socket settings shared by the streamer (M3) and the engine (M4) (plan §6.2).

The streamer requests a ~4 MB send buffer so a 1,000-record burst (S4) does not block
the sender. Linux caps the request at net.core.wmem_max and getsockopt then reports
twice the granted size (the kernel's bookkeeping overhead), so the *effective* value is
probed on a real socket and recorded in TEST_ENVIRONMENT.run_conditions for every run
(FR15, Table 3.14) instead of being assumed.
"""
from __future__ import annotations

import socket
from pathlib import Path

SEND_BUFFER = 4 * 1024 * 1024  # requested SO_SNDBUF (plan §6.2)


def _sysctl(name: str) -> int | None:
    try:
        return int(Path("/proc/sys/net/core", name).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None  # not Linux


def effective_socket_buffers() -> dict:
    """Probe what the kernel grants for the data path's socket options."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        rcv_default = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER)
        snd = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    return {"so_sndbuf_requested": SEND_BUFFER, "so_sndbuf_effective": snd,
            "so_rcvbuf_effective": rcv_default,  # engine does not change SO_RCVBUF
            "net_core_wmem_max": _sysctl("wmem_max"),
            "net_core_rmem_max": _sysctl("rmem_max")}


def sndbuf_warning(granted: int) -> str | None:
    """Warning text if the granted send buffer is below the 4 MB request, else None."""
    if granted >= SEND_BUFFER:
        return None
    return (f"WARNING: send buffer granted {granted} bytes < requested {SEND_BUFFER} "
            f"(net.core.wmem_max={_sysctl('wmem_max')}); a burst may block the sender and "
            f"delay t0. Raise it with: sudo sysctl -w net.core.wmem_max={2 * SEND_BUFFER}")
