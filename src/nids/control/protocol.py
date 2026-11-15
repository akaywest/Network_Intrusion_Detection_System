"""Loopback control channel (M7; plan §7.1, Chapter 3 §3.3.1; FR2, FR8, NFR6).

The engine (127.0.0.1:9101) and the streamer (127.0.0.1:9102) each run one control
thread that accepts one-line JSON commands and answers each with one JSON line that
always carries "ok". Commands never travel on the data path (port 9100), so they add
nothing to L (§3.3.1).

  engine    prepare   {"cmd":"prepare","model":"rf","task":"binary","seed":42,"dt_ms":100,
                       "mode":"steady","scenario":"S1","n":5000[,"dashboard_open":true]}
                      -> {"ok":true,"run_id":17}
            stop      cancel a prepared run, or end a running one if the stream is lost
            set_rate  {"cmd":"set_rate","dt_ms":20,"from_seq":k}  record a demo rate change
            status    state, active run, counters, last finished run
  streamer  start     {"cmd":"start","run_id":17,"dt_ms":100,"mode":"steady","n":5000,
                       "sample_seed":42[,"burst_block":1000]}
            stop      send eos now (the engine then flushes t4 and sets ended_at)
            set_rate  {"cmd":"set_rate","dt_ms":20}  change dt mid-run (demo only, FR2)
            status    state, active run, records sent, last finished run

Replies: {"ok":true,...} or {"ok":false,"error":"..."}. A bad command never stops a
service. Every socket here binds or connects to 127.0.0.1 only (NFR6).
"""
from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Callable

LOOPBACK = "127.0.0.1"
MAX_LINE = 64 * 1024
CONN_TIMEOUT_S = 300.0  # idle control connections are closed after this

MODES = ("steady", "burst")
TASKS = ("binary", "multi")
END_REASONS = ("complete", "stopped", "disconnected", "cancelled", "shutdown")

log = logging.getLogger("nids.control")


class ProtocolError(ValueError):
    """A command line that is not valid JSON or does not match the command schema."""


# ---------------------------------------------------------------- field checks
def _int(v, name, minimum=None):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ProtocolError(f"{name} must be an integer")
    if minimum is not None and v < minimum:
        raise ProtocolError(f"{name} must be >= {minimum}")
    return v


def _pos_number(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0 \
            or v != v or v == float("inf"):
        raise ProtocolError(f"{name} must be a positive number")
    return float(v)


def _choice(v, name, options):
    if v not in options:
        raise ProtocolError(f"{name} must be one of {list(options)}")
    return v


def _bool(v, name):
    if v is not None and not isinstance(v, bool):
        raise ProtocolError(f"{name} must be true, false or null")
    return v


def _str(v, name, optional=False):
    if v is None and optional:
        return None
    if not isinstance(v, str) or not v or len(v) > 64:
        raise ProtocolError(f"{name} must be a non-empty string (<= 64 chars)")
    return v


# Field schema for each command: name -> (checker, required)
SCHEMAS: dict[str, dict[str, tuple[Callable, bool]]] = {
    "prepare": {
        "model": (lambda v: _str(v, "model"), True),
        "task": (lambda v: _choice(v, "task", TASKS), False),
        "seed": (lambda v: _int(v, "seed"), True),
        "dt_ms": (lambda v: _pos_number(v, "dt_ms"), True),
        "mode": (lambda v: _choice(v, "mode", MODES), False),
        "scenario": (lambda v: _str(v, "scenario", optional=True), False),
        "n": (lambda v: None if v is None else _int(v, "n", 1), False),
        # declared by the caller, recorded in run_conditions (§3.3.7: the design runs
        # with the dashboard open; the dashboard-closed S1 set is the comparison)
        "dashboard_open": (lambda v: _bool(v, "dashboard_open"), False),
    },
    "start": {
        "run_id": (lambda v: _int(v, "run_id", 1), True),
        "dt_ms": (lambda v: _pos_number(v, "dt_ms"), True),
        "mode": (lambda v: _choice(v, "mode", MODES), False),
        "n": (lambda v: _int(v, "n", 1), True),
        "sample_seed": (lambda v: _int(v, "sample_seed"), False),
        "burst_block": (lambda v: _int(v, "burst_block", 1), False),
    },
    "set_rate": {
        "dt_ms": (lambda v: _pos_number(v, "dt_ms"), True),
        "from_seq": (lambda v: None if v is None else _int(v, "from_seq", 0), False),
    },
    "stop": {},
    "status": {},
}

ENGINE_CMDS = frozenset({"prepare", "stop", "set_rate", "status"})
STREAMER_CMDS = frozenset({"start", "stop", "set_rate", "status"})


def encode(obj: dict) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def decode(line: bytes | str) -> dict:
    if isinstance(line, bytes):
        if len(line) > MAX_LINE:
            raise ProtocolError("command line too long")
        line = line.decode("utf-8", errors="replace")
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"not JSON: {e.msg}") from None
    if not isinstance(msg, dict):
        raise ProtocolError("a command must be a JSON object")
    return msg


def validate(msg: dict, allowed: frozenset[str]) -> dict:
    """Check msg against the schema of its cmd; return a clean dict (cmd + known fields)."""
    cmd = msg.get("cmd")
    if cmd not in allowed:
        raise ProtocolError(f"unknown cmd {cmd!r}; this service accepts {sorted(allowed)}")
    schema = SCHEMAS[cmd]
    unknown = set(msg) - set(schema) - {"cmd"}
    if unknown:
        raise ProtocolError(f"unknown field(s) for {cmd}: {sorted(unknown)}")
    out = {"cmd": cmd}
    for name, (check, required) in schema.items():
        if name in msg:
            out[name] = check(msg[name])
        elif required:
            raise ProtocolError(f"{cmd} needs {name!r}")
    return out


# ---------------------------------------------------------------- server
class ControlServer(threading.Thread):
    """One daemon thread answering one-line JSON commands on 127.0.0.1:port.

    Each connection gets its own short-lived thread, so a client that keeps its
    connection open (the dashboard) cannot block another (run_matrix), and a slow
    prepare does not hold up status. `handler(cmd_dict) -> reply_dict` must therefore be
    thread-safe (the services guard their state with a lock and do the real work on
    their main thread); any exception it raises becomes {"ok":false,"error":...} and
    the server carries on.
    The socket is bound in __init__, so `self.port` is known before start() (port 0
    picks a free port for tests).
    """

    def __init__(self, port: int, handler: Callable[[dict], dict],
                 allowed: frozenset[str], host: str = LOOPBACK, name: str = "control"):
        super().__init__(name=name, daemon=True)
        if host != LOOPBACK:
            raise ValueError("the control channel binds 127.0.0.1 only (NFR6)")
        self.handler, self.allowed = handler, allowed
        self.sock = socket.create_server((host, port))
        self.sock.settimeout(0.5)
        self.port = self.sock.getsockname()[1]
        self._closing = threading.Event()

    def run(self) -> None:
        while not self._closing.is_set():
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break  # socket closed
            threading.Thread(target=self._serve, args=(conn,), daemon=True,
                             name=f"{self.name}-conn").start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            self._serve_lines(conn)

    def _serve_lines(self, conn: socket.socket) -> None:
        conn.settimeout(CONN_TIMEOUT_S)
        try:
            for line in conn.makefile("rb"):
                if not line.strip():
                    continue
                conn.sendall(encode(self.dispatch(line)))
        except (OSError, TimeoutError):
            pass  # client went away; nothing to clean up

    def dispatch(self, line: bytes) -> dict:
        try:
            cmd = validate(decode(line), self.allowed)
        except ProtocolError as e:
            return {"ok": False, "error": str(e)}
        try:
            reply = self.handler(cmd)
        except Exception as e:  # noqa: BLE001  a bad command must never stop the service
            log.exception("control command %s failed", cmd.get("cmd"))
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return reply if "ok" in reply else {"ok": True, **reply}

    def close(self) -> None:
        self._closing.set()
        self.sock.close()


# ---------------------------------------------------------------- client side
def request(port: int, msg: dict, host: str = LOOPBACK, timeout: float = 10.0) -> dict:
    """Send one command and return the reply (raises OSError if nobody listens)."""
    if host != LOOPBACK:
        raise ValueError("the control channel is loopback only (NFR6)")
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(encode(msg))
        buf = s.makefile("rb").readline(MAX_LINE + 1)
    if not buf:
        raise ConnectionError(f"no reply from control port {port}")
    return decode(buf)
