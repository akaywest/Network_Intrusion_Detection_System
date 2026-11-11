"""TEST_ENVIRONMENT capture (M5; plan §5.1, FR15, Table 3.14, NFR4).

Collects the hardware and software description that every MODEL_RUN links to:
CPU model, physical cores, RAM, storage type of the disk holding the database, OS,
Python version, library versions (importlib.metadata) and the run conditions
(PRAGMA synchronous, journal mode, inference threads, power state).

get_or_create_env() reuses an existing row when the description is identical, so
repeated initialisation does not multiply environment rows.
"""
from __future__ import annotations

import json
import os
import platform
import sqlite3
import subprocess
from importlib import metadata
from pathlib import Path

import psutil

from nids.storage.db import SYNCHRONOUS

# Distribution names as installed by pip (plan §2.1 install list + TF's Keras).
LIBRARIES = ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "streamlit",
             "plotly", "pyarrow", "PyYAML", "psutil", "py-cpuinfo", "joblib",
             "matplotlib", "pytest")

FIELDS = ("cpu_model", "cpu_cores", "ram_gb", "storage_type", "os_name",
          "python_version", "library_versions", "run_conditions")


def cpu_model() -> str:
    try:
        import cpuinfo
        brand = cpuinfo.get_cpu_info().get("brand_raw")
        if brand:
            return brand
    except Exception:  # py-cpuinfo can fail on unusual platforms; fall back below
        pass
    return platform.processor() or "unknown"


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _base_disks(dev_name: str) -> list[str]:
    """Resolve a partition / device-mapper name (sda2, nvme0n1p2, dm-0) to its disks."""
    sys_dev = Path("/sys/class/block") / dev_name
    if not sys_dev.exists():
        return []
    slaves = sys_dev / "slaves"  # LVM / LUKS: follow down to the physical disks
    if slaves.is_dir() and any(slaves.iterdir()):
        return [d for s in sorted(slaves.iterdir()) for d in _base_disks(s.name)]
    real = sys_dev.resolve()
    if (real / "partition").exists():  # partition -> parent disk directory
        return [real.parent.name]
    return [real.name]


def storage_type(path: str | Path) -> str:
    """Storage technology of the disk holding `path` (Table 3.14), e.g. 'HDD (sata, ...)'."""
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    if platform.system() != "Linux":
        return "unknown"
    best = None
    for part in psutil.disk_partitions(all=False):
        mp = Path(part.mountpoint)
        if (path == mp or mp in path.parents) and (best is None or len(str(mp)) > len(str(best.mountpoint))):
            best = part
    if best is None or not best.device.startswith("/dev/"):
        return "unknown"
    disks = _base_disks(Path(os.path.realpath(best.device)).name)
    if not disks:
        return "unknown"
    descs = []
    for disk in disks:
        q = Path("/sys/block") / disk
        rota = _read(q / "queue" / "rotational")
        kind = "NVMe SSD" if disk.startswith("nvme") else ("HDD" if rota == "1" else "SSD")
        model = _read(q / "device" / "model") or ""
        descs.append(f"{kind} ({disk}{', ' + model if model else ''})")
    return "; ".join(descs)


def os_name() -> str:
    try:
        pretty = platform.freedesktop_os_release().get("PRETTY_NAME")
    except OSError:
        pretty = None
    base = pretty or f"{platform.system()} {platform.version()}"
    return f"{base}, kernel {platform.release()}" if platform.system() == "Linux" else base


def library_versions() -> dict[str, str]:
    out = {}
    for lib in LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "not installed"
    out["sqlite"] = sqlite3.sqlite_version  # the SQLite library linked into Python
    return out


def power_state() -> dict:
    """Power source, power profile and CPU governor (§3.3.11: same power plan every run)."""
    state: dict = {}
    for ps in Path("/sys/class/power_supply").glob("*"):
        if _read(ps / "type") == "Mains":
            online = _read(ps / "online")
            state["on_ac_power"] = online == "1" if online is not None else None
    try:
        r = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True,
                           timeout=5, check=False)
        if r.returncode == 0:
            state["power_profile"] = r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    gov = _read(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))
    if gov:
        state["cpu_governor"] = gov
    return state


def run_conditions(extra: dict | None = None) -> dict:
    cond = {
        "sqlite_synchronous": SYNCHRONOUS,           # plan §5.1; drives t_ser
        "sqlite_journal_mode": "WAL",
        "inference_threads": 1,                      # §3.3.10, single-thread inference
        "logical_cpus": psutil.cpu_count(logical=True),
        **power_state(),
    }
    if extra:
        cond.update(extra)
    return cond


def collect(db_path: str | Path, extra_conditions: dict | None = None) -> dict:
    """Return the TEST_ENVIRONMENT row (without env_id) as a dict of column -> value."""
    return {
        "cpu_model": cpu_model(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
        "storage_type": storage_type(db_path),
        "os_name": os_name(),
        "python_version": platform.python_version(),
        "library_versions": json.dumps(library_versions(), sort_keys=True),
        "run_conditions": json.dumps(run_conditions(extra_conditions), sort_keys=True),
    }


def get_or_create_env(con: sqlite3.Connection, env: dict) -> int:
    """Return the env_id of an identical TEST_ENVIRONMENT row, inserting one if needed (FR15)."""
    values = tuple(env[f] for f in FIELDS)
    row = con.execute(
        "SELECT env_id FROM test_environment WHERE cpu_model IS ? AND cpu_cores IS ? "
        "AND ram_gb IS ? AND storage_type IS ? AND os_name IS ? AND python_version IS ? "
        "AND library_versions IS ? AND run_conditions IS ? ORDER BY env_id DESC LIMIT 1",
        values).fetchone()
    if row:
        return int(row[0])
    cur = con.execute(
        "INSERT INTO test_environment (cpu_model, cpu_cores, ram_gb, storage_type, os_name, "
        "python_version, library_versions, run_conditions) VALUES (?,?,?,?,?,?,?,?)", values)
    return int(cur.lastrowid)
