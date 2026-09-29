"""Jalankan ulang app desktop setelah proses ini exit.

Dipakai tombol "Restart Backend" di tab Machine Settings. Di mode local-only
backend ter-embed di proses client, jadi "restart backend" berarti "restart
proses ini". Di mode split yang dimulai scripts/run_desktop.py, subprocess
backend cuma dihentikan saat launcher exit, jadi instance baru tidak boleh
mulai sebelum yang lama benar-benar hilang (port / COM / kamera bisa
bentrok). Makanya perlu tarian dua langkah:

  1. `schedule_relaunch()` spawn helper terpisah
     (`python -m client_tk.app.services.app_restart --wait-pid <pid> ...`).
  2. Pemanggil mematikan app secara normal.
  3. Helper menunggu pid itu hilang, lalu menjalankan command line yang sama
     lagi dengan stdout/stderr ditambahkan ke `<data_root>/restart.log`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _detached_flags() -> dict:
    if os.name == "nt":
        return {
            "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            "close_fds": True,
        }
    return {"start_new_session": True, "close_fds": True}


def schedule_relaunch(argv: list[str], *, cwd: str | os.PathLike, log_path: str | os.PathLike) -> subprocess.Popen:
    """Spawn waiter terpisah yang menjalankan ulang `argv` setelah proses ini exit."""
    cmd = [
        sys.executable, "-m", "client_tk.app.services.app_restart",
        "--wait-pid", str(os.getpid()),
        "--cwd", str(cwd),
        "--log", str(log_path),
        "--",
        *argv,
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_detached_flags(),
    )


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, wintypes.DWORD(pid))
        if not handle:
            return False
        try:
            # WAIT_TIMEOUT (0x102) berarti masih jalan; WAIT_OBJECT_0 berarti sudah exit.
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_exit(pid: int, *, timeout_s: float = 60.0, poll_s: float = 0.25) -> bool:
    """Blok sampai `pid` hilang. Return False kalau masih hidup setelah `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(poll_s)
    return not _pid_alive(pid)


def relaunch(argv: list[str], *, cwd: str | os.PathLike, log_path: str | os.PathLike) -> subprocess.Popen:
    """Jalankan `argv` secara terpisah, tambahkan outputnya ke `log_path`."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    log.write(f"\n=== relaunch {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(argv)}\n".encode("utf-8"))
    log.flush()
    return subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        **_detached_flags(),
    )


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tunggu proses exit, lalu jalankan ulang command.")
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    ns = parser.parse_args(args)
    argv = [a for a in ns.argv if a != "--"] if ns.argv and ns.argv[0] == "--" else ns.argv
    if not argv:
        return 2
    # Beri proses lama waktu ekstra untuk melepas kamera / COM port meskipun
    # pid-nya sudah hilang (handle driver bisa sedikit lag dari exit proses).
    if not wait_for_exit(ns.wait_pid, timeout_s=ns.timeout):
        return 3
    time.sleep(1.0)
    relaunch(argv, cwd=ns.cwd, log_path=ns.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
