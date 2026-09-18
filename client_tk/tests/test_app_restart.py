"""client_tk/app/services/app_restart.py — relaunch-after-exit helper."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client_tk.app.services import app_restart


class WaitForExitTest(unittest.TestCase):
    def test_returns_true_once_process_is_gone(self) -> None:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.6)"])
        started = time.monotonic()
        self.assertTrue(app_restart.wait_for_exit(proc.pid, timeout_s=10, poll_s=0.05))
        self.assertGreaterEqual(time.monotonic() - started, 0.4)
        proc.wait(timeout=5)

    def test_times_out_while_process_alive(self) -> None:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            self.assertFalse(app_restart.wait_for_exit(proc.pid, timeout_s=0.3, poll_s=0.05))
        finally:
            proc.kill()
            proc.wait(timeout=5)


class ScheduleRelaunchTest(unittest.TestCase):
    def test_helper_command_carries_pid_cwd_log_and_argv(self) -> None:
        with mock.patch.object(app_restart.subprocess, "Popen") as popen:
            app_restart.schedule_relaunch(
                ["python.exe", "scripts/run_desktop.py", "--split"],
                cwd="D:/work", log_path="D:/data/restart.log",
            )
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[:3], [sys.executable, "-m", "client_tk.app.services.app_restart"])
        self.assertIn("--wait-pid", cmd)
        self.assertEqual(cmd[cmd.index("--wait-pid") + 1], str(os.getpid()))
        self.assertEqual(cmd[cmd.index("--cwd") + 1], "D:/work")
        self.assertEqual(cmd[cmd.index("--log") + 1], "D:/data/restart.log")
        self.assertEqual(cmd[cmd.index("--") + 1:], ["python.exe", "scripts/run_desktop.py", "--split"])
        # detached from the parent so it survives the app closing
        kwargs = popen.call_args.kwargs
        if os.name == "nt":
            self.assertTrue(kwargs["creationflags"] & subprocess.DETACHED_PROCESS)
        else:
            self.assertTrue(kwargs["start_new_session"])


class HelperMainTest(unittest.TestCase):
    def test_main_waits_then_relaunches_argv(self) -> None:
        """End to end: the helper waits for a short-lived pid, then runs the argv, which
        writes a marker file. Output goes to the log file."""
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "relaunched.txt"
            log = Path(tmp) / "restart.log"
            victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
            argv = [sys.executable, "-c", f"open(r'{marker}', 'w').write('ok'); print('hello from relaunch')"]
            _real_sleep = time.sleep
            with mock.patch.object(app_restart.time, "sleep", side_effect=lambda s: _real_sleep(min(s, 0.05))):
                rc = app_restart.main(["--wait-pid", str(victim.pid), "--cwd", tmp, "--log", str(log), "--", *argv])
            victim.wait(timeout=5)
            self.assertEqual(rc, 0)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not marker.exists():
                time.sleep(0.05)
            self.assertTrue(marker.exists(), "relaunched command did not run")
            time.sleep(0.3)
            self.assertIn("hello from relaunch", log.read_text(encoding="utf-8", errors="replace"))


if __name__ == "__main__":
    unittest.main()
