from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AuthStorageWiringTest(unittest.TestCase):
    def _run_container_probe(self, backend: str) -> None:
        with tempfile.TemporaryDirectory(prefix="qc-suite-wiring-") as data_root:
            env = dict(os.environ)
            env.update(
                {
                    "PYTHONPATH": str(PROJECT_ROOT),
                    "QC_SUITE_DATA_ROOT": data_root,
                    "QC_SUITE_DATABASE_BACKEND": backend,
                    "QC_SUITE_PLC_ENABLED": "0",
                    "MSSQL_SERVER": "dummy",
                    "MSSQL_DATABASE": "dummy",
                    "MSSQL_USERNAME": "dummy",
                    "MSSQL_PASSWORD": "dummy",
                    "POSTGRESQL_HOST": "127.0.0.1",
                    "POSTGRESQL_DATABASE": "dummy",
                    "POSTGRESQL_USERNAME": "dummy",
                    "POSTGRESQL_PASSWORD": "dummy",
                }
            )
            script = textwrap.dedent(
                f"""
                from __future__ import annotations

                import sys
                import types

                package_name = "{'sqlserver' if backend == 'sqlserver' else 'postgres'}"

                cv2 = types.ModuleType("cv2")
                cv2.IMWRITE_JPEG_QUALITY = 1
                cv2.THRESH_BINARY = 0
                cv2.THRESH_BINARY_INV = 1
                cv2.THRESH_OTSU = 8
                cv2.RETR_EXTERNAL = 0
                cv2.CHAIN_APPROX_SIMPLE = 0
                cv2.FONT_HERSHEY_SIMPLEX = 0
                cv2.LINE_AA = 0
                sys.modules["cv2"] = cv2
                numpy = types.ModuleType("numpy")
                numpy.uint8 = object()
                numpy.ndarray = object
                sys.modules["numpy"] = numpy
                pymodbus = types.ModuleType("pymodbus")
                pymodbus_client = types.ModuleType("pymodbus.client")
                dummy_client = type("DummyModbusClient", (), {{"__init__": lambda self, *a, **k: None}})
                pymodbus_client.ModbusSerialClient = dummy_client
                pymodbus_client.ModbusTcpClient = dummy_client
                pymodbus_exceptions = types.ModuleType("pymodbus.exceptions")
                pymodbus_exceptions.ConnectionException = type("ConnectionException", (Exception,), {{}})
                pymodbus_exceptions.ModbusIOException = type("ModbusIOException", (Exception,), {{}})
                pymodbus_framer = types.ModuleType("pymodbus.framer")
                pymodbus_framer.FramerType = type("FramerType", (), {{"RTU": "rtu"}})
                sys.modules["pymodbus"] = pymodbus
                sys.modules["pymodbus.client"] = pymodbus_client
                sys.modules["pymodbus.exceptions"] = pymodbus_exceptions
                sys.modules["pymodbus.framer"] = pymodbus_framer

                def install_fake(name, class_name):
                    module = types.ModuleType(name)
                    cls = type(class_name, (), {{
                        "__init__": lambda self, *args, **kwargs: None,
                    }})
                    setattr(module, class_name, cls)
                    sys.modules[name] = module
                    return cls

                expected_users_cls = install_fake(
                    f"backend.app.repositories.{{package_name}}.users_repository",
                    "{'SqlServerUsersRepository' if backend == 'sqlserver' else 'PostgresUsersRepository'}",
                )
                expected_mirror_cls = install_fake(
                    f"backend.app.repositories.{{package_name}}.inspection_mirror_repository",
                    "{'SqlServerInspectionMirrorRepository' if backend == 'sqlserver' else 'PostgresInspectionMirrorRepository'}",
                )

                import backend.app.core.container as container
                from backend.app.core.security import TokenStore
                from backend.app.repositories.auth_audit_repository import AuthAuditRepository

                blocked = [
                    f"backend.app.repositories.{{package_name}}.auth_audit_repository",
                    f"backend.app.repositories.{{package_name}}.session_store",
                ]
                imported_blocked = [name for name in blocked if name in sys.modules]
                if imported_blocked:
                    raise AssertionError(f"legacy modules imported: {{imported_blocked}}")
                if not isinstance(container.audit_repo, AuthAuditRepository):
                    raise AssertionError(type(container.audit_repo).__name__)
                if not isinstance(container.token_store, TokenStore):
                    raise AssertionError(type(container.token_store).__name__)
                if not isinstance(container.users_repo, expected_users_cls):
                    raise AssertionError(type(container.users_repo).__name__)
                if not isinstance(container.inspection_sql_mirror_repo, expected_mirror_cls):
                    raise AssertionError(type(container.inspection_sql_mirror_repo).__name__)
                """
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            if result.returncode != 0:
                self.fail(
                    f"container wiring probe failed for {backend}\n"
                    f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )

    def test_sqlserver_uses_local_audit_and_memory_sessions(self) -> None:
        self._run_container_probe("sqlserver")

    def test_postgresql_uses_local_audit_and_memory_sessions(self) -> None:
        self._run_container_probe("postgresql")


if __name__ == "__main__":
    unittest.main()
