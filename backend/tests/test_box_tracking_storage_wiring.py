from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BoxTrackingStorageWiringTest(unittest.TestCase):
    def _run_container_probe(self, backend: str) -> None:
        with tempfile.TemporaryDirectory(prefix="qc-suite-box-wiring-") as data_root:
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

                def install_fake(name, class_name, methods=None):
                    module = types.ModuleType(name)
                    attrs = {{"__init__": lambda self, *args, **kwargs: None}}
                    attrs.update(methods or {{}})
                    cls = type(class_name, (), attrs)
                    setattr(module, class_name, cls)
                    sys.modules[name] = module
                    return cls

                expected_users_cls = install_fake(
                    f"backend.app.repositories.{{package_name}}.users_repository",
                    "{'SqlServerUsersRepository' if backend == 'sqlserver' else 'PostgresUsersRepository'}",
                    methods={{"apply_machine_settings": lambda self, *a, **k: None}},
                )
                expected_mirror_cls = install_fake(
                    f"backend.app.repositories.{{package_name}}.inspection_mirror_repository",
                    "{'SqlServerInspectionMirrorRepository' if backend == 'sqlserver' else 'PostgresInspectionMirrorRepository'}",
                )
                expected_box_cls = install_fake(
                    f"backend.app.repositories.{{package_name}}.box_luggage_repository",
                    "{'SqlServerBoxLuggageRepository' if backend == 'sqlserver' else 'PostgresBoxLuggageRepository'}",
                )
                expected_okdatapart_cls = install_fake(
                    f"backend.app.repositories.{{package_name}}.ok_data_part_repository",
                    "{'SqlServerOkDataPartRepository' if backend == 'sqlserver' else 'PostgresOkDataPartRepository'}",
                )

                import backend.app.core.container as container
                from backend.app.services.box_tracking_service import BoxTrackingService

                if not isinstance(container.box_luggage_repo, expected_box_cls):
                    raise AssertionError(type(container.box_luggage_repo).__name__)
                if not isinstance(container.ok_data_part_repo, expected_okdatapart_cls):
                    raise AssertionError(type(container.ok_data_part_repo).__name__)
                if not isinstance(container.box_tracking_service, BoxTrackingService):
                    raise AssertionError(type(container.box_tracking_service).__name__)
                if container.inspection_session_service._box_tracking_service is not container.box_tracking_service:
                    raise AssertionError("InspectionSessionService not wired with the container's box_tracking_service")
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
                    f"box tracking container wiring probe failed for {backend}\n"
                    f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )

    def test_sqlserver_uses_sqlserver_box_repos(self) -> None:
        self._run_container_probe("sqlserver")

    def test_postgresql_uses_postgresql_box_repos(self) -> None:
        self._run_container_probe("postgresql")


if __name__ == "__main__":
    unittest.main()
