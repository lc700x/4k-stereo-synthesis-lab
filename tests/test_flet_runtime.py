from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import gui.flet_runtime as flet_runtime


def test_missing_package_logs_system_and_vpn_hint(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(flet_runtime, "PACKAGES_DIR", tmp_path / "packages")
    monkeypatch.setattr(flet_runtime, "CLIENTS_DIR", tmp_path / "clients")
    monkeypatch.setattr(flet_runtime, "get_os_name", lambda: "Windows")
    monkeypatch.setattr(flet_runtime, "get_arch", lambda: "amd64")
    monkeypatch.setattr(flet_runtime, "get_flet_desktop_artifact_name", lambda: "flet-windows.zip")

    with caplog.at_level(logging.ERROR):
        result = flet_runtime.ensure_vendored_flet_view()

    output = caplog.text
    assert result is None
    assert "Detected system: Windows amd64" in output
    assert "flet-windows.zip" in output
    assert "Please turn on VPN and run the program again." in output


def test_extract_logs_prepare_message_and_sets_view_path(monkeypatch, tmp_path, caplog):
    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()
    with zipfile.ZipFile(packages_dir / "flet-windows.zip", "w") as archive:
        archive.writestr("flet/flet.exe", "")

    monkeypatch.setattr(flet_runtime, "PACKAGES_DIR", packages_dir)
    monkeypatch.setattr(flet_runtime, "CLIENTS_DIR", tmp_path / "clients")
    monkeypatch.setattr(flet_runtime, "get_os_name", lambda: "Windows")
    monkeypatch.setattr(flet_runtime, "get_arch", lambda: "amd64")
    monkeypatch.setattr(flet_runtime, "get_flet_desktop_artifact_name", lambda: "flet-windows.zip")
    monkeypatch.delenv("FLET_VIEW_PATH", raising=False)

    with caplog.at_level(logging.INFO):
        result = flet_runtime.ensure_vendored_flet_view()

    output = caplog.text
    assert result == str(tmp_path / "clients" / "flet-windows" / "flet")
    assert "Detected system: Windows amd64" in output
    assert "Preparing Flet GUI package: flet-windows.zip" in output
    assert (Path(result) / "flet.exe").is_file()