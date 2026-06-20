"""Vendored Flet desktop client provisioning."""
from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

from flet.utils import safe_tar_extractall, safe_zip_extractall
from utils.platform_info import get_arch, get_flet_desktop_artifact_name, get_linux_distro_id, get_os_name

from .paths import GUI_DIR

CLIENTS_DIR = Path(GUI_DIR) / "flet_clients"
PACKAGES_DIR = Path(GUI_DIR) / "flet_packages"

_LINUX_FALLBACK_ARTIFACTS = (
    "flet-linux-ubuntu22.04-light-amd64.tar.gz",
)


def ensure_vendored_flet_view() -> str | None:
    """Ensure a bundled Flet client exists for this OS and set FLET_VIEW_PATH."""
    artifact = _select_artifact_name()
    if artifact is None:
        _print_missing_package_message()
        return None

    archive_path = PACKAGES_DIR / artifact
    extract_dir = CLIENTS_DIR / _archive_stem(artifact)
    view_path = _view_path_for_platform(extract_dir)
    if not _view_path_ready(view_path):
        _print_prepare_message(artifact)
        _extract_archive(archive_path, extract_dir)
        view_path = _view_path_for_platform(extract_dir)
        if not _view_path_ready(view_path):
            raise FileNotFoundError(
                f"Flet desktop client was extracted, but no runnable view was found in {extract_dir}"
            )

    os.environ["FLET_VIEW_PATH"] = str(view_path)
    return str(view_path)


def _select_artifact_name() -> str | None:
    for candidate in _artifact_candidates():
        if (PACKAGES_DIR / candidate).is_file():
            return candidate
    return None


def _artifact_candidates() -> list[str]:
    artifact = _current_artifact_name()
    candidates = [artifact] if artifact else []
    if _is_linux():
        candidates.extend(name for name in _LINUX_FALLBACK_ARTIFACTS if name not in candidates)
    return candidates


def _current_artifact_name() -> str | None:
    return get_flet_desktop_artifact_name()


def _is_linux() -> bool:
    return get_os_name() == "Linux"


def _system_label() -> str:
    os_name = get_os_name()
    if os_name == "Linux":
        return f"Linux {get_linux_distro_id()} {get_arch()}"
    return f"{os_name} {get_arch()}"


def _print_prepare_message(artifact: str) -> None:
    print(f"[Flet GUI] Detected system: {_system_label()}")
    print(f"[Flet GUI] Preparing Flet GUI package: {artifact}")


def _print_missing_package_message() -> None:
    candidates = ", ".join(_artifact_candidates()) or "unknown"
    print(f"[Flet GUI] Detected system: {_system_label()}")
    print(f"[Flet GUI] Missing matching Flet GUI package in: {PACKAGES_DIR}")
    print(f"[Flet GUI] Expected package candidate(s): {candidates}")
    print("[Flet GUI] Please turn on VPN and run the program again.")


def _archive_stem(file_name: str) -> str:
    if file_name.endswith(".tar.gz"):
        return file_name[:-7]
    if file_name.endswith(".zip"):
        return file_name[:-4]
    return Path(file_name).stem


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    tmp_dir = extract_dir.with_name(f"{extract_dir.name}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as archive:
                safe_zip_extractall(archive, str(tmp_dir))
        else:
            with tarfile.open(archive_path, "r:gz") as archive:
                safe_tar_extractall(archive, str(tmp_dir))

        shutil.rmtree(extract_dir, ignore_errors=True)
        tmp_dir.rename(extract_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _view_path_for_platform(extract_dir: Path) -> Path:
    if get_os_name() == "Windows":
        parent = _find_file_parent(extract_dir, "flet.exe")
        return parent or (extract_dir / "flet")

    system_name = get_os_name()
    if system_name == "Darwin":
        return extract_dir

    parent = _find_file_parent(extract_dir, "flet")
    return parent or (extract_dir / "flet")


def _view_path_ready(view_path: Path) -> bool:
    if get_os_name() == "Windows":
        return (view_path / "flet.exe").is_file()

    system_name = get_os_name()
    if system_name == "Darwin":
        if not view_path.is_dir():
            return False
        return any(path.name.endswith(".app") and path.is_dir() for path in view_path.iterdir())

    return (view_path / "flet").is_file()


def _find_file_parent(root: Path, file_name: str) -> Path | None:
    if not root.exists():
        return None
    for path in root.rglob(file_name):
        if path.is_file():
            return path.parent
    return None