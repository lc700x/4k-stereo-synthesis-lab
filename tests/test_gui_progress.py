import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))
from gui.process import GUIProcessMixin


def test_child_status_marker_routes_to_status_logger(caplog):
    gui = GUIProcessMixin()

    with caplog.at_level(logging.INFO, logger="status"):
        gui._log_child_line("[D2S_STATUS] Downloading weights")

    assert "Downloading weights" in caplog.text


def test_tqdm_download_line_updates_gui_progress_panel():
    gui = GUIProcessMixin()
    item = (
        logging.INFO,
        "child",
        "23:08:43",
        "[23:08:43] [INFO] [child] Downloading (incomplete total...):  68%|######    | 67.1M/99.2M [00:29<00:12, 2.59MB/s]",
    )

    event = gui._progress_event(item)

    assert event["desc"] == "Downloading (incomplete total...)"
    assert event["percent"] == 68.0
    assert event["downloaded"] == "67.1M"
    assert event["size"] == "99.2M"
    assert event["speed"] == "2.59MB/s"
    assert event["eta"] == "00:12"


def test_tqdm_fetch_warning_line_updates_gui_progress_panel():
    gui = GUIProcessMixin()
    item = (
        logging.WARNING,
        "child",
        "23:08:43",
        "[23:08:43] [WARNING] [child] Fetching 5 files:  80%|########  | 4/5 [00:29<00:07,  7.64s/it]",
    )

    event = gui._progress_event(item)

    assert event["desc"] == "Fetching 5 files"
    assert event["percent"] == 80.0
    assert event["downloaded"] == "4"
    assert event["size"] == "5"
