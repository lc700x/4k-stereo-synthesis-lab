import sys
import types


FAKE_MSS_MONITORS = [
    {"left": 0, "top": 0, "width": 9600, "height": 2160},
    {"left": 5760, "top": 0, "width": 3840, "height": 2160},
    {"left": 3840, "top": 0, "width": 1920, "height": 1080},
    {"left": 0, "top": 0, "width": 3840, "height": 2160},
]


class FakeMssContext:
    monitors = FAKE_MSS_MONITORS

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMssModule:
    @staticmethod
    def mss():
        return FakeMssContext()


class FakeWin32Api:
    @staticmethod
    def EnumDisplayMonitors():
        return [
            ("h5", None, (5760, 0, 9600, 2160)),
            ("h2", None, (3840, 0, 5760, 1080)),
            ("h1", None, (0, 0, 3840, 2160)),
        ]

    @staticmethod
    def GetMonitorInfo(handle):
        return {
            "h5": {"Device": r"\\.\DISPLAY5"},
            "h2": {"Device": r"\\.\DISPLAY2"},
            "h1": {"Device": r"\\.\DISPLAY1"},
        }[handle]


def test_list_monitors_uses_windows_display_numbers(monkeypatch):
    from gui import capture_sources

    monkeypatch.setattr(capture_sources, "OS_NAME", "Windows")
    monkeypatch.setitem(sys.modules, "mss", FakeMssModule)
    monkeypatch.setitem(sys.modules, "win32api", FakeWin32Api)

    monitors = capture_sources.list_monitors()

    assert [mon["display_number"] for mon in monitors] == [1, 2, 3]
    assert [mon["device_display_number"] for mon in monitors] == [1, 2, 5]
    assert [mon["capture_index"] for mon in monitors] == [3, 2, 1]

