"""OS-level actions: cross-platform application launch, window management,
clipboard access, and keystroke injection.

Designed to work on Linux, macOS, and Windows.
"""
from __future__ import annotations

import asyncio
import platform
import subprocess
import sys
from typing import List, Optional

from nova.core.logger import get_logger

logger = get_logger(__name__)

_PLATFORM = platform.system()  # 'Linux', 'Darwin', 'Windows'


# ---------------------------------------------------------------------------
# Application launch
# ---------------------------------------------------------------------------

async def launch_application(name: str) -> bool:
    """Launch an application by name, cross-platform.

    Args:
        name: Human-readable application name, e.g. 'Chrome', 'VS Code'.

    Returns:
        True if launch succeeded, False otherwise.
    """
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _launch_sync, name)
        logger.info("Launched application: %s", name)
        return True
    except Exception as exc:
        logger.error("Failed to launch '%s': %s", name, exc)
        return False


def _launch_sync(name: str) -> None:
    """Blocking launch — run in executor."""
    cmd_map = _build_command_map(name)
    for cmd in cmd_map:
        try:
            if _PLATFORM == "Windows":
                subprocess.Popen(cmd, shell=True)
            else:
                subprocess.Popen(cmd, shell=False, start_new_session=True)
            return
        except FileNotFoundError:
            continue
    raise RuntimeError(f"Could not find a launcher for '{name}'")


def _build_command_map(name: str) -> List:
    """Return ordered list of candidate commands for *name*."""
    lower = name.lower().strip()
    known = {
        "chrome": {
            "Linux": ["google-chrome", "chromium-browser", "chromium"],
            "Darwin": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"],
            "Windows": ["start chrome"],
        },
        "firefox": {
            "Linux": ["firefox"],
            "Darwin": ["/Applications/Firefox.app/Contents/MacOS/firefox"],
            "Windows": ["start firefox"],
        },
        "vs code": {
            "Linux": ["code"],
            "Darwin": ["code"],
            "Windows": ["code"],
        },
        "terminal": {
            "Linux": ["x-terminal-emulator", "gnome-terminal", "xterm"],
            "Darwin": ["open", "-a", "Terminal"],
            "Windows": ["start", "cmd"],
        },
    }
    for key, platforms in known.items():
        if key in lower:
            candidates = platforms.get(_PLATFORM, [])
            return [[c] if not isinstance(c, list) else c for c in candidates]
    # Generic fallback: try the name directly
    return [[lower]]


# ---------------------------------------------------------------------------
# Window management
# ---------------------------------------------------------------------------

def list_open_windows() -> List[str]:
    """Return a list of currently open window titles."""
    if _PLATFORM == "Linux":
        return _list_windows_linux()
    elif _PLATFORM == "Darwin":
        return _list_windows_macos()
    elif _PLATFORM == "Windows":
        return _list_windows_windows()
    return []


def _list_windows_linux() -> List[str]:
    try:
        output = subprocess.check_output(
            ["wmctrl", "-l"], text=True, stderr=subprocess.DEVNULL
        )
        return [
            " ".join(line.split()[3:]).strip()
            for line in output.splitlines() if line.strip()
        ]
    except FileNotFoundError:
        logger.warning("wmctrl not found. Install with: sudo apt install wmctrl")
        return []


def _list_windows_macos() -> List[str]:
    script = 'tell application "System Events" to get name of every window of every process'
    try:
        result = subprocess.check_output(
            ["osascript", "-e", script], text=True, stderr=subprocess.DEVNULL
        )
        return [w.strip() for w in result.split(",") if w.strip()]
    except Exception:
        return []


def _list_windows_windows() -> List[str]:
    try:
        import ctypes
        import ctypes.wintypes
        titles: List[str] = []
        def enum_callback(hwnd, _):
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
                if buf.value:
                    titles.append(buf.value)
        ctypes.windll.user32.EnumWindows(
            ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)(enum_callback),
            0
        )
        return titles
    except Exception as exc:
        logger.error("Win32 window enumeration failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------

def read_clipboard() -> str:
    """Read current clipboard content as a string."""
    try:
        import pyperclip
        return pyperclip.paste() or ""
    except ImportError:
        return _read_clipboard_fallback()


def write_clipboard(text: str) -> None:
    """Write *text* to the system clipboard."""
    try:
        import pyperclip
        pyperclip.copy(text)
    except ImportError:
        _write_clipboard_fallback(text)


def _read_clipboard_fallback() -> str:
    if _PLATFORM == "Linux":
        try:
            return subprocess.check_output(["xclip", "-o", "-sel", "clip"], text=True)
        except Exception:
            return ""
    return ""


def _write_clipboard_fallback(text: str) -> None:
    if _PLATFORM == "Linux":
        proc = subprocess.Popen(
            ["xclip", "-sel", "clip"], stdin=subprocess.PIPE
        )
        proc.communicate(text.encode())


# ---------------------------------------------------------------------------
# Keystrokes
# ---------------------------------------------------------------------------

async def send_keystrokes(keys: str) -> None:
    """Send keyboard input to the focused application.

    Args:
        keys: xdotool-style key string, e.g. 'ctrl+c', 'Return', 'alt+F4'.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send_keys_sync, keys)


def _send_keys_sync(keys: str) -> None:
    if _PLATFORM == "Linux":
        subprocess.run(["xdotool", "key", keys], check=True)
    elif _PLATFORM == "Darwin":
        # Map common key names for osascript
        _osx_keystroke(keys)
    elif _PLATFORM == "Windows":
        import pyautogui
        pyautogui.hotkey(*keys.split("+"))


def _osx_keystroke(keys: str) -> None:
    parts = keys.split("+")
    using = []
    key = parts[-1]
    for mod in parts[:-1]:
        if mod == "ctrl":
            using.append("control down")
        elif mod == "alt":
            using.append("option down")
        elif mod == "shift":
            using.append("shift down")
        elif mod == "cmd":
            using.append("command down")
    using_str = ", ".join(using)
    using_clause = f" using {{{using_str}}}" if using else ""
    script = f'tell application "System Events" to keystroke "{key}"{using_clause}'
    subprocess.run(["osascript", "-e", script], check=True)
