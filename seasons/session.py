"""What happens when the player presses Play: packs, mod order, game start."""
import ctypes
import ctypes.wintypes as wt
import json
import os
import shutil
import subprocess
import time

from . import build, profiles

GAME_PROCESS = "eurotrucks2.exe"
STEAM_URL = "steam://rungameid/227300"
NORMAL_WINDOW_WIDTH = 640  # the game's first window is about 136 wide, any real mode is wider
STARTUP_GRACE = 3.0        # when left alone the game resizes itself in about a second
POLL_SECONDS = 1.0


class GameRunning(Exception):
    """The game holds the mods and overwrites profiles on exit, so nothing may change under it."""


class NotBuilt(Exception):
    """The map pack does not exist yet."""


def game_running():
    done = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {GAME_PROCESS}", "/NH"], capture_output=True,
                          text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return GAME_PROCESS.lower() in done.stdout.lower()


def load_state(paths):
    """The map pack's state, or None when it hasn't been built."""
    if not (os.path.exists(paths.state) and all(os.path.exists(pack) for pack in paths.map_packs)):
        return None
    with open(paths.state, encoding="utf8") as fh:
        return json.load(fh)


def complete_state(paths, state, log=lambda _text: None):
    """Fill in what older builds didn't record. Reads the map, which takes about a minute."""
    if "sectors" in state and "game_stamp" in state:
        return state
    log("Считываю карту игры, это нужно один раз")
    state.update(build.field_record(paths.game))
    state.setdefault("game_stamp", build.game_stamp(paths.game))
    with open(paths.state, "w", encoding="utf8") as fh:
        json.dump(state, fh)
    return state


def game_updated(paths, state):
    """True when the game's files differ from those the map pack was built from."""
    return state.get("game_stamp") not in (None, build.game_stamp(paths.game))


def _install(source, target):
    """Make the mod folder hold a pack, without copying it when the system allows a link.

    A link costs no space on the game's drive and always shows the current pack. Windows allows
    links only with Developer Mode on or with administrator rights; without them the pack is copied.
    A copy left half-written by an interrupted run is removed first.
    """
    tmp = target + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    if os.path.islink(target) and os.path.realpath(target) == os.path.realpath(source):
        return False
    try:
        os.symlink(os.path.abspath(source), tmp)
    except OSError:
        if os.path.exists(target):
            a, b = os.stat(source), os.stat(target)
            if a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime):
                return False
        shutil.copy2(source, tmp)
    os.replace(tmp, target)
    return True


def drop_navigation_cache(paths, profile_name):
    """Set aside the game's navigation cache of a profile, which the game would otherwise keep using.

    The cache is built from the map on first use and is never checked again. Once the map has
    changed, the game reads node indices that no longer exist and crashes. It rebuilds the cache
    on its next start; the old one is kept as a .stale file.
    """
    path = os.path.join(paths.docs, "cache", profiles.folder_name(profile_name) + ".navcache")
    if os.path.exists(path):
        os.replace(path, path + ".stale")


def prepare(paths, state, date, profile_name, log=lambda _text: None):
    """Set everything up for one game session and return the profile that was arranged."""
    if state is None:
        raise NotBuilt()
    if game_running():
        raise GameRunning()
    log("Собираю данные на выбранную дату")
    build.build_date_pack(state, date, paths.date_pack)
    os.makedirs(paths.mods, exist_ok=True)
    log("Копирую моды в папку игры")
    map_changed = any([_install(pack, os.path.join(paths.mods, os.path.basename(pack))) for pack in paths.map_packs])
    _install(paths.date_pack, os.path.join(paths.mods, os.path.basename(paths.date_pack)))
    if map_changed:
        drop_navigation_cache(paths, profile_name)
    log("Расставляю моды в профиле")
    profile = profiles.find_or_create(paths.profiles, profile_name)
    profiles.apply_order(profile)
    return profile


def start_game():
    os.startfile(STEAM_URL)


def _game_window():
    """The game's visible top level window, or None."""
    done = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {GAME_PROCESS}", "/NH", "/FO", "CSV"],
                          capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if GAME_PROCESS.lower() not in done.stdout.lower():
        return None
    pid = int(done.stdout.split('","')[1])
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def visit(hwnd, _param):
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(visit, 0)
    return found[0] if found else None


def _window_width(hwnd):
    rect = wt.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.right - rect.left


def _activate(hwnd):
    """Give the window focus. A tap of Alt is what lets a background process do that on Windows."""
    user32 = ctypes.windll.user32
    user32.keybd_event(0x12, 0, 0, 0)
    user32.keybd_event(0x12, 0, 2, 0)
    if user32.IsIconic(hwnd):
        user32.ShowWindowAsync(hwnd, 9)  # the plain call would wait for the busy game to respond
    user32.SetForegroundWindow(hwnd)


def bring_game_up(timeout=180.0, find=_game_window, width=_window_width, activate=_activate, sleep=time.sleep):
    """Make sure the game reaches its normal window.

    It opens tiny and goes to its real size only while it has focus, so a click elsewhere at that
    moment leaves it tiny. Returns True once the window is full sized, False if it never got there.
    """
    waited = 0.0
    seen_for = 0.0
    while waited < timeout:
        hwnd = find()
        if hwnd:
            if width(hwnd) >= NORMAL_WINDOW_WIDTH:
                return True
            if seen_for >= STARTUP_GRACE:
                activate(hwnd)
            seen_for += POLL_SECONDS
        sleep(POLL_SECONDS)
        waited += POLL_SECONDS
    return False
