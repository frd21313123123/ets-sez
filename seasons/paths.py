"""Where the game, its documents folder and the project's build output live."""
import ctypes
import os
import re
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_FOLDER = os.path.join("steamapps", "common", "Euro Truck Simulator 2")
FALLBACK_GAME = os.path.join(r"D:\SteamLibrary", GAME_FOLDER)
MAP_PACK = "ets_seasons_map.scs"
DATE_PACK = "ets_seasons_date.scs"


@dataclass(frozen=True)
class Paths:
    game: str
    docs: str
    out: str

    @property
    def mods(self):
        return os.path.join(self.docs, "mod")

    @property
    def profiles(self):
        return os.path.join(self.docs, "profiles")

    @property
    def map_pack(self):
        return os.path.join(self.out, MAP_PACK)

    @property
    def date_pack(self):
        return os.path.join(self.out, DATE_PACK)

    @property
    def state(self):
        return self.map_pack + ".state.json"

    @property
    def settings(self):
        return os.path.join(self.out, "launcher.json")

    @property
    def extracted(self):
        return os.path.join(self.out, "extracted")


def find_game():
    """The game folder in any of the Steam libraries, or the usual guess when Steam can't say."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            steam = winreg.QueryValueEx(key, "SteamPath")[0]
        with open(os.path.join(steam, "steamapps", "libraryfolders.vdf"), encoding="utf8") as fh:
            libraries = re.findall(r'"path"\s+"([^"]+)"', fh.read())
    except OSError:
        return FALLBACK_GAME
    for lib in libraries:
        path = os.path.join(lib.replace("\\\\", "\\"), GAME_FOLDER)
        if os.path.exists(os.path.join(path, "def.scs")):
            return path
    return FALLBACK_GAME


def find_docs():
    """The game's folder in the real Documents, which Windows may have moved from the profile."""
    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf)
    return os.path.join(buf.value, "Euro Truck Simulator 2")


def default_paths():
    return Paths(find_game(), find_docs(), os.path.join(ROOT, "build"))
