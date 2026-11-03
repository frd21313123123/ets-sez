"""Local game profiles: listing them, creating one and arranging its active mods.

The game encrypts a profile each time it saves one, with plain text inside. It also loads plain
text and encrypts it again on its next save, so an edited profile is simply written as text.
Only local profiles are handled: Steam Cloud ones live in Steam's folder and are synced by it.
In active_mods a higher index means a higher priority, the last entry is on top in the game.
"""
import os
import re
import shutil
import time
from dataclasses import dataclass

from . import scsc

# Lowest priority first, the same order as in profile.sii.
SEASON_MODS = (("ets_seasons_map", "Seasons: map data"), ("ets_seasons_date", "Seasons: date"))
DEFAULT_PROFILE = "Сезоны"
BACKUP_SUFFIX = ".seasons_bak"

_MOD_LINE = re.compile(r'^[ \t]*active_mods\[\d*\]:[ \t]*"([^"]*)"[ \t]*\r?\n', re.M)
_COUNT_LINE = re.compile(r"^([ \t]*)active_mods:[ \t]*\d+([ \t]*\r?\n)", re.M)
_NAME_LINE = re.compile(r'^\s*profile_name:\s*"([^"]*)"', re.M)

TEMPLATE = """SiiNunit
{{
user_profile : _nameless.0.1 {{
 face: 0
 brand: scania_r
 map_path: "/map/europe.mbd"
 logo: logo_0
 company_name: "{name}"
 male: true
 cached_experience: 0
 cached_distance: 0
 user_data: 0
 active_mods: 0
 customization: 0
 cached_stats: 0
 cached_discovery: 0
 profile_name: "{name}"
 creation_time: {now}
 save_time: {now}
 version: 6
 online_user_name: ""
 online_password: ""
}}
}}
"""


@dataclass
class Profile:
    name: str
    folder: str
    mods: list

    @property
    def file(self):
        return os.path.join(self.folder, "profile.sii")

    @property
    def foreign_mods(self):
        ours = {m[0] for m in SEASON_MODS}
        return [m for m in self.mods if m.split("|")[0] not in ours]


def folder_name(name):
    return name.encode("utf8").hex().upper()


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def read_text(path):
    """A profile's text whether the game left it encrypted or not; None if it holds something else."""
    raw = _read(path)
    if scsc.is_encrypted(raw):
        raw = scsc.decode(raw)
    return raw.decode("utf8", "replace") if raw.startswith(b"SiiNunit") else None


def _unescape(value):
    """The game writes non-ASCII characters in names as \\xNN bytes."""
    raw = re.sub(rb"\\x([0-9a-fA-F]{2})", lambda m: bytes([int(m.group(1), 16)]), value.encode("utf8"))
    return raw.decode("utf8", "replace")


def read_mods(text):
    return _MOD_LINE.findall(text)


def arrange(text):
    """Return profile text with the season mods on top of the list, other mods keeping their order."""
    match = _COUNT_LINE.search(text)
    if not match:
        raise ValueError("no active_mods in this profile")
    ours = {m[0] for m in SEASON_MODS}
    others = [m for m in read_mods(text) if m.split("|")[0] not in ours]
    mods = others + [f"{pkg}|{title}" for pkg, title in SEASON_MODS]
    indent, eol = match.group(1), match.group(2)
    lines = [f"{indent}active_mods: {len(mods)}{eol}"]
    lines += [f'{indent}active_mods[{i}]: "{m}"{eol}' for i, m in enumerate(mods)]
    stripped = _MOD_LINE.sub("", text)
    return _COUNT_LINE.sub(lambda _m: "".join(lines), stripped, count=1)


def list_profiles(profiles_dir):
    """Local profiles the launcher can edit; ones it can't read are left out."""
    out = []
    if not os.path.isdir(profiles_dir):
        return out
    for entry in sorted(os.listdir(profiles_dir)):
        folder = os.path.join(profiles_dir, entry)
        path = os.path.join(folder, "profile.sii")
        if not os.path.isfile(path):
            continue
        text = read_text(path)
        if text is None:
            continue
        name = _NAME_LINE.search(text)
        out.append(Profile(_unescape(name.group(1)) if name else entry, folder, read_mods(text)))
    return sorted(out, key=lambda p: p.name.lower())


def create_profile(profiles_dir, name):
    folder = os.path.join(profiles_dir, folder_name(name))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "profile.sii")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf8", newline="\n") as fh:
            fh.write(TEMPLATE.format(name=name, now=int(time.time())))
    return Profile(name, folder, [])


def find_or_create(profiles_dir, name):
    for p in list_profiles(profiles_dir):
        if p.name == name:
            return p
    return create_profile(profiles_dir, name)


def apply_order(profile):
    """Update the mod list in place, touching the file only if the order is wrong.

    The version from right before the change stays next to it as a backup.
    """
    text = read_text(profile.file)
    if text is None:
        raise ValueError(f"не удалось прочитать профиль «{profile.name}»")
    updated = arrange(text)
    if updated == text:
        profile.mods = read_mods(text)
        return False
    shutil.copy2(profile.file, profile.file + BACKUP_SUFFIX)
    tmp = profile.file + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(updated.encode("utf8"))
    os.replace(tmp, profile.file)
    profile.mods = read_mods(updated)
    return True
