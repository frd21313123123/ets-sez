"""Command line: python -m seasons <map|date|verify|preview|install>. The window is seasons.launcher."""
import argparse
import datetime as dt
import json
import os
import shutil

from . import build, verify
from .paths import default_paths


def parse_date(text):
    return dt.date.today() if text == "today" else dt.date.fromisoformat(text)


def main():
    paths = default_paths()
    p = argparse.ArgumentParser(prog="seasons")
    p.add_argument("--game", default=paths.game)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="build the map pack from the game's files")
    m.add_argument("--top", type=int, default=800, help="terrain materials that receive snow")
    m.add_argument("--textures", default=None, help="folder of game files extracted with SCS's tool")
    sub.add_parser("verify", help="check the built packs against the game's files")
    d = sub.add_parser("date", help="build the date pack")
    d.add_argument("--date", default="today")
    d.add_argument("--install", action="store_true", help="also copy it into the game's mod folder")
    v = sub.add_parser("preview", help="draw snow cover on the map")
    v.add_argument("--date", default=None, help="one date; by default twelve panels of the current year")
    sub.add_parser("install", help="copy both packs into the game's mod folder")
    a = p.parse_args()

    os.makedirs(paths.out, exist_ok=True)
    if a.cmd == "map":
        build.build_map_pack(a.game, paths.map_pack, top=a.top, textures_dir=a.textures)
    elif a.cmd == "verify":
        problems = verify.verify(a.game, paths.map_pack, paths.date_pack)
        print("\n".join(problems[:40]) if problems else "no problems found")
        return 1 if problems else 0
    elif a.cmd == "date":
        with open(paths.state, encoding="utf8") as fh:
            state = json.load(fh)
        date = parse_date(a.date)
        build.build_date_pack(state, date, paths.date_pack)
        print(f"date pack for {date}: {paths.date_pack}")
        if a.install:
            shutil.copy2(paths.date_pack, os.path.join(paths.mods, os.path.basename(paths.date_pack)))
    elif a.cmd == "preview":
        today = dt.date.today()
        dates = [parse_date(a.date)] if a.date else [dt.date(today.year, mo, 15) for mo in range(1, 13)]
        target = os.path.join(paths.out, "preview.png")
        located = build.preview_png(a.game, target, dates)
        print(f"{target} ({len(located)} stations on the map)")
    elif a.cmd == "install":
        for source in (paths.map_pack, paths.date_pack):
            shutil.copy2(source, os.path.join(paths.mods, os.path.basename(source)))
            print("installed", os.path.basename(source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
