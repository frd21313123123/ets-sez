"""Offline integrity checks of the built packs against the game's own files."""
import json
import re
import struct
import zipfile
import zlib
from collections import Counter

from . import build, hashfs, materials as mats, model, roads, sectors as sec, textures as tex, tokens

_CLIMATE_RE = re.compile(r"climate_profile\s*:\s*climate\.(\w+)")
_VARIANT_RE = re.compile(r'material_def\s*:\s*terrain_mat\.(\w+)\s*\{\s*path:\s*"([^"]+)"')
_SOURCE_RE = re.compile(r'source\s*:\s*"([^"]+)"')


def _balanced(text):
    return text.count("{") == text.count("}")


def worst_cover_step(sectors):
    """The largest difference in snow cover between neighbouring sectors over the year."""
    table = model.ZoneTable()
    worst = 0.0
    for day in range(1, 366, 5):
        covers = build.cover_map(sectors, table, day)
        for (x, z), cover in covers.items():
            for near in ((x + 1, z), (x, z + 1)):
                if near in covers:
                    worst = max(worst, abs(covers[near] - cover))
    return worst


def unit_rule_problems(units):
    """The rules of the game's .sii loader. Breaking one makes it discard the whole file."""
    out = []
    counts = Counter(name for _kind, name, _body in units)
    repeated = [name for name, n in counts.items() if n > 1]
    if repeated:
        out.append(f"{len(repeated)} units defined more than once, e.g. {repeated[0]}")
    long_parts = [name for name in counts if any(len(part) > 12 for part in name.split("."))]
    if long_parts:
        out.append(f"unit name part over 12 chars: {long_parts[0]}")
    owners = Counter()
    for _kind, name, body in units:
        for ref in roads.references(body):
            owners[ref] += 1
            if ref not in counts:
                out.append(f"{name} refers to {ref}, which belongs to another file")
    shared = [name for name, n in owners.items() if n > 1]
    if shared:
        out.append(f"{len(shared)} anonymous units have several owners, e.g. {shared[0]}")
    return out[:6]


def corrupt_entry(archive):
    """Like ZipFile.testzip, for an archive where several names can share one entry."""
    seen = set()
    for info in archive.infolist():
        if info.header_offset in seen:
            continue
        seen.add(info.header_offset)
        if zlib.crc32(roads.read_shared(archive, info)) != info.CRC:
            return info.filename
    return None


def prefab_link_problems(units):
    """A prefab may only link to prefabs of its own file, or the loader discards the file."""
    defined = {name for _kind, name, _body in units}
    return [f"{name} links to {target}, which this file doesn't define"
            for _kind, name, body in units for target in roads.prefab_links(body)
            if f"prefab.{target}" not in defined][:6]


def _check_prefabs(mz, map_names, problems, log):
    """Prefab copies refer to their models, each with geometry. Returns (prefab tokens, models)."""
    name = "def/world/prefab.ets_seasons.sii"
    if name not in map_names:
        return set(), []
    text = mz.read(name).decode("utf8", "replace")
    if not _balanced(text):
        problems.append("prefabs: unbalanced braces")
    units = roads.parse_units(text)
    problems.extend(unit_rule_problems(units))
    problems.extend(prefab_link_problems(units))
    for path in set(re.findall(r'"(/prefab_seasons/z\d+/m\d+\.pmd)"', text)):
        if path[1:] not in map_names:
            problems.append(f"prefab refers to missing model {path}")
    prefabs = [unit.split(".", 1)[1] for kind, unit, _body in units if kind == "prefab_model"]
    log(f"prefabs: {len(prefabs)}")
    return {tokens.encode(v) for v in prefabs}, sorted(n for n in map_names if n.startswith("prefab_seasons/")
                                                       and n.endswith(".pmd"))


def _check_roads(mz, dz, map_names, date_names, problems, log, prefab_models=()):
    """Road copies must form a closed set: look -> model -> material -> texture. Returns look tokens."""
    name = "def/world/road_look.ets_seasons.sii"
    if name not in map_names:
        return set()
    text = mz.read(name).decode("utf8", "replace")
    if not _balanced(text):
        problems.append("road looks: unbalanced braces")
    units = roads.parse_units(text)
    looks = [name.split(".", 1)[1] for kind, name, _body in units if kind == "road_look"]
    problems.extend(unit_rule_problems(units))
    models = sorted(n for n in map_names if re.fullmatch(r"road_template/z\d+/m\d+\.pmd", n)) + list(prefab_models)
    for path in set(re.findall(r'"(/road_template/z\d+/m\d+\.pmd)"', text)):
        if path[1:] not in map_names:
            problems.append(f"road look refers to missing model {path}")
    used_materials = set()
    for model_name in models:
        if model_name[:-1] + "g" not in map_names:
            problems.append(f"{model_name}: no geometry file")
        for raw in re.findall(rb"/automat/r[0-9a-z]/[0-9a-f]{16}\.mat", mz.read(model_name)):
            used_materials.add(raw.decode("ascii"))
    for path in used_materials:
        if path[1:] not in date_names:
            problems.append(f"road material {path} missing from the date pack")
            break
    for path in sorted(n for n in date_names if n.startswith("automat/r")):
        body = dz.read(path).decode("utf8", "replace")
        if not _balanced(body):
            problems.append(f"{path}: unbalanced braces")
        for src in _SOURCE_RE.findall(body):
            if "/material/seasons/" in src and src[1:] not in map_names:
                problems.append(f"{path}: missing baked texture {src}")
    log(f"road looks: {len(looks)}, models: {len(models)}, materials in use: {len(used_materials)}")
    return {tokens.encode(v) for v in looks}


def verify(game_dir, map_pack, date_pack, log=print):
    problems = []
    with open(map_pack + ".state.json", encoding="utf8") as fh:
        state = json.load(fh)
    step = worst_cover_step(state["sectors"])
    log(f"largest snow step between neighbouring sectors: {step:.2f}")
    if step > 1 / model.SNOW_LEVELS:
        problems.append(f"snow jumps by more than one level between sectors: {step:.2f}")
    files = hashfs.GameFiles(game_dir)
    stock = mats.load_materials(files)
    stock_tokens = {mats.token_value(n) for n in stock}
    with zipfile.ZipFile(map_pack) as mz, zipfile.ZipFile(date_pack) as dz:
        map_names, date_names = set(mz.namelist()), set(dz.namelist())

        for z, label in ((mz, "map"), (dz, "date")):
            for need in ("manifest.sii", "description.txt"):
                if need not in z.namelist():
                    problems.append(f"{label}: no {need}")
            bad = corrupt_entry(z)
            if bad:
                problems.append(f"{label}: corrupt entry {bad}")

        defs = mz.read("def/world/terrain_material.ets_seasons.sii").decode()
        variants = dict(_VARIANT_RE.findall(defs))
        if not _balanced(defs):
            problems.append("material definitions: unbalanced braces")
        too_long = [v for v in variants if len(v) > 12]
        if too_long:
            problems.append(f"tokens over 12 chars: {too_long[:3]}")
        for v, path in variants.items():
            if path.lstrip("/") not in date_names:
                problems.append(f"{v}: material {path} missing from the date pack")
                break
        log(f"material variants: {len(variants)}")

        checked = 0
        for name in date_names:
            if not name.endswith(".mat"):
                continue
            text = dz.read(name).decode("utf8", "replace")
            if not _balanced(text):
                problems.append(f"{name}: unbalanced braces")
            for src in _SOURCE_RE.findall(text):
                rel = src.lstrip("/")
                if not src.startswith("/"):
                    problems.append(f"{name}: relative texture {src}")
                elif rel not in map_names and not any(a.has(rel) for a in files.archives):
                    problems.append(f"{name}: texture {src} exists nowhere")
            checked += 1
        log(f"materials checked: {checked}")

        tobjs = [n for n in map_names if n.endswith(".tobj")]
        for name in tobjs:
            raw = mz.read(name)
            length = struct.unpack_from("<I", raw, 40)[0]
            target = raw[48:48 + length].decode()
            if target.lstrip("/") not in map_names:
                problems.append(f"{name}: points to missing {target}")
        for name in [n for n in map_names if n.endswith(".dds")][:8]:
            raw = mz.read(name)
            h, w = struct.unpack_from("<II", raw, 12)
            img = tex.decode_surface(raw[148:148 + ((w + 3) // 4) * ((h + 3) // 4) * 16], w, h, 78)
            if img.shape[:2] != (h, w):
                problems.append(f"{name}: wrong size")
        log(f"textures: {len(tobjs)}")

        climates = set(_CLIMATE_RE.findall(dz.read("def/climate.ets_seasons.sii").decode()))
        sector_list = sec.list_sectors(files)
        known = {mats.token_value(v) for v in variants} | stock_tokens
        all_mats = sec.MaterialSet(known)
        by_name = {s.name: s for s in sector_list}
        desc_seen = patched = swapped = road_swaps = prefab_swaps = 0
        road_wanted = {struct.pack("<Q", v): v for v in state.get("road_tokens", [])}
        prefab_wanted = set(state.get("prefab_tokens", []))
        prefab_variants, prefab_models = _check_prefabs(mz, map_names, problems, log)
        road_variants = _check_roads(mz, dz, map_names, date_names, problems, log, prefab_models)
        for name in map_names:
            m = re.fullmatch(r"map/europe/(sec[+-]\d{4}[+-]\d{4})\.(\w+)", name)
            if not m:
                continue
            s = by_name.get(m.group(1))
            if s is None:
                problems.append(f"{name}: sector not in game")
                continue
            data = mz.read(name)
            original = s.archive.read(f"map/europe/{s.name}.{m.group(2)}")
            if len(data) != len(original):
                problems.append(f"{name}: size changed")
                continue
            if m.group(2) == "desc":
                desc_seen += 1
                token = sec.desc_climate(data)
                if token not in climates:
                    problems.append(f"{name}: climate {token} undefined")
                if data[:24] != original[:24] or data[32:] != original[32:]:
                    problems.append(f"{name}: changed outside the climate token")
                continue
            patched += 1
            before = sec.find_tables(original, sec.MaterialSet(stock_tokens))
            after = sec.find_tables(data, all_mats)
            if [t[0] for t in before] != [t[0] for t in after]:
                problems.append(f"{name}: terrain tables moved")
            allowed = bytearray(data)
            for start, n in after:
                for j in range(n):
                    pos = start + j * 10
                    if allowed[pos:pos + 8] != original[pos:pos + 8]:
                        swapped += 1
                        allowed[pos:pos + 8] = original[pos:pos + 8]
            if m.group(2) == "base":
                for pos, _value in roads.find_refs(original, s, road_wanted):
                    token = struct.unpack_from("<Q", data, pos)[0]
                    if allowed[pos:pos + 8] != original[pos:pos + 8]:
                        if token not in road_variants:
                            problems.append(f"{name}: road look {tokens.decode(token)} undefined")
                        road_swaps += 1
                        allowed[pos:pos + 8] = original[pos:pos + 8]
                for pos, _value in roads.find_prefab_refs(original, s, prefab_wanted):
                    token = struct.unpack_from("<Q", data, pos)[0]
                    if allowed[pos:pos + 8] != original[pos:pos + 8]:
                        if token not in prefab_variants:
                            problems.append(f"{name}: prefab {tokens.decode(token)} undefined")
                        prefab_swaps += 1
                        allowed[pos:pos + 8] = original[pos:pos + 8]
            if bytes(allowed) != original:
                problems.append(f"{name}: bytes changed outside material tokens")
        log(f"sector descriptions: {desc_seen} of {len(sector_list)}")
        log(f"sector files patched: {patched}, material tokens differing: {swapped}, road pieces: {road_swaps}, "
            f"prefabs: {prefab_swaps}")
        if desc_seen != len(sector_list):
            problems.append("not every sector received a climate")
    return problems
