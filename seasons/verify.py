"""Offline integrity checks of the built packs against the game's own files."""
import json
import re
import struct
import zipfile
import zlib
from collections import Counter

from . import build, hashfs, mapdata as md, materials as mats, model, roads, sectors as sec, textures as tex, tokens, \
    zones as zn

_CLIMATE_RE = re.compile(r"climate_profile\s*:\s*climate\.(\w+)")
_VARIANT_RE = re.compile(r'material_def\s*:\s*terrain_mat\.(\w+)\s*\{\s*path:\s*"([^"]+)"')
_SOURCE_RE = re.compile(r'source\s*:\s*"([^"]+)"')
_COPY_MODEL_RE = re.compile(r'"(/(?:prefab_seasons|veg_seasons)/z\d+/m\d+\.pmd)"')
_COPY_MATERIAL_RE = re.compile(rb"/automat/r[0-9a-z]/[0-9a-f]{16}\.mat")
_LEGACY_RE = re.compile(r'"(/material/seasons/legacy/z\d+/\d+\.mat)"')
_COPY_FILES = {"look": "road_look", "edge": "road_edge", "prefab": "prefab", "corner": "prefab_corner",
               "vegetation": "vegetation", "model": "model", "curve": "curve_model", "far": "far_model",
               "terrain_edge": "terrain_edge", "road_material": "road_material", "sidewalk": "road_sidewalk"}


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


class Packs:
    """All the map packs read as one."""

    def __init__(self, paths):
        self.zips = [zipfile.ZipFile(path) for path in paths.map_packs]
        self.where = {}
        for zf in self.zips:
            for name in zf.namelist():
                self.where[name] = zf

    def read(self, name):
        zf = self.where[name]
        return roads.read_shared(zf, zf.getinfo(name))

    def close(self):
        for zf in self.zips:
            zf.close()


def _check_copies(packs, date_names, problems, log):
    """Copies must form a closed set: unit -> model -> material -> texture. Returns tokens per kind."""
    names = set(packs.where)
    defined = {}
    models = set()
    for kind, base in _COPY_FILES.items():
        name = f"def/world/{base}.ets_seasons.sii"
        if name not in names:
            problems.append(f"{name} is missing")
            continue
        text = packs.read(name).decode("utf8", "replace")
        if not _balanced(text):
            problems.append(f"{name}: unbalanced braces")
        units = roads.parse_units(text)
        problems.extend(f"{name}: {p}" for p in unit_rule_problems(units))
        if kind == "prefab":
            problems.extend(prefab_link_problems(units))
        # Anonymous units start with a dot. Nothing on the map can refer to those.
        defined[kind] = {tokens.encode(unit.split(".", 1)[1]) for _kind, unit, _body in units if unit[0] != "."}
        models.update(_COPY_MODEL_RE.findall(text))
        for path in _LEGACY_RE.findall(text):
            if path[1:] not in date_names:
                problems.append(f"{name}: material {path} missing from the date pack")
        log(f"{name}: {len(units)} units")
    tree_defs = "def/world/vegetation_model.ets_seasons.sii"
    if tree_defs in names:
        text = packs.read(tree_defs).decode("utf8", "replace")
        problems.extend(f"{tree_defs}: {p}" for p in unit_rule_problems(roads.parse_units(text)))
        models.update(_COPY_MODEL_RE.findall(text))

    used = set()
    for path in sorted(models):
        rel = path[1:]
        if rel not in names:
            problems.append(f"missing model {path}")
            continue
        if rel[:-1] + "g" not in names:
            problems.append(f"{rel}: no geometry file")
        for raw in _COPY_MATERIAL_RE.findall(packs.read(rel)):
            used.add(raw.decode("ascii"))
    absent = [p for p in used if p[1:] not in date_names]
    if absent:
        problems.append(f"{len(absent)} materials missing from the date pack, e.g. {absent[0]}")
    log(f"models referred to: {len(models)}, materials in use: {len(used)}")
    return defined


def _check_sector(args):
    """Worker: what changed in one sector file may only be what the plan says."""
    name, x, z, position, originals, patched = args
    problems = []
    vocab, plan = build._WORK["vocab"], build._WORK["plan"]
    all_materials = build._WORK["all_materials"]
    stock_materials = build._WORK["stock_materials"]
    swapped = Counter()
    try:
        slots = zn.sector_slots(originals, x, z, vocab)
    except md.LayoutError as err:
        return [f"{name}: {err}"], swapped, 0
    low, frac = zn.split_position(position, model.ZONE_COUNT)
    allowed = {kind: bytearray(patched[kind]) for kind in patched}
    for kind, data in patched.items():
        if len(data) != len(originals[kind]):
            problems.append(f"{name}.{kind}: size changed")
            allowed.pop(kind)
    for slot in slots:
        if slot.file not in allowed:
            continue
        original = originals[slot.file][slot.offset:slot.offset + 8]
        if allowed[slot.file][slot.offset:slot.offset + 8] == original:
            continue
        zone = zn.item_zone(slot.uid, low, frac)
        want = plan[slot.kind].get((zone, struct.unpack("<Q", original)[0]))
        got = struct.unpack_from("<Q", allowed[slot.file], slot.offset)[0]
        if want != got:
            problems.append(f"{name}.{slot.file}: {slot.kind} token at {slot.offset} is {tokens.decode(got)}")
        swapped[slot.kind] += 1
        allowed[slot.file][slot.offset:slot.offset + 8] = original
    tables = 0
    for kind, data in allowed.items():
        before = sec.find_tables(originals[kind], stock_materials)
        after = sec.find_tables(patched[kind], all_materials)
        if [t[0] for t in before] != [t[0] for t in after]:
            problems.append(f"{name}.{kind}: terrain tables moved")
        for start, n in after:
            for j in range(n):
                pos = start + j * 10
                if data[pos:pos + 8] != originals[kind][pos:pos + 8]:
                    tables += 1
                    data[pos:pos + 8] = originals[kind][pos:pos + 8]
        if bytes(data) != originals[kind]:
            problems.append(f"{name}.{kind}: bytes changed outside known tokens")
    return problems, swapped, tables


def verify(game_dir, paths, log=print):
    problems = []
    with open(paths.state, encoding="utf8") as fh:
        state = json.load(fh)
    with open(paths.plan, encoding="utf8") as fh:
        plan = {kind: {(zone, token): new for zone, token, new in rows} for kind, rows in json.load(fh).items()}
    step = worst_cover_step(state["sectors"])
    log(f"largest snow step between neighbouring sectors: {step:.2f}")
    if step > 1 / model.SNOW_LEVELS:
        problems.append(f"snow jumps by more than one level between sectors: {step:.2f}")
    files = hashfs.GameFiles(game_dir)
    stock = mats.load_materials(files)
    stock_tokens = {mats.token_value(n) for n in stock}
    packs = Packs(paths)
    try:
        with zipfile.ZipFile(paths.date_pack) as dz:
            map_names, date_names = set(packs.where), set(dz.namelist())
            for z in (*packs.zips, dz):
                for need in ("manifest.sii", "description.txt"):
                    if need not in z.namelist():
                        problems.append(f"{z.filename}: no {need}")
                bad = corrupt_entry(z)
                if bad:
                    problems.append(f"{z.filename}: corrupt entry {bad}")
            log("archives read back with matching checksums")

            defs = packs.read("def/world/terrain_material.ets_seasons.sii").decode()
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
                raw = packs.read(name)
                length = struct.unpack_from("<I", raw, 40)[0]
                target = raw[48:48 + length].decode()
                if target.lstrip("/") not in map_names:
                    problems.append(f"{name}: points to missing {target}")
            for name in [n for n in map_names if n.endswith(".dds")][:8]:
                raw = packs.read(name)
                h, w = struct.unpack_from("<II", raw, 12)
                img = tex.decode_surface(raw[148:148 + ((w + 3) // 4) * ((h + 3) // 4) * 16], w, h, 78)
                if img.shape[:2] != (h, w):
                    problems.append(f"{name}: wrong size")
            log(f"textures: {len(tobjs)}")

            climates = set(_CLIMATE_RE.findall(dz.read("def/climate.ets_seasons.sii").decode()))
            defined = _check_copies(packs, date_names, problems, log)
            for kind, table in plan.items():
                if kind in defined:
                    absent = [new for new in table.values() if new not in defined[kind]]
                    if absent:
                        problems.append(f"{kind}: {len(absent)} planned copies are not defined, "
                                        f"e.g. {tokens.decode(absent[0])}")

            sector_list = sec.list_sectors(files)
            by_name = {s.name: s for s in sector_list}
            # The positions in the state are rounded, while an item's zone depends on the exact one.
            located = {k: tuple(v) for k, v in state["located"].items()}
            positions = build.plan_sectors(sector_list, located, model.STATIONS)
            vocab, _defs = roads.load_vocabulary(files)
            vocab.center = stock_tokens
            variant_tokens = {mats.token_value(v) for v in variants}
            build._init_worker({"vocab": vocab, "plan": plan,
                                "all_materials": sec.MaterialSet(variant_tokens | stock_tokens),
                                "stock_materials": sec.MaterialSet(stock_tokens)})
            descs = patched = tables = 0
            swaps = Counter()
            grouped = {}
            for name in map_names:
                m = re.fullmatch(r"map/europe/(sec[+-]\d{4}[+-]\d{4})\.(\w+)", name)
                if not m:
                    continue
                if m.group(1) not in by_name:
                    problems.append(f"{name}: sector not in game")
                    continue
                if m.group(2) == "desc":
                    s = by_name[m.group(1)]
                    data = packs.read(name)
                    original = s.archive.read(f"map/europe/{s.name}.desc")
                    descs += 1
                    token = sec.desc_climate(data)
                    if token not in climates:
                        problems.append(f"{name}: climate {token} undefined")
                    if data[:24] != original[:24] or data[32:] != original[32:]:
                        problems.append(f"{name}: changed outside the climate token")
                else:
                    grouped.setdefault(m.group(1), []).append(m.group(2))
            for sector_name, kinds in grouped.items():
                s = by_name[sector_name]
                originals = build._sector_files(s.archive, s.name)
                changed = {kind: packs.read(f"map/europe/{s.name}.{kind}") for kind in kinds}
                found, here, in_tables = _check_sector((s.name, s.x, s.z, positions[s.name], originals, changed))
                problems.extend(found)
                swaps.update(here)
                tables += in_tables
                patched += len(kinds)
            log(f"sector descriptions: {descs} of {len(sector_list)}")
            log(f"sector files patched: {patched}, terrain table tokens: {tables}, swapped: {dict(swaps)}")
            if descs != len(sector_list):
                problems.append("not every sector received a climate")
    finally:
        packs.close()
    return problems
