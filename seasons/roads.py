"""Snow on road surfaces: per-zone copies of road looks, prefabs, their models and their materials.

A road look or a prefab points to a model, and the model lists its materials by path. So a zone
gets its own look or prefab, its own copy of the model with those paths changed, and its own
materials. Prefab geometry is too big to repeat for every zone, so its copies share one archive
entry under several names.
"""
import copy
import os
import re
import struct
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import materials as mats, model, sectors as sec, textures as tex, tokens

_SII_TOKEN_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|#[^\n]*|//[^\n]*|/\*.*?\*/|[{}]', re.S)
_HEADER_RE = re.compile(r"(\w+)\s*:\s*([\w.]+)\s*$")
_REFERENCE_RE = re.compile(r"(:\s*)(\.[\w.]+)")
_MODEL_RE = re.compile(r'"(/road_template/[^"]+\.pmd)"')
_PREFAB_MODEL_RE = re.compile(r'"(/[^"]+\.pmd)"')
_PREFAB_LINK_RE = re.compile(r"(:\s*prefab\.)(\w+)")
_NAMED_PREFAB_RE = re.compile(rb"(?:\bprefab\.|\bprefab_name:\s*)(\w+)")
_MATERIAL_RE = re.compile(rb"/automat/[0-9a-f]{2}/[0-9a-f]{16}\.mat")
_EFFECT_RE = re.compile(r'effect\s*:\s*"([^"]+)"')
_SUBSTANCE_RE = re.compile(r'substance\s*:\s*"([^"]*)"')
_TEXTURE_RE = re.compile(r'texture\s*:\s*"(\w+)"\s*\{(.*?)\}', re.S)
_SOURCE_RE = re.compile(r'source\s*:\s*"([^"]+)"')
_SNOWY_SLOTS = ("texture_base", "texture_over")
_ZONE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
_ROAD_SUBSTANCES = {"road", "road_dirt", "road_coarse", "dirt", "gravel"}
_GROUND_SUBSTANCES = {"grass"}
# Some surfaces name no substance, so their textures have to tell.
_ROAD_SOURCE_RE = re.compile(r"/material/road/|/material/(terrain|tiled)/([\w/]*/)?(sidewalk|cobble|paving|asphalt)")
_GROUND_SOURCE_RE = re.compile(r"/material/terrain/")
# Wear and dirt drawn over a road. Everything else that is transparent is markings, which stay.
_FADING_RE = re.compile(r"/(road_cracks|road_decal|decal_|overlay_|rumble_stripe)|_decal\b")

PREFAB_ITEM = 4
PREFAB_MODEL_AT = 57  # from the start of a prefab item to its model token

BAKE_WIDTH = 2048  # one repeat spans the width of a road, about 14 m, so this is 7 mm per texel
# A road always keeps some bare tracks, so its top level isn't fully white.
LEVEL_COVER = {1: 0.25, 2: 0.5, 3: 0.75, 4: 0.92}
PACKED_SNOW_TINT = 0.92


@dataclass
class Look:
    name: str
    kind: str
    body: str
    related: list = field(default_factory=list)  # further top level units that belong to this look

    @property
    def models(self):
        found = _MODEL_RE.findall(self.body)
        for _kind, _name, body in self.related:
            found += _MODEL_RE.findall(body)
        return sorted(set(found))


def parse_units(text):
    """List the (kind, name, body) of each unit in .sii text.

    Goes by braces rather than layout, since some of the game's files are indented and some not.
    Quoted strings and comments may contain anything, so those are skipped as whole tokens.
    """
    units = []
    depth = 0
    header_from = body_from = 0
    header = None
    for m in _SII_TOKEN_RE.finditer(text):
        token = m.group()
        if token == "{":
            depth += 1
            if depth == 2:
                source = text[header_from:m.start()]
                header = _HEADER_RE.search(re.sub(r"#[^\n]*|//[^\n]*|/\*.*?\*/", "", source, flags=re.S).strip())
                body_from = m.end()
            elif depth == 1:
                header_from = m.end()
        elif token == "}":
            if depth == 2:
                if header:
                    units.append((header.group(1), header.group(2), text[body_from:m.start()]))
                header_from = m.end()
            depth -= 1
    return units


def references(body):
    """Names of anonymous units this body points to, the ones that start with a dot."""
    return [m.group(2) for m in _REFERENCE_RE.finditer(body)]


def load_looks(files):
    """Every road look with a template model, together with all the units it refers to."""
    looks, units = {}, {}
    for _fname, raw in files.read_all("def/world", "road_look"):
        for kind, name, body in parse_units(raw.decode("utf8", "replace")):
            if kind == "road_look" and name.startswith("road."):
                short = name.split(".", 1)[1]
                if short not in looks and re.fullmatch(r"[0-9a-z_]{1,12}", short):
                    looks[short] = Look(short, kind, body)
            else:
                units.setdefault(name, (kind, name, body))
    for look in looks.values():
        waiting = references(look.body)
        seen = set()
        while waiting:
            name = waiting.pop()
            if name in seen or name not in units:
                continue
            seen.add(name)
            look.related.append(units[name])
            waiting += references(units[name][2])
    return {k: v for k, v in looks.items() if v.models}


def variant_name(zone, index):
    return f"r{zone}_{index}"


def zone_material_path(path, zone):
    """Same length as the original, so a model file can be patched in place."""
    parts = path.split("/")
    parts[2] = "r" + _ZONE_CHARS[zone]
    return "/".join(parts)


def zone_model_path(zone, index):
    return f"/road_template/z{zone}/m{index}.pmd"


def rename_look(look, new_name, models):
    """A full copy of the look under a new name, pointing to the given models.

    The game lets only one owner refer to an anonymous unit, so the copy can't share them with the
    original: each one it refers to is copied as well, under a name that can't collide.
    """
    renamed = {name: f".rv.{new_name}.{i}" for i, (_kind, name, _body) in enumerate(look.related)}

    def retarget(body):
        body = _MODEL_RE.sub(lambda m: f'"{models.get(m.group(1), m.group(1))}"', body)
        return _REFERENCE_RE.sub(lambda m: m.group(1) + renamed.get(m.group(2), m.group(2)), body)

    out = [f"{look.kind} : road.{new_name}\n{{{retarget(look.body)}}}\n"]
    for kind, name, body in look.related:
        out.append(f"{kind} : {renamed[name]}\n{{{retarget(body)}}}\n")
    return "".join(out)


def patch_model(pmd, replacements):
    """Swap material paths inside a model file. Paths keep their length, so nothing else moves."""
    out = pmd
    for old, new in replacements.items():
        if len(old) != len(new):
            raise ValueError(f"material path length changed: {old}")
        out = out.replace(old.encode("ascii"), new.encode("ascii"))
    return out


def find_refs(data, sector, wanted):
    """Offsets of road look tokens in a sector's .base file.

    A road item stores its bounding box right before the look. What follows the look varies from
    road to road, so only the box can tell: it has to lie in this very sector, which numbers that
    merely look like a token won't manage.
    """
    size = 4000.0
    x0, x1 = (sector.x - 1) * size, (sector.x + 2) * size
    z0, z1 = (sector.z - 1) * size, (sector.z + 2) * size
    refs = []
    for raw, value in wanted.items():
        p = data.find(raw)
        while p >= 0:
            if p >= 49:
                min_x, _y, min_z = struct.unpack_from("<3f", data, p - 49)
                max_x, _y2, max_z = struct.unpack_from("<3f", data, p - 29)
                if x0 <= min_x <= max_x <= x1 and z0 <= min_z <= max_z <= z1:
                    refs.append((p, value))
            p = data.find(raw, p + 1)
    return sorted(refs)


def patch_refs(data, refs, zone_low, frac, index_of, zones):
    """Swap road looks for their zone copies; returns (data, swaps).

    Road pieces split between two neighbouring zones the same way terrain tables do.
    """
    out = bytearray(data)
    swapped = 0
    for p, value in refs:
        take_upper = sec.content_choice(data, p - 49, 57, frac)
        zone = zone_low + 1 if take_upper else zone_low
        index = index_of.get(value)
        if index is None or zone not in zones:
            continue
        struct.pack_into("<Q", out, p, tokens.encode(variant_name(zone, index)))
        swapped += 1
    return bytes(out), swapped


def snow_mode(effect, substance, base_source):
    """How snow treats a material: "road", "ground", "fade", or None when it stays as it is."""
    if "decal" in effect or ".a." in effect:  # the alpha channel is opacity here, so it can't be painted over
        return "fade" if substance in _ROAD_SUBSTANCES or _FADING_RE.search(base_source) else None
    if substance in _GROUND_SUBSTANCES:
        return "ground"
    if substance in _ROAD_SUBSTANCES or _ROAD_SOURCE_RE.search(base_source):
        return "road"
    return "ground" if _GROUND_SOURCE_RE.search(base_source) else None


def _material_info(files, path, textures_dir):
    """The material's text and the textures to bake, or None if snow doesn't touch it."""
    raw = files.read(path)
    if raw is None:
        return None
    text = mats.absolute_sources(raw.decode("utf8", "replace"), path)
    effect = _EFFECT_RE.search(text)
    substance = _SUBSTANCE_RE.search(text)
    if not effect or "none" in effect.group(1):
        return None
    slots = {}
    for slot, body in _TEXTURE_RE.findall(text):
        src = _SOURCE_RE.search(body)
        if slot in _SNOWY_SLOTS and src:
            dds = os.path.join(textures_dir, src.group(1).lstrip("/").rsplit(".", 1)[0] + ".dds")
            if not os.path.exists(dds):
                return None
            slots[slot] = src.group(1)
    if "texture_base" not in slots:
        return None
    mode = snow_mode(effect.group(1), substance.group(1) if substance else "", slots["texture_base"])
    return {"path": path, "text": text, "slots": slots, "mode": mode} if mode else None


def baked_path(key, level):
    return f"/material/seasons/road/{key}_{level}.tobj"


def _bake_job(job):
    """Worker: every snow level of one texture. Runs in a separate process."""
    textures_dir, read_image, path, mode, noise_from, key, snow_rgb = job
    base = read_image(textures_dir, path, BAKE_WIDTH)
    if base is None:
        return path, key, None
    h, w = base.shape[:2]
    if mode == "road":
        source = base if noise_from == path else read_image(textures_dir, noise_from, BAKE_WIDTH)
        noise = tex.resampled(tex.road_noise(base if source is None else source), h, w)
        snow = tex.packed_snow(h, w, snow_rgb)
        paint = lambda cover: tex.bake_road(base, snow, noise, cover)
    elif mode == "ground":
        tile = read_image(textures_dir, "/material/terrain/snow.tobj", 256)
        tile = tex.flat_snow(256) if tile is None else tile
        noise = tex.periodic_noise(256, seed=19)
        paint = lambda cover: tex.bake(base, tile, noise, cover)
    else:
        paint = lambda cover: tex.fade_decal(base, cover)
    return path, key, {level: tex.write_dds(paint(cover)) for level, cover in LEVEL_COVER.items()}


def bake_jobs(found):
    """One entry per distinct texture: (path, mode, texture its snow pattern follows).

    The layers of a two layer road share their UV, so the pattern comes from the base layer of
    both, or snow wouldn't line up where one layer blends into the other.
    """
    jobs = {}
    for info in found:
        for slot, path in info["slots"].items():
            jobs.setdefault(path, (path, info["mode"], info["slots"]["texture_base"]))
    return list(jobs.values())


def _bake_textures(found, zf, textures_dir, read_image, log):
    """Bake every distinct road texture. Returns {tobj path: key}."""
    stock = read_image(textures_dir, "/material/terrain/snow.tobj", 256)
    rgb = tex.SNOW_RGB if stock is None else stock[:, :, :3].mean(axis=(0, 1))
    snow_rgb = tuple(float(v) * PACKED_SNOW_TINT for v in rgb)
    jobs = bake_jobs(found)
    work = [(textures_dir, read_image, path, mode, noise_from, f"r{i}", snow_rgb)
            for i, (path, mode, noise_from) in enumerate(jobs)]
    keys = {}
    with ProcessPoolExecutor() as pool:
        for path, key, levels in pool.map(_bake_job, work):
            if levels is None:
                continue
            for level, data in levels.items():
                dds = f"/material/seasons/road/{key}_{level}.dds"
                zf.writestr(dds[1:], data)
                zf.writestr(baked_path(key, level)[1:], tex.write_tobj(dds))
            keys[path] = key
            if len(keys) % 50 == 0:
                log(f"  road textures baked: {len(keys)}/{len(jobs)}")
    log(f"road textures baked: {len(keys)} of {len(jobs)}, {Counter(j[1] for j in jobs)}")
    return keys


def snowy_text(info, level, keys):
    """A road material with snow baked into its surface textures."""
    text = info["text"]
    for path in info["slots"].values():
        text = text.replace(f'"{path}"', f'"{baked_path(keys[path], level)}"')
    return text


def tally(sector_list, positions, zones, looks):
    """How often each look occurs in sectors that ever see snow."""
    wanted = {struct.pack("<Q", tokens.encode(k)): k for k in looks}
    counts = Counter()
    for s in sector_list:
        low = min(int(positions[s.name]), model.ZONE_COUNT - 2)
        if low not in zones and low + 1 not in zones:
            continue
        data = sec.read_sector(s, "base")
        if data:
            for _p, name in find_refs(data, s, wanted):
                counts[name] += 1
    return counts


def prefabs_known_by_name(files):
    """Prefabs the game looks up by their name, such as garages: a copy wouldn't be recognised."""
    names = set()
    for fname, raw in files.read_all("def", ""):
        if fname.endswith(".sii"):
            names.update(m.decode("ascii") for m in _NAMED_PREFAB_RE.findall(raw))
    return names


def without_prefabs(prefabs, names):
    """Leave out `names` and whatever links to them, as a copy can't link outside its own file."""
    kept = {k: v for k, v in prefabs.items() if k not in names}
    while True:
        broken = [k for k, v in kept.items() if any(t not in kept for t in prefab_links(v))]
        if not broken:
            return kept
        for name in broken:
            del kept[name]


def load_prefabs(files):
    """Every prefab_model unit that can be copied as it is: {name: body}."""
    prefabs = {}
    for fname, raw in files.read_all("def/world", "prefab"):
        if fname.startswith("prefab_corner"):
            continue
        for kind, name, body in parse_units(raw.decode("utf8", "replace")):
            short = name.split(".", 1)[1] if "." in name else name
            if kind == "prefab_model" and not references(body):
                prefabs.setdefault(short, body)
    return without_prefabs(prefabs, prefabs_known_by_name(files))


def find_prefab_refs(data, sector, wanted):
    """Offsets of prefab model tokens in a sector's .base file: [(offset, token)].

    A prefab item starts with its type and id, followed by its bounding box and then the model.
    `wanted` is the set of model tokens; the type and the box tell real items from lookalikes.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size < PREFAB_MODEL_AT + 8:
        return []
    limit = arr.size - PREFAB_MODEL_AT - 8
    starts = np.nonzero((arr[:limit] == PREFAB_ITEM) & (arr[1:limit + 1] == 0) & (arr[2:limit + 2] == 0) &
                        (arr[3:limit + 3] == 0))[0]
    size = 4000.0
    x0, x1 = (sector.x - 1) * size, (sector.x + 2) * size
    z0, z1 = (sector.z - 1) * size, (sector.z + 2) * size
    refs = []
    for start in starts.tolist():
        p = start + PREFAB_MODEL_AT
        value = struct.unpack_from("<Q", data, p)[0]
        if value not in wanted:
            continue
        min_x, _y, min_z = struct.unpack_from("<3f", data, start + 12)
        max_x, _y2, max_z = struct.unpack_from("<3f", data, start + 32)
        if x0 <= min_x <= max_x <= x1 and z0 <= min_z <= max_z <= z1:
            refs.append((p, value))
    return refs


def prefab_zone(data, p, zone_low, frac):
    """The zone a prefab goes to, decided by the item's own bytes as with terrain tables."""
    return zone_low + 1 if sec.content_choice(data, p - PREFAB_MODEL_AT, PREFAB_MODEL_AT, frac) else zone_low


def patch_prefab_refs(original, patched, refs, zone_low, frac, index_of, zones):
    """Swap prefab models for their zone copies in `patched`; returns (data, swaps)."""
    out = bytearray(patched)
    swapped = 0
    for p, value in refs:
        zone = prefab_zone(original, p, zone_low, frac)
        index = index_of.get(value)
        if index is None or zone not in zones:
            continue
        struct.pack_into("<Q", out, p, tokens.encode(prefab_name(zone, index)))
        swapped += 1
    return bytes(out), swapped


def prefab_name(zone, index):
    return f"sp{zone}_{index}"


def prefab_model_path(zone, index):
    """Fixed width, so all the names of one shared archive entry have the same length."""
    return f"/prefab_seasons/z{zone:02d}/m{index:05d}.pmd"


def tally_prefabs(sector_list, positions, zones, prefabs):
    """Which zones each prefab appears in: {name: set of zones}."""
    by_token = {tokens.encode(k): k for k in prefabs}
    wanted = set(by_token)
    found = defaultdict(set)
    for s in sector_list:
        low = min(int(positions[s.name]), model.ZONE_COUNT - 2)
        if low not in zones and low + 1 not in zones:
            continue
        data = sec.read_sector(s, "base")
        if not data:
            continue
        frac = positions[s.name] - low
        for p, value in find_prefab_refs(data, s, wanted):
            zone = prefab_zone(data, p, low, frac)
            if zone in zones:
                found[by_token[value]].add(zone)
    spread_over_links(found, prefabs)
    return found


def prefab_links(body):
    """Other prefabs this one points to: its versions for the other side of traffic."""
    return [m.group(2) for m in _PREFAB_LINK_RE.finditer(body)]


def spread_over_links(found, prefabs):
    """Give linked prefabs the same zones, since a copy has to link to a copy from its own zone."""
    waiting = list(found)
    while waiting:
        name = waiting.pop()
        for target in prefab_links(prefabs[name]):
            if target in prefabs and not found[name] <= found[target]:
                found[target] |= found[name]
                waiting.append(target)


def with_linked(names, prefabs):
    """`names` plus all it links to: the loader discards a whole file over one link it can't resolve."""
    out = set(names)
    waiting = list(out)
    while waiting:
        for target in prefab_links(prefabs[waiting.pop()]):
            if target in prefabs and target not in out:
                out.add(target)
                waiting.append(target)
    return out


def copied_prefab(body, zone, model_paths, index_of):
    """A prefab's body for one zone: its models and its links point to copies from that zone."""
    body = _PREFAB_MODEL_RE.sub(lambda m: f'"{model_paths[m.group(1)]}"' if m.group(1) in model_paths else m.group(0),
                                body)
    return _PREFAB_LINK_RE.sub(lambda m: m.group(1) + prefab_name(zone, index_of[m.group(2)]), body)


def add_alias(zf, source, target):
    """Another name for an archive entry that was already written, costing no extra data.

    The game finds files through the archive's central directory alone. Names should keep their
    length, as the local header of the entry keeps saying the first one.
    """
    info = copy.copy(zf.getinfo(source))
    info.filename = info.orig_filename = target
    zf.filelist.append(info)
    zf.NameToInfo[target] = info


def read_shared(zf, info):
    """Read an entry that may have other names: zipfile itself refuses to open those."""
    zf.fp.seek(info.header_offset)
    head = zf.fp.read(30)
    name_length, extra_length = struct.unpack_from("<HH", head, 26)
    zf.fp.seek(info.header_offset + 30 + name_length + extra_length)
    data = zf.fp.read(info.compress_size)
    return zlib.decompress(data, -15) if info.compress_type else data


def _write_prefabs(zf, files, prefab_zones, prefabs, model_mats, usable_paths, log):
    """Copy the prefabs that have a snowy surface. Returns {prefab token: index}."""
    model_zones = defaultdict(set)
    for name, zones_here in prefab_zones.items():
        for path in _PREFAB_MODEL_RE.findall(prefabs[name]):
            if model_mats.get(path, set()) & usable_paths:
                model_zones[path] |= zones_here
    model_index = {path: i for i, path in enumerate(sorted(model_zones))}
    entries = 0
    for path, index in model_index.items():
        raw, geometry = files.read(path), files.read(path[:-1] + "g")
        collision = files.read(path[:-1] + "c")
        swaps = model_mats[path] & usable_paths
        first = None
        for zone in sorted(model_zones[path]):
            target = prefab_model_path(zone, index)[1:]
            zf.writestr(target, patch_model(raw, {n: zone_material_path(n, zone) for n in swaps}))
            if first is None:
                first = target
                zf.writestr(target[:-1] + "g", geometry)
                if collision is not None:
                    zf.writestr(target[:-1] + "c", collision)
            else:
                add_alias(zf, first[:-1] + "g", target[:-1] + "g")
                if collision is not None:
                    add_alias(zf, first[:-1] + "c", target[:-1] + "c")
            entries += 1
    log(f"prefab models: {len(model_index)}, copies for zones: {entries}")

    chosen = sorted(with_linked((name for name in prefab_zones
                                 if any(p in model_index for p in _PREFAB_MODEL_RE.findall(prefabs[name]))), prefabs))
    index_of = {name: i for i, name in enumerate(chosen)}
    token_index = {tokens.encode(name): i for name, i in index_of.items()}
    defs = ["SiiNunit\n{\n"]
    for name, i in index_of.items():
        for zone in sorted(prefab_zones[name]):
            paths = {path: prefab_model_path(zone, index) for path, index in model_index.items()
                     if zone in model_zones[path]}
            body = copied_prefab(prefabs[name], zone, paths, index_of)
            defs.append(f"prefab_model : prefab.{prefab_name(zone, i)}\n{{{body}}}\n")
    defs.append("}\n")
    zf.writestr("def/world/prefab.ets_seasons.sii", "".join(defs))
    log(f"prefabs with snow: {len(chosen)} of {len(prefab_zones)}, units: {len(defs) - 2}")
    return token_index


def build_into(zf, files, sector_list, positions, zones, textures_dir, read_image, log):
    """Write looks, prefabs, models and textures into the map pack.

    Returns ({look token: index}, {prefab token: index}, material records for the date pack).
    """
    looks = load_looks(files)
    counts = tally(sector_list, positions, set(zones), looks)
    used = [looks[name] for name, _n in counts.most_common()]
    log(f"road looks in snowy sectors: {len(used)} of {len(looks)}, references: {sum(counts.values())}")
    prefabs = load_prefabs(files)
    prefab_zones = tally_prefabs(sector_list, positions, set(zones), prefabs)
    log(f"prefabs in snowy sectors: {len(prefab_zones)} of {len(prefabs)}")

    model_data = {}
    material_info = {}

    def learn_materials(raw):
        names = {m.decode("ascii") for m in _MATERIAL_RE.findall(raw)}
        for name in names:
            if name not in material_info:
                material_info[name] = _material_info(files, name, textures_dir)
        return names

    for look in used:
        for path in look.models:
            if path in model_data:
                continue
            raw = files.read(path)
            geometry = files.read(path[:-1] + "g")
            if raw is None or geometry is None:
                continue
            model_data[path] = (raw, geometry)
            learn_materials(raw)
    prefab_model_mats = {}
    for name in prefab_zones:
        for path in _PREFAB_MODEL_RE.findall(prefabs[name]):
            if path not in prefab_model_mats:
                raw = files.read(path)
                if raw is not None and files.read(path[:-1] + "g") is not None:
                    prefab_model_mats[path] = learn_materials(raw)
    usable = [info for info in material_info.values() if info]
    log(f"road models: {len(model_data)}, prefab models: {len(prefab_model_mats)}, "
        f"surface materials: {len(usable)} of {len(material_info)}")

    keys = _bake_textures(usable, zf, textures_dir, read_image, log)
    usable = [i for i in usable if all(p in keys for p in i["slots"].values())]
    usable_paths = {i["path"] for i in usable}
    prefab_index = _write_prefabs(zf, files, prefab_zones, prefabs, prefab_model_mats, usable_paths, log)

    model_index = {}
    for path, (raw, _geometry) in model_data.items():
        names = {m.decode("ascii") for m in _MATERIAL_RE.findall(raw)}
        if names & usable_paths:
            model_index[path] = (len(model_index), sorted(names & usable_paths))
    for zone in zones:
        for path, (index, names) in model_index.items():
            raw, geometry = model_data[path]
            target = zone_model_path(zone, index)
            swaps = {n: zone_material_path(n, zone) for n in names}
            zf.writestr(target[1:], patch_model(raw, swaps))
            zf.writestr(target[1:-1] + "g", geometry)
    log(f"road models copied: {len(model_index)} for each of {len(zones)} zones")

    chosen = [look for look in used if any(p in model_index for p in look.models)]
    defs = ["SiiNunit\n{\n"]
    token_index = {}
    for i, look in enumerate(chosen):
        token_index[tokens.encode(look.name)] = i
        for zone in zones:
            models = {p: zone_model_path(zone, model_index[p][0]) for p in look.models if p in model_index}
            defs.append(rename_look(look, variant_name(zone, i), models))
    defs.append("}\n")
    zf.writestr("def/world/road_look.ets_seasons.sii", "".join(defs))
    records = [{**info, "keys": {p: keys[p] for p in info["slots"].values()}} for info in usable]
    log(f"road looks with snow: {len(chosen)}")
    return token_index, prefab_index, records


def date_files(records, zone, level):
    """Yield (zip path, material text) for one zone at one snow level."""
    for info in records:
        target = zone_material_path(info["path"], zone).lstrip("/")
        yield target, info["text"] if level == 0 else snowy_text(info, level, info["keys"])
