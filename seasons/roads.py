"""Snow on what stands on the map: per-zone copies of roads, prefabs, trees and their materials.

A road look, a road edge, a prefab, a corner or a tree points to a model, and the model lists its
materials by path. So a zone gets its own unit, its own copy of the model with those paths changed,
and its own materials. Geometry is too big to repeat for every zone, so copies share one archive
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

from . import materials as mats, pmx, textures as tex, tokens, zones as zn

_SII_TOKEN_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|#[^\n]*|//[^\n]*|/\*.*?\*/|[{}]', re.S)
_HEADER_RE = re.compile(r"(\w+)\s*:\s*([\w.]+)\s*$")
_REFERENCE_RE = re.compile(r"(:\s*)(\.[\w.]+)")
_MODEL_RE = re.compile(r'"(/road_template/[^"]+\.pmd)"')
_ANY_MODEL_RE = re.compile(r'"(/[^"]+\.pmd)"')
_PREFAB_LINK_RE = re.compile(r"(:\s*prefab\.)(\w+)")
_NAMED_PREFAB_RE = re.compile(rb"(?:\bprefab\.|\bprefab_name:\s*)(\w+)")
_MATERIAL_RE = re.compile(rb"/automat/[0-9a-f]{2}/[0-9a-f]{16}\.mat")
_EFFECT_RE = re.compile(r'effect\s*:\s*"([^"]+)"')
_SUBSTANCE_RE = re.compile(r'substance\s*:\s*"([^"]*)"')
_TEXTURE_RE = re.compile(r'texture\s*:\s*"(\w+)"\s*\{(.*?)\}', re.S)
_SOURCE_RE = re.compile(r'source\s*:\s*"([^"]+)"')
_SLOT_BLOCK_RE = re.compile(r'(texture\s*:\s*"(\w+)"\s*\{[^}]*?source\s*:\s*")([^"]+)(")', re.S)
_PAINTED_SLOT_RE = re.compile(r"texture_(base|over)(_\d+)?")
_MULT_SLOT_RE = re.compile(r"texture_mult(_\d+)?")
_ZONE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
_ROAD_SUBSTANCES = {"road", "road_dirt", "road_coarse", "dirt", "gravel"}
_GROUND_SUBSTANCES = {"grass"}
# Some surfaces name no substance, so their textures have to tell.
_ROAD_SOURCE_RE = re.compile(r"/material/road/|/material/tiled/|/material/(terrain|ground)/([\w/]*/)?"
                             r"(sidewalk|cobble|paving|asphalt)")
_GROUND_SOURCE_RE = re.compile(r"/material/terrain/|/material/ground/")
# Laid in small pieces rather than along the road, so snow on them has no direction.
_TILED_SOURCE_RE = re.compile(r"/tiled/|sidewalk|pavement|cobble|paving|curb|kerb")
_NATURE_SOURCE_RE = re.compile(r"/vegetation/|/asset/(stone|rock|hill)/|/model2?/(rock|hill)/")
# A surface wide enough to hold lanes. Anything else a road substance is on is a yard or a parking lot.
_LANE_SOURCE_RE = re.compile(r"road_template|/material/road/")
_NATURE_MODEL_RE = re.compile(r"/vegetation/|/rock/|/hill|/mountain|/stone|/cliff|/tree|/forest")
# These have lights or glass in them, or draw nothing to put snow on.
_UNTOUCHED_EFFECT_RE = re.compile(r"none|shadowonly|retroreflective|lamp|lightmap|baked|glass|water|\.lum\b")
_SEE_THROUGH_EFFECT_RE = re.compile(r"leaves|\.a\.|asafew|billboard|decal")
# Wear and dirt drawn over a road. Everything else that is transparent is markings, which stay.
_FADING_RE = re.compile(r"/(road_cracks|road_decal|decal_|overlay_|rumble_stripe)|_decal\b")
_LANE_RE = re.compile(r"lanes_(left|right)\[\d*\]\s*:")
_TEMPLATE_RE = re.compile(r'template_(left|right)\s*:\s*"(/[^"]+\.pmd)"')
_COMPATIBLE_EDGE_RE = re.compile(r"(compatible_edges_\w+\[\d*\]\s*:\s*)(\w+)")
_CORNER_RE = re.compile(r"(corner\d\[\d*\]\s*:\s*)(\w+)")
_VEGETATION_MODEL_RE = re.compile(r"(model\[\d*\]\s*:\s*)(vegetmodel\.[\w.]+)")
_MATERIAL_PATH_RE = re.compile(r'((?:path|kerb)\s*:\s*")([^"]+)(")')
_SIMPLE_TOKEN_RE = re.compile(r"[0-9a-z_]{1,12}")

ROAD_WIDTH = 2048   # a road's texture spans about 14 m across, so this is 7 mm per texel
DETAIL_WIDTH = 1024
# A road always keeps some bare tracks, so its top level isn't fully white.
LEVEL_COVER = {1: 0.25, 2: 0.5, 3: 0.75, 4: 0.92}
# Nobody drives on a sidewalk, so snow settles there sooner and stays whole.
TILED_COVER = {1: 0.4, 2: 0.7, 3: 0.92, 4: 1.0}
# Trees shake light snow off, and a light dusting of a whole forest would read as dust.
TREE_COVER = {2: 0.25, 3: 0.45, 4: 0.65}
PACKED_SNOW_TINT = 0.92

WHEEL_OFFSET_M = 1.05   # from the middle of a lane
SHOULDER_IN_MODEL_M = 0.25
MIN_LANE_M, MAX_LANE_M = 2.4, 7.0
MIN_SURFACE_WIDTH_M = 2.5
MIN_SURFACE_U = 0.05
RUT_MERGE_M = 0.25
RUT_MIN_SHARE = 0.05

# Old roads have no model: this is where the game's look definitions put a lane on the texture.
LEGACY_RUTS = {"us": [0.5 + 0.252 * 1.2 / 4.5, 0.5 + 0.252 * 3.3 / 4.5], "per_metre": 0.252 / 4.5}


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
                if short not in looks and _SIMPLE_TOKEN_RE.fullmatch(short):
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
    return looks


def load_units(files, prefix, kind, tokens_only=True):
    """Units of one kind that can be copied as they are: {short name: (full name, body)}.

    The name of a unit a sector refers to has to fit a token, unlike one only other units refer to.
    """
    out = {}
    for _fname, raw in files.read_all("def/world", prefix):
        for unit_kind, name, body in parse_units(raw.decode("utf8", "replace")):
            short = name.split(".", 1)[1] if "." in name else name
            if unit_kind == kind and not references(body) and (not tokens_only or _SIMPLE_TOKEN_RE.fullmatch(short)):
                out.setdefault(short, (name, body))
    return out


def variant_name(zone, index):
    return f"r{zone}_{index}"


def zone_material_path(path, zone):
    """Same length as the original, so a model file can be patched in place."""
    parts = path.split("/")
    parts[2] = "r" + _ZONE_CHARS[zone]
    return "/".join(parts)


def legacy_material_path(index, zone):
    return f"/material/seasons/legacy/z{zone}/{index}.mat"


def rename_look(look, new_name, models, edges=None):
    """A full copy of the look under a new name, pointing to the given models and edges.

    The game lets only one owner refer to an anonymous unit, so the copy can't share them with the
    original: each one it refers to is copied as well, under a name that can't collide.
    """
    renamed = {name: f".rv.{new_name}.{i}" for i, (_kind, name, _body) in enumerate(look.related)}
    edges = edges or {}

    def retarget(body):
        body = _ANY_MODEL_RE.sub(lambda m: f'"{models.get(m.group(1), m.group(1))}"', body)
        body = _COMPATIBLE_EDGE_RE.sub(lambda m: m.group(1) + edges.get(m.group(2), m.group(2)), body)
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


def snow_mode(effect, substance, base_source):
    """How snow treats a material: "road", "tiled", "ground", "fade", "tree", or None when it stays as it is."""
    if _UNTOUCHED_EFFECT_RE.search(effect):
        return None
    see_through = _SEE_THROUGH_EFFECT_RE.search(effect) or "difopa" in base_source
    if _NATURE_SOURCE_RE.search(base_source):
        return "tree" if see_through else "ground"
    if "decal" in effect or ".a." in effect:  # the alpha channel is opacity here, so it can't be painted over
        return "fade" if substance in _ROAD_SUBSTANCES or _FADING_RE.search(base_source) else None
    if substance in _GROUND_SUBSTANCES:
        return "ground"
    if substance in _ROAD_SUBSTANCES or _ROAD_SOURCE_RE.search(base_source):
        lanes = _LANE_SOURCE_RE.search(base_source) and not _TILED_SOURCE_RE.search(base_source)
        return "road" if lanes else "tiled"
    return "ground" if _GROUND_SOURCE_RE.search(base_source) else None


def covers_for(mode):
    return {"tiled": TILED_COVER, "tree": TREE_COVER}.get(mode, LEVEL_COVER)


def _dds_path(textures_dir, source):
    return os.path.join(textures_dir, source.lstrip("/").rsplit(".", 1)[0] + ".dds")


def _material_info(files, path, textures_dir, missing=None):
    """The material's text and the textures to bake, or None if snow doesn't touch it."""
    raw = files.read(path)
    if raw is None:
        return None
    text = mats.absolute_sources(raw.decode("utf8", "replace"), path)
    effect = _EFFECT_RE.search(text)
    substance = _SUBSTANCE_RE.search(text)
    if not effect:
        return None
    slots = {}
    for slot, body in _TEXTURE_RE.findall(text):
        src = _SOURCE_RE.search(body)
        if src and (_PAINTED_SLOT_RE.fullmatch(slot) or _MULT_SLOT_RE.fullmatch(slot)):
            slots[slot] = src.group(1)
    if "texture_base" not in slots:
        return None
    mode = snow_mode(effect.group(1), substance.group(1) if substance else "", slots["texture_base"])
    if not mode:
        return None
    if mode not in ("road", "tiled"):
        slots = {slot: src for slot, src in slots.items() if not _MULT_SLOT_RE.fullmatch(slot)}
    absent = [src for src in slots.values() if not os.path.exists(_dds_path(textures_dir, src))]
    if absent:
        if missing is not None:
            missing.update(absent)
        return None
    return {"path": path, "text": text, "slots": slots, "mode": mode}


def baked_path(key, level):
    return f"/material/seasons/road/{key}_{level}.tobj"


def slot_mode(info, slot):
    """How one texture of a material is baked."""
    return "mult" if _MULT_SLOT_RE.fullmatch(slot) else info["mode"]


def _bake_job(job):
    """Worker: every snow level of one texture. Runs in a separate process."""
    textures_dir, read_image, path, mode, key, snow_rgb, ruts, per_metre = job
    if mode == "tree":
        with open(_dds_path(textures_dir, path), "rb") as fh:
            chain = tex.read_dds_chain(fh.read(), DETAIL_WIDTH)
        if chain is None:
            return path, mode, key, None
        return path, mode, key, {level: tex.write_dds(chain[0], tex.bake_cut_out(chain, cover))
                                 for level, cover in TREE_COVER.items()}
    base = read_image(textures_dir, path, ROAD_WIDTH if mode == "road" else DETAIL_WIDTH)
    if base is None:
        return path, mode, key, None
    h, w = base.shape[:2]
    covers = covers_for(mode)
    if mode == "road":
        noise = tex.road_noise(base, ruts=ruts, per_metre=per_metre)
        rut = tex.rut_field(h, w, ruts, per_metre) if ruts else None
        snow = tex.packed_snow(h, w, snow_rgb)
        paint = lambda cover: tex.bake_road(base, snow, noise, cover, rut)
    elif mode == "tiled":
        noise = tex.resampled(tex.periodic_noise(256, seed=29), h, w)
        snow = tex.packed_snow(h, w, tuple(v / PACKED_SNOW_TINT for v in snow_rgb))
        paint = lambda cover: tex.bake_road(base, snow, noise, cover)
    elif mode == "ground":
        tile = read_image(textures_dir, "/material/terrain/snow.tobj", 256)
        tile = tex.flat_snow(256) if tile is None else tile
        noise = tex.periodic_noise(256, seed=19)
        paint = lambda cover: tex.bake(base, tile, noise, cover)
    elif mode == "mult":
        paint = lambda cover: tex.neutral_mult(base, cover)
    else:
        paint = lambda cover: tex.fade_decal(base, cover)
    return path, mode, key, {level: tex.write_dds(paint(cover)) for level, cover in covers.items()}


def bake_jobs(found):
    """One entry per distinct way to bake a texture: (path, mode)."""
    jobs = set()
    for info in found:
        for slot, path in info["slots"].items():
            jobs.add((path, slot_mode(info, slot)))
    return sorted(jobs)


def merge_ruts(positions, per_metre):
    """Join wheel track positions closer than a tyre, dropping those hardly ever driven on."""
    reach = RUT_MERGE_M * per_metre
    groups = []
    for u, weight in sorted(positions.items()):
        if groups and u - groups[-1][-1][0] <= reach:
            groups[-1].append((u, weight))
        else:
            groups.append([(u, weight)])
    merged = [(sum(u * w for u, w in g) / sum(w for _u, w in g), sum(w for _u, w in g)) for g in groups]
    if not merged:
        return []
    top = max(w for _u, w in merged)
    return [round(u, 4) for u, w in merged if w >= RUT_MIN_SHARE * top]


def lane_tracks(left, right, lanes):
    """X positions of the wheel tracks of `lanes` equal lanes between two edges of a road."""
    if lanes < 1:
        return []
    width = (right - left - 2 * SHOULDER_IN_MODEL_M) / lanes
    if not MIN_LANE_M <= width <= MAX_LANE_M:
        return []
    start = left + SHOULDER_IN_MODEL_M
    out = []
    for lane in range(lanes):
        middle = start + (lane + 0.5) * width
        out += [middle - WHEEL_OFFSET_M, middle + WHEEL_OFFSET_M]
    return out


def _piece_columns(piece, channel):
    """Across the road: distinct X positions with their mean U, for interpolating along a surface."""
    columns = defaultdict(list)
    for x, u in zip(piece.positions[:, 0].tolist(), piece.uvs[channel][:, 0].tolist()):
        columns[round(x, 2)].append(u)
    xs = sorted(columns)
    return np.array(xs), np.array([sum(columns[x]) / len(columns[x]) for x in xs])


def _tally_tracks(tally, scales, materials, pieces, tracks, weight):
    """Add where these wheel tracks fall on each texture, going through each surface's UV."""
    for piece in pieces:
        info = materials[piece.slot]
        for slot, path in info["slots"].items():
            if slot_mode(info, slot) != "road":
                continue
            channel = min(0 if slot == "texture_base" else 1, len(piece.uvs) - 1)
            u = piece.uvs[channel][:, 0]
            if u.max() - u.min() < MIN_SURFACE_U:
                continue
            xs, us = _piece_columns(piece, channel)
            if xs.size < 2:
                continue
            per_metre = float(np.median(np.abs(np.diff(us) / np.maximum(np.diff(xs), 1e-3))))
            if per_metre <= 0:
                continue
            for track in tracks:
                if xs[0] <= track <= xs[-1]:
                    tally[path][round(float(np.interp(track, xs, us)) % 1.0, 3)] += weight
                    scales[path][round(per_metre, 3)] += weight


def road_ruts(looks, counts, read_model, material_of):
    """Where the wheels run on each road texture: {texture: {"us": [...], "per_metre": ...}}.

    A road texture holds several lanes side by side and each model maps its own part of it, so
    this goes through the geometry of every look in use, weighted by how often the look occurs.
    `read_model` gives a model's (material table, pieces) and `material_of` a material's info.
    """
    tally = defaultdict(Counter)
    scales = defaultdict(Counter)
    for look in looks:
        templates = dict(_TEMPLATE_RE.findall(look.body))
        lanes = Counter(_LANE_RE.findall(look.body))
        for side, path in templates.items():
            model = read_model(path)
            if model is None:
                continue
            tables, pieces = model
            count = lanes[side] if len(templates) == 2 else lanes["left"] + lanes["right"]
            for table in tables:  # a look of the model, often one per country
                materials = {i: material_of(name) for i, name in enumerate(table)}
                surfaces = [p for p in pieces if p.slot in materials and materials[p.slot]
                            and materials[p.slot]["mode"] == "road"
                            and p.positions[:, 0].max() - p.positions[:, 0].min() >= MIN_SURFACE_WIDTH_M]
                if not surfaces:
                    continue
                left = min(float(p.positions[:, 0].min()) for p in surfaces)
                right = max(float(p.positions[:, 0].max()) for p in surfaces)
                tracks = lane_tracks(left, right, count)
                _tally_tracks(tally, scales, materials, surfaces, tracks, counts.get(look.name, 1))
    out = {}
    for path, positions in tally.items():
        per_metre = scales[path].most_common(1)[0][0]
        us = merge_ruts(positions, per_metre)
        if us:
            out[path] = {"us": us, "per_metre": per_metre}
    return out


def _bake_textures(found, packs, textures_dir, read_image, ruts, log):
    """Bake every distinct texture. Returns {(tobj path, mode): key}."""
    stock = read_image(textures_dir, "/material/terrain/snow.tobj", 256)
    rgb = tex.SNOW_RGB if stock is None else stock[:, :, :3].mean(axis=(0, 1))
    snow_rgb = tuple(float(v) * PACKED_SNOW_TINT for v in rgb)
    jobs = bake_jobs(found)
    # A second layer of a road blends into the first, so without a place of its own it follows that one.
    for info in found:
        if info["mode"] == "road" and info["slots"].get("texture_base") in ruts:
            for slot, path in info["slots"].items():
                if slot_mode(info, slot) == "road":
                    ruts.setdefault(path, ruts[info["slots"]["texture_base"]])
    work = []
    for i, (path, mode) in enumerate(jobs):
        track = ruts.get(path, {}) if mode == "road" else {}
        work.append((textures_dir, read_image, path, mode, f"r{i}", snow_rgb, track.get("us"), track.get("per_metre")))
    keys = {}
    with ProcessPoolExecutor() as pool:
        for path, mode, key, levels in pool.map(_bake_job, work):
            if levels is None:
                continue
            target = packs.nature if mode == "tree" else packs.roads()
            for level, data in levels.items():
                dds = f"/material/seasons/road/{key}_{level}.dds"
                target.writestr(dds[1:], data)
                target.writestr(baked_path(key, level)[1:], tex.write_tobj(dds))
            keys[path, mode] = key
            if len(keys) % 100 == 0:
                log(f"  textures baked: {len(keys)}/{len(jobs)}")
    with_ruts = sum(1 for path, mode in keys if mode == "road" and path in ruts)
    road_total = sum(1 for _path, mode in keys if mode == "road")
    log(f"textures baked: {len(keys)} of {len(jobs)}, {dict(Counter(j[1] for j in jobs))}; "
        f"road textures with wheel tracks from geometry: {with_ruts} of {road_total}")
    return keys


def snowy_text(info, level, keys):
    """A material with snow baked into its textures. `keys` gives a slot its baked texture."""
    if level not in covers_for(info["mode"]):
        return info["text"]

    def swap(m):
        key = keys.get(m.group(2))
        return m.group(1) + baked_path(key, level) + m.group(4) if key else m.group(0)

    return _SLOT_BLOCK_RE.sub(swap, info["text"])


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


def prefab_name(zone, index):
    return f"sp{zone}_{index}"


def zone_model_path(root, zone, index):
    """Fixed width, so all the names of one shared archive entry have the same length."""
    return f"/{root}/z{zone:02d}/m{index:05d}.pmd"


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


def copied_prefab(body, zone, model_paths, index_of, corners=None):
    """A prefab's body for one zone: its models, links and corners point to copies from that zone."""
    body = _ANY_MODEL_RE.sub(lambda m: f'"{model_paths[m.group(1)]}"' if m.group(1) in model_paths else m.group(0), body)
    if corners:
        body = _CORNER_RE.sub(lambda m: m.group(1) + corners.get(m.group(2), m.group(2)), body)
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


class Catalog:
    """What's known about the game's materials, and which zones end up using which."""

    def __init__(self, files, textures_dir):
        self.files = files
        self.textures_dir = textures_dir
        self.info = {}
        self.missing = set()
        self.names = {}
        self.geometry = {}

    def learn(self, path):
        """Learn a model's materials. Returns its bytes, or None if the model isn't complete."""
        if path not in self.names:
            raw = self.files.read(path)
            if raw is None or self.files.read(path[:-1] + "g") is None:
                self.names[path] = None
            else:
                names = {m.decode("ascii") for m in _MATERIAL_RE.findall(raw)}
                self.names[path] = names
                for name in names:
                    if name not in self.info:
                        self.info[name] = _material_info(self.files, name, self.textures_dir, self.missing)
        return self.names[path]

    def read_model(self, path):
        """A model's material table and pieces, for working out where lanes lie."""
        if self.learn(path) is None:
            return None
        return pmx.material_tables(self.files.read(path)), pmx.pieces(self.files.read(path[:-1] + "g"))

    def usable(self):
        return [info for info in self.info.values() if info]


class ModelCopies:
    """Copies of models for zones, sharing their geometry in one archive."""

    def __init__(self, zf, files, catalog, usable_paths, used_by_zone, root):
        self.zf = zf
        self.files = files
        self.catalog = catalog
        self.usable_paths = usable_paths
        self.used_by_zone = used_by_zone
        self.root = root
        self.index = {}
        self.first = {}
        self.written = set()

    def copy(self, path, zone):
        """The zone's copy of a model, or None when snow wouldn't change anything on it."""
        names = self.catalog.learn(path)
        swaps = (names or set()) & self.usable_paths
        if not swaps:
            return None
        index = self.index.setdefault(path, len(self.index))
        target = zone_model_path(self.root, zone, index)
        if target in self.written:
            return target
        self.written.add(target)
        for name in swaps:
            self.used_by_zone[name].add(zone)
        raw = self.files.read(path)
        self.zf.writestr(target[1:], patch_model(raw, {n: zone_material_path(n, zone) for n in swaps}))
        for ext in ("g", "c", "a"):
            name = target[1:-1] + ext
            if path not in self.first:
                data = self.files.read(path[:-1] + ext)
                if data is not None:
                    self.zf.writestr(name, data)
            else:
                source = self.first[path][:-1] + ext
                if source in self.zf.NameToInfo:
                    add_alias(self.zf, source, name)
        self.first.setdefault(path, target[1:])
        return target

    def copies(self, paths, zone):
        out = {}
        for path in paths:
            target = self.copy(path, zone)
            if target:
                out[path] = target
        return out


def _tally_by_kind(tally):
    """{kind: {token: set of zones}} and, for owned kinds, {kind: {(zone, token): owners}}."""
    zones_of = defaultdict(lambda: defaultdict(set))
    owners = defaultdict(lambda: defaultdict(set))
    uses = defaultdict(Counter)
    for (kind, zone, token, owner), count in tally.items():
        zones_of[kind][token].add(zone)
        uses[kind][token] += count
        if owner:
            owners[kind][zone, token].add(owner)
    return zones_of, owners, uses


def load_vocabulary(files):
    """Everything that can be swapped on the map, by token. Returns (vocabulary, definitions)."""
    looks = load_looks(files)
    prefabs = load_prefabs(files)
    defs = {
        "looks": looks,
        "prefabs": prefabs,
        "edges": load_units(files, "road_edge", "road_edge"),
        "terrain_edges": load_units(files, "terrain_edge", "terrain_edge"),
        "corners": load_units(files, "prefab_corner", "prefab_corner"),
        "vegetation": load_units(files, "vegetation.", "vegetation_data"),
        "vegetation_models": load_units(files, "vegetation_model", "vegetation_model", tokens_only=False),
        "models": {k: v for k, v in load_units(files, "model", "model_def").items() if _NATURE_MODEL_RE.search(v[1])},
        "curves": {k: v for k, v in load_units(files, "curve_model", "curve_model").items()
                   if _NATURE_MODEL_RE.search(v[1])},
        "far": load_units(files, "far_model", "far_model_def"),
        "road_materials": load_units(files, "road_material", "road_material_def"),
        "sidewalks": load_units(files, "road_sidewalk", "material_def"),
    }
    words = lambda names: {tokens.encode(k) for k in names}
    vocab = zn.Vocabulary(
        look=words(looks), legacy_look=words(k for k, v in looks.items() if not v.models),
        edge=words(defs["edges"]), road_material=words(defs["road_materials"]),
        prefab=words(prefabs), corner=words(defs["corners"]), vegetation=words(defs["vegetation"]),
        sidewalk=words(defs["sidewalks"]), model=words(defs["models"]), curve=words(defs["curves"]),
        far=words(defs["far"]), terrain_edge=words(defs["terrain_edges"]))
    return vocab, defs


def _names_by_token(names):
    return {tokens.encode(k): k for k in names}


def _write_unit_file(zf, name, units):
    zf.writestr(name, "SiiNunit\n{\n" + "".join(units) + "}\n")


def _simple_copies(defs, zones_of, models, kind_prefix, unit_kind):
    """Copy units whose only outside references are their models. Returns (units, {(zone, token): new})."""
    units, plan = [], {}
    by_token = _names_by_token(defs)
    ordered = sorted(zones_of.items(), key=lambda kv: by_token.get(kv[0], ""))
    for index, (token, zones_here) in enumerate(ordered):
        name = by_token.get(token)
        if name is None:
            continue
        body = defs[name][1]
        paths = _ANY_MODEL_RE.findall(body)
        for zone in sorted(zones_here):
            targets = models.copies(paths, zone)
            if not targets:
                continue
            new = f"{kind_prefix}{zone}_{index}"
            units.append(f"{unit_kind} : {defs[name][0].split('.', 1)[0]}.{new}\n{{"
                         + _ANY_MODEL_RE.sub(lambda m: f'"{targets.get(m.group(1), m.group(1))}"', body) + "}\n")
            plan[zone, token] = tokens.encode(new)
    return units, plan


def _learn_all(catalog, defs, zones_of):
    """Find the materials of everything that's on a snowy part of the map."""
    for look_token in zones_of["look"]:
        name = _names_by_token(defs["looks"]).get(look_token)
        if name:
            for path in defs["looks"][name].models:
                catalog.learn(path)
    for kind, table in (("edge", "edges"), ("terrain_edge", "terrain_edges"), ("corner", "corners"),
                        ("model", "models"), ("curve", "curves"), ("far", "far")):
        by_token = _names_by_token(defs[table])
        for token in zones_of[kind]:
            name = by_token.get(token)
            if name:
                for path in _ANY_MODEL_RE.findall(defs[table][name][1]):
                    catalog.learn(path)
    by_token = _names_by_token(defs["prefabs"])
    for token in zones_of["prefab"]:
        name = by_token.get(token)
        if name:
            for path in _ANY_MODEL_RE.findall(defs["prefabs"][name]):
                catalog.learn(path)
    veg_models = defs["vegetation_models"]
    by_token = _names_by_token(defs["vegetation"])
    for token in zones_of["vegetation"]:
        name = by_token.get(token)
        if name:
            for model in _VEGETATION_MODEL_RE.findall(defs["vegetation"][name][1]):
                short = model[1].split(".", 1)[1]
                if short in veg_models:
                    for path in _ANY_MODEL_RE.findall(veg_models[short][1]):
                        catalog.learn(path)


def _write_vegetation(zf, defs, zones_of, models, log):
    """Per-zone vegetation schemes and their tree models. Returns the plan for vegetation tokens."""
    by_token = _names_by_token(defs["vegetation"])
    veg_models = defs["vegetation_models"]
    model_index = {name: i for i, name in enumerate(sorted(veg_models))}
    units, model_units, plan, copied = [], [], {}, {}

    def tree_copy(short, zone):
        """The name of a tree model's unit for a zone, or None."""
        key = zone, short
        if key not in copied:
            body = veg_models[short][1]
            targets = models.copies(_ANY_MODEL_RE.findall(body), zone)
            if targets:
                new = f"vm{zone}_{model_index[short]}"
                model_units.append(f"vegetation_model : vegetmodel.{new}\n{{"
                                   + _ANY_MODEL_RE.sub(lambda m: f'"{targets.get(m.group(1), m.group(1))}"', body)
                                   + "}\n")
                copied[key] = f"vegetmodel.{new}"
            else:
                copied[key] = None
        return copied[key]

    for index, (token, zones_here) in enumerate(sorted(zones_of["vegetation"].items())):
        name = by_token.get(token)
        if name is None:
            continue
        body = defs["vegetation"][name][1]
        for zone in sorted(zones_here):
            changed = []

            def swap(m):
                short = m.group(2).split(".", 1)[1]
                new = tree_copy(short, zone) if short in veg_models else None
                if new:
                    changed.append(new)
                return m.group(1) + (new or m.group(2))

            new_body = _VEGETATION_MODEL_RE.sub(swap, body)
            if changed:
                new = f"vz{zone}_{index}"
                units.append(f"vegetation_data : veg.{new}\n{{{new_body}}}\n")
                plan[zone, token] = tokens.encode(new)
    _write_unit_file(zf, "def/world/vegetation_model.ets_seasons.sii", model_units)
    _write_unit_file(zf, "def/world/vegetation.ets_seasons.sii", units)
    log(f"vegetation schemes: {len(units)}, tree models: {len(model_units)}, model copies: {len(models.written)}")
    return plan


def _write_legacy(zf, defs, zones_of, catalog, log):
    """Roads from before templates name their materials themselves. Returns (plan, records)."""
    records = {}

    def record(path):
        if path not in records:
            info = _material_info(catalog.files, path, catalog.textures_dir, catalog.missing)
            records[path] = None if info is None else {**info, "legacy": len(records), "zones": set()}
        return records[path]

    plans = {}
    for kind, table, prefix, namespace, unit, filename in (
            ("road_material", "road_materials", "lm", "road_mat", "road_material_def",
             "def/world/road_material.ets_seasons.sii"),
            ("sidewalk", "sidewalks", "ls", "sidewalk", "material_def", "def/world/road_sidewalk.ets_seasons.sii")):
        by_token = _names_by_token(defs[table])
        units, plan = [], {}
        for index, (token, zones_here) in enumerate(sorted(zones_of[kind].items())):
            name = by_token.get(token)
            if name is None:
                continue
            body = defs[table][name][1]
            found = [record(m.group(2)) for m in _MATERIAL_PATH_RE.finditer(body)]
            if not any(found):
                continue
            for zone in sorted(zones_here):
                def swap(m):
                    info = records.get(m.group(2))
                    if info is None:
                        return m.group(0)
                    info["zones"].add(zone)
                    return m.group(1) + legacy_material_path(info["legacy"], zone) + m.group(3)

                new = f"{prefix}{zone}_{index}"
                units.append(f"{unit} : {namespace}.{new}\n{{{_MATERIAL_PATH_RE.sub(swap, body)}}}\n")
                plan[zone, token] = tokens.encode(new)
        _write_unit_file(zf, filename, units)
        plans[kind] = plan
    used = [info for info in records.values() if info and info["zones"]]
    log(f"old road materials: {len(used)}, units: {sum(len(p) for p in plans.values())}")
    return plans, used


def _write_prefabs(zf, defs, zones_of, owners, models, log):
    """Copy the prefabs and their corners. Returns the plans for both."""
    prefabs = defs["prefabs"]
    by_token = _names_by_token(prefabs)
    found = defaultdict(set)
    for token, zones_here in zones_of["prefab"].items():
        if token in by_token:
            found[by_token[token]] |= zones_here
    spread_over_links(found, prefabs)
    model_paths = {}
    for name, zones_here in found.items():
        for zone in zones_here:
            model_paths[name, zone] = models.copies(_ANY_MODEL_RE.findall(prefabs[name]), zone)
    wanted = [name for name in found if any(model_paths.get((name, z)) for z in found[name])]
    chosen = sorted(with_linked(wanted, prefabs))
    index_of = {name: i for i, name in enumerate(chosen)}
    plan = {}
    for name in chosen:
        for zone in found[name]:
            plan[zone, tokens.encode(name)] = tokens.encode(prefab_name(zone, index_of[name]))

    # Only corners of prefabs that have a copy can be swapped, since a stock prefab wouldn't list them.
    corner_tokens = _names_by_token(defs["corners"])
    corner_plan, corner_units, corners_of = {}, [], defaultdict(dict)
    for index, ((zone, token), owning) in enumerate(sorted(owners["corner"].items())):
        name = corner_tokens.get(token)
        if name is None or not any((zone, owner) in plan for owner in owning):
            continue
        body = defs["corners"][name][1]
        targets = models.copies(_ANY_MODEL_RE.findall(body), zone)
        if not targets:
            continue
        new = f"cz{zone}_{index}"
        corner_units.append(f"prefab_corner : corner.{new}\n{{"
                            + _ANY_MODEL_RE.sub(lambda m: f'"{targets.get(m.group(1), m.group(1))}"', body) + "}\n")
        corner_plan[zone, token] = tokens.encode(new)
        corners_of[zone][name] = new
    _write_unit_file(zf, "def/world/prefab_corner.ets_seasons.sii", corner_units)

    units = []
    for name in chosen:
        for zone in sorted(found[name]):
            body = copied_prefab(prefabs[name], zone, model_paths.get((name, zone)) or {}, index_of, corners_of[zone])
            units.append(f"prefab_model : prefab.{prefab_name(zone, index_of[name])}\n{{{body}}}\n")
    _write_unit_file(zf, "def/world/prefab.ets_seasons.sii", units)
    log(f"prefabs with snow: {len(chosen)} of {len(found)}, units: {len(units)}, corners: {len(corner_units)}")
    return plan, corner_plan


def _write_roads(zf, defs, zones_of, models, log):
    """Copy road edges and looks. Returns the plans for both."""
    edge_units, edge_plan = _simple_copies(defs["edges"], zones_of["edge"], models, "ez", "road_edge")
    _write_unit_file(zf, "def/world/road_edge.ets_seasons.sii", edge_units)
    edge_tokens = _names_by_token(defs["edges"])
    edge_names = defaultdict(dict)
    for (zone, token), new in edge_plan.items():
        edge_names[zone][edge_tokens[token]] = tokens.decode(new)

    looks = defs["looks"]
    by_token = _names_by_token(looks)
    units, plan = [], {}
    for index, (token, zones_here) in enumerate(sorted(zones_of["look"].items())):
        name = by_token.get(token)
        if name is None or not looks[name].models:
            continue
        for zone in sorted(zones_here):
            targets = models.copies(looks[name].models, zone)
            if not targets and not edge_names[zone]:
                continue
            new = variant_name(zone, index)
            units.append(rename_look(looks[name], new, targets, edge_names[zone]))
            plan[zone, token] = tokens.encode(new)
    _write_unit_file(zf, "def/world/road_look.ets_seasons.sii", units)
    log(f"road looks: {len(plan)} copies of {len(zones_of['look'])}, edges: {len(edge_units)}")
    return plan, edge_plan


def build_into(packs, files, vocab, defs, tally, textures_dir, read_image, log):
    """Write all per-zone copies into the packs. Returns (plan for sectors, material records)."""
    zones_of, owners, uses = _tally_by_kind(tally)
    catalog = Catalog(files, textures_dir)
    _learn_all(catalog, defs, zones_of)
    usable = catalog.usable()
    log(f"materials on the snowy map: {len(catalog.info)}, can take snow: {len(usable)}, "
        f"{dict(Counter(i['mode'] for i in usable))}")
    if catalog.missing:
        log(f"materials left out, textures not extracted: {len(catalog.missing)}, e.g. {sorted(catalog.missing)[:3]}")

    used_looks = [defs["looks"][name] for token, name in _names_by_token(defs["looks"]).items()
                  if token in zones_of["look"]]
    look_counts = {tokens.decode(token): n for token, n in uses["look"].items()}
    ruts = road_ruts(used_looks, look_counts, catalog.read_model, catalog.info.get)
    # Old roads use their own materials, which get wheel tracks from the look's standard layout.
    legacy_plans, legacy_records = _write_legacy(packs.map, defs, zones_of, catalog, log)
    for info in legacy_records:
        if info["mode"] == "road":
            ruts.setdefault(info["slots"]["texture_base"], LEGACY_RUTS)

    keys = _bake_textures(usable + legacy_records, packs, textures_dir, read_image, ruts, log)
    complete = lambda info: all((path, slot_mode(info, slot)) in keys for slot, path in info["slots"].items())
    usable = [info for info in usable if complete(info)]
    usable_paths = {info["path"] for info in usable}
    used_by_zone = defaultdict(set)

    surface = ModelCopies(packs.map, files, catalog, usable_paths, used_by_zone, "prefab_seasons")
    plan = defaultdict(dict)
    plan["look"], plan["edge"] = _write_roads(packs.map, defs, zones_of, surface, log)
    plan["prefab"], plan["corner"] = _write_prefabs(packs.map, defs, zones_of, owners, surface, log)
    edge_units, plan["terrain_edge"] = _simple_copies(defs["terrain_edges"], zones_of["terrain_edge"], surface, "tz",
                                                      "terrain_edge")
    _write_unit_file(packs.map, "def/world/terrain_edge.ets_seasons.sii", edge_units)
    log(f"model copies on roads: {len(surface.written)} of {len(surface.index)} models")

    nature = ModelCopies(packs.nature, files, catalog, usable_paths, used_by_zone, "veg_seasons")
    plan["vegetation"] = _write_vegetation(packs.map, defs, zones_of, nature, log)
    for kind, table, prefix, unit, filename in (("model", "models", "mz", "model_def", "model"),
                                                ("curve", "curves", "cv", "curve_model", "curve_model"),
                                                ("far", "far", "fz", "far_model_def", "far_model")):
        units, plan[kind] = _simple_copies(defs[table], zones_of[kind], nature, prefix, unit)
        _write_unit_file(packs.map, f"def/world/{filename}.ets_seasons.sii", units)
        log(f"{kind} copies: {len(units)}")
    log(f"model copies in nature: {len(nature.written)} of {len(nature.index)} models")
    plan.update(legacy_plans)

    records = []
    for info in usable:
        zones_here = used_by_zone.get(info["path"])
        if zones_here:
            records.append(_record(info, keys, zones_here))
    # A unit of an old road already points to these, so they stay even if a texture couldn't be baked.
    records += [_record(info, keys, info["zones"]) for info in legacy_records]
    log(f"materials in use: {len(records)}, files for the date pack: {sum(len(r['zones']) for r in records)}")
    return plan, records


def _record(info, keys, zones_here):
    baked = {slot: keys.get((path, slot_mode(info, slot))) for slot, path in info["slots"].items()}
    return {**info, "zones": sorted(zones_here), "keys": {slot: key for slot, key in baked.items() if key}}


def date_files(records, zone, level):
    """Yield (zip path, material text) for one zone at one snow level."""
    for info in records:
        if zone not in info["zones"]:
            continue
        if "legacy" in info:
            target = legacy_material_path(info["legacy"], zone).lstrip("/")
        else:
            target = zone_material_path(info["path"], zone).lstrip("/")
        yield target, info["text"] if level == 0 else snowy_text(info, level, info["keys"])
