"""Snow on road surfaces: per-zone copies of road looks, their models and their materials.

A road look points to a model, and the model lists its materials by path. So a zone gets its own
look, its own copy of the model with those paths changed, and its own materials. Junctions are
separate prefab models shared by the whole map, so they stay clean.
"""
import os
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import materials as mats, model, sectors as sec, textures as tex, tokens

_SII_TOKEN_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|#[^\n]*|//[^\n]*|/\*.*?\*/|[{}]', re.S)
_HEADER_RE = re.compile(r"(\w+)\s*:\s*([\w.]+)\s*$")
_REFERENCE_RE = re.compile(r"(:\s*)(\.[\w.]+)")
_MODEL_RE = re.compile(r'"(/road_template/[^"]+\.pmd)"')
_MATERIAL_RE = re.compile(rb"/automat/[0-9a-f]{2}/[0-9a-f]{16}\.mat")
_EFFECT_RE = re.compile(r'effect\s*:\s*"([^"]+)"')
_SUBSTANCE_RE = re.compile(r'substance\s*:\s*"([^"]*)"')
_TEXTURE_RE = re.compile(r'texture\s*:\s*"(\w+)"\s*\{(.*?)\}', re.S)
_SOURCE_RE = re.compile(r'source\s*:\s*"([^"]+)"')
_SNOWY_SLOTS = ("texture_base", "texture_over")
_ZONE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"

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


def _material_info(files, path, textures_dir):
    """The material's text and the textures to bake, or None if it isn't a snowable road surface."""
    raw = files.read(path)
    if raw is None:
        return None
    text = mats.absolute_sources(raw.decode("utf8", "replace"), path)
    effect = _EFFECT_RE.search(text)
    substance = _SUBSTANCE_RE.search(text)
    if not effect or "none" in effect.group(1) or not substance or substance.group(1) != "road":
        return None
    slots = {}
    for slot, body in _TEXTURE_RE.findall(text):
        src = _SOURCE_RE.search(body)
        if slot in _SNOWY_SLOTS and src:
            dds = os.path.join(textures_dir, src.group(1).lstrip("/").rsplit(".", 1)[0] + ".dds")
            if not os.path.exists(dds):
                return None
            slots[slot] = src.group(1)
    return {"path": path, "text": text, "slots": slots} if "texture_base" in slots else None


def baked_path(key, level):
    return f"/material/seasons/road/{key}_{level}.tobj"


def _bake_textures(found, zf, textures_dir, read_image, log):
    """Bake every distinct road texture. Returns {tobj path: key}."""
    size = 512
    snow = read_image(textures_dir, "/material/terrain/snow.tobj", size)
    if snow is None:
        snow = tex.flat_snow(size)
    snow = snow.astype("float32")
    snow[:, :, :3] *= PACKED_SNOW_TINT
    snow = snow.astype("uint8")
    plain_noise = tex.periodic_noise(size, seed=19)
    kinds = {}
    for info in found:
        for slot, path in info["slots"].items():
            kinds.setdefault(path, slot)
    keys = {}
    for path, slot in kinds.items():
        base = read_image(textures_dir, path, size)
        if base is None:
            continue
        noise = tex.road_noise(base) if slot == "texture_base" else plain_noise
        key = f"r{len(keys)}"
        for level, cover in LEVEL_COVER.items():
            dds = f"/material/seasons/road/{key}_{level}.dds"
            zf.writestr(dds[1:], tex.write_dds(tex.bake(base, snow, noise, cover)))
            zf.writestr(baked_path(key, level)[1:], tex.write_tobj(dds))
        keys[path] = key
    log(f"road textures baked: {len(keys)} of {len(kinds)}")
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


def build_into(zf, files, sector_list, positions, zones, textures_dir, read_image, log):
    """Write looks, models and textures into the map pack.

    Returns ({look token: index}, material records for the date pack).
    """
    looks = load_looks(files)
    counts = tally(sector_list, positions, set(zones), looks)
    used = [looks[name] for name, _n in counts.most_common()]
    log(f"road looks in snowy sectors: {len(used)} of {len(looks)}, references: {sum(counts.values())}")

    model_data = {}
    material_info = {}
    for look in used:
        for path in look.models:
            if path in model_data:
                continue
            raw = files.read(path)
            geometry = files.read(path[:-1] + "g")
            if raw is None or geometry is None:
                continue
            model_data[path] = (raw, geometry)
            for m in set(_MATERIAL_RE.findall(raw)):
                name = m.decode("ascii")
                if name not in material_info:
                    material_info[name] = _material_info(files, name, textures_dir)
    usable = [info for info in material_info.values() if info]
    log(f"road models: {len(model_data)}, surface materials: {len(usable)} of {len(material_info)}")

    keys = _bake_textures(usable, zf, textures_dir, read_image, log)
    usable = [i for i in usable if all(p in keys for p in i["slots"].values())]
    usable_paths = {i["path"] for i in usable}

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
    return token_index, records


def date_files(records, zone, level):
    """Yield (zip path, material text) for one zone at one snow level."""
    for info in records:
        target = zone_material_path(info["path"], zone).lstrip("/")
        yield target, info["text"] if level == 0 else snowy_text(info, level, info["keys"])
