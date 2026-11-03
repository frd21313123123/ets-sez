"""Terrain material definitions and per-zone material text."""
import posixpath
import re
from dataclasses import dataclass, field

from . import tokens

_DEF_RE = re.compile(r"material_def\s*:\s*terrain_mat\.(\w+)\s*\{(.*?)\}", re.S)
_DETAIL_RE = re.compile(r"detail_vegetation_def\s*:\s*\S+\s*\{(.*?)\}", re.S)
_TEXTURE_RE = re.compile(r'texture\s*:\s*"(\w+)"\s*\{(.*?)\}', re.S)
_SOURCE_RE = re.compile(r'source\s*:\s*"([^"]+)"')
_EXCLUDED = re.compile(r"water|asphalt|concrete|road|sidewalk|pavement|tile|artificial|stamp|parklot|bridge|"
                       r"lava|inv|underground|hell|halloween", re.I)
_LINEAR_SNOW = (0.58, 0.61, 0.65)
_SRGB_SNOW = (200, 205, 211)


@dataclass
class TerrainMaterial:
    name: str
    path: str
    extra: str = ""       # color/name attributes carried over to the variants
    text: str = ""
    base_texture: str = ""


def load_materials(files):
    """Every terrain material_def in the installed archives, keyed by unit name."""
    out = {}
    for _fname, raw in files.read_all("def/world", "terrain_material"):
        for name, body in _DEF_RE.findall(raw.decode("utf8", "replace")):
            path = re.search(r'path\s*:\s*"([^"]+)"', body)
            if not path or name in out:
                continue
            extra = "".join(f"\t{line.strip()}\n" for line in body.splitlines()
                            if line.strip().startswith(("color", "name")))
            out[name] = TerrainMaterial(name, path.group(1), extra)
    return out


def absolute_sources(text, mat_path):
    """Make texture paths absolute so the material can live in another folder."""
    folder = posixpath.dirname(mat_path)

    def fix(m):
        src = m.group(1)
        return f'source : "{src if src.startswith("/") else posixpath.join(folder, src)}"'

    return _SOURCE_RE.sub(fix, text)


def attach_text(materials, files):
    """Read each material's .mat file; returns those that can receive baked snow."""
    usable = {}
    for m in materials.values():
        if _EXCLUDED.search(m.path) or _EXCLUDED.search(m.name):
            continue
        raw = files.read(m.path)
        if raw is None:
            continue
        m.text = absolute_sources(raw.decode("utf8", "replace"), m.path)
        for slot, body in _TEXTURE_RE.findall(m.text):
            if slot == "texture_base":
                src = _SOURCE_RE.search(body)
                if src:
                    m.base_texture = src.group(1)
        if m.base_texture:
            usable[m.name] = m
    return usable


def mean_color(material):
    """The material's average colour in sRGB, from its far colour settings."""
    m = re.search(r"far_color\s*:\s*\{([^}]*)\}", material.text)
    if m:
        vals = [max(0.0, float(v)) for v in m.group(1).split(",")][:3]
        return tuple(min(255, int(255 * v ** (1 / 2.2))) for v in vals)
    m = re.search(r"aux\[2\]\s*:\s*\{([^}]*)\}", material.text)
    if m:
        return tuple(int(float(v)) for v in m.group(1).split(",")[:3])
    return (92, 104, 70)


def variant_name(zone, index):
    return f"z{zone}_{index}"


def snowy_text(material, baked_path, cover):
    """The material text with its base texture swapped and far colour moved towards snow."""
    text = material.text.replace(f'"{material.base_texture}"', f'"{baked_path}"', 1)

    def far(m):
        vals = [float(v) for v in m.group(1).split(",")]
        mixed = [v * (1 - cover) + s * cover for v, s in zip(vals, _LINEAR_SNOW)]
        return "far_color : { " + " , ".join(f"{v:.6f}" for v in mixed) + " }"

    text = re.sub(r"far_color\s*:\s*\{([^}]*)\}", far, text)

    def aux2(m):
        vals = [float(v) for v in m.group(1).split(",")]
        mixed = [v * (1 - cover) + s * cover for v, s in zip(vals[:3], _SRGB_SNOW)] + vals[3:]
        return "aux[2] : { " + ", ".join(f"{int(v)}" for v in mixed) + " }"

    text = re.sub(r"aux\[2\]\s*:\s*\{([^}]*)\}", aux2, text)
    if cover >= 1.0:
        text = re.sub(r'substance\s*:\s*"[^"]*"', 'substance : "snow"', text)
    return text


def detail_vegetation(files):
    """Map texture path -> list of detail grass model lines, so baked textures keep their grass."""
    out = {}
    for _fname, raw in files.read_all("def/world", "detail_vegetation_def"):
        for body in _DETAIL_RE.findall(raw.decode("utf8", "replace")):
            tex = re.search(r'texture\s*:\s*"([^"]+)"', body)
            if tex:
                lines = [ln.strip() for ln in body.splitlines()
                         if ln.strip() and not ln.strip().startswith("texture")]
                out.setdefault(tex.group(1), lines)
    return out


def token_value(name):
    return tokens.encode(name)
