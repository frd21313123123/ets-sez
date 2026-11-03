"""Sector files: enumerating them, locating cities and rewriting terrain material tables."""
import hashlib
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from . import tokens

SECTOR_RE = re.compile(r"sec([+-]\d{4})([+-]\d{4})\.desc")
SECTOR_KINDS = ("base", "aux", "data")
MAX_MATERIALS = 40
MAX_COLORS = 64
ROTATIONS_OK = 3600  # material rotations are small values; anything larger is not a table


@dataclass(frozen=True)
class Sector:
    x: int
    z: int
    name: str
    archive: object


def list_sectors(files):
    out = {}
    for a in files.archives:
        for n in a.listdir("map/europe") or []:
            m = SECTOR_RE.fullmatch(n)
            if m and n not in out:
                out[n] = Sector(int(m.group(1)), int(m.group(2)), n[:-5], a)
    return sorted(out.values(), key=lambda s: (s.z, s.x))


def read_sector(sector, kind):
    return sector.archive.read(f"map/europe/{sector.name}.{kind}")


def locate_stations(sectors, stations):
    """Find each station's sector by the place its city token occurs most in map data."""
    wanted = {struct.pack("<Q", tokens.encode(k)): k for k in stations if len(k) >= 4}
    hits = defaultdict(Counter)
    for s in sectors:
        data = read_sector(s, "base") or b""
        for raw, key in wanted.items():
            c = data.count(raw)
            if c:
                hits[key][(s.x, s.z)] += c
    return {key: c.most_common(1)[0][0] for key, c in hits.items()}


def find_tables(data, materials):
    """Locate terrain tables: u16 n, n*(u64 token, u16 rot), u16 c, c*rgba, u16 rows, u16 cols, u32 rows*cols.

    Returns a list of (offset of first token, n). The rows*cols check makes false hits practically impossible.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size < 24:
        return []
    limit = arr.size - 24
    cand = np.nonzero((arr[:limit] >= 1) & (arr[:limit] <= MAX_MATERIALS) & (arr[1:limit + 1] == 0))[0]
    if cand.size == 0:
        return []
    value = np.zeros(cand.size, dtype=np.uint64)
    for k in range(8):
        value |= arr[cand + 2 + k].astype(np.uint64) << np.uint64(8 * k)
    cand = cand[np.isin(value, materials.sorted_array, assume_unique=False)]
    out = []
    next_free = 0
    for i in cand.tolist():
        if i < next_free:
            continue
        n = data[i]
        p = i + 2 + n * 10
        if p + 2 > len(data):
            continue
        ok = True
        for j in range(n):
            t, rot = struct.unpack_from("<QH", data, i + 2 + j * 10)
            if t not in materials.tokens or rot > ROTATIONS_OK:
                ok = False
                break
        if not ok:
            continue
        colors = struct.unpack_from("<H", data, p)[0]
        q = p + 2 + colors * 4
        if colors > MAX_COLORS or q + 8 > len(data):
            continue
        rows, cols, quads = struct.unpack_from("<HHI", data, q)
        if rows * cols != quads or quads == 0:
            continue
        out.append((i + 2, n))
        next_free = q + 8
    return out


class MaterialSet:
    def __init__(self, token_values):
        self.tokens = set(token_values)
        self.sorted_array = np.array(sorted(self.tokens), dtype=np.uint64)


def content_choice(data, start, length, frac):
    """True with probability `frac`, decided by the bytes themselves.

    Hashing the content makes copies of one item in different files agree on their zone.
    """
    digest = hashlib.blake2b(data[start:start + length], digest_size=2).digest()
    return (digest[0] | digest[1] << 8) / 65536.0 < frac


def patch_tables(data, tables, zone_low, frac, variant_of):
    """Return `data` with material tokens swapped for their per-zone variants.

    Each table goes to zone_low or zone_low + 1 with probability `frac`, which dissolves
    the sector edge into a patchwork rather than a straight line.
    """
    out = bytearray(data)
    swapped = 0
    for start, n in tables:
        zone = zone_low + 1 if content_choice(data, start, n * 10 + 32, frac) else zone_low
        for j in range(n):
            pos = start + j * 10
            t = struct.unpack_from("<Q", data, pos)[0]
            repl = variant_of(t, zone)
            if repl is not None:
                struct.pack_into("<Q", out, pos, repl)
                swapped += 1
    return bytes(out), swapped


def patch_desc(data, climate_token):
    """Set a sector's climate; the token sits at offset 24 of the 40 byte version 3 file."""
    if len(data) != 40 or struct.unpack_from("<I", data, 0)[0] != 3:
        raise ValueError("unknown sector description format")
    out = bytearray(data)
    struct.pack_into("<Q", out, 24, climate_token)
    return bytes(out)


def desc_climate(data):
    return tokens.decode(struct.unpack_from("<Q", data, 24)[0])
