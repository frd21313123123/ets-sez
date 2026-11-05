"""Exact layout of sector files, format version 907: where each token of each map item sits.

.data holds the cosmetic part of roads and prefabs. It is walked field by field and has to end
exactly on its end marker, which proves every offset in it. Items in .base and .aux are found by
their header. .aux can't be walked without knowing every item type, so what is searched for there
either has a fixed place after the header or is checked against the fields around it.

Layouts: https://github.com/sk-zk/map-docs/wiki
"""
import struct
from dataclasses import dataclass, field

import numpy as np

FORMAT_VERSION = 907
HEADER_SIZE = 16
END_MARKER = 0xFFFFFFFFFFFFFFFF

TERRAIN, ROAD, PREFAB, MODEL, BEZIER, CURVE, FAR_MODEL = 1, 3, 4, 5, 39, 44, 43

SECTOR_SIZE = 4000.0
BOX_MIN_AT, BOX_MAX_AT = 12, 32  # k-DOP: five minimums followed by five maximums
ROAD_LOOK_AT = 61
ROAD_EDGES_AT = ROAD_LOOK_AT + 40       # right_edge_right, right_edge_left, left_edge_right, left_edge_left
ROAD_MATERIAL_AT = ROAD_LOOK_AT + 112   # only legacy roads, which have no template model
ROAD_MIN_SIZE = 120                     # from the look to past the last token used here
PREFAB_MODEL_AT = 57
MODEL_TOKEN_AT = 57
CURVE_TOKEN_AT = 72
TERRAIN_SIDES_AT = 139  # after nodes, node offsets, lengths, seed and railings
BEZIER_VEGETATION_AT = 265  # after control points, tessellation, node and seed
FAR_COUNT_AT = 65
FAR_MODEL_STRIDE = 20  # a token and a scale

VEGETATION_SIZE = 16
BEZIER_VEGETATION_SIZE = 11
VEGETATION_PART_SIZE = 32
PREFAB_NODE_SIZE = 6 + 2 * VEGETATION_SIZE
PREFAB_NODES = 6
SPHERE_SIZE = 20
EDGE_OVERRIDE_SIZE = 20
VARIANT_OVERRIDE_SIZE = 12
MAX_ARRAY = 100000


class LayoutError(ValueError):
    """A sector file doesn't follow the known format, most likely after a game update."""


@dataclass
class Item:
    kind: int
    offset: int
    uid: int


@dataclass
class DataSlots:
    """Offsets of swappable tokens in a .data file: lists of (uid, offset)."""
    vegetation: list = field(default_factory=list)
    sidewalks: list = field(default_factory=list)
    center_materials: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    corners: list = field(default_factory=list)
    items: int = 0


def _in_sector(data, start, x, z):
    min_x, min_y, min_z = struct.unpack_from("<3f", data, start + BOX_MIN_AT)
    max_x, max_y, max_z = struct.unpack_from("<3f", data, start + BOX_MAX_AT)
    x0, x1 = (x - 1) * SECTOR_SIZE, (x + 2) * SECTOR_SIZE
    z0, z1 = (z - 1) * SECTOR_SIZE, (z + 2) * SECTOR_SIZE
    return x0 <= min_x <= max_x <= x1 and z0 <= min_z <= max_z <= z1 and min_y <= max_y


def find_items(data, kinds, x, z):
    """Items of the given types, told from lookalikes by a bounding box lying in this sector."""
    arr = np.frombuffer(data, dtype=np.uint8)
    limit = arr.size - 80
    if limit <= 0:
        return []
    zeros = (arr[1:limit + 1] == 0) & (arr[2:limit + 2] == 0) & (arr[3:limit + 3] == 0)
    out = []
    for kind in kinds:
        for start in np.nonzero((arr[:limit] == kind) & zeros)[0].tolist():
            if _in_sector(data, start, x, z):
                out.append(Item(kind, start, struct.unpack_from("<Q", data, start + 4)[0]))
    return sorted(out, key=lambda item: item.offset)


def _u16(data, p):
    return struct.unpack_from("<H", data, p)[0]


def _u32(data, p):
    return struct.unpack_from("<I", data, p)[0]


def _count(data, p):
    n = _u32(data, p)
    if n > MAX_ARRAY:
        raise LayoutError(f"array of {n} at {p}")
    return n


def _skip_quad_info(data, p):
    """quad_info: materials, colours, one u32 per quad, then vertex and normal overrides."""
    p += 2 + _u16(data, p) * 10
    p += 2 + _u16(data, p) * 4
    rows, cols, quads = struct.unpack_from("<HHI", data, p)
    if rows * cols != quads:
        raise LayoutError(f"quad table of {rows}x{cols} says {quads} at {p}")
    p += 8 + quads * 4
    for _ in range(2):
        p += 4 + _count(data, p) * 16
    return p


def _road_payload(data, p, uid, slots):
    p += 8  # overlay scheme
    for _side in range(2):
        p += 24 + 2  # two side models, terrain size
        slots.vegetation += [(uid, p + i * VEGETATION_SIZE) for i in range(3)]
        p += 3 * VEGETATION_SIZE
        slots.sidewalks.append((uid, p))
        p = _skip_quad_info(data, p + 8)
    slots.center_materials.append((uid, p))
    p += 14 + 8  # material, colour, rotation; seed, previous length
    slots.vegetation.append((uid, p))
    p += 12 + 8  # centre vegetation, no detail vegetation ranges
    p += 4 + _count(data, p) * SPHERE_SIZE
    for _ in range(2):
        p += 4 + _count(data, p) * 8
    for _ in range(2):
        n = _count(data, p)
        slots.edges += [(uid, p + 4 + 4 + i * EDGE_OVERRIDE_SIZE) for i in range(n)]
        p += 4 + n * EDGE_OVERRIDE_SIZE
    for _ in range(2):
        p += 4 + _count(data, p) * VARIANT_OVERRIDE_SIZE
    return p + 32  # looks of the four edges


def _prefab_payload(data, p, uid, slots):
    p += 8  # look
    for i in range(PREFAB_NODES):
        slots.vegetation += [(uid, p + 6 + j * VEGETATION_SIZE) for j in range(2)]
        p += PREFAB_NODE_SIZE
    p += 4 + _count(data, p) * VEGETATION_PART_SIZE
    p += 4  # seed
    p += 4 + _count(data, p) * SPHERE_SIZE
    slots.corners += [(uid, p + i * 24) for i in range(PREFAB_NODES)]
    p += PREFAB_NODES * 24
    for _ in range(PREFAB_NODES):
        p = _skip_quad_info(data, p)
    return p


def parse_data(data, kinds):
    """Walk a .data file. `kinds` maps a uid to its item type, known from the .base file.

    Raises LayoutError unless the walk ends exactly at the end marker.
    """
    if struct.unpack_from("<I", data, 0)[0] != FORMAT_VERSION:
        raise LayoutError(f"format version {struct.unpack_from('<I', data, 0)[0]}, expected {FORMAT_VERSION}")
    slots = DataSlots()
    p = HEADER_SIZE
    try:
        while True:
            uid = struct.unpack_from("<Q", data, p)[0]
            p += 8
            if uid == END_MARKER:
                break
            kind = kinds.get(uid)
            if kind == ROAD:
                p = _road_payload(data, p, uid, slots)
            elif kind == PREFAB:
                p = _prefab_payload(data, p, uid, slots)
            else:
                raise LayoutError(f"item {uid:x} of type {kind} in .data at {p}")
            slots.items += 1
    except struct.error as err:
        raise LayoutError(f"ran past the end at {p}") from err
    if p != len(data):
        raise LayoutError(f"end marker at {p}, the file has {len(data)} bytes")
    return slots


def terrain_slots(data, item):
    """A terrain item in .aux: (offsets of vegetation tokens, offsets of its two edge tokens)."""
    p = item.offset + TERRAIN_SIDES_AT
    vegetation = []
    for _side in range(2):
        p += 2 + 12 + 12  # size, profile, previous profile
        vegetation += [p + i * VEGETATION_SIZE for i in range(3)]
        p += 3 * VEGETATION_SIZE + 4
    p += 4 + _count(data, p) * SPHERE_SIZE
    for _side in range(2):
        p = _skip_quad_info(data, p)
    return vegetation, [p, p + 16]


def bezier_slots(item):
    """A bezier patch in .aux: offsets of its vegetation tokens. Its structs are shorter than others'."""
    return [item.offset + BEZIER_VEGETATION_AT + i * BEZIER_VEGETATION_SIZE for i in range(3)]


def far_model_slots(data, item, tokens):
    """Offsets of the model tokens of a far model item, empty when it just lists other items."""
    count_at = item.offset + FAR_COUNT_AT
    if count_at + 4 > len(data):
        return []
    n = _u32(data, count_at)
    if n == 0 or n > 256 or count_at + 4 + n * FAR_MODEL_STRIDE > len(data):
        return []
    slots = [count_at + 4 + i * FAR_MODEL_STRIDE for i in range(n)]
    if all(struct.unpack_from("<Q", data, p)[0] in tokens for p in slots):
        return slots
    return []
