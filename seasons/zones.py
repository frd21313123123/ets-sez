"""Which tokens of a sector point to something that has a copy for a snow zone, and swapping them.

Every item goes to one of two neighbouring zones as decided by its uid, so everything belonging to
a road, from its look and edges in .base to its sidewalks and vegetation in .data, ends up in one zone.
"""
import struct
from collections import Counter, namedtuple
from dataclasses import dataclass, field

from . import mapdata as md, sectors as sec

Slot = namedtuple("Slot", "kind file offset uid owner")

# A copied road or prefab may list only its own edges and corners, so these wait for their owner.
OWNER_KIND = {"edge": "look", "corner": "prefab"}


@dataclass
class Vocabulary:
    """Tokens of every kind of thing that can be swapped."""
    look: set = field(default_factory=set)
    legacy_look: set = field(default_factory=set)  # roads without a template model, which name their materials
    edge: set = field(default_factory=set)
    road_material: set = field(default_factory=set)
    prefab: set = field(default_factory=set)
    corner: set = field(default_factory=set)
    vegetation: set = field(default_factory=set)
    sidewalk: set = field(default_factory=set)
    center: set = field(default_factory=set)
    model: set = field(default_factory=set)
    curve: set = field(default_factory=set)
    far: set = field(default_factory=set)
    terrain_edge: set = field(default_factory=set)


def token_at(data, p):
    return struct.unpack_from("<Q", data, p)[0]


def _known(data, offsets, words):
    """Every one of these holds a known token or none, which tells real items from lookalikes."""
    for p in offsets:
        value = token_at(data, p)
        if value and value not in words:
            return False
    return True


def _base_slots(base, x, z, vocab):
    items = md.find_items(base, (md.ROAD, md.PREFAB), x, z)
    kinds = {item.uid: item.kind for item in items}
    owners = {}
    slots = []
    for item in items:
        if item.kind == md.ROAD:
            at = item.offset + md.ROAD_LOOK_AT
            look = token_at(base, at) if at + md.ROAD_MIN_SIZE <= len(base) else 0
            if look in vocab.look:
                owners[item.uid] = look
                slots.append(Slot("look", "base", at, item.uid, 0))
                slots += [Slot("edge", "base", item.offset + md.ROAD_EDGES_AT + i * 8, item.uid, look) for i in range(4)]
                if look in vocab.legacy_look:
                    slots.append(Slot("road_material", "base", item.offset + md.ROAD_MATERIAL_AT, item.uid, 0))
        else:
            at = item.offset + md.PREFAB_MODEL_AT
            model = token_at(base, at) if at + 8 <= len(base) else 0
            if model in vocab.prefab:
                owners[item.uid] = model
                slots.append(Slot("prefab", "base", at, item.uid, 0))
    return slots, kinds, owners


def _data_slots(data, kinds, owners, vocab):
    found = md.parse_data(data, kinds)
    slots = [Slot("vegetation", "data", p, uid, 0) for uid, p in found.vegetation]
    slots += [Slot("sidewalk", "data", p, uid, 0) for uid, p in found.sidewalks if owners.get(uid) in vocab.legacy_look]
    slots += [Slot("center", "data", p, uid, 0) for uid, p in found.center_materials]
    slots += [Slot("edge", "data", p, uid, owners.get(uid, 0)) for uid, p in found.edges]
    slots += [Slot("corner", "data", p, uid, owners.get(uid, 0)) for uid, p in found.corners]
    return slots


def _aux_slots(aux, x, z, vocab):
    slots = []
    for item in md.find_items(aux, (md.TERRAIN, md.BEZIER, md.MODEL, md.CURVE, md.FAR_MODEL), x, z):
        if item.kind == md.MODEL:
            at = item.offset + md.MODEL_TOKEN_AT
            if at + 8 <= len(aux) and token_at(aux, at) in vocab.model:
                slots.append(Slot("model", "aux", at, item.uid, 0))
        elif item.kind == md.CURVE:
            at = item.offset + md.CURVE_TOKEN_AT
            if at + 8 <= len(aux) and token_at(aux, at) in vocab.curve:
                slots.append(Slot("curve", "aux", at, item.uid, 0))
        elif item.kind == md.FAR_MODEL:
            slots += [Slot("far", "aux", p, item.uid, 0) for p in md.far_model_slots(aux, item, vocab.far)]
        elif item.kind == md.BEZIER:
            offsets = md.bezier_slots(item)
            if offsets[-1] + 8 <= len(aux) and _known(aux, offsets, vocab.vegetation):
                slots += [Slot("vegetation", "aux", p, item.uid, 0) for p in offsets]
        else:
            try:
                vegetation, edges = md.terrain_slots(aux, item)
                real = edges[-1] + 8 <= len(aux) and _known(aux, vegetation, vocab.vegetation) \
                    and _known(aux, edges, vocab.terrain_edge)
            except (md.LayoutError, struct.error):
                continue
            if real:
                slots += [Slot("vegetation", "aux", p, item.uid, 0) for p in vegetation]
                slots += [Slot("terrain_edge", "aux", p, item.uid, 0) for p in edges]
    return slots


def sector_slots(files, x, z, vocab):
    """All swappable tokens of a sector. `files` holds the bytes of base, data and aux.

    Raises mapdata.LayoutError if a .data file doesn't follow the known format.
    """
    slots, kinds, owners = _base_slots(files.get("base") or b"", x, z, vocab)
    if files.get("data"):
        slots += _data_slots(files["data"], kinds, owners, vocab)
    if files.get("aux"):
        slots += _aux_slots(files["aux"], x, z, vocab)
    return slots


def split_position(position, zone_count):
    """(lower zone, share of the upper one) of a sector's place on the temperature scale."""
    low = min(int(position), zone_count - 2)
    return low, position - low


def item_zone(uid, low, frac):
    return low + 1 if sec.content_choice(struct.pack("<Q", uid), 0, 8, frac) else low


def survey(files, slots, low, frac, vocab, zones):
    """How often each token occurs per zone: {(kind, zone, token, owner): count}."""
    tally = Counter()
    words = vars(vocab)
    for slot in slots:
        value = token_at(files[slot.file], slot.offset)
        if value in words[slot.kind]:
            zone = item_zone(slot.uid, low, frac)
            if zone in zones:
                tally[slot.kind, zone, value, slot.owner] += 1
    return tally


def apply_plan(originals, patched, slots, low, frac, plan):
    """Swap tokens in `patched`, a {file: bytearray}. Returns {kind: swaps}.

    `plan` maps a kind to {(zone, token): new token}.
    """
    swaps = Counter()
    for slot in slots:
        zone = item_zone(slot.uid, low, frac)
        new = plan[slot.kind].get((zone, token_at(originals[slot.file], slot.offset)))
        if new is None:
            continue
        owner_kind = OWNER_KIND.get(slot.kind)
        if owner_kind and (zone, slot.owner) not in plan[owner_kind]:
            continue
        struct.pack_into("<Q", patched[slot.file], slot.offset, new)
        swaps[slot.kind] += 1
    return swaps
