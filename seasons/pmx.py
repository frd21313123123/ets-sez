"""Just enough of the game's model files to learn where a road's lanes lie on its textures."""
import struct
from dataclasses import dataclass

import numpy as np

PMG_MAGIC = b"gmP"
PIECE_SIZE = 100
NO_OFFSET = 0xFFFFFFFF


@dataclass
class Piece:
    slot: int          # index into the model's material table
    positions: object  # (n, 3)
    uvs: list          # one (n, 2) array per UV channel


def material_tables(pmd):
    """The material paths of each of a model's looks, by slot."""
    _version, slots, looks = struct.unpack_from("<3I", pmd, 0)
    table_at = struct.unpack_from("<I", pmd, 14 * 4)[0]
    offsets = struct.unpack_from(f"<{slots * looks}I", pmd, table_at)
    names = [pmd[at:pmd.index(b"\0", at)].decode("ascii") for at in offsets]
    return [names[i * slots:(i + 1) * slots] for i in range(looks)]


def pieces(pmg):
    """Mesh pieces that have texture coordinates. An unknown geometry format gives an empty list."""
    if pmg[1:4] != PMG_MAGIC or pmg[0] != 0x15:
        return []
    count = struct.unpack_from("<I", pmg, 4)[0]
    table_at = struct.unpack_from("<I", pmg, 0x54)[0]
    out = []
    for i in range(count):
        fields = struct.unpack_from("<25I", pmg, table_at + i * PIECE_SIZE)
        vertices, channels, slot, stride, vertex_at, uv_at = fields[1], fields[3], fields[4], fields[15], fields[16], fields[18]
        if uv_at == NO_OFFSET or vertices == 0 or stride % 4:
            continue
        rows = np.frombuffer(pmg, dtype="<f4", count=vertices * stride // 4, offset=vertex_at).reshape(vertices, stride // 4)
        first = (uv_at - vertex_at) // 4
        uvs = [rows[:, first + 2 * c:first + 2 * c + 2] for c in range(channels) if first + 2 * c + 2 <= rows.shape[1]]
        if uvs:
            out.append(Piece(slot, rows[:, :3], uvs))
    return out
