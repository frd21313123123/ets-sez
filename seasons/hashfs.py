"""Read-only access to the game's HashFS v2 archives."""
import glob
import os
import struct
import zlib

M = (1 << 64) - 1
K0, K1, K2, K3 = 0xc3a5c85c97cb3127, 0xb492b66fbe98f273, 0x9ae16a3b2f90404f, 0xc949d7c7509e6557
KMUL = 0x9ddfea08eb382d69

META_PLAIN, META_DIR = 128, 129

# Archives that hold no map or world data the mod needs.
SKIPPED_ARCHIVES = {"version.scs", "effect.scs", "core.scs", "locale.scs", "base_cfg.scs", "base_vehicle.scs"}


def _f64(s, p):
    return struct.unpack_from("<Q", s, p)[0]


def _f32(s, p):
    return struct.unpack_from("<I", s, p)[0]


def _rot(v, n):
    return v if n == 0 else ((v >> n) | (v << (64 - n))) & M


def _mix(v):
    return v ^ (v >> 47)


def _h16(u, v):
    a = ((u ^ v) * KMUL) & M
    a ^= a >> 47
    b = ((v ^ a) * KMUL) & M
    b ^= b >> 47
    return (b * KMUL) & M


def _weak(s, p, a, b):
    w, x, y, z = _f64(s, p), _f64(s, p + 8), _f64(s, p + 16), _f64(s, p + 24)
    a = (a + w) & M
    b = _rot((b + a + z) & M, 21)
    c = a
    a = (a + x + y) & M
    b = (b + _rot(a, 44)) & M
    return (a + z) & M, (b + c) & M


def city64(s):
    """CityHash64 1.0.3, the variant HashFS uses for its paths."""
    n = len(s)
    if n <= 16:
        if n > 8:
            a, b = _f64(s, 0), _f64(s, n - 8)
            return _h16(a, _rot((b + n) & M, n & 63)) ^ b
        if n >= 4:
            return _h16((n + (_f32(s, 0) << 3)) & M, _f32(s, n - 4))
        if n > 0:
            y = s[0] + (s[n >> 1] << 8)
            z = n + (s[n - 1] << 2)
            return (_mix(((y * K2) ^ (z * K3)) & M) * K2) & M
        return K2
    if n <= 32:
        a = (_f64(s, 0) * K1) & M
        b = _f64(s, 8)
        c = (_f64(s, n - 8) * K2) & M
        d = (_f64(s, n - 16) * K0) & M
        return _h16((_rot((a - b) & M, 43) + _rot(c, 30) + d) & M, (a + _rot(b ^ K3, 20) - c + n) & M)
    if n <= 64:
        z = _f64(s, 24)
        a = (_f64(s, 0) + (n + _f64(s, n - 16)) * K0) & M
        b = _rot((a + z) & M, 52)
        c = _rot(a, 37)
        a = (a + _f64(s, 8)) & M
        c = (c + _rot(a, 7)) & M
        a = (a + _f64(s, 16)) & M
        vf = (a + z) & M
        vs = (b + _rot(a, 31) + c) & M
        a = (_f64(s, 16) + _f64(s, n - 32)) & M
        z = _f64(s, n - 8)
        b = _rot((a + z) & M, 52)
        c = _rot(a, 37)
        a = (a + _f64(s, n - 24)) & M
        c = (c + _rot(a, 7)) & M
        a = (a + _f64(s, n - 16)) & M
        wf = (a + z) & M
        ws = (b + _rot(a, 31) + c) & M
        r = _mix(((vf + ws) * K2 + (wf + vs) * K0) & M)
        return (_mix((r * K0 + vs) & M) * K2) & M
    x = _f64(s, n - 40)
    y = (_f64(s, n - 16) + _f64(s, n - 56)) & M
    z = _h16((_f64(s, n - 48) + n) & M, _f64(s, n - 24))
    v = _weak(s, n - 64, n, z)
    w = _weak(s, n - 32, (y + K1) & M, x)
    x = (x * K1 + _f64(s, 0)) & M
    left = (n - 1) & ~63
    p = 0
    while True:
        x = (_rot((x + y + v[0] + _f64(s, p + 8)) & M, 37) * K1) & M
        y = (_rot((y + v[1] + _f64(s, p + 48)) & M, 42) * K1) & M
        x ^= w[1]
        y = (y + v[0] + _f64(s, p + 40)) & M
        z = (_rot((z + w[0]) & M, 33) * K1) & M
        v = _weak(s, p, (v[1] * K1) & M, (x + w[0]) & M)
        w = _weak(s, p + 32, (z + w[1]) & M, (y + _f64(s, p + 16)) & M)
        z, x = x, z
        p += 64
        left -= 64
        if left == 0:
            break
    return _h16((_h16(v[0], w[0]) + _mix(y) * K1 + z) & M, (_h16(v[1], w[1]) + x) & M)


class Archive:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.f = open(path, "rb")
        h = self.f.read(64)
        magic, self.version, self.salt, _ = struct.unpack_from("<4sHH4s", h, 0)
        if magic != b"SCS#" or self.version != 2:
            raise ValueError(f"{self.name}: not a HashFS v2 archive")
        count, elen, _mc, mlen, estart, mstart, _sec, _pl = struct.unpack_from("<IIIIQQQB", h, 12)
        self.f.seek(estart)
        et = zlib.decompress(self.f.read(elen))
        self.f.seek(mstart)
        self.meta = zlib.decompress(self.f.read(mlen))
        self.entries = {}
        for i in range(count):
            hsh, midx, mcnt, _fl = struct.unpack_from("<QIHH", et, i * 16)
            self.entries[hsh] = (midx, mcnt)

    def _key(self, path):
        s = path.strip("/").encode()
        return city64(str(self.salt).encode() + s if self.salt else s)

    def has(self, path):
        return self._key(path) in self.entries

    def _metas(self, path):
        e = self.entries.get(self._key(path))
        if e is None:
            return None
        out = {}
        for i in range(e[1]):
            raw = _f32(self.meta, (e[0] + i) * 4)
            out[raw >> 24] = raw & 0xFFFFFF
        return out

    def _blob(self, index):
        csz, sz, _pad, block = struct.unpack_from("<IIII", self.meta, index * 4)
        csz &= 0x0FFFFFFF
        sz &= 0x0FFFFFFF
        self.f.seek(block * 16)
        d = self.f.read(csz)
        return zlib.decompress(d) if csz != sz else d

    def read(self, path):
        m = self._metas(path)
        if not m:
            return None
        for kind in (META_PLAIN, META_DIR):
            if kind in m:
                return self._blob(m[kind])
        return None

    def listdir(self, path):
        m = self._metas(path)
        if not m or META_DIR not in m:
            return None
        d = self._blob(m[META_DIR])
        n = _f32(d, 0)
        pos, out = 4 + n, []
        for length in d[4:4 + n]:
            out.append(d[pos:pos + length].decode("utf8", "replace"))
            pos += length
        return out



class GameFiles:
    """Union view of the installed archives, DLCs taking priority over the base game."""

    def __init__(self, game_dir):
        self.game_dir = game_dir
        archives = []
        for path in sorted(glob.glob(os.path.join(game_dir, "*.scs"))):
            if os.path.basename(path) in SKIPPED_ARCHIVES:
                continue
            try:
                archives.append(Archive(path))
            except ValueError:
                continue
        archives.sort(key=lambda a: not a.name.startswith("dlc_"))
        self.archives = archives

    def read(self, path):
        for a in self.archives:
            d = a.read(path)
            if d is not None:
                return d
        return None

    def read_all(self, directory, prefix):
        """Yield (name, bytes) for files in `directory` starting with `prefix`, from every archive."""
        for a in self.archives:
            for name in a.listdir(directory) or []:
                if name.startswith(prefix) and not name.startswith("/"):
                    d = a.read(f"{directory}/{name}")
                    if d is not None:
                        yield name, d
