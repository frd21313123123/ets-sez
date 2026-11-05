"""BC1/BC3 codec, DDS and tobj writers, and baking snow into terrain textures."""
import struct

import numpy as np

BC1_FORMATS = {71, 72}
BC3_FORMATS = {77, 78}
BC3_SRGB = 78
SNOW_RGB = (200, 205, 211)
SNOW_ALPHA = 40
ROAD_SNOW_ALPHA = 30  # its share of the specular: asphalt holds about 20, the game's terrain snow 123
WEAR_FLOOR = 1.5      # brightness difference across columns that still counts as a wheel path
RUT_WIDTH_M = 0.28    # half width of a wheel track
RUT_WANDER_M = 0.07   # how far a track wanders from its line
RUT_DEPTH = 3.0       # against streaks of unit variance: tracks are the last to get snow
RUT_STREAK_SHARE = 0.45
RUT_DARKENING = 0.42  # packed snow in a track against fresh snow beside it
RUT_GLOSS = 45
MULT_NEUTRAL = 188.0  # sRGB value of 0.5 in linear light
TREE_SNOW_RGB = (214, 220, 228)
SOLID_MIP = 4         # 1/16 of a texture's width, wider than twigs but narrower than a bark strip
BARK_SNOW = 0.12


def _rgb565(c):
    r = ((c >> 11) & 31) * 255 // 31
    g = ((c >> 5) & 63) * 255 // 63
    b = (c & 31) * 255 // 31
    return np.stack([r, g, b], axis=-1).astype(np.int32)


def _decode_colors(blocks):
    """blocks: (n, 8) uint8 BC1 colour blocks. Returns (n, 16, 3) int32."""
    c0 = blocks[:, 0].astype(np.int32) | blocks[:, 1].astype(np.int32) << 8
    c1 = blocks[:, 2].astype(np.int32) | blocks[:, 3].astype(np.int32) << 8
    p0, p1 = _rgb565(c0), _rgb565(c1)
    four = (c0 > c1)[:, None]
    p2 = np.where(four, (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four, (p0 + 2 * p1) // 3, 0)
    pal = np.stack([p0, p1, p2, p3], axis=1)
    bits = blocks[:, 4].astype(np.int64) | blocks[:, 5].astype(np.int64) << 8 | \
        blocks[:, 6].astype(np.int64) << 16 | blocks[:, 7].astype(np.int64) << 24
    idx = (bits[:, None] >> (np.arange(16) * 2)) & 3
    return np.take_along_axis(pal, idx[:, :, None], axis=1)


def _decode_alpha(blocks):
    """blocks: (n, 8) uint8 BC3 alpha blocks. Returns (n, 16) int32."""
    a0, a1 = blocks[:, 0].astype(np.int32), blocks[:, 1].astype(np.int32)
    w = np.arange(1, 7)[None, :]
    six = (a0[:, None] * (7 - w) + a1[:, None] * w) // 7
    w5 = np.arange(1, 5)[None, :]
    five = (a0[:, None] * (5 - w5) + a1[:, None] * w5) // 5
    alt = np.concatenate([five, np.zeros((len(a0), 1), np.int32), np.full((len(a0), 1), 255)], axis=1)
    rest = np.where((a0 > a1)[:, None], six, alt)
    pal = np.concatenate([a0[:, None], a1[:, None], rest], axis=1)
    bits = np.zeros(len(a0), dtype=np.int64)
    for i in range(6):
        bits |= blocks[:, 2 + i].astype(np.int64) << (8 * i)
    idx = (bits[:, None] >> (np.arange(16) * 3)) & 7
    return np.take_along_axis(pal, idx, axis=1)


def _bc1_alpha(blocks):
    """BC1 can hold one bit of alpha: index 3 of a block whose first colour isn't the larger one."""
    c0 = blocks[:, 0].astype(np.int32) | blocks[:, 1].astype(np.int32) << 8
    c1 = blocks[:, 2].astype(np.int32) | blocks[:, 3].astype(np.int32) << 8
    bits = blocks[:, 4].astype(np.int64) | blocks[:, 5].astype(np.int64) << 8 | \
        blocks[:, 6].astype(np.int64) << 16 | blocks[:, 7].astype(np.int64) << 24
    idx = (bits[:, None] >> (np.arange(16) * 2)) & 3
    return np.where((c0 <= c1)[:, None] & (idx == 3), 0, 255).astype(np.int32)


def decode_surface(data, width, height, dxgi_format):
    """Decode tightly packed blocks to an (h, w, 4) uint8 image."""
    bw, bh = (width + 3) // 4, (height + 3) // 4
    size = 8 if dxgi_format in BC1_FORMATS else 16
    raw = np.frombuffer(data, dtype=np.uint8, count=bw * bh * size).reshape(-1, size)
    if dxgi_format in BC1_FORMATS:
        rgb, a = _decode_colors(raw), _bc1_alpha(raw)
    else:
        rgb, a = _decode_colors(raw[:, 8:]), _decode_alpha(raw[:, :8])
    px = np.concatenate([rgb, a[:, :, None]], axis=2).reshape(bh, bw, 4, 4, 4)
    img = px.transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:height, :width].astype(np.uint8)


def _dds_mips(data):
    """(width, height, format, block bytes) of each mip in DDS bytes. Empty if the format is unsupported."""
    if data[:4] != b"DDS ":
        return
    h, w = struct.unpack_from("<II", data, 12)
    mips = max(1, struct.unpack_from("<I", data, 28)[0])
    fourcc = data[84:88]
    pos = 128
    if fourcc == b"DXT1":
        fmt = 71
    elif fourcc == b"DXT5":
        fmt = 77
    elif fourcc == b"DX10":
        fmt = struct.unpack_from("<I", data, 128)[0]
        pos = 148
    else:
        return
    if fmt not in BC1_FORMATS | BC3_FORMATS:
        return
    size = 8 if fmt in BC1_FORMATS else 16
    for level in range(mips):
        lw, lh = max(1, w >> level), max(1, h >> level)
        length = ((lw + 3) // 4) * ((lh + 3) // 4) * size
        if pos + length > len(data):
            return
        yield lw, lh, fmt, data[pos:pos + length]
        pos += length


def read_dds_chain(data, want_width):
    """Decode every mip from the one nearest to want_width down. None if the format is unsupported."""
    chain = [decode_surface(raw, lw, lh, fmt) for lw, lh, fmt, raw in _dds_mips(data) if lw <= want_width]
    return chain or None


def read_dds(data, want_width):
    """Decode the mip nearest to want_width from DDS bytes. None if the format is unsupported."""
    for lw, lh, fmt, raw in _dds_mips(data):
        if lw <= want_width:
            return decode_surface(raw, lw, lh, fmt)
    return None


def flat_base(rgb, size, seed=11):
    """Stand-in terrain texture of one colour, used while real textures are unavailable."""
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 7.0, (size, size, 1))
    out = np.clip(np.array(rgb, np.float32)[None, None, :] + grain, 0, 255)
    alpha = np.full((size, size, 1), 20, np.float32)
    return np.concatenate([out, alpha], axis=2).astype(np.uint8)


def _encode_colors(px):
    """px: (n, 16, 3) int32. Returns (n, 8) uint8 BC1 blocks, always in four colour mode."""
    mean = px.mean(axis=1, keepdims=True)
    d = px - mean
    cov_axis = d[:, :, :, None] * d[:, :, None, :]
    cov = cov_axis.sum(axis=1)
    # Two power iterations from the widest channel find the block's main colour axis.
    axis = (px.max(axis=1) - px.min(axis=1)).astype(np.float64) + 1e-3
    for _ in range(2):
        axis = np.einsum("nij,nj->ni", cov, axis)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    proj = np.einsum("nij,nj->ni", d, axis)
    lo = np.take_along_axis(px, proj.argmin(axis=1)[:, None, None].repeat(3, 2), axis=1)[:, 0]
    hi = np.take_along_axis(px, proj.argmax(axis=1)[:, None, None].repeat(3, 2), axis=1)[:, 0]
    pack = lambda c: (c[:, 0] >> 3) << 11 | (c[:, 1] >> 2) << 5 | c[:, 2] >> 3
    c0, c1 = pack(hi), pack(lo)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    same = c0 == c1
    c1 = np.where(same & (c0 > 0), c0 - 1, c1)
    c0 = np.where(same & (c0 == 0), 1, c0)
    p0, p1 = _rgb565(c0), _rgb565(c1)
    pal = np.stack([p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3], axis=1)
    dist = ((px[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(axis=3)
    idx = dist.argmin(axis=2).astype(np.int64)
    bits = (idx << (np.arange(16) * 2)).sum(axis=1)
    out = np.zeros((len(px), 8), np.uint8)
    out[:, 0], out[:, 1] = c0 & 255, c0 >> 8
    out[:, 2], out[:, 3] = c1 & 255, c1 >> 8
    for i in range(4):
        out[:, 4 + i] = (bits >> (8 * i)) & 255
    return out


def _encode_alpha(a):
    """a: (n, 16) int32. Returns (n, 8) uint8 BC3 alpha blocks in eight value mode."""
    a0, a1 = a.max(axis=1), a.min(axis=1)
    same = a0 == a1
    a0 = np.where(same & (a0 < 255), a0 + 1, a0)
    a1 = np.where(same & (a1 == 255), 254, a1)
    w = np.arange(1, 7)[None, :]
    pal = np.concatenate([a0[:, None], a1[:, None], (a0[:, None] * (7 - w) + a1[:, None] * w) // 7], axis=1)
    idx = np.abs(a[:, :, None] - pal[:, None, :]).argmin(axis=2).astype(np.int64)
    bits = (idx << (np.arange(16) * 3)).sum(axis=1)
    out = np.zeros((len(a), 8), np.uint8)
    out[:, 0], out[:, 1] = a0, a1
    for i in range(6):
        out[:, 2 + i] = (bits >> (8 * i)) & 255
    return out


def encode_bc3(img):
    """Encode an (h, w, 4) uint8 image to tightly packed BC3 blocks."""
    h, w = img.shape[:2]
    bw, bh = (w + 3) // 4, (h + 3) // 4
    padded = np.pad(img, ((0, bh * 4 - h), (0, bw * 4 - w), (0, 0)), mode="edge").astype(np.int32)
    blocks = padded.reshape(bh, 4, bw, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4)
    out = np.concatenate([_encode_alpha(blocks[:, :, 3]), _encode_colors(blocks[:, :, :3])], axis=1)
    return out.tobytes()


def mip_chain(img):
    """Box filtered chain from img down to 1x1."""
    chain = [img]
    while chain[-1].shape[0] > 1 or chain[-1].shape[1] > 1:
        cur = chain[-1].astype(np.int32)
        h, w = cur.shape[:2]
        if h > 1:
            cur = (cur[0:h // 2 * 2:2] + cur[1:h // 2 * 2:2]) // 2
        if w > 1:
            cur = (cur[:, 0:w // 2 * 2:2] + cur[:, 1:w // 2 * 2:2]) // 2
        chain.append(cur.astype(np.uint8))
    return chain


def write_dds(img, chain=None):
    """Return sRGB BC3 DDS bytes with a full mip chain for an (h, w, 4) image, as the game's own textures are.

    A `chain` given by the caller is written as it is, for textures whose lower mips must not be averaged.
    """
    h, w = img.shape[:2]
    chain = mip_chain(img) if chain is None else chain
    header = struct.pack(
        "<4s7I44x2I4s5I5I", b"DDS ", 124, 0x000A1007, h, w, max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 16,
        0, len(chain), 32, 0x4, b"DX10", 0, 0, 0, 0, 0, 0x401008, 0, 0, 0, 0)
    dx10 = struct.pack("<5I", BC3_SRGB, 3, 0, 1, 0)
    return header + dx10 + b"".join(encode_bc3(m) for m in chain)


def write_tobj(dds_path):
    """Binary texture object for a colour texture, laid out as in the game's terrain and road tobj files."""
    path = dds_path.encode("ascii")
    head = bytearray(48)
    head[0:4] = bytes.fromhex("010ab170")
    head[20] = 1
    head[24] = 2
    head[26:29] = b"\x01\x01\x01"
    head[37] = 1
    struct.pack_into("<I", head, 40, len(path))
    return bytes(head) + path


def periodic_noise(size, seed=7):
    """Seamless value in 0..1 with uniform distribution, so a threshold equals exact coverage."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((size, size))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    freq = np.sqrt(fx * fx + fy * fy)
    freq[0, 0] = 1.0
    shaped = np.fft.irfft2(np.fft.rfft2(white) / freq ** 1.15, s=(size, size))
    order = shaped.ravel().argsort()
    rank = np.empty(order.size, dtype=np.float32)
    rank[order] = np.linspace(0, 1, order.size, dtype=np.float32)
    return rank.reshape(size, size)


def _fractal(h, w, rng, stretch=1.0, lowest=1.0, slope=1.7):
    """Seamless noise of unit variance whose scale is tied to the repeat, not to the texel.

    `lowest` is the largest feature in cycles per repeat: anything wider stays flat, so no part
    of the texture ends up with more of a value than another. `stretch` lengthens features along V,
    and a steeper `slope` leaves less fine detail, which keeps thresholded edges from crumbling.
    """
    fy = np.fft.fftfreq(h, 1 / h)[:, None] * stretch
    fx = np.fft.rfftfreq(w, 1 / w)[None, :]
    freq = np.maximum(np.sqrt(fx * fx + fy * fy), lowest)
    out = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((h, w))) / freq ** slope, s=(h, w))
    return (out - out.mean()) / out.std()


def _smooth_columns(columns, width):
    width = max(1, width)
    n = columns.size
    wrapped = np.concatenate([columns, columns, columns])
    return np.convolve(wrapped, np.ones(width) / width, mode="same")[n:2 * n]


def rut_field(h, w, ruts, per_metre, seed=41):
    """How deep in a wheel track each texel lies, 0..1. `ruts` are U positions of the track centres.

    Each track wanders a little along the road, with whole waves per repeat so the texture stays
    seamless, and with a wave of its own so that the tracks of a pair don't move in step.
    """
    rng = np.random.default_rng(seed)
    v = np.arange(h, dtype=np.float32)[:, None] / h
    u = np.arange(w, dtype=np.float32)[None, :] / w
    field = np.zeros((h, w), np.float32)
    half_width = RUT_WIDTH_M * per_metre
    wander = RUT_WANDER_M * per_metre
    for centre in ruts:
        drift = np.zeros_like(v)
        for wave, share in ((1, 0.6), (2, 0.3), (5, 0.1)):
            drift += share * np.sin(2 * np.pi * (wave * v + rng.random()))
        off = (u - centre - wander * drift + 0.5) % 1.0 - 0.5
        field = np.maximum(field, np.exp(-(off / half_width) ** 2))
    return field


def road_noise(base, seed=23, ruts=None, per_metre=None):
    """Threshold map for a road texture, whose U axis runs across the road and V along it.

    Snow gathers in streaks along the road. With known wheel tracks it stays out of them first, so
    every level shows a pair of tracks per lane. Otherwise only the darker bands of the texture
    tell where wheels go, and a texture spans the whole width of a road, so only the difference
    from the nearby columns counts as wear: a side that is merely darker overall must not lose its
    snow. Like periodic_noise the result is uniform, so a threshold equals exact coverage.
    """
    h, w = base.shape[:2]
    streaks = _fractal(h, w, np.random.default_rng(seed), stretch=5.0, lowest=10.0)
    if ruts:
        return _uniform(streaks * RUT_STREAK_SHARE + rut_field(h, w, ruts, per_metre) * RUT_DEPTH).reshape(h, w)
    columns = _smooth_columns(base[:, :, :3].astype(np.float32).mean(axis=(0, 2)), w // 128)
    # A median ignores a step in brightness, such as the edge of a shoulder, but not a narrow band.
    reach = max(3, w // 3) | 1
    wrapped = np.concatenate([columns[-(reach // 2):], columns, columns[:reach // 2]])
    wear = np.median(np.lib.stride_tricks.sliding_window_view(wrapped, reach), axis=1) - columns
    wear = np.clip(wear / max(wear.std(), WEAR_FLOOR), -2.0, 2.0)
    return _uniform(streaks + 0.6 * wear[None, :]).reshape(h, w)


def _uniform(values):
    order = values.ravel().argsort()
    rank = np.empty(order.size, dtype=np.float32)
    rank[order] = np.linspace(0, 1, order.size, dtype=np.float32)
    return rank


def packed_snow(h, w, rgb, seed=31):
    """Snow compacted by traffic, covering one repeat of a road texture without tiling."""
    rng = np.random.default_rng(seed)
    broad = _fractal(h, w, rng, stretch=2.0, lowest=6.0)
    fine = rng.standard_normal((h, w))
    shade = (broad * 5.0 + fine * 1.5)[:, :, None]
    color = np.clip(np.array(rgb, np.float32)[None, None, :] + shade, 0, 255)
    alpha = np.clip(ROAD_SNOW_ALPHA + broad * 5.0, 0, 255)[:, :, None]
    return np.concatenate([color, alpha], axis=2).astype(np.uint8)


def resampled(noise, h, w):
    """The same pattern for a texture of another size."""
    ny = (np.arange(h) * noise.shape[0] // h)[:, None]
    nx = (np.arange(w) * noise.shape[1] // w)[None, :]
    return noise[ny, nx]


def snow_mask(noise, cover, softness=0.07):
    """Snow weight per texel. Masks of growing cover are nested, so levels blend into each other."""
    if cover >= 1.0:
        return np.ones_like(noise)
    return np.clip((cover + softness - noise) / (2 * softness), 0.0, 1.0) * min(1.0, cover / softness)


def bake(base, snow, noise, cover):
    """Blend snow into `base` (h, w, 4) by coverage. Snow repeats as a tile, noise stretches to fit."""
    h, w = base.shape[:2]
    sy = (np.arange(h) % snow.shape[0])[:, None]
    sx = (np.arange(w) % snow.shape[1])[None, :]
    ny = (np.arange(h) * noise.shape[0] // h)[:, None]
    nx = (np.arange(w) * noise.shape[1] // w)[None, :]
    mask = snow_mask(noise[ny, nx], cover)[:, :, None]
    tile = snow[sy, sx].astype(np.float32)
    out = base.astype(np.float32) * (1 - mask) + tile * mask
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def bake_road(base, snow, noise, cover, rut=None):
    """Snow on a road surface: all the images have the same size.

    Where it thins out the snow is dirtier, and the bare asphalt next to it is wet: darker and
    glossier. In a wheel track what snow remains is packed: grey and slick, so tracks still show
    where the snow has covered them.
    """
    mask = snow_mask(noise, cover)[:, :, None]
    deep = snow_mask(noise, max(0.0, cover - 0.12), 0.12)[:, :, None]
    wet = snow_mask(noise, min(1.0, cover + 0.15), 0.15)[:, :, None]
    bare = base.astype(np.float32)
    bare[:, :, :3] *= 1 - 0.25 * wet
    bare[:, :, 3:] += 30 * wet
    top = snow.astype(np.float32)
    top[:, :, :3] *= 0.86 + 0.14 * deep
    if rut is not None:
        pressed = rut[:, :, None]
        top[:, :, :3] *= 1 - RUT_DARKENING * pressed
        top[:, :, 3:] += RUT_GLOSS * pressed
    out = bare * (1 - mask) + top * mask
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def fade_decal(decal, cover):
    """A dark road decal under snow: its opacity goes as the road's snow comes."""
    out = decal.copy()
    out[:, :, 3] = (out[:, :, 3] * (1.0 - cover)).astype(np.uint8)
    return out


def neutral_mult(mult, cover):
    """A detail texture multiplied over a road would darken its snow, so it fades to no effect.

    It tiles at another scale than the snow pattern, so this can't follow the snow's shape.
    The shader doubles it after reading sRGB, which makes this value leave the colour unchanged.
    """
    out = mult.astype(np.float32)
    out[:, :, :3] += (MULT_NEUTRAL - out[:, :, :3]) * cover
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _exposed(chain, h, w):
    """How much snow a part of a tree atlas can hold, 0..1.

    Leaves and twigs have see-through texels around them, while bark is a solid block. A lower mip
    shows the difference, as its alpha is high only where everything around a texel is opaque.
    """
    level = chain[min(SOLID_MIP, len(chain) - 1)]
    ys = (np.arange(h) * level.shape[0] // h)[:, None]
    xs = (np.arange(w) * level.shape[1] // w)[None, :]
    solid = np.clip((level[ys, xs, 3].astype(np.float32) - 200) / 50, 0, 1)
    return 1 - (1 - BARK_SNOW) * solid


def bake_cut_out(chain, cover, seed=53):
    """Snow on something with a cut out shape, such as leaves: the colour changes, the shape doesn't.

    Each mip keeps the game's own alpha, as averaging it again would thin out trees in the distance.
    Lighter parts take snow first, which are the ones facing up in tree textures.
    """
    top = chain[0]
    h, w = top.shape[:2]
    rgb = top[:, :, :3].astype(np.float32)
    light = rgb.mean(axis=2)
    spread = max(float(light.std()), 1.0)
    noise = _fractal(h, w, np.random.default_rng(seed), lowest=4.0, slope=1.2)
    threshold = _uniform(noise - 0.8 * (light - light.mean()) / spread).reshape(h, w)
    mask = snow_mask(threshold, cover) * _exposed(chain, h, w)
    masks = [mask]
    while len(masks) < len(chain):
        cur = masks[-1]
        ch, cw = chain[len(masks)].shape[:2]
        ys = (np.arange(ch) * cur.shape[0] // ch)[:, None]
        xs = (np.arange(cw) * cur.shape[1] // cw)[None, :]
        ys2 = ((np.arange(ch) * 2 + 1) * cur.shape[0] // (2 * ch))[:, None]
        xs2 = ((np.arange(cw) * 2 + 1) * cur.shape[1] // (2 * cw))[None, :]
        masks.append((cur[ys, xs] + cur[ys2, xs2] + cur[ys, xs2] + cur[ys2, xs]) / 4)
    white = np.array(TREE_SNOW_RGB, np.float32)
    out = []
    for level, m in zip(chain, masks):
        painted = level.astype(np.float32)
        painted[:, :, :3] += (white - painted[:, :, :3]) * m[:, :, None]
        out.append(np.clip(painted + 0.5, 0, 255).astype(np.uint8))
    return out


def flat_snow(size):
    """Fallback snow tile when the game's own cannot be read."""
    rng = np.random.default_rng(3)
    grain = rng.normal(0, 3.0, (size, size, 1))
    rgb = np.clip(np.array(SNOW_RGB, np.float32)[None, None, :] + grain, 0, 255)
    alpha = np.full((size, size, 1), SNOW_ALPHA, np.float32)
    return np.concatenate([rgb, alpha], axis=2).astype(np.uint8)
