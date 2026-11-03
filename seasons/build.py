"""Builds the two mod packages.

ets_seasons_map  - rarely rebuilt: patched sectors, material definitions, baked snow textures.
ets_seasons_date - rebuilt on every launch: date, climates and which snow level each zone shows.
"""
import json
import os
import re
import struct
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import hashfs, materials as mats, model, roads, sectors as sec, textures as tex, tokens

GAME_VERSIONS = "1.61.*"
BAKE_WIDTH = 512
CLIMATE_GEOMETRY = """\tmap_projection: lambert_conic
\tstandard_paralel_1: 37
\tstandard_paralel_2: 65

\tmap_origin: (50, 15)
\tmap_offset: (16660.0, 4150.0)
\tmap_factor: (-1.71570875e-4, 1.729241463e-4)
"""
# Reference temperature of the stock weather profiles, which span about 16..31 degrees.
PROFILE_MEAN_TEMPERATURE = 23.0
CLIMATE_CLASSES = {"default": "d", "cold": "d", "arid": "a", "desert": "x"}
CLASS_BAD_WEATHER = {"d": 0.07, "a": 0.03, "x": 0.01}


def manifest(name, title, version, category, description):
    return (f'SiiNunit\n{{\nmod_package : .package_name\n{{\n\tpackage_version: "{version}"\n'
            f'\tdisplay_name: "{title}"\n\tauthor: "ets-sez"\n\tcategory[]: "{category}"\n'
            f'\tcompatible_versions[]: "{GAME_VERSIONS}"\n\tdescription_file: "description.txt"\n}}\n}}\n'), description


def _write_common(z, name, title, version, category, description):
    text, desc = manifest(name, title, version, category, description)
    z.writestr("manifest.sii", text)
    z.writestr("description.txt", desc)


def climate_token(cls, zone):
    return f"sz{cls}{zone:02d}"


def plan_sectors(sector_list, located, stations):
    """Continuous zone coordinate for every sector."""
    table = {located[k]: stations[k] for k in located}
    return {s.name: model.zone_position(model.field(s.x, s.z, table)[0]) for s in sector_list}


def game_stamp(game_dir):
    """Changes whenever a game update touches the map or the definitions."""
    out = []
    for name in ("base_map.scs", "def.scs"):
        st = os.stat(os.path.join(game_dir, name))
        out += [st.st_size, int(st.st_mtime)]
    return out


def _field_entries(sector_list, located, positions):
    return {"sectors": [[s.x, s.z, round(positions[s.name], 3)] for s in sector_list],
            "located": {k: list(v) for k, v in located.items()}}


def field_record(game_dir):
    """Sector positions on the temperature field, for drawing the map without opening the archives."""
    sector_list = sec.list_sectors(hashfs.GameFiles(game_dir))
    located = sec.locate_stations(sector_list, model.STATIONS)
    return _field_entries(sector_list, located, plan_sectors(sector_list, located, model.STATIONS))


def tally_materials(files, sector_list, materials):
    """Count terrain area per material token over the whole map."""
    token_set = sec.MaterialSet(mats.token_value(n) for n in materials)
    tally = Counter()
    for s in sector_list:
        for kind in sec.SECTOR_KINDS:
            data = sec.read_sector(s, kind)
            if not data:
                continue
            for start, n in sec.find_tables(data, token_set):
                for j in range(n):
                    tally[struct.unpack_from("<Q", data, start + j * 10)[0]] += 1
    return tally


def choose_materials(usable, tally, top):
    by_area = sorted(usable.values(), key=lambda m: -tally.get(mats.token_value(m.name), 0))
    return by_area[:top]


def baked_path(key, level):
    return f"/material/seasons/{key}_{level}.tobj"


def read_extracted(root, tobj_path, width):
    """Decode a texture from a folder produced by SCS's archive extractor."""
    path = os.path.join(root, tobj_path.lstrip("/").rsplit(".", 1)[0] + ".dds")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return tex.read_dds(fh.read(), width)


def bake_all(chosen, zf, textures_dir, log):
    """Bake a texture per snow level per distinct base texture; returns {base texture: key}.

    The game packs its textures with GDeflate, so real ones come from a folder extracted with
    SCS's own tool. Without that folder each material gets a flat stand-in of its mean colour.
    """
    snow = read_extracted(textures_dir, "/material/terrain/snow.tobj", BAKE_WIDTH) if textures_dir else None
    if snow is None:
        snow = tex.flat_snow(BAKE_WIDTH)
    noise = tex.periodic_noise(BAKE_WIDTH)
    keys = {}
    real = flat = 0
    for m in chosen:
        if m.base_texture in keys:
            continue
        base = read_extracted(textures_dir, m.base_texture, BAKE_WIDTH) if textures_dir else None
        if base is None:
            base = tex.flat_base(mats.mean_color(m), BAKE_WIDTH // 4)
            flat += 1
        else:
            real += 1
        key = f"t{len(keys)}"
        for level in range(1, model.SNOW_LEVELS + 1):
            baked = tex.bake(base, snow, noise, level / model.SNOW_LEVELS)
            dds = f"/material/seasons/{key}_{level}.dds"
            zf.writestr(dds[1:], tex.write_dds(baked))
            zf.writestr(baked_path(key, level)[1:], tex.write_tobj(dds))
        keys[m.base_texture] = key
    log(f"textures baked: {real} from the game, {flat} flat stand-ins")
    return keys


def stock_climate_weights(files):
    """The weather weights of the stock arid and desert climates, which copies must repeat."""
    text = files.read("def/climate.sii").decode("utf8", "replace")
    out = {}
    for cls, source in (("a", "arid"), ("x", "desert")):
        start = text.find(f"climate.{source} ")
        block = text[start:text.find("}", start)]
        out[cls] = re.findall(r"weights\[\]:\s*(\S+)", block)
    return out


def _patch_sector(job):
    """Worker: patch one sector's files. Runs in a separate process."""
    path, name, x, z, position, known, zones, indices, road_index, prefab_index = job
    archive = hashfs.Archive(path)
    token_set = sec.MaterialSet(known)
    low = min(int(position), model.ZONE_COUNT - 2)
    frac = position - low
    index_of = dict(indices)

    def variant_of(tok, zone):
        idx = index_of.get(tok)
        if idx is None or zone not in zones:
            return None
        return tokens.encode(mats.variant_name(zone, idx))

    out = {}
    swapped = road_total = prefab_total = 0
    for kind in sec.SECTOR_KINDS:
        data = archive.read(f"map/europe/{name}.{kind}")
        if not data:
            continue
        tables = sec.find_tables(data, token_set)
        patched, n = sec.patch_tables(data, tables, low, frac, variant_of)
        if kind == "base" and road_index:
            wanted = {struct.pack("<Q", value): value for value in road_index}
            refs = roads.find_refs(data, sec.Sector(x, z, name, None), wanted)
            patched, road_swaps = roads.patch_refs(patched, refs, low, frac, road_index, zones)
            n += road_swaps
            road_total += road_swaps
        if kind == "base" and prefab_index:
            refs = roads.find_prefab_refs(data, sec.Sector(x, z, name, None), set(prefab_index))
            patched, prefab_swaps = roads.patch_prefab_refs(data, patched, refs, low, frac, prefab_index, zones)
            n += prefab_swaps
            prefab_total += prefab_swaps
        if n:
            out[kind] = patched
            swapped += n
    return name, out, swapped, road_total, prefab_total


def build_map_pack(game_dir, out_path, top=800, textures_dir=None, workers=None, log=print):
    started = time.time()
    files = hashfs.GameFiles(game_dir)
    sector_list = sec.list_sectors(files)
    log(f"sectors: {len(sector_list)}")
    located = sec.locate_stations(sector_list, model.STATIONS)
    log(f"stations located on the map: {len(located)} of {len(model.STATIONS)}")
    positions = plan_sectors(sector_list, located, model.STATIONS)
    materials = mats.load_materials(files)
    usable = mats.attach_text(materials, files)
    log(f"terrain materials: {len(materials)}, can receive snow: {len(usable)}")
    table = model.ZoneTable()
    zones = sorted(z for z in range(model.ZONE_COUNT) if table.snows_ever(z))
    log(f"zones that ever see snow: {len(zones)} of {model.ZONE_COUNT}")

    known = [mats.token_value(n) for n in materials]
    tally = tally_materials(files, sector_list, materials)
    chosen = choose_materials(usable, tally, top)
    total = sum(tally.values()) or 1
    covered = sum(tally.get(mats.token_value(m.name), 0) for m in chosen)
    log(f"materials chosen: {len(chosen)}, {100 * covered / total:.0f}% of terrain tables")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        _write_common(zf, "ets_seasons_map", "Seasons: map data", "0.1", "map",
                      "Snow-ready terrain for the base map and DLCs. Needs 'Seasons: date' too.")
        keys = bake_all(chosen, zf, textures_dir, log)
        final = [m for m in chosen if m.base_texture in keys]
        indices = [(mats.token_value(m.name), i) for i, m in enumerate(final)]

        defs = ["SiiNunit\n{\n"]
        for zone in zones:
            for i, m in enumerate(final):
                defs.append(f"material_def : terrain_mat.{mats.variant_name(zone, i)}\n{{\n"
                            f'\tpath: "/material/seasons/z{zone}/{i}.mat"\n{m.extra}}}\n')
        defs.append("}\n")
        zf.writestr("def/world/terrain_material.ets_seasons.sii", "".join(defs))

        veg = mats.detail_vegetation(files)
        lines = ["SiiNunit\n{\n"]
        for base_texture, key in keys.items():
            models = veg.get(base_texture)
            if not models:
                continue
            # Grass stays under light snow and is buried from the third level on.
            for level in range(1, model.SNOW_LEVELS - 1):
                lines.append(f"detail_vegetation_def : grassdef.s{key}_{level}\n{{\n"
                             f'\ttexture: "{baked_path(key, level)}"\n')
                lines.extend(f"\t{ln}\n" for ln in models)
                lines.append("}\n")
        lines.append("}\n")
        zf.writestr("def/world/detail_vegetation_def.ets_seasons.sii", "".join(lines))

        road_index, prefab_index, road_records = {}, {}, []
        if textures_dir:
            road_index, prefab_index, road_records = roads.build_into(zf, files, sector_list, positions, zones,
                                                                      textures_dir, read_extracted, log)

        jobs = []
        for s in sector_list:
            desc = s.archive.read(f"map/europe/{s.name}.desc")
            cls = CLIMATE_CLASSES.get(sec.desc_climate(desc), "d")
            nearest = int(round(positions[s.name]))
            zf.writestr(f"map/europe/{s.name}.desc",
                        sec.patch_desc(desc, tokens.encode(climate_token(cls, nearest))))
            low = min(int(positions[s.name]), model.ZONE_COUNT - 2)
            if low in zones or low + 1 in zones:
                jobs.append((s.archive.path, s.name, s.x, s.z, positions[s.name], known, set(zones), indices,
                             road_index, prefab_index))

        swapped = road_swaps = prefab_swaps = done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, out, n, roads_here, prefabs_here in pool.map(_patch_sector, jobs, chunksize=4):
                for kind, data in out.items():
                    zf.writestr(f"map/europe/{name}.{kind}", data)
                swapped += n
                road_swaps += roads_here
                prefab_swaps += prefabs_here
                done += 1
                if done % 100 == 0:
                    log(f"  sectors patched: {done}/{len(jobs)}")
        log(f"sectors patched: {done}, references swapped: {swapped}, "
            f"of them road pieces: {road_swaps}, prefabs: {prefab_swaps}")

    state = {
        **_field_entries(sector_list, located, positions),
        "game_stamp": game_stamp(game_dir),
        "road_materials": road_records,
        "road_tokens": sorted(road_index),
        "prefab_tokens": sorted(prefab_index),
        "zones": zones,
        "climate_weights": stock_climate_weights(files),
        "materials": [{"name": m.name, "path": m.path, "base_texture": m.base_texture,
                       "text": m.text, "key": keys[m.base_texture]} for m in final],
    }
    with open(out_path + ".state.json", "w", encoding="utf8") as fh:
        json.dump(state, fh)
    log(f"map pack: {os.path.getsize(out_path) / 2 ** 30:.2f} GB in {time.time() - started:.0f}s")
    return state


def climate_text(day, table, weights):
    out = ["SiiNunit\n{\n"]
    for cls in CLASS_BAD_WEATHER:
        rows = weights.get(cls, [])
        block = f"\tweights: {len(rows)}\n" + "".join(f"\tweights[]: {w}\n" for w in rows) if rows else ""
        for zone in range(model.ZONE_COUNT):
            # Arid and desert zones keep the small warm bias of the stock climates.
            offset = table.temperature(zone, day) - PROFILE_MEAN_TEMPERATURE + {"d": 0, "a": 2, "x": 6}[cls]
            out.append(f"climate_profile : climate.{climate_token(cls, zone)} {{\n"
                       f"\tclear_skyboxes_count: 3\n\tbad_weather_factor: {CLASS_BAD_WEATHER[cls]}\n"
                       f"\twetting_factor: 0.1\n\tdrying_factor: 0.01\n\n{CLIMATE_GEOMETRY}\n"
                       f"\tcopy: default\n{block}\ttemperatures: 1\n\ttemperatures[]: {offset:+.1f}\n}}\n\n")
    out.append("}\n")
    return "".join(out)


def env_text(date):
    return ("SiiNunit\n{\nenv_profile : env.data {\n\n\tlamps_on_elevation: 5.0\n\n"
            f"\tday_in_year: {model.day_of_year(date)}\n\tsummer_time: {model.summer_time(date)}\n\n"
            "\tthunderstorm_probability: 0.37\n\n}\n}\n")


def build_date_pack(state, date, out_path):
    """Everything that depends on the date. Fast enough to run before each game start."""
    table = model.ZoneTable()
    day = model.day_of_year(date)
    final = []
    for raw in state["materials"]:
        m = mats.TerrainMaterial(raw["name"], raw["path"], "", raw["text"], raw["base_texture"])
        final.append((m, raw["key"]))
    tmp = out_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        _write_common(zf, "ets_seasons_date", "Seasons: date", date.isoformat(), "weather_setup",
                      f"Season state for {date.isoformat()}. Rewritten when the launcher runs.")
        zf.writestr("def/env_data.sii", env_text(date))
        zf.writestr("def/climate.ets_seasons.sii", climate_text(day, table, state["climate_weights"]))
        for zone in state["zones"]:
            level = table.level(zone, day)
            cover = level / model.SNOW_LEVELS
            for i, (m, key) in enumerate(final):
                text = m.text if level == 0 else mats.snowy_text(m, baked_path(key, level), cover)
                zf.writestr(f"material/seasons/z{zone}/{i}.mat", text)
            for target, text in roads.date_files(state.get("road_materials", []), zone, level):
                zf.writestr(target, text)
    os.replace(tmp, out_path)


def cover_map(sectors, table, day):
    """Snow cover per sector, blending neighbouring zones like the map pack does."""
    out = {}
    for x, z, p in sectors:
        low = min(int(p), model.ZONE_COUNT - 2)
        frac = p - low
        out[(x, z)] = table.cover(low, day) * (1 - frac) + table.cover(low + 1, day) * frac
    return out


def preview_png(game_dir, out_path, dates):
    """A map panel per date with snow cover by sector."""
    from PIL import Image, ImageDraw

    record = field_record(game_dir)
    located = record["located"]
    table = model.ZoneTable()
    xs = [s[0] for s in record["sectors"]]
    zs = [s[1] for s in record["sectors"]]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    cell = 3 if len(dates) > 1 else 8
    w, h = (x1 - x0 + 1) * cell, (z1 - z0 + 1) * cell
    cols = 4 if len(dates) > 1 else 1
    rows = (len(dates) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + 8), rows * (h + 22)), (24, 28, 34))
    draw = ImageDraw.Draw(sheet)
    for i, date in enumerate(dates):
        ox, oy = (i % cols) * (w + 8) + 4, (i // cols) * (h + 22) + 18
        day = model.day_of_year(date)
        draw.text((ox, oy - 15), date.isoformat(), fill=(230, 230, 230))
        for (x, z), cover in cover_map(record["sectors"], table, day).items():
            green, white = np.array((64, 92, 58)), np.array((240, 244, 250))
            color = tuple(int(v) for v in green + (white - green) * cover)
            px, pz = ox + (x - x0) * cell, oy + (z - z0) * cell
            draw.rectangle([px, pz, px + cell - 1, pz + cell - 1], fill=color)
    sheet.save(out_path)
    return located
