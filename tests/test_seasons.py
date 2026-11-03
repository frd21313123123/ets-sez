import datetime as dt
import struct
import unittest

import numpy as np

from seasons import hashfs, model, sectors, textures, tokens


class CalendarTests(unittest.TestCase):
    def test_day_of_year(self):
        self.assertEqual(model.day_of_year(dt.date(2026, 11, 3)), 307)
        self.assertEqual(model.day_of_year(dt.date(2026, 6, 21)), 172)  # the game's own default

    def test_summer_time(self):
        self.assertEqual(model.summer_time(dt.date(2026, 11, 3)), 0)
        self.assertEqual(model.summer_time(dt.date(2026, 7, 1)), 1)
        self.assertEqual(model.summer_time(dt.date(2026, 3, 28)), 0)  # day before the change
        self.assertEqual(model.summer_time(dt.date(2026, 3, 29)), 1)


class HashAndTokenTests(unittest.TestCase):
    def test_city_hash_matches_archive_paths(self):
        self.assertEqual(hashfs.city64(b""), 0x9ae16a3b2f90404f)
        self.assertEqual(hashfs.city64(b"def"), 0x2c6f469efb31c45a)  # seen in def.scs
        long_path = b"material/ground/ground/grassland_sandy_soil_dry_01_shrubs_01_nrm.tobj"  # in base_share.scs
        self.assertEqual(hashfs.city64(long_path), 0xaca555055c9e3413)

    def test_tokens(self):
        self.assertEqual(tokens.encode("default"), 0x1573c3e700)
        self.assertEqual(tokens.decode(tokens.encode("szd07")), "szd07")
        with self.assertRaises(ValueError):
            tokens.encode("z" * 13)


class SnowModelTests(unittest.TestCase):
    def test_snow_is_ordered_by_cold(self):
        table = model.ZoneTable()
        for day in (15, 60, 100, 330):
            covers = [table.cover(z, day) for z in range(model.ZONE_COUNT)]
            for colder, warmer in zip(covers, covers[1:]):
                self.assertGreaterEqual(colder + 1e-9, warmer)

    def test_summer_has_no_snow_and_winter_has(self):
        table = model.ZoneTable()
        self.assertTrue(all(table.level(z, 200) == 0 for z in range(model.ZONE_COUNT)))
        self.assertEqual(table.level(0, 30), model.SNOW_LEVELS)
        self.assertEqual(table.level(model.ZONE_COUNT - 1, 30), 0)

    def test_neighbouring_zones_never_jump_more_than_one_level(self):
        table = model.ZoneTable()
        for day in range(1, 366):
            levels = [table.level(z, day) for z in range(model.ZONE_COUNT)]
            for a, b in zip(levels, levels[1:]):
                self.assertLessEqual(abs(a - b), 1, f"day {day}: {levels}")

    def test_snow_changes_gradually_through_the_year(self):
        table = model.ZoneTable()
        for z in range(model.ZONE_COUNT):
            for day in range(1, 365):
                self.assertLessEqual(abs(table.level(z, day) - table.level(z, day + 1)), 1)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.mats = sectors.MaterialSet([tokens.encode("grass"), tokens.encode("dirt")])

    def table(self, toks, rows=2, cols=3):
        out = struct.pack("<H", len(toks))
        for t in toks:
            out += struct.pack("<QH", tokens.encode(t), 0)
        out += struct.pack("<H", 1) + b"\xff\xff\xff\xff"
        out += struct.pack("<HHI", rows, cols, rows * cols) + b"\0" * 24
        return out

    def test_finds_real_tables_only(self):
        blob = b"\x01\0" * 5 + self.table(["grass", "dirt"]) + b"\0" * 16
        found = sectors.find_tables(blob, self.mats)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][1], 2)

    def test_rejects_wrong_quad_count(self):
        bad = bytearray(self.table(["grass"]))
        struct.pack_into("<I", bad, 2 + 10 + 2 + 4 + 4, 99)
        self.assertEqual(sectors.find_tables(bytes(bad) + b"\0" * 16, self.mats), [])

    def test_rejects_unknown_material(self):
        self.assertEqual(sectors.find_tables(self.table(["stone"]) + b"\0" * 16, self.mats), [])

    def test_patch_keeps_size_and_swaps_only_known(self):
        blob = self.table(["grass", "dirt"]) + b"\0" * 16
        found = sectors.find_tables(blob, self.mats)
        grass = tokens.encode("grass")
        out, n = sectors.patch_tables(blob, found, 3, 0.0,
                                      lambda t, z: tokens.encode(f"z{z}_0") if t == grass else None)
        self.assertEqual(len(out), len(blob))
        self.assertEqual(n, 1)
        self.assertEqual(struct.unpack_from("<Q", out, 2)[0], tokens.encode("z3_0"))
        self.assertEqual(struct.unpack_from("<Q", out, 12)[0], tokens.encode("dirt"))

    def test_desc(self):
        desc = struct.pack("<I", 3) + b"\0" * 20 + struct.pack("<QQ", tokens.encode("cold"), tokens.encode("default"))
        self.assertEqual(sectors.desc_climate(desc), "cold")
        out = sectors.patch_desc(desc, tokens.encode("szd03"))
        self.assertEqual(sectors.desc_climate(out), "szd03")
        self.assertEqual(out[32:], desc[32:])


class TextureTests(unittest.TestCase):
    def gradient(self, size=32):
        y, x = np.mgrid[0:size, 0:size]
        return np.stack([x * 8, y * 8, (x + y) * 4, np.full_like(x, 128)], axis=2).astype(np.uint8)

    def test_bc3_round_trip_is_close(self):
        img = self.gradient()
        decoded = textures.decode_surface(textures.encode_bc3(img), 32, 32, 77)
        self.assertLess(np.abs(decoded.astype(int) - img.astype(int)).mean(), 8)

    def test_noise_threshold_equals_coverage(self):
        noise = textures.periodic_noise(64)
        for cover in (0.25, 0.5, 0.75):
            self.assertAlmostEqual(textures.snow_mask(noise, cover).mean(), cover, delta=0.02)

    def test_snow_levels_are_nested(self):
        noise = textures.periodic_noise(64)
        masks = [textures.snow_mask(noise, c) for c in (0.25, 0.5, 0.75, 1.0)]
        for less, more in zip(masks, masks[1:]):
            self.assertTrue((less <= more + 1e-6).all())

    def test_tobj_layout_matches_game_sample(self):
        # Header taken from a loose tobj shipped in a working mod.
        sample = bytes.fromhex("010ab170" + "00" * 16 + "01000000" + "0200" + "030303" + "00" * 8 + "01" + "0000"
                               + "26" + "00" * 7)
        out = textures.write_tobj("/model2/props/garbage/debris_02_02.dds")
        self.assertEqual(out[:48], sample[:48])
        self.assertEqual(out[48:], b"/model2/props/garbage/debris_02_02.dds")

    def test_dds_round_trip_picks_requested_mip(self):
        data = textures.write_dds(self.gradient(32))
        self.assertEqual(textures.read_dds(data, 32).shape, (32, 32, 4))
        self.assertEqual(textures.read_dds(data, 16).shape, (16, 16, 4))
        self.assertIsNone(textures.read_dds(b"not a dds file", 16))

    def test_dds_size_is_consistent(self):
        data = textures.write_dds(self.gradient(32))
        self.assertEqual(data[:4], b"DDS ")
        self.assertEqual(len(data), 128 + sum(len(textures.encode_bc3(m)) for m in textures.mip_chain(self.gradient(32))))


if __name__ == "__main__":
    unittest.main()
