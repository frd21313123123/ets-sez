import datetime as dt
import json
import os
import shutil
import struct
import tempfile
import unittest
import zlib
from unittest import mock

from seasons import profiles, scsc, session
from seasons.paths import Paths

SAMPLE = """SiiNunit
{
user_profile : _nameless.0.1 {
 company_name: "Test"
 active_mods: 2
 active_mods[0]: "base_map|Base map"
 active_mods[1]: "map_fix|Map fix"
 customization: 0
 profile_name: "Test"
}
}
"""
OURS = ['ets_seasons_map|Seasons: map data', 'ets_seasons_roads|Seasons: roads', 'ets_seasons_roads2|Seasons: roads 2',
        'ets_seasons_nature|Seasons: nature', 'ets_seasons_date|Seasons: date']


class ScscTests(unittest.TestCase):
    def test_aes256_block_matches_the_standard_vector(self):
        # FIPS-197, appendix C.3.
        key = bytes(range(32))
        cipher = bytes.fromhex("8ea2b7ca516745bfeafc49904b496089")
        round_keys = scsc._expand_key(key)
        self.assertEqual(scsc.decrypt_block(cipher, round_keys).hex(), "00112233445566778899aabbccddeeff")

    def test_cbc_matches_the_standard_vector(self):
        # NIST SP 800-38A, F.2.6.
        key = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
        iv = bytes(range(16))
        cipher = bytes.fromhex("f58c4c04d6e5f1ba779eabfb5f7bfbd69cfc4e967edb808d679f777bc6702c7d")
        self.assertEqual(scsc.decrypt_cbc(cipher, key, iv).hex(),
                         "6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51")

    def test_decode_unwraps_header_padding_and_compression(self):
        plain = b"SiiNunit\n{\n}\n"
        body = zlib.compress(plain)
        pad = 16 - len(body) % 16
        raw = b"ScsC" + bytes(32) + bytes(16) + struct.pack("<I", len(plain)) + body + bytes([pad]) * pad
        with mock.patch.object(scsc, "decrypt_cbc", lambda data, key, iv: data):
            self.assertEqual(scsc.decode(raw), plain)


class ArrangeTests(unittest.TestCase):
    def test_season_mods_go_on_top_and_others_keep_order(self):
        text = profiles.arrange(SAMPLE)
        self.assertEqual(profiles.read_mods(text), ["base_map|Base map", "map_fix|Map fix"] + OURS)
        self.assertIn(f"active_mods: {2 + len(OURS)}", text)
        self.assertIn(f'active_mods[{1 + len(OURS)}]: "ets_seasons_date|Seasons: date"', text)

    def test_repeating_changes_nothing(self):
        once = profiles.arrange(SAMPLE)
        self.assertEqual(profiles.arrange(once), once)

    def test_misplaced_season_mods_are_moved(self):
        text = SAMPLE.replace('"base_map|Base map"', '"ets_seasons_date|old title"')
        self.assertEqual(profiles.read_mods(profiles.arrange(text)), ["map_fix|Map fix"] + OURS)

    def test_other_attributes_survive(self):
        text = profiles.arrange(SAMPLE)
        for line in ('company_name: "Test"', "customization: 0", 'profile_name: "Test"'):
            self.assertIn(line, text)

    def test_windows_line_endings_survive(self):
        text = profiles.arrange(SAMPLE.replace("\n", "\r\n"))
        self.assertEqual(text.count("\n"), text.count("\r\n"))
        self.assertEqual(len(profiles.read_mods(text)), 2 + len(OURS))

    def test_no_mod_list_is_an_error(self):
        with self.assertRaises(ValueError):
            profiles.arrange("SiiNunit\n{\n}\n")


class ProfileFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def test_new_profile_folder_is_the_hex_of_its_name(self):
        p = profiles.create_profile(self.dir, "Сезоны")
        self.assertEqual(os.path.basename(p.folder), "D0A1D0B5D0B7D0BED0BDD18B")
        self.assertEqual(profiles.list_profiles(self.dir)[0].name, "Сезоны")

    def test_apply_keeps_one_backup_of_the_original(self):
        folder = os.path.join(self.dir, "AA")
        os.makedirs(folder)
        path = os.path.join(folder, "profile.sii")
        with open(path, "w", encoding="utf8", newline="") as fh:
            fh.write(SAMPLE)
        p = profiles.list_profiles(self.dir)[0]
        self.assertTrue(profiles.apply_order(p))
        self.assertFalse(profiles.apply_order(p))
        with open(path + profiles.BACKUP_SUFFIX, encoding="utf8", newline="") as fh:
            self.assertEqual(fh.read(), SAMPLE)
        self.assertEqual(p.foreign_mods, ["base_map|Base map", "map_fix|Map fix"])

    def encrypted_profile(self, text):
        """A profile as the game leaves it. Decryption itself is covered by ScscTests."""
        folder = os.path.join(self.dir, "BB")
        os.makedirs(folder)
        path = os.path.join(folder, "profile.sii")
        with open(path, "wb") as fh:
            fh.write(b"ScsC" + text.encode("utf8"))
        return path

    def decoding(self):
        return mock.patch.object(scsc, "decode", lambda raw: raw[4:])

    def test_encrypted_profile_is_listed_with_its_real_name(self):
        self.encrypted_profile(SAMPLE.replace('profile_name: "Test"', r'profile_name: "\xd0\xa1\xd0\xb5\xd0\xb7"'))
        with self.decoding():
            listed = profiles.list_profiles(self.dir)
        self.assertEqual(listed[0].name, "Сез")
        self.assertEqual(listed[0].mods, ["base_map|Base map", "map_fix|Map fix"])

    def test_mods_are_changed_in_an_existing_encrypted_profile(self):
        text = SAMPLE.replace(' customization: 0', ' customization: 870\n cached_experience: 335')
        path = self.encrypted_profile(text)
        with self.decoding():
            p = profiles.list_profiles(self.dir)[0]
            self.assertTrue(profiles.apply_order(p))
        with open(path, encoding="utf8") as fh:
            saved = fh.read()
        self.assertEqual(profiles.read_mods(saved), ["base_map|Base map", "map_fix|Map fix"] + OURS)
        self.assertIn("customization: 870", saved)
        self.assertIn("cached_experience: 335", saved)
        with open(path + profiles.BACKUP_SUFFIX, "rb") as fh:
            self.assertEqual(fh.read(), b"ScsC" + text.encode("utf8"))

    def test_encrypted_profile_in_right_order_is_not_touched(self):
        path = self.encrypted_profile(profiles.arrange(SAMPLE))
        before = os.stat(path).st_mtime_ns
        with self.decoding():
            p = profiles.list_profiles(self.dir)[0]
            self.assertFalse(profiles.apply_order(p))
        self.assertEqual(os.stat(path).st_mtime_ns, before)
        self.assertFalse(os.path.exists(path + profiles.BACKUP_SUFFIX))

    def test_existing_profile_is_found_not_recreated(self):
        self.encrypted_profile(SAMPLE)
        with self.decoding():
            found = profiles.find_or_create(self.dir, "Test")
        self.assertEqual(os.path.basename(found.folder), "BB")
        self.assertEqual(os.listdir(self.dir), ["BB"])

    def test_profile_of_unknown_format_is_not_listed(self):
        folder = os.path.join(self.dir, "CC")
        os.makedirs(folder)
        with open(os.path.join(folder, "profile.sii"), "wb") as fh:
            fh.write(b"BSII\x00\x01\x02")
        self.assertEqual(profiles.list_profiles(self.dir), [])

    def test_template_has_every_attribute_of_a_working_profile(self):
        text = profiles.TEMPLATE.format(name="X", now=1)
        for key in ("face", "brand", "map_path", "logo", "company_name", "male", "user_data", "active_mods",
                    "customization", "profile_name", "creation_time", "save_time", "version", "online_user_name"):
            self.assertIn(f" {key}:", text)


class BringGameUpTests(unittest.TestCase):
    def run_with(self, windows, widths, timeout=30.0):
        """Feeds a scripted sequence of what the watcher sees; the last entry repeats."""
        self.activated = 0
        finds = iter(windows)
        sizes = iter(widths)
        last = {"f": None, "w": 136}

        def find():
            last["f"] = next(finds, last["f"])
            return last["f"]

        def width(_hwnd):
            last["w"] = next(sizes, last["w"])
            return last["w"]

        def activate(_hwnd):
            self.activated += 1

        return session.bring_game_up(timeout, find, width, activate, lambda _s: None)

    def test_leaves_the_game_alone_when_it_resizes_by_itself(self):
        self.assertTrue(self.run_with([None, None, 1], [136, 2560]))
        self.assertEqual(self.activated, 0)

    def test_gives_focus_when_the_game_stays_tiny(self):
        self.assertTrue(self.run_with([1], [136] * 6 + [2560]))
        self.assertGreater(self.activated, 0)

    def test_gives_up_when_the_game_never_appears(self):
        self.assertFalse(self.run_with([None], []))
        self.assertEqual(self.activated, 0)

    def test_gives_up_when_the_window_never_grows(self):
        self.assertFalse(self.run_with([1], [136], timeout=10.0))


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.paths = Paths(game=os.path.join(self.dir, "game"), docs=os.path.join(self.dir, "docs"),
                           out=os.path.join(self.dir, "out"))
        os.makedirs(self.paths.out)
        for pack in self.paths.map_packs:
            with open(pack, "wb") as fh:
                fh.write(b"map")
        self.state = {"zones": [], "materials": [], "climate_weights": {}}

    def test_stops_when_the_game_is_running(self):
        with mock.patch.object(session, "game_running", return_value=True):
            with self.assertRaises(session.GameRunning):
                session.prepare(self.paths, self.state, dt.date(2027, 1, 20), "Сезоны")
        self.assertFalse(os.path.exists(self.paths.mods))

    def test_stops_when_nothing_was_built(self):
        with self.assertRaises(session.NotBuilt):
            session.prepare(self.paths, None, dt.date(2027, 1, 20), "Сезоны")

    def test_installs_mods_and_arranges_a_new_profile(self):
        with mock.patch.object(session, "game_running", return_value=False):
            profile = session.prepare(self.paths, self.state, dt.date(2027, 1, 20), "Сезоны")
        self.assertEqual(sorted(os.listdir(self.paths.mods)),
                         sorted(f"{mod.split('|')[0]}.scs" for mod in OURS))
        self.assertEqual(profile.mods, OURS)
        listed = profiles.list_profiles(self.paths.profiles)
        self.assertEqual([p.name for p in listed], ["Сезоны"])
        self.assertEqual(listed[0].mods, OURS)

    def test_a_build_with_a_pack_missing_counts_as_not_built(self):
        with open(self.paths.state, "w", encoding="utf8") as fh:
            json.dump(self.state, fh)
        self.assertEqual(session.load_state(self.paths), self.state)
        os.remove(self.paths.nature_pack)
        self.assertIsNone(session.load_state(self.paths))

    def test_state_file_records_field_only_once(self):
        with open(self.paths.state, "w", encoding="utf8") as fh:
            json.dump(self.state, fh)
        full = {**self.state, "sectors": [[0, 0, 1.0]], "game_stamp": [1]}
        self.assertIs(session.complete_state(self.paths, full), full)


if __name__ == "__main__":
    unittest.main()
