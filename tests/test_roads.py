import struct
import unittest

import numpy as np

from seasons import roads, sectors, textures, tokens, verify

SECTOR = sectors.Sector(8, -31, "sec+0008-0031", None)


def road_item(look, x=33000.0, z=-122000.0):
    """The bytes around a look token in a real road item."""
    head = b"\x11" * 7 + struct.pack("<3f", x, 10.0, z) + b"\0" * 8 + struct.pack("<3f", x + 50, 20.0, z + 50)
    head += b"\x0a" * 17  # the box starts 49 bytes before the token
    return head + struct.pack("<Q", tokens.encode(look)) + b"\0" * 16 + b"\x22" * 8


class RefTests(unittest.TestCase):
    def wanted(self, *names):
        return {struct.pack("<Q", tokens.encode(n)): tokens.encode(n) for n in names}

    def test_finds_real_item(self):
        item = road_item("un01")
        refs = roads.find_refs(item, SECTOR, self.wanted("un01"))
        self.assertEqual([r[1] for r in refs], [tokens.encode("un01")])

    def test_ignores_short_token_lookalike(self):
        # The same small number with no bounding box of this sector around it.
        blob = b"\0" * 80 + struct.pack("<Q", tokens.encode("un01")) + b"\0" * 40
        self.assertEqual(roads.find_refs(blob, SECTOR, self.wanted("un01")), [])

    def test_ignores_box_from_elsewhere(self):
        item = road_item("un01", x=-90000.0)
        self.assertEqual(roads.find_refs(item, SECTOR, self.wanted("un01")), [])

    def test_patch_swaps_only_refs_and_keeps_size(self):
        item = road_item("un01")
        value = tokens.encode("un01")
        refs = roads.find_refs(item, SECTOR, {struct.pack("<Q", value): value})
        out, n = roads.patch_refs(item, refs, 4, 0.0, {value: 7}, {4, 5})
        self.assertEqual((len(out), n), (len(item), 1))
        pos = refs[0][0]
        self.assertEqual(struct.unpack_from("<Q", out, pos)[0], tokens.encode("r4_7"))
        self.assertEqual(out[:pos], item[:pos])
        self.assertEqual(out[pos + 8:], item[pos + 8:])

    def test_zone_without_snow_stays_original(self):
        item = road_item("un01")
        value = tokens.encode("un01")
        refs = roads.find_refs(item, SECTOR, {struct.pack("<Q", value): value})
        out, n = roads.patch_refs(item, refs, 20, 0.0, {value: 7}, {4, 5})
        self.assertEqual((out, n), (item, 0))


class CopyTests(unittest.TestCase):
    def test_material_path_keeps_length(self):
        old = "/automat/67/672a03c6ba2f0402.mat"
        for zone in (0, 9, 15, 27):
            new = roads.zone_material_path(old, zone)
            self.assertEqual(len(new), len(old))
            self.assertTrue(new.endswith("/672a03c6ba2f0402.mat"))
        self.assertEqual(roads.zone_material_path(old, 15), "/automat/rf/672a03c6ba2f0402.mat")

    def test_model_patch_changes_only_paths(self):
        pmd = b"\x01\x02/automat/67/672a03c6ba2f0402.mat\0/automat/aa/bbbbbbbbbbbbbbbb.mat\0\x03"
        swaps = {"/automat/67/672a03c6ba2f0402.mat": "/automat/r3/672a03c6ba2f0402.mat"}
        out = roads.patch_model(pmd, swaps)
        self.assertEqual(len(out), len(pmd))
        self.assertIn(b"/automat/r3/672a03c6ba2f0402.mat", out)
        self.assertIn(b"/automat/aa/bbbbbbbbbbbbbbbb.mat", out)
        with self.assertRaises(ValueError):
            roads.patch_model(pmd, {"/automat/67/672a03c6ba2f0402.mat": "/too/short.mat"})

    def test_rename_covers_look_variants_and_models(self):
        body = ('\n\tname: "un"\n\ttemplate_right: "/road_template/un/a.pmd"\n'
                "\tlanes_left[]: traffic_lane.road.local\n"
                "\tlane_variants[]: .lane.variant.road.un11.full\n")
        related = [("road_template_variant", ".lane.variant.road.un11.full", '\n\ttemplate_right: "/road_template/un/b.pmd"\n')]
        look = roads.Look("un11", "road_look", body, related)
        self.assertEqual(look.models, ["/road_template/un/a.pmd", "/road_template/un/b.pmd"])
        text = roads.rename_look(look, "r3_7", {"/road_template/un/a.pmd": "/road_template/z3/m0.pmd",
                                                "/road_template/un/b.pmd": "/road_template/z3/m1.pmd"})
        self.assertIn("road_look : road.r3_7", text)
        self.assertIn("lane_variants[]: .rv.r3_7.0", text)
        self.assertIn("road_template_variant : .rv.r3_7.0", text)
        self.assertIn('"/road_template/z3/m0.pmd"', text)
        self.assertIn('"/road_template/z3/m1.pmd"', text)
        self.assertIn("traffic_lane.road.local", text)
        self.assertNotIn("road.un11", text)

    def test_copies_of_a_look_share_nothing_with_each_other(self):
        look = roads.Look("a1", "road_look", "\n\tlane_variants[]: .tmpl_var.road.other\n",
                          [("road_template_variant", ".tmpl_var.road.other", "\n\tx: 1\n")])
        text = roads.rename_look(look, "r1_0", {}) + roads.rename_look(look, "r2_0", {})
        units = roads.parse_units("SiiNunit\n{\n" + text + "}\n")
        self.assertEqual(verify.unit_rule_problems(units), [])
        self.assertNotIn(".tmpl_var.road.other", text)


class ParseTests(unittest.TestCase):
    def test_indented_units_stay_separate(self):
        text = ('SiiNunit\n{\n\troad_look : road.a1\n\t{\n\t\tname: "first"\n\t}\n\n'
                '\troad_look : road.b2\n\t{\n\t\tname: "second"\n\t}\n}\n')
        units = roads.parse_units(text)
        self.assertEqual([u[1] for u in units], ["road.a1", "road.b2"])
        self.assertNotIn("second", units[0][2])

    def test_braces_in_strings_and_comments_do_not_count(self):
        text = ('SiiNunit\n{\n# old { comment\nroad_look : road.a1\n{\n\tname: "odd } name"\n'
                "\t// another }\n\tx: 1\n}\nroad_look : road.b2\n{\n}\n}\n")
        self.assertEqual([u[1] for u in roads.parse_units(text)], ["road.a1", "road.b2"])

    def test_references_are_only_dotted_values(self):
        body = "\n\tlane_variants[]: .lane.a\n\tlanes_left[]: traffic_lane.road.local\n\tname: \"x\"\n"
        self.assertEqual(roads.references(body), [".lane.a"])


class LoaderRuleTests(unittest.TestCase):
    def test_the_mistakes_that_broke_the_game_are_caught(self):
        duplicate = [("road_look", "road.a", ""), ("road_look", "road.a", "")]
        self.assertTrue(verify.unit_rule_problems(duplicate))
        outside = [("road_look", "road.a", "\n\tlane_variants[]: .tmpl_var.road.b\n")]
        self.assertTrue(verify.unit_rule_problems(outside))
        shared = [("road_look", "road.a", "\n\tv[]: .x\n"), ("road_look", "road.b", "\n\tv[]: .x\n"), ("v", ".x", "")]
        self.assertTrue(verify.unit_rule_problems(shared))
        too_long = [("road_look", "road.thirteenchars", "")]
        self.assertTrue(verify.unit_rule_problems(too_long))

    def test_clean_file_passes(self):
        good = [("road_look", "road.a", "\n\tv[]: .x\n\tlanes[]: traffic_lane.road.local\n"), ("v", ".x", "")]
        self.assertEqual(verify.unit_rule_problems(good), [])

    def test_snowy_text_swaps_only_baked_slots(self):
        info = {"text": 'texture : "texture_base" {\n source : "/a.tobj"\n}\ntexture : "texture_nmap" {\n source : "/n.tobj"\n}\n',
                "slots": {"texture_base": "/a.tobj"}}
        text = roads.snowy_text(info, 2, {"/a.tobj": "r5"})
        self.assertIn('"/material/seasons/road/r5_2.tobj"', text)
        self.assertIn('"/n.tobj"', text)


class RoadNoiseTests(unittest.TestCase):
    def setUp(self):
        base = np.full((128, 128, 4), 120, np.uint8)
        base[:, 40:56, :3] = 60  # a dark worn wheel path
        self.base = base
        self.noise = textures.road_noise(base)

    def test_threshold_equals_coverage(self):
        for cover in roads.LEVEL_COVER.values():
            self.assertAlmostEqual(textures.snow_mask(self.noise, cover).mean(), cover, delta=0.03)

    def test_wheel_path_gets_less_snow(self):
        mask = textures.snow_mask(self.noise, 0.5)
        self.assertLess(mask[:, 42:54].mean(), mask[:, 80:120].mean())

    def test_snow_runs_along_the_road(self):
        n = self.noise
        along = np.abs(np.diff(n, axis=0)).mean()
        across = np.abs(np.diff(n, axis=1)).mean()
        self.assertLess(along, across)

    def test_top_level_keeps_some_bare_tracks(self):
        self.assertLess(textures.snow_mask(self.noise, roads.LEVEL_COVER[4]).mean(), 1.0)


if __name__ == "__main__":
    unittest.main()
