import collections
import io
import struct
import unittest
import zipfile

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

    def test_finds_item_with_a_variant_after_the_look(self):
        for after in (tokens.encode("local_road"), 0):
            item = road_item("un01")[:-24] + struct.pack("<QQ", after, tokens.encode("no_lines")) + b"\x22" * 8
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


def prefab_item(model, x=33000.0, z=-122000.0, kind=4):
    """A prefab item up to its model token, laid out as in the game's sector files."""
    head = struct.pack("<I", kind) + b"\x11" * 8 + struct.pack("<3f", x, 10.0, z) + b"\0" * 8
    head += struct.pack("<3f", x + 50, 20.0, z + 50) + b"\0" * 8 + b"\x0a" * 5
    return head + struct.pack("<QQ", tokens.encode(model), tokens.encode("default")) + b"\x22" * 8


class PrefabTests(unittest.TestCase):
    def test_finds_real_item(self):
        item = b"\x33" * 9 + prefab_item("un_1403g")
        value = tokens.encode("un_1403g")
        self.assertEqual(roads.find_prefab_refs(item, SECTOR, {value}), [(9 + 57, value)])

    def test_ignores_other_items_and_places(self):
        value = tokens.encode("un_1403g")
        self.assertEqual(roads.find_prefab_refs(prefab_item("un_1403g", kind=5), SECTOR, {value}), [])
        self.assertEqual(roads.find_prefab_refs(prefab_item("un_1403g", x=-90000.0), SECTOR, {value}), [])
        self.assertEqual(roads.find_prefab_refs(prefab_item("other"), SECTOR, {value}), [])

    def test_patch_swaps_only_the_model(self):
        item = prefab_item("un_1403g")
        value = tokens.encode("un_1403g")
        refs = roads.find_prefab_refs(item, SECTOR, {value})
        out, n = roads.patch_prefab_refs(item, item, refs, 4, 0.0, {value: 7}, {4, 5})
        self.assertEqual((len(out), n), (len(item), 1))
        self.assertEqual(struct.unpack_from("<Q", out, 57)[0], tokens.encode("sp4_7"))
        self.assertEqual(out[:57], item[:57])
        self.assertEqual(out[65:], item[65:])

    def test_zone_without_snow_stays_original(self):
        item = prefab_item("un_1403g")
        value = tokens.encode("un_1403g")
        refs = roads.find_prefab_refs(item, SECTOR, {value})
        self.assertEqual(roads.patch_prefab_refs(item, item, refs, 20, 0.0, {value: 7}, {4, 5}), (item, 0))

    PREFABS = {
        "cross": '\n\tmodel_desc: "/prefab2/cross.pmd"\n\tprefab_desc: "/prefab2/cross.ppd"\n'
                 "\tmodel_left_hand_traffic: prefab.crosslh\n\tcompatible_roads[0]: road.un11\n",
        "crosslh": '\n\tmodel_desc: "/prefab2/crosslh.pmd"\n\tmodel_right_hand_traffic: prefab.cross\n',
        "other": '\n\tmodel_desc: "/prefab2/other.pmd"\n',
    }

    def test_linked_versions_share_their_zones(self):
        found = collections.defaultdict(set, {"cross": {3, 4}, "crosslh": {4, 9}})
        roads.spread_over_links(found, self.PREFABS)
        self.assertEqual(found["cross"], {3, 4, 9})
        self.assertEqual(found["crosslh"], {3, 4, 9})
        self.assertNotIn("other", found)

    def test_with_linked_pulls_in_the_other_version(self):
        self.assertEqual(roads.with_linked(["cross"], self.PREFABS), {"cross", "crosslh"})
        self.assertEqual(roads.with_linked(["other"], self.PREFABS), {"other"})

    def test_copy_points_to_copies_of_its_zone(self):
        body = roads.copied_prefab(self.PREFABS["cross"], 4, {"/prefab2/cross.pmd": "/prefab_seasons/z04/m00007.pmd"},
                                   {"cross": 0, "crosslh": 1})
        self.assertIn('model_desc: "/prefab_seasons/z04/m00007.pmd"', body)
        self.assertIn('prefab_desc: "/prefab2/cross.ppd"', body)
        self.assertIn("model_left_hand_traffic: prefab.sp4_1", body)
        self.assertIn("compatible_roads[0]: road.un11", body)

    def test_prefab_known_by_name_is_left_out_with_its_links(self):
        kept = roads.without_prefabs(self.PREFABS, {"crosslh"})
        self.assertEqual(sorted(kept), ["other"])
        self.assertEqual(sorted(roads.without_prefabs(self.PREFABS, set())), ["cross", "crosslh", "other"])

    def test_names_the_game_looks_up_are_found(self):
        class Files:
            def read_all(self, directory, prefix):
                yield "game_data.sii", b"tiny_garages[]: \tprefab.poor_garage\nsmall_garages[]: prefab.81\n"
                yield "europe.portal_overrides.sii", b"portal_prefab_override : x {\n prefab_name: 300\n}\n"
                yield "readme.txt", b"prefab.not_this\n"
        self.assertEqual(roads.prefabs_known_by_name(Files()), {"poor_garage", "81", "300"})

    def test_link_outside_the_file_is_caught(self):
        good = [("prefab_model", "prefab.sp4_0", "\n\tmodel_left_hand_traffic: prefab.sp4_1\n"),
                ("prefab_model", "prefab.sp4_1", "\n\tmodel_right_hand_traffic: prefab.sp4_0\n")]
        self.assertEqual(verify.prefab_link_problems(good), [])
        bad = [("prefab_model", "prefab.sp4_0", "\n\tmodel_left_hand_traffic: prefab.un_14001lh\n")]
        self.assertTrue(verify.prefab_link_problems(bad))

    def test_copies_of_a_model_have_names_of_one_length(self):
        self.assertEqual(len(roads.prefab_model_path(3, 7)), len(roads.prefab_model_path(27, 1945)))


class SharedEntryTests(unittest.TestCase):
    def test_alias_shares_data_and_can_be_read(self):
        buffer = io.BytesIO()
        payload = b"geometry" * 500
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("before.txt", b"x")
            zf.writestr("a/z03.pmg", payload)
            size = buffer.tell()
            roads.add_alias(zf, "a/z03.pmg", "a/z15.pmg")
            self.assertEqual(buffer.tell(), size)
        with zipfile.ZipFile(buffer) as zf:
            self.assertEqual(sorted(zf.namelist()), ["a/z03.pmg", "a/z15.pmg", "before.txt"])
            first, second = zf.getinfo("a/z03.pmg"), zf.getinfo("a/z15.pmg")
            self.assertEqual(first.header_offset, second.header_offset)
            self.assertEqual(roads.read_shared(zf, second), payload)
            self.assertIsNone(verify.corrupt_entry(zf))


class SnowModeTests(unittest.TestCase):
    ASPHALT = "/material/road/road_template/un/asp_new_01_difspec.tobj"

    def test_every_kind_of_road_surface_gets_snow(self):
        for substance in ("road", "road_dirt", "road_coarse", "dirt", "gravel"):
            self.assertEqual(roads.snow_mode("eut2.dif.spec.weight.rfx", substance, "/material/x.tobj"), "road")

    def test_surface_without_a_substance_goes_by_its_texture(self):
        self.assertEqual(roads.snow_mode("eut2.dif.spec.weight.rfx", "", self.ASPHALT), "road")
        self.assertEqual(roads.snow_mode("eut2.dif.spec.rfx", "", "/material/tiled/sidewalk_it_01.tobj"), "road")
        self.assertEqual(roads.snow_mode("eut2.dif.spec.rfx", "", "/material/terrain/grass_city_fr.tobj"), "ground")

    def test_markings_stay_and_wear_fades(self):
        decal = "eut2.dif.spec.decal.over.rfx"
        self.assertIsNone(roads.snow_mode(decal, "", "/road_template/un/default/lines_local_difopa.tobj"))
        self.assertIsNone(roads.snow_mode(decal, "", "/material/special/lines_eu_01.tobj"))
        self.assertEqual(roads.snow_mode(decal, "", "/material/decal/harddecal/road_cracks_01_difopa.tobj"), "fade")
        self.assertEqual(roads.snow_mode(decal, "road_dirt", "/material/road/road_template/ibe/overlay_3_ibe.tobj"),
                         "fade")

    def test_opacity_in_alpha_is_not_painted_over(self):
        self.assertEqual(roads.snow_mode("eut2.dif.spec.weight.a.rfx", "road_dirt", self.ASPHALT), "fade")

    def test_walls_and_rails_stay(self):
        self.assertIsNone(roads.snow_mode("eut2.dif.spec.rfx", "", "/model/rail/rail.tobj"))
        self.assertIsNone(roads.snow_mode("eut2.dif.spec.rfx", "concrete", "/material/subset/wall.tobj"))

    def test_both_layers_follow_the_base_pattern(self):
        info = {"mode": "road", "slots": {"texture_base": "/a.tobj", "texture_over": "/b.tobj"}}
        self.assertEqual(sorted(roads.bake_jobs([info])),
                         [("/a.tobj", "road", "/a.tobj"), ("/b.tobj", "road", "/a.tobj")])


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

    def test_darker_side_of_the_road_keeps_its_snow(self):
        base = np.full((512, 512, 4), 120, np.uint8)
        base[:, :256, :3] = 80
        mask = textures.snow_mask(textures.road_noise(base), 0.5)
        self.assertAlmostEqual(mask[:, :256].mean(), mask[:, 256:].mean(), delta=0.1)

    def test_snow_scale_does_not_depend_on_texture_size(self):
        # The share of the road between two thresholds' edges stays alike at any resolution.
        small = np.full((256, 256, 4), 120, np.uint8)
        big = np.full((1024, 1024, 4), 120, np.uint8)
        edge = lambda n: (np.abs(np.diff(textures.snow_mask(n, 0.5), axis=1)) > 0.2).mean()
        self.assertLess(edge(textures.road_noise(big)), edge(textures.road_noise(small)))

    def test_bake_keeps_bare_surface_and_covers_the_rest(self):
        base = np.full((64, 64, 4), 60, np.uint8)
        snow = textures.packed_snow(64, 64, (180, 185, 190))
        noise = textures.periodic_noise(64)
        out = textures.bake_road(base, snow, noise, 0.5)
        mask = textures.snow_mask(noise, 0.5)
        self.assertGreater(out[mask > 0.99][:, 0].mean(), 130)
        self.assertLess(out[mask < 0.01][:, 0].mean(), 60)

    def test_decal_fades_with_snow(self):
        decal = np.full((8, 8, 4), 200, np.uint8)
        self.assertEqual(textures.fade_decal(decal, 0.75)[0, 0, 3], 50)
        self.assertEqual(textures.fade_decal(decal, 0.75)[0, 0, 0], 200)


if __name__ == "__main__":
    unittest.main()
