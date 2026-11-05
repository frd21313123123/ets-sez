import collections
import io
import struct
import unittest
import zipfile

import numpy as np

from seasons import mapdata, pmx, roads, textures, tokens, verify, zones

X, Z = 8, -31
BOX = (33000.0, -122000.0)


def item(kind, uid, token=None, token_at=None, size=400, x=BOX[0], z=BOX[1]):
    """A map item laid out as in the game's sector files, with one token in place."""
    data = bytearray(size)
    struct.pack_into("<IQ", data, 0, kind, uid)
    struct.pack_into("<3f", data, 12, x, 10.0, z)
    struct.pack_into("<3f", data, 32, x + 50, 20.0, z + 50)
    if token:
        struct.pack_into("<Q", data, token_at, tokens.encode(token))
    return bytes(data)


def road_item(uid, look, edges=()):
    data = bytearray(item(mapdata.ROAD, uid, look, mapdata.ROAD_LOOK_AT))
    for i, edge in enumerate(edges):
        struct.pack_into("<Q", data, mapdata.ROAD_EDGES_AT + i * 8, tokens.encode(edge))
    return bytes(data)


def quad_info(material="grass1"):
    return (struct.pack("<H", 1) + struct.pack("<QH", tokens.encode(material), 0) + struct.pack("<H", 1)
            + b"\xff" * 4 + struct.pack("<HHI", 1, 2, 2) + b"\0" * 8 + struct.pack("<II", 0, 0))


def road_data(uid, vegetation="v2_14", sidewalk="", center="grass1", edge_override=None):
    side = b"\0" * 24 + b"\0" * 2 + struct.pack("<Q", tokens.encode(vegetation)) + b"\0" * 8 + b"\0" * 32
    side += struct.pack("<Q", tokens.encode(sidewalk) if sidewalk else 0) + quad_info()
    out = struct.pack("<Q", uid) + b"\0" * 8 + side + side
    out += struct.pack("<Q", tokens.encode(center)) + b"\0" * 6 + b"\0" * 8
    out += struct.pack("<Q", tokens.encode(vegetation)) + b"\0" * 4 + b"\0" * 8 + struct.pack("<I", 0)
    out += struct.pack("<I", 0) * 2
    if edge_override:
        out += struct.pack("<I", 1) + struct.pack("<hH", 0, 1) + struct.pack("<Q", tokens.encode(edge_override)) + b"\0" * 8
    else:
        out += struct.pack("<I", 0)
    out += struct.pack("<I", 0) * 3 + b"\0" * 32
    return out


def prefab_data(uid, corner="blke260", vegetation="v2_14"):
    out = struct.pack("<Q", uid) + b"\0" * 8
    node = b"\0" * 6 + struct.pack("<Q", tokens.encode(vegetation)) + b"\0" * 8 + b"\0" * 16
    out += node * 6 + struct.pack("<I", 0) + b"\0" * 4 + struct.pack("<I", 0)
    out += struct.pack("<Q", tokens.encode(corner)) + b"\0" * 16 + b"\0" * 24 * 5
    empty = struct.pack("<H", 0) + struct.pack("<H", 0) + struct.pack("<HHI", 0, 0, 0) + struct.pack("<II", 0, 0)
    return out + empty * 6


def data_file(*payloads):
    head = struct.pack("<I", mapdata.FORMAT_VERSION) + struct.pack("<Q", tokens.encode("euro2")) + struct.pack("<I", 3)
    return head + b"".join(payloads) + struct.pack("<Q", mapdata.END_MARKER)


def vocabulary():
    words = lambda *names: {tokens.encode(n) for n in names}
    return zones.Vocabulary(look=words("un11", "look1"), legacy_look=words("look1"), edge=words("un_sh_15", "un_crb_15"),
                            road_material=words("road0"), prefab=words("un_1403g"), corner=words("blke260"),
                            vegetation=words("v2_14"), sidewalk=words("sw1"), center=words("grass1"),
                            model=words("hill1"), curve=words("hedge1"), far=words("far1"), terrain_edge=words("tre_01"))


class DataFileTests(unittest.TestCase):
    def test_walk_finds_every_slot_and_ends_on_the_marker(self):
        data = data_file(road_data(1, edge_override="un_sh_15"), prefab_data(2))
        slots = mapdata.parse_data(data, {1: mapdata.ROAD, 2: mapdata.PREFAB})
        self.assertEqual(slots.items, 2)
        values = lambda found: [struct.unpack_from("<Q", data, p)[0] for _uid, p in found]
        self.assertIn(tokens.encode("v2_14"), values(slots.vegetation))
        self.assertEqual(values(slots.center_materials), [tokens.encode("grass1")])
        self.assertEqual(values(slots.edges), [tokens.encode("un_sh_15")])
        self.assertIn(tokens.encode("blke260"), values(slots.corners))

    def test_a_file_that_does_not_end_on_the_marker_is_refused(self):
        data = data_file(road_data(1))
        with self.assertRaises(mapdata.LayoutError):
            mapdata.parse_data(data + b"\0", {1: mapdata.ROAD})
        with self.assertRaises(mapdata.LayoutError):
            mapdata.parse_data(data[:-20], {1: mapdata.ROAD})

    def test_another_format_version_is_refused(self):
        data = bytearray(data_file(road_data(1)))
        struct.pack_into("<I", data, 0, 908)
        with self.assertRaises(mapdata.LayoutError):
            mapdata.parse_data(bytes(data), {1: mapdata.ROAD})

    def test_an_item_that_is_not_in_base_is_refused(self):
        with self.assertRaises(mapdata.LayoutError):
            mapdata.parse_data(data_file(road_data(1)), {})


class ItemTests(unittest.TestCase):
    def test_box_from_elsewhere_is_a_lookalike(self):
        self.assertEqual(mapdata.find_items(item(mapdata.ROAD, 1, x=-90000.0), (mapdata.ROAD,), X, Z), [])
        found = mapdata.find_items(b"\x33" * 5 + item(mapdata.ROAD, 1), (mapdata.ROAD,), X, Z)
        self.assertEqual([(i.offset, i.uid) for i in found], [(5, 1)])

    def test_far_model_that_lists_items_has_no_tokens(self):
        data = bytearray(item(mapdata.FAR_MODEL, 1))
        found = mapdata.find_items(bytes(data), (mapdata.FAR_MODEL,), X, Z)[0]
        self.assertEqual(mapdata.far_model_slots(bytes(data), found, {tokens.encode("far1")}), [])
        struct.pack_into("<I", data, mapdata.FAR_COUNT_AT, 1)
        struct.pack_into("<Q", data, mapdata.FAR_COUNT_AT + 4, tokens.encode("far1"))
        self.assertEqual(mapdata.far_model_slots(bytes(data), found, {tokens.encode("far1")}), [mapdata.FAR_COUNT_AT + 4])
        self.assertEqual(mapdata.far_model_slots(bytes(data), found, {tokens.encode("other")}), [])


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.base = road_item(1, "un11", ("un_sh_15", "un_crb_15"))
        self.data = data_file(road_data(1, edge_override="un_sh_15"))
        self.files = {"base": self.base, "data": self.data, "aux": b""}
        self.vocab = vocabulary()
        self.slots = zones.sector_slots(self.files, X, Z, self.vocab)

    def plan(self, **tables):
        plan = collections.defaultdict(dict)
        for kind, new in tables.items():
            plan[kind] = new
        return plan

    def patch(self, plan):
        patched = {k: bytearray(v) for k, v in self.files.items()}
        return patched, zones.apply_plan(self.files, patched, self.slots, 4, 0.0, plan)

    def test_everything_belonging_to_a_road_is_found(self):
        kinds = collections.Counter(s.kind for s in self.slots)
        self.assertEqual(kinds["look"], 1)
        self.assertEqual(kinds["edge"], 5)  # four in .base, one override in .data
        self.assertEqual(kinds["vegetation"], 7)
        self.assertEqual(kinds["center"], 1)

    def test_road_keeps_its_size_and_changes_only_tokens(self):
        look, edge = tokens.encode("un11"), tokens.encode("un_sh_15")
        plan = self.plan(look={(4, look): tokens.encode("r4_7")}, edge={(4, edge): tokens.encode("ez4_1")})
        patched, swaps = self.patch(plan)
        self.assertEqual(swaps["look"], 1)
        self.assertEqual(swaps["edge"], 2)
        self.assertEqual(len(patched["base"]), len(self.base))
        self.assertEqual(struct.unpack_from("<Q", patched["base"], mapdata.ROAD_LOOK_AT)[0], tokens.encode("r4_7"))
        self.assertEqual(struct.unpack_from("<Q", patched["base"], mapdata.ROAD_EDGES_AT)[0], tokens.encode("ez4_1"))
        self.assertEqual(struct.unpack_from("<Q", patched["base"], mapdata.ROAD_EDGES_AT + 8)[0], tokens.encode("un_crb_15"))

    def test_edge_waits_for_its_road(self):
        edge = tokens.encode("un_sh_15")
        patched, swaps = self.patch(self.plan(edge={(4, edge): tokens.encode("ez4_1")}))
        self.assertEqual(swaps["edge"], 0)
        self.assertEqual(bytes(patched["base"]), self.base)

    def test_zone_without_a_copy_stays_original(self):
        look = tokens.encode("un11")
        patched, swaps = self.patch(self.plan(look={(9, look): tokens.encode("r9_7")}))
        self.assertEqual(sum(swaps.values()), 0)
        self.assertEqual(bytes(patched["base"]), self.base)

    def test_a_road_with_a_template_keeps_its_old_material_token(self):
        self.assertFalse([s for s in self.slots if s.kind in ("road_material", "sidewalk")])

    def test_old_road_gives_up_its_materials(self):
        files = {"base": road_item(1, "look1"), "data": data_file(road_data(1, sidewalk="sw1")), "aux": b""}
        kinds = collections.Counter(s.kind for s in zones.sector_slots(files, X, Z, self.vocab))
        self.assertEqual(kinds["road_material"], 1)
        self.assertEqual(kinds["sidewalk"], 2)

    def test_all_parts_of_an_item_land_in_one_zone(self):
        chosen = {zones.item_zone(77, 4, 0.5) for _ in range(5)}
        self.assertEqual(len(chosen), 1)

    def test_survey_counts_by_zone_and_owner(self):
        counted = zones.survey(self.files, self.slots, 4, 0.0, self.vocab, {4, 5})
        self.assertEqual(counted["look", 4, tokens.encode("un11"), 0], 1)
        self.assertEqual(counted["edge", 4, tokens.encode("un_sh_15"), tokens.encode("un11")], 2)
        self.assertFalse(zones.survey(self.files, self.slots, 4, 0.0, self.vocab, {9}))

    def test_prefab_corner_is_owned_by_its_prefab(self):
        base = item(mapdata.PREFAB, 2, "un_1403g", mapdata.PREFAB_MODEL_AT)
        files = {"base": base, "data": data_file(prefab_data(2)), "aux": b""}
        slots = zones.sector_slots(files, X, Z, self.vocab)
        corners = [s for s in slots if s.kind == "corner" and zones.token_at(files["data"], s.offset)]
        self.assertEqual(corners[0].owner, tokens.encode("un_1403g"))

    def test_nature_on_aux_is_found_by_its_place(self):
        aux = item(mapdata.MODEL, 3, "hill1", mapdata.MODEL_TOKEN_AT) + item(mapdata.CURVE, 4, "hedge1", mapdata.CURVE_TOKEN_AT)
        files = {"base": b"", "data": None, "aux": aux}
        kinds = collections.Counter(s.kind for s in zones.sector_slots(files, X, Z, self.vocab))
        self.assertEqual((kinds["model"], kinds["curve"]), (1, 1))
        other = item(mapdata.MODEL, 3, "house1", mapdata.MODEL_TOKEN_AT)
        self.assertFalse(zones.sector_slots({"base": b"", "data": None, "aux": other}, X, Z, self.vocab))


class PrefabTests(unittest.TestCase):
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
        self.assertEqual(len(roads.zone_model_path("prefab_seasons", 3, 7)), len(roads.zone_model_path("prefab_seasons", 27, 1945)))

    def test_copy_lists_the_corners_of_its_zone(self):
        body = "\n\tcorner0[]: blke260\n\tcorner1[]: blke261\n\tcorner0: 1\n"
        out = roads.copied_prefab(body, 4, {}, {}, {"blke260": "cz4_1"})
        self.assertIn("corner0[]: cz4_1", out)
        self.assertIn("corner1[]: blke261", out)


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
            self.assertEqual(roads.snow_mode("eut2.dif.spec.weight.rfx", substance, self.ASPHALT), "road")

    def test_surface_without_a_substance_goes_by_its_texture(self):
        self.assertEqual(roads.snow_mode("eut2.dif.spec.weight.rfx", "", self.ASPHALT), "road")
        self.assertEqual(roads.snow_mode("eut2.dif.spec.rfx", "", "/material/terrain/grass_city_fr.tobj"), "ground")

    def test_sidewalks_and_yards_have_no_lanes(self):
        self.assertEqual(roads.snow_mode("eut2.dif.spec.rfx", "", "/material/tiled/sidewalk_it_01.tobj"), "tiled")
        self.assertEqual(roads.snow_mode("eut2.dif.spec.rfx", "", "/material/road/sidewalk.tobj"), "tiled")
        self.assertEqual(roads.snow_mode("eut2.dif.spec.rfx", "road", "/material/ground/artificial/asphalt_02.tobj"), "tiled")

    def test_markings_stay_and_wear_fades(self):
        decal = "eut2.dif.spec.decal.over.rfx"
        self.assertIsNone(roads.snow_mode(decal, "", "/road_template/un/default/lines_local_difopa.tobj"))
        self.assertIsNone(roads.snow_mode(decal, "", "/material/special/lines_eu_01.tobj"))
        self.assertEqual(roads.snow_mode(decal, "", "/material/decal/harddecal/road_cracks_01_difopa.tobj"), "fade")
        self.assertEqual(roads.snow_mode(decal, "road_dirt", "/material/road/road_template/ibe/overlay_3_ibe.tobj"),
                         "fade")

    def test_opacity_in_alpha_is_not_painted_over(self):
        self.assertEqual(roads.snow_mode("eut2.dif.spec.weight.a.rfx", "road_dirt", self.ASPHALT), "fade")

    def test_walls_rails_and_lights_stay(self):
        self.assertIsNone(roads.snow_mode("eut2.dif.spec.rfx", "", "/model/rail/rail.tobj"))
        self.assertIsNone(roads.snow_mode("eut2.dif.spec.rfx", "concrete", "/material/subset/wall.tobj"))
        self.assertIsNone(roads.snow_mode("eut2.lightmap.night.rfx", "", "/asset/vegetation/x/y.tobj"))
        self.assertIsNone(roads.snow_mode("eut2.none.rfx", "road", self.ASPHALT))

    def test_trees_keep_their_shape_and_hills_are_ground(self):
        self.assertEqual(roads.snow_mode("eut2.leaves.shadow.rfx", "", "/asset/vegetation/deciduous/a/b_difopa.tobj"), "tree")
        self.assertEqual(roads.snow_mode("eut2.dif.spec.a.rfx", "", "/model2/vegetation/far/treewall.tobj"), "tree")
        self.assertEqual(roads.snow_mode("eut2.dif.spec.tsnmapuv.rfx", "", "/model2/rock/hills/mountain5.tobj"), "ground")
        self.assertEqual(roads.snow_mode("eut2.dif.spec.decal.over.rfx", "", "/asset/rock/sandstone/a.tobj"), "tree")

    def test_a_detail_layer_is_baked_its_own_way(self):
        info = {"mode": "road", "slots": {"texture_base": "/a.tobj", "texture_over": "/b.tobj", "texture_mult": "/a.tobj"}}
        self.assertEqual(roads.bake_jobs([info]), [("/a.tobj", "mult"), ("/a.tobj", "road"), ("/b.tobj", "road")])


class RutTests(unittest.TestCase):
    def test_a_lane_has_two_tracks(self):
        tracks = roads.lane_tracks(-4.75, 4.75, 2)
        self.assertEqual(len(tracks), 4)
        self.assertAlmostEqual(tracks[1] - tracks[0], 2 * roads.WHEEL_OFFSET_M)
        self.assertAlmostEqual(tracks[0] + tracks[3], 0.0)

    def test_nonsense_lanes_give_no_tracks(self):
        self.assertEqual(roads.lane_tracks(-1.0, 1.0, 2), [])
        self.assertEqual(roads.lane_tracks(-20.0, 20.0, 1), [])
        self.assertEqual(roads.lane_tracks(-4.0, 4.0, 0), [])

    def test_close_positions_merge_and_rare_ones_go(self):
        merged = roads.merge_ruts({0.200: 10, 0.205: 10, 0.5: 20, 0.8: 0.5}, 0.07)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0], 0.2025, places=3)
        self.assertAlmostEqual(merged[1], 0.5, places=3)

    def test_tracks_are_found_on_the_texture_through_uv(self):
        positions = np.array([[-4.75, 0, 0], [4.75, 0, 0], [-4.75, 0, -15], [4.75, 0, -15]], np.float32)
        uvs = np.array([[0.0, 0], [0.6, 0], [0.0, 1], [0.6, 1]], np.float32)
        piece = pmx.Piece(0, positions, [uvs])
        info = {"mode": "road", "slots": {"texture_base": "/a.tobj"}}
        look = roads.Look("un11", "road_look", '\n\ttemplate_right: "/m.pmd"\n\tlanes_left[]: a\n\tlanes_right[]: a\n')
        found = roads.road_ruts([look], {"un11": 3}, lambda _p: ([["/m.mat"]], [piece]), lambda _n: info)
        self.assertEqual(len(found["/a.tobj"]["us"]), 4)
        self.assertAlmostEqual(found["/a.tobj"]["per_metre"], 0.6 / 9.5, places=2)
        self.assertAlmostEqual(found["/a.tobj"]["us"][0], (roads.lane_tracks(-4.75, 4.75, 2)[0] + 4.75) * 0.6 / 9.5, places=2)

    def test_road_on_a_second_look_of_a_model_counts_too(self):
        positions = np.array([[-4.75, 0, 0], [4.75, 0, 0]], np.float32)
        piece = pmx.Piece(0, positions, [np.array([[0.0, 0], [0.6, 0]], np.float32)])
        materials = {"/stone.mat": None, "/asphalt.mat": {"mode": "road", "slots": {"texture_base": "/a.tobj"}}}
        look = roads.Look("un11", "road_look", '\n\ttemplate_right: "/m.pmd"\n\tlanes_right[]: a\n\tlanes_left[]: a\n')
        found = roads.road_ruts([look], {}, lambda _p: ([["/stone.mat"], ["/asphalt.mat"]], [piece]), materials.get)
        self.assertIn("/a.tobj", found)


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
        info = {"mode": "road",
                "text": 'texture : "texture_base" {\n source : "/a.tobj"\n}\ntexture : "texture_mult" {\n source : "/a.tobj"\n}\n'
                        'texture : "texture_nmap" {\n source : "/n.tobj"\n}\n',
                "slots": {"texture_base": "/a.tobj", "texture_mult": "/a.tobj"}}
        text = roads.snowy_text(info, 2, {"texture_base": "r5", "texture_mult": "r6"})
        self.assertIn('"/material/seasons/road/r5_2.tobj"', text)
        self.assertIn('"/material/seasons/road/r6_2.tobj"', text)
        self.assertIn('"/n.tobj"', text)
        self.assertNotIn('"/a.tobj"', text)

    def test_a_light_dusting_leaves_trees_as_they_are(self):
        info = {"mode": "tree", "text": 'texture : "texture_base" {\n source : "/a.tobj"\n}\n', "slots": {"texture_base": "/a.tobj"}}
        self.assertEqual(roads.snowy_text(info, 1, {"texture_base": "r5"}), info["text"])
        self.assertIn("r5_3", roads.snowy_text(info, 3, {"texture_base": "r5"}))

    def test_date_pack_writes_a_material_only_where_it_is_used(self):
        records = [{"path": "/automat/67/672a03c6ba2f0402.mat", "text": "t", "mode": "road", "keys": {}, "zones": [3]},
                   {"path": "/material/road/asphalt.mat", "text": "t", "mode": "road", "keys": {}, "zones": [3], "legacy": 5}]
        self.assertEqual([t for t, _ in roads.date_files(records, 4, 1)], [])
        self.assertEqual([t for t, _ in roads.date_files(records, 3, 1)],
                         ["automat/r3/672a03c6ba2f0402.mat", "material/seasons/legacy/z3/5.mat"])


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


class WheelTrackTests(unittest.TestCase):
    RUTS = [0.25, 0.4]
    PER_METRE = 0.07

    def setUp(self):
        self.base = np.full((256, 256, 4), 90, np.uint8)
        self.noise = textures.road_noise(self.base, ruts=self.RUTS, per_metre=self.PER_METRE)
        self.field = textures.rut_field(256, 256, self.RUTS, self.PER_METRE)

    def in_track(self):
        return self.field > 0.8

    def test_coverage_stays_exact(self):
        for cover in roads.LEVEL_COVER.values():
            self.assertAlmostEqual(textures.snow_mask(self.noise, cover).mean(), cover, delta=0.03)

    def test_tracks_keep_clear_of_snow_longest(self):
        mask = textures.snow_mask(self.noise, 0.75)
        self.assertLess(mask[self.in_track()].mean(), 0.1)
        self.assertGreater(mask[~self.in_track()].mean(), 0.75)

    def test_tracks_show_under_full_snow(self):
        snow = textures.packed_snow(256, 256, (180, 185, 190))
        out = textures.bake_road(self.base, snow, self.noise, 1.0, self.field)
        self.assertLess(out[self.in_track()][:, 0].mean(), out[~self.in_track()][:, 0].mean() - 30)

    def test_track_is_seamless_along_the_road(self):
        self.assertLess(np.abs(self.field[0] - self.field[-1]).mean(), 0.05)

    def test_track_wraps_across_the_texture(self):
        field = textures.rut_field(64, 256, [0.999], self.PER_METRE)
        self.assertGreater(field[:, 0].mean(), 0.5)


class DetailTextureTests(unittest.TestCase):
    def test_detail_layer_goes_neutral_under_snow(self):
        mult = np.full((8, 8, 4), 60, np.uint8)
        self.assertEqual(textures.neutral_mult(mult, 1.0)[0, 0, 0], int(textures.MULT_NEUTRAL))
        self.assertEqual(textures.neutral_mult(mult, 0.0)[0, 0, 0], 60)
        self.assertEqual(textures.neutral_mult(mult, 1.0)[0, 0, 3], 60)


class TreeTests(unittest.TestCase):
    def chain(self):
        """A leafy half with holes and a solid strip of bark, like the game's tree atlases."""
        img = np.zeros((256, 256, 4), np.uint8)
        img[:, :, :3] = (40, 70, 30)
        img[::2, ::2, 3] = 255
        img[:, :64, 3] = 255
        return textures.mip_chain(img)

    def test_shape_stays_on_every_mip(self):
        chain = self.chain()
        for before, after in zip(chain, textures.bake_cut_out(chain, 0.6)):
            self.assertTrue((before[:, :, 3] == after[:, :, 3]).all())

    def test_leaves_take_snow_and_bark_hardly_any(self):
        chain = self.chain()
        top = textures.bake_cut_out(chain, 0.6)[0]
        self.assertGreater(top[::2, 128::2, 0].mean(), 110)
        self.assertLess(top[:, :48, 0].mean(), 80)

    def test_more_cover_is_more_snow(self):
        chain = self.chain()
        light = textures.bake_cut_out(chain, 0.25)[0][:, :, 0].mean()
        heavy = textures.bake_cut_out(chain, 0.65)[0][:, :, 0].mean()
        self.assertLess(light, heavy)

    def test_chain_reads_back_as_written(self):
        chain = textures.mip_chain(np.full((32, 32, 4), 128, np.uint8))
        data = textures.write_dds(chain[0], chain)
        self.assertEqual(len(textures.read_dds_chain(data, 32)), len(chain))
        self.assertEqual(textures.read_dds_chain(data, 8)[0].shape[0], 8)


if __name__ == "__main__":
    unittest.main()
