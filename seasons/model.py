"""Calendar, per-sector temperature field and the snow cover simulation."""
import datetime as dt
import math

ZONE_STEP = 1.0       # degrees of January temperature per zone
ZONE_COLDEST = -15.0
ZONE_COUNT = 28
SNOW_LEVELS = 4       # baked snow textures per material, level 0 is the original texture

# Rough climate normals (January, July mean in deg C), keyed by the game's city token.
# Positions come from the map itself, so the field follows the game's own geography.
STATIONS = {
    "london": (5, 18.5), "edinburgh": (4, 15), "calais": (4.5, 17.5),
    "paris": (5, 20), "lyon": (3.5, 22), "marseille": (7, 25), "bordeaux": (7, 21), "brest": (7, 17),
    "amsterdam": (3.5, 18), "brussel": (3.5, 18), "berlin": (0.5, 19.5), "hamburg": (1.5, 18),
    "frankfurt": (2, 20), "munchen": (-0.5, 19), "wien": (0.5, 21), "innsbruck": (-2, 18.5),
    "zurich": (0.5, 19), "geneve": (2, 20), "prague": (-1, 19), "warszawa": (-1.5, 19.5),
    "gdansk": (-1, 18), "krakow": (-2, 19.5), "bratislava": (0.5, 21), "budapest": (1, 22),
    "milano": (2.5, 24), "roma": (8, 25), "napoli": (9, 25), "palermo": (12, 26), "torino": (2, 23),
    "madrid": (6.5, 25), "barcelona": (9.5, 24.5), "sevilla": (11, 28), "lisboa": (11.5, 23),
    "bilbao": (9, 20), "valencia": (11, 25), "zaragoza": (6.5, 24.5), "porto": (10, 20.5),
    "athens": (10, 29), "thessaloniki": (6, 27), "sofia": (0, 21), "bucuresti": (-1, 23),
    "istanbul": (6, 24), "beograd": (1.5, 23), "zagreb": (0.5, 21), "ljubljana": (0, 21),
    "sarajevo": (-1, 20), "tirana": (7, 25), "kobenhavn": (1, 17.5), "aalborg": (0.5, 16.5),
    "oslo": (-4, 17), "bergen": (2, 15), "stavanger": (2, 15), "trondheim": (-2, 14), "tromso": (-3.5, 12),
    "narvik": (-4, 13), "bodo": (-1.5, 13), "alta": (-8, 13.5), "honningsvag": (-4, 9.5),
    "stockholm": (-1.5, 18.5), "goteborg": (-1, 17.5), "malmo": (0, 17.5), "sundsvall": (-6, 16),
    "umea": (-7.5, 15.5), "lulea": (-11, 16.5), "kiruna": (-12, 13), "helsinki": (-4, 18),
    "tampere": (-7, 17), "oulu": (-9.5, 16.5), "rovaniemi": (-12, 15), "ivalo": (-13, 14.5),
    "tallinn": (-4, 17), "riga": (-4, 18), "vilnius": (-5, 18), "kaliningrad": (-2, 18),
    "petersburg": (-6, 19), "pskov": (-6.5, 18),
}

# Snow cover follows the mean temperature linearly between these two values. Keeping this band
# wider than a zone step is what guarantees a soft snow line on the map.
_COVER_STARTS_AT = 1.5
_COVER_FULL_AT = -6.0
_COVER_RISE_DAYS = 7.0   # new snow settles quickly
_COVER_MELT_DAYS = 16.0  # old snow lingers into spring
_COLDEST_DAY = 20
_WARMEST_DAY = 201


def day_of_year(date):
    return date.timetuple().tm_yday


def _last_sunday(year, month):
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def summer_time(date):
    """1 while European daylight saving time is on, else 0."""
    return int(_last_sunday(date.year, 3) <= date < _last_sunday(date.year, 10))


def field(x, z, located, softness=3.0):
    """Inverse distance blend of the stations at sector (x, z), returning (january, july).

    The softness trades closeness to the stations for smoothness: at 3 neighbouring sectors
    still stay within one snow level, which `seasons verify` checks on the real map.
    """
    wj = wa = ws = 0.0
    for (sx, sz), (jan, jul) in located.items():
        w = 1.0 / (((x - sx) ** 2 + (z - sz) ** 2 + softness ** 2) ** 2)
        ws += w
        wj += w * jan
        wa += w * jul
    return wj / ws, wa / ws


def zone_position(jan):
    """Continuous zone coordinate of a January temperature, 0 being the coldest zone."""
    return max(0.0, min(ZONE_COUNT - 1.0, (jan - ZONE_COLDEST) / ZONE_STEP))


def zone_january(zone):
    return ZONE_COLDEST + zone * ZONE_STEP


def zone_july(zone):
    # Warmer winters come with warmer summers across the map; slope fitted to STATIONS.
    return 17.0 + 0.55 * zone_january(zone)


def temperature(jan, jul, day):
    """Mean temperature on a day of the year, following the lagged seasonal curve."""
    if _COLDEST_DAY <= day <= _WARMEST_DAY:
        t = (day - _COLDEST_DAY) / (_WARMEST_DAY - _COLDEST_DAY)
    else:
        t = 1 - ((day - _WARMEST_DAY) % 365) / (365 - (_WARMEST_DAY - _COLDEST_DAY))
    return jan + (jul - jan) * (1 - math.cos(math.pi * t)) / 2


def snow_year(jan, jul):
    """Snow cover (0..1) for days 1..365 after the yearly cycle has settled."""
    cover = 0.0
    covers = [0.0] * 366
    for _ in range(2):
        for offset in range(365):
            day = (offset + 212) % 365 + 1  # start in August, always snow free
            t = temperature(jan, jul, day)
            target = max(0.0, min(1.0, (_COVER_STARTS_AT - t) / (_COVER_STARTS_AT - _COVER_FULL_AT)))
            days = _COVER_RISE_DAYS if target > cover else _COVER_MELT_DAYS
            cover += (target - cover) / days
            covers[day] = cover
    return covers


def level_from_cover(cover):
    return int(round(cover * SNOW_LEVELS))


class ZoneTable:
    """Precomputed per-zone values so a date lookup costs nothing."""

    def __init__(self):
        self.covers = [snow_year(zone_january(z), zone_july(z)) for z in range(ZONE_COUNT)]

    def cover(self, zone, day):
        return self.covers[zone][min(day, 365)]

    def level(self, zone, day):
        return level_from_cover(self.cover(zone, day))

    def snows_ever(self, zone):
        return any(self.level(zone, d) > 0 for d in range(1, 366))

    def temperature(self, zone, day):
        return temperature(zone_january(zone), zone_july(zone), min(day, 365))
