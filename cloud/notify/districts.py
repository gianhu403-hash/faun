"""Forestry districts (лесничества) with geographic bounding boxes.

Each district defines the monitoring zone assigned to rangers
who register for it. When adding a new district, just add an entry
to the DISTRICTS dict.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class District:
    slug: str
    name_ru: str
    region_ru: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


DISTRICTS: dict[str, District] = {
    "varnavino": District(
        slug="varnavino",
        name_ru="Варнавинское лесничество",
        region_ru="Нижегородская область",
        lat_min=57.05,
        lat_max=57.55,
        lon_min=44.60,
        lon_max=45.40,
    ),
}
