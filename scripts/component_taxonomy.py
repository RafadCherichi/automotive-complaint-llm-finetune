"""component taxonomy: derived from real frequency counts over the full NHTSA flat file
(2,159,966 vehicle rows -- see scripts/tally_components.py output). Top ~18 real
categories by volume, near-duplicate raw strings collapsed together (e.g. ENGINE +
ENGINE AND ENGINE COOLING -> ENGINE), everything else buckets to OTHER.
Per blueprint.md Section 5 / 5a.
"""

_RAW_TO_BUCKET = {
    "ELECTRICAL SYSTEM": "ELECTRICAL SYSTEM",
    "POWER TRAIN": "POWER TRAIN",
    "ENGINE": "ENGINE",
    "ENGINE AND ENGINE COOLING": "ENGINE",
    "AIR BAGS": "AIR BAGS",
    "STEERING": "STEERING",
    "SERVICE BRAKES, HYDRAULIC": "BRAKES",
    "SERVICE BRAKES": "BRAKES",
    "SERVICE BRAKES, AIR": "BRAKES",
    "PARKING BRAKE": "BRAKES",
    "STRUCTURE": "STRUCTURE",
    "SUSPENSION": "SUSPENSION",
    "VEHICLE SPEED CONTROL": "VEHICLE SPEED CONTROL",
    "FUEL/PROPULSION SYSTEM": "FUEL SYSTEM",
    "FUEL SYSTEM, GASOLINE": "FUEL SYSTEM",
    "FUEL SYSTEM, OTHER": "FUEL SYSTEM",
    "EXTERIOR LIGHTING": "EXTERIOR LIGHTING",
    "VISIBILITY": "VISIBILITY",
    "VISIBILITY/WIPER": "VISIBILITY",
    "TIRES": "TIRES/WHEELS",
    "WHEELS": "TIRES/WHEELS",
    "SEAT BELTS": "SEAT BELTS",
    "SEATS": "SEATS",
    "FORWARD COLLISION AVOIDANCE": "ADAS/DRIVER ASSIST",
    "ELECTRONIC STABILITY CONTROL (ESC)": "ADAS/DRIVER ASSIST",
    "LANE DEPARTURE": "ADAS/DRIVER ASSIST",
    "BACK OVER PREVENTION": "ADAS/DRIVER ASSIST",
    "LATCHES/LOCKS/LINKAGES": "LATCHES/LOCKS/LINKAGES",
    "EQUIPMENT": "EQUIPMENT",
    "UNKNOWN OR OTHER": "OTHER",
}

COMPONENT_CATEGORIES = sorted(set(_RAW_TO_BUCKET.values())) + ["OTHER"]


def bucket_component(raw_top_level: str) -> str:
    return _RAW_TO_BUCKET.get(raw_top_level.strip().upper(), "OTHER")
