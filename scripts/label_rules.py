"""Label-derivation rules for the four target fields.
Everything here is a deterministic function of NHTSA's own columns (COMPDESC, CDESCR,
CRASH, FIRE, INJURED, DEATHS) -- no manual labeling, no LLM-generated labels.
Per blueprint.md Section 5 / 5a.
"""
import re

# --- safety_risk / severity -------------------------------------------------

def safety_risk(crash: bool, fire: bool, injured: int, deaths: int) -> str:
    return "yes" if (crash or fire or injured > 0 or deaths > 0) else "no"


def severity(crash: bool, fire: bool, injured: int, deaths: int) -> str:
    if injured > 0 or deaths > 0:
        return "high"
    if crash or fire:
        return "medium"
    return "low"


# --- defect_type -------------------------------------------------------------
# Two-tier matching, in this order:
#  1. Fire flag / explicit fire-narrative and the unintended-acceleration phrase are
#     checked first -- these are specific enough that they're safe to detect from free
#     text and matter regardless of which component NHTSA filed the complaint under.
#  2. Everything else is matched primarily against COMPDESC (NHTSA's own structured
#     component field), not the free-text narrative. Narrative words like "brake" or
#     "steering" show up constantly as incidental scene-setting ("I was braking when...")
#     even in complaints that aren't about brakes at all, so free-text keyword matching
#     produces a lot of false positives. COMPDESC is a curated NHTSA classification, so
#     it's a much higher-precision signal. Narrative keywords are only used as a fallback
#     for the handful of categories that have no reliable COMPDESC signature (e.g.
#     software/infotainment, stalling).
# Section 5a's 15-category taxonomy.

_FIRE_PATTERN = re.compile(r"\bFIRE\b|\bSMOKE\b|\bFLAME|\bBURN(ED|ING)?\b|\bEXPLOD")
_ACCEL_PATTERN = re.compile(
    r"UNINTENDED ACCELERAT|SUDDEN ACCELERAT|UNCOMMANDED ACCELERAT"
)
# VEHICLE SPEED CONTROL (cruise control) complaints go both directions -- unwanted
# speeding up AND unwanted slowing down/stalling. Only route to UNINTENDED
# ACCELERATION when there's no power-loss language; otherwise it's a stalling/power
# complaint that happens to be filed under the speed-control component.
_DECEL_STALL_PATTERN = re.compile(
    r"NO LONGER ACCELERATE|WOULD ?N.?T ACCELERATE|COULD ?N.?T ACCELERATE|"
    r"FAILED TO ACCELERATE|LOST (POWER|ACCELERATION)|LOSS OF POWER|\bSTALL"
)

_COMPDESC_RULES = [
    ("BRAKE FAILURE", ["BRAKE"]),
    ("STEERING LOSS", ["STEERING"]),
    ("TRANSMISSION FAILURE", ["TRANSMISSION", "POWER TRAIN"]),
    ("SUSPENSION FAILURE", ["SUSPENSION"]),
    ("TIRE/WHEEL FAILURE", ["TIRES", "WHEEL"]),
    ("FUEL SYSTEM LEAK", ["FUEL"]),
    ("ELECTRICAL FAULT", ["ELECTRICAL"]),
    ("AIRBAG NON-DEPLOYMENT", ["AIR BAG", "AIRBAG"]),
    ("SEAT BELT FAILURE", ["SEAT BELT"]),
    ("STRUCTURAL/CORROSION", ["STRUCTURE"]),
    ("SOFTWARE/INFOTAINMENT/ADAS", [
        "FORWARD COLLISION", "LANE DEPARTURE", "ELECTRONIC STABILITY",
        "BACK OVER", "ADAPTIVE CRUISE",
    ]),
    ("ENGINE/STALLING/POWER LOSS", ["ENGINE"]),
]

_NARRATIVE_FALLBACK_RULES = [
    ("SOFTWARE/INFOTAINMENT/ADAS", [r"\bSOFTWARE\b", r"INFOTAINMENT"]),
    ("ENGINE/STALLING/POWER LOSS", [r"\bSTALL", r"LOSS OF POWER", r"LOST POWER"]),
]
_NARRATIVE_FALLBACK_COMPILED = [
    (label, [re.compile(p) for p in patterns]) for label, patterns in _NARRATIVE_FALLBACK_RULES
]

DEFECT_TYPE_CATEGORIES = ["FIRE/SMOKE"] + [
    label for label, _ in _COMPDESC_RULES if label != "UNINTENDED ACCELERATION"
] + ["UNINTENDED ACCELERATION", "OTHER"]


def defect_type(compdesc_raw: str, narrative: str, fire_flag: bool) -> str:
    narrative_u = narrative.upper()
    compdesc_u = compdesc_raw.upper()

    if fire_flag or _FIRE_PATTERN.search(narrative_u):
        return "FIRE/SMOKE"
    if _ACCEL_PATTERN.search(narrative_u):
        return "UNINTENDED ACCELERATION"

    if "VEHICLE SPEED CONTROL" in compdesc_u and "BRAKE" not in compdesc_u and "STEERING" not in compdesc_u:
        if _DECEL_STALL_PATTERN.search(narrative_u):
            return "ENGINE/STALLING/POWER LOSS"
        return "UNINTENDED ACCELERATION"

    for label, keys in _COMPDESC_RULES:
        if any(k in compdesc_u for k in keys):
            return label

    for label, patterns in _NARRATIVE_FALLBACK_COMPILED:
        if any(p.search(narrative_u) for p in patterns):
            return label

    return "OTHER"
