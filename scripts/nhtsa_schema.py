# Column layout for NHTSA's FLAT_CMPL.txt, per static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt
# (51 tab-delimited columns, no header row in the source file).
COLUMNS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT",
    "CRASH", "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC",
    "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES",
    "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN",
    "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN",
    "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED", "DOT",
    "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN",
    "MANUF_DT", "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME",
    "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP",
    "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN",
    "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]

USECOLS = [
    "CMPLID", "ODINO", "MAKETXT", "MODELTXT", "YEARTXT",
    "CRASH", "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC",
    "CDESCR", "PROD_TYPE",
]

READ_CSV_KWARGS = dict(
    sep="\t",
    header=None,
    names=COLUMNS,
    usecols=USECOLS,
    dtype=str,
    na_filter=False,
    encoding="cp1252",
    encoding_errors="replace",
    engine="c",
    on_bad_lines="skip",
    quoting=3,  # csv.QUOTE_NONE -- free text may contain stray quote chars
)
