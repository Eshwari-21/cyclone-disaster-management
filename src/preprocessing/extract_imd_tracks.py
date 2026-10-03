import re
import pdfplumber
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PDF_PATH = Path(
    "data/raw/imd/33_82f41b_33_4d5ff5_Best_Track_2024_ (1).pdf"
)

OUTPUT_PATH = Path(
    "data/processed/imd_best_track_2024_clean.csv"
)


# ============================================================
# IMD 2024 CYCLONE EVENTS
# ============================================================

EVENTS = {
    1: {
        "name": "REMAL",
        "pages": [2, 3],
    },
    2: {
        "name": "July Depression",
        "pages": [4],
    },
    3: {
        "name": "August Deep Depression",
        "pages": [5],
    },
    4: {
        "name": "ASNA",
        "pages": [6, 7],
    },
    5: {
        "name": "August Depression",
        "pages": [8],
    },
    6: {
        "name": "September Deep Depression",
        "pages": [9],
    },
    7: {
        "name": "September Depression",
        "pages": [10],
    },
    8: {
        "name": "September Deep Depression Bangladesh",
        "pages": [11, 12],
    },
    9: {
        "name": "Arabian Sea Depression",
        "pages": [13],
    },
    10: {
        "name": "October Bay of Bengal Depression",
        "pages": [14],
    },
    11: {
        "name": "DANA",
        "pages": [15, 16],
    },
    12: {
        "name": "Fengal",
        "pages": [17, 18],
    },
    13: {
        "name": "December Depression",
        "pages": [19],
    },
}


# Map PDF page number -> event ID and event name
PAGE_TO_EVENT = {}

for event_id, info in EVENTS.items():
    for page in info["pages"]:
        PAGE_TO_EVENT[page] = (
            event_id,
            info["name"]
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """
    Convert a cell to clean text.
    """
    if value is None:
        return ""

    return str(value).replace("\n", " ").strip()


def is_date(value):
    """
    Accept only exact IMD date format:

        DD.MM.YY

    Examples:
        24.05.24 -> True
        21.75°N   -> False
        12.05N    -> False
        79.9E     -> False
    """

    value = clean_text(value)

    return bool(
        re.fullmatch(
            r"\d{2}\.\d{2}\.\d{2}",
            value
        )
    )


def to_float(value):
    """
    Convert a table value to float.

    '-' and empty values become None.
    """

    if value is None:
        return None

    value = clean_text(value)

    if value == "" or value == "-":
        return None

    try:
        return float(value)

    except ValueError:
        return None


def make_timestamp(date_value, time_value):
    """
    Combine date + UTC time into pandas timestamp.
    """

    if not date_value or not time_value:
        return None

    time_value = clean_text(time_value)

    if not time_value.isdigit():
        return None

    if len(time_value) not in [3, 4]:
        return None

    timestamp = pd.to_datetime(
        f"{date_value} {time_value}",
        dayfirst=True,
        errors="coerce"
    )

    if pd.isna(timestamp):
        return None

    return timestamp


def build_record(
    event_id,
    event_name,
    timestamp,
    date_value,
    time_value,
    latitude,
    longitude,
    ci_no,
    ecp_hpa,
    msw_kt,
    delta_p_hpa,
    category,
):
    """
    Create one clean cyclone track record.
    """

    return {
        "event_id": event_id,
        "event_name": event_name,
        "timestamp_utc": timestamp,
        "date": date_value,
        "time_utc": time_value,
        "latitude": latitude,
        "longitude": longitude,
        "ci_no": ci_no,
        "ecp_hpa": ecp_hpa,
        "msw_kt": msw_kt,
        "delta_p_hpa": delta_p_hpa,
        "category": category,
    }


# ============================================================
# STANDARD IMD TABLE PARSER
# ============================================================

def parse_standard_row(
    row,
    current_date,
    event_id,
    event_name,
):
    """
    Parse normal 9-column IMD track tables.

    Expected structure:

    Date
    Time
    Latitude
    Longitude
    C.I.
    ECP
    MSW
    Delta P
    Category
    """

    if not row or len(row) < 9:
        return None, current_date

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_value = clean_text(row[0])

    # Only update date if this cell is a real DD.MM.YY date.
    #
    # This prevents narrative rows such as:
    # "latitude 21.75°N and longitude 89.20°E"
    # from corrupting current_date.
    if is_date(date_value):
        current_date = date_value

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time_value = clean_text(row[1])

    timestamp = make_timestamp(
        current_date,
        time_value
    )

    if timestamp is None:
        return None, current_date

    # --------------------------------------------------------
    # LATITUDE / LONGITUDE
    # --------------------------------------------------------

    latitude = to_float(row[2])
    longitude = to_float(row[3])

    if latitude is None or longitude is None:
        return None, current_date

    # --------------------------------------------------------
    # OTHER FIELDS
    # --------------------------------------------------------

    ci_no = to_float(row[4])
    ecp_hpa = to_float(row[5])
    msw_kt = to_float(row[6])
    delta_p_hpa = to_float(row[7])

    category = clean_text(row[8])

    if not category:
        return None, current_date

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    record = build_record(
        event_id=event_id,
        event_name=event_name,
        timestamp=timestamp,
        date_value=current_date,
        time_value=time_value,
        latitude=latitude,
        longitude=longitude,
        ci_no=ci_no,
        ecp_hpa=ecp_hpa,
        msw_kt=msw_kt,
        delta_p_hpa=delta_p_hpa,
        category=category,
    )

    return record, current_date


# ============================================================
# PAGE 9 PARSER
# ============================================================

def parse_page_9_row(
    row,
    current_date,
    event_id,
    event_name,
):
    """
    Page 9 has a slightly different column arrangement.

    Some rows use:

        ... delta_p, blank, MSW, blank, Category

    while others use:

        ... delta_p, MSW, blank, Category
    """

    if not row or len(row) < 11:
        return None, current_date

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_value = clean_text(row[0])

    if is_date(date_value):
        current_date = date_value

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time_value = clean_text(row[1])

    timestamp = make_timestamp(
        current_date,
        time_value
    )

    if timestamp is None:
        return None, current_date

    # --------------------------------------------------------
    # LAT / LONG
    # --------------------------------------------------------

    latitude = to_float(row[3])
    longitude = to_float(row[4])

    if latitude is None or longitude is None:
        return None, current_date

    # --------------------------------------------------------
    # OTHER FIELDS
    # --------------------------------------------------------

    ci_no = to_float(row[5])
    ecp_hpa = to_float(row[6])
    delta_p_hpa = to_float(row[7])

    # Page 9 has two arrangements.
    msw_early = to_float(row[9])

    if msw_early is not None:

        # Earlier arrangement
        msw_kt = msw_early
        category = clean_text(row[11])

    else:

        # Later arrangement
        msw_kt = to_float(row[8])
        category = clean_text(row[10])

    if not category:
        return None, current_date

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    record = build_record(
        event_id=event_id,
        event_name=event_name,
        timestamp=timestamp,
        date_value=current_date,
        time_value=time_value,
        latitude=latitude,
        longitude=longitude,
        ci_no=ci_no,
        ecp_hpa=ecp_hpa,
        msw_kt=msw_kt,
        delta_p_hpa=delta_p_hpa,
        category=category,
    )

    return record, current_date


# ============================================================
# PAGE 19 PARSER
# ============================================================

def parse_page_19_row(
    row,
    current_date,
    event_id,
    event_name,
):
    """
    Page 19 has a special 27-column table layout.

    Relevant positions:

        Time       -> index 4
        Latitude   -> index 7
        Longitude  -> index 10
        C.I.       -> index 13
        ECP        -> index 16
        Delta P    -> index 19
        MSW        -> index 22
        Category   -> index 25
    """

    if not row or len(row) < 26:
        return None, current_date

    # --------------------------------------------------------
    # DATE ROW
    # --------------------------------------------------------

    # Some page-19 rows contain the date as a separate
    # table row. Detect only exact DD.MM.YY strings.
    for value in row:

        text = clean_text(value)

        if is_date(text):

            current_date = text

            # This row contains only the date,
            # not a track observation.
            return None, current_date

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time_value = clean_text(row[4])

    timestamp = make_timestamp(
        current_date,
        time_value
    )

    if timestamp is None:
        return None, current_date

    # --------------------------------------------------------
    # LAT / LONG
    # --------------------------------------------------------

    latitude = to_float(row[7])
    longitude = to_float(row[10])

    if latitude is None or longitude is None:
        return None, current_date

    # --------------------------------------------------------
    # OTHER FIELDS
    # --------------------------------------------------------

    ci_no = to_float(row[13])
    ecp_hpa = to_float(row[16])
    delta_p_hpa = to_float(row[19])
    msw_kt = to_float(row[22])

    category = clean_text(row[25])

    if not category:
        return None, current_date

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    record = build_record(
        event_id=event_id,
        event_name=event_name,
        timestamp=timestamp,
        date_value=current_date,
        time_value=time_value,
        latitude=latitude,
        longitude=longitude,
        ci_no=ci_no,
        ecp_hpa=ecp_hpa,
        msw_kt=msw_kt,
        delta_p_hpa=delta_p_hpa,
        category=category,
    )

    return record, current_date


# ============================================================
# MAIN EXTRACTION
# ============================================================

rows = []


with pdfplumber.open(PDF_PATH) as pdf:

    # IMPORTANT:
    # current_date is defined OUTSIDE the page loop.
    #
    # This allows continuation pages to inherit the date
    # from the previous page.
    current_date = None

    # Used to detect when we move from one cyclone event
    # to another.
    previous_event_id = None

    for page_number, page in enumerate(
        pdf.pages,
        start=1
    ):

        # Skip pages that do not contain one of our
        # cyclone track tables.
        if page_number not in PAGE_TO_EVENT:
            continue

        event_id, event_name = PAGE_TO_EVENT[
            page_number
        ]

        # ----------------------------------------------------
        # EVENT CHANGE
        # ----------------------------------------------------

        # Reset date only when starting a NEW event.
        #
        # Do NOT reset it between continuation pages.
        if event_id != previous_event_id:

            current_date = None
            previous_event_id = event_id

        # ----------------------------------------------------
        # SPECIAL PAGE 19
        # ----------------------------------------------------

        # Page 19 begins with a time row before its date
        # label, so initialize it from the event table.
        if page_number == 19:

            current_date = "20.12.24"

        # ----------------------------------------------------
        # EXTRACT TABLES
        # ----------------------------------------------------

        tables = page.extract_tables()

        if not tables:
            continue

        # ----------------------------------------------------
        # PROCESS TABLES
        # ----------------------------------------------------

        for table in tables:

            if not table:
                continue

            for row in table:

                record = None

                # Page 9 has special column structure.
                if page_number == 9:

                    record, current_date = parse_page_9_row(
                        row,
                        current_date,
                        event_id,
                        event_name,
                    )

                # Page 19 has special 27-column structure.
                elif page_number == 19:

                    record, current_date = parse_page_19_row(
                        row,
                        current_date,
                        event_id,
                        event_name,
                    )

                # All other track pages use standard structure.
                else:

                    record, current_date = parse_standard_row(
                        row,
                        current_date,
                        event_id,
                        event_name,
                    )

                # Add valid observation.
                if record is not None:

                    rows.append(record)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates(
    subset=[
        "event_id",
        "timestamp_utc",
    ]
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    by=[
        "event_id",
        "timestamp_utc",
    ]
).reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

print(
    "Clean rows:",
    len(df)
)


# ------------------------------------------------------------
# EVENTS FOUND
# ------------------------------------------------------------

print("\nEvents found:")

print(
    df[
        [
            "event_id",
            "event_name",
        ]
    ]
    .drop_duplicates()
    .sort_values("event_id")
    .to_string(index=False)
)


# ------------------------------------------------------------
# CHECK EVENT IDs
# ------------------------------------------------------------

expected_event_ids = set(
    range(1, 14)
)

found_event_ids = set(
    df["event_id"]
    .astype(int)
    .unique()
)

missing_event_ids = (
    expected_event_ids
    - found_event_ids
)

print(
    "\nMissing event IDs:",
    len(missing_event_ids)
)

if missing_event_ids:

    print(
        "Missing:",
        sorted(missing_event_ids)
    )


# ------------------------------------------------------------
# ROWS PER EVENT
# ------------------------------------------------------------

print("\nRows per event:")

print(
    df.groupby(
        [
            "event_id",
            "event_name",
        ]
    )
    .size()
    .to_string()
)


# ------------------------------------------------------------
# DUPLICATE CHECK
# ------------------------------------------------------------

print(
    "\nDuplicate event + timestamp:"
)

duplicates = df.duplicated(
    subset=[
        "event_id",
        "timestamp_utc",
    ]
).sum()

print(duplicates)


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values:")

print(
    df.isna()
    .sum()
    .to_string()
)


# ============================================================
# SAVE CSV
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nSaved to:",
    OUTPUT_PATH
)