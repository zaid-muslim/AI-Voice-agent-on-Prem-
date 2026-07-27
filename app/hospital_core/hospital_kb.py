"""
Dummy knowledge base for Riverside General (fictional test hospital).

This is DATA ONLY - no retrieval mechanism lives here, that's rag.py.

SCOPE SPLIT (important, keep this boundary as the project grows):
  - THIS FILE: static, informational content - hours, departments, doctor
    bios, policies, insurance, FAQs. Answers "what/when/who/how" questions.
  - booking.py: dynamic, transactional data - actual open appointment slots,
    real-time availability, bookings. Answers "is there an opening" and
    "book me one" - RAG should never try to answer those from this file.

CONSISTENCY WITH booking.py: the doctors and departments named here are the
SAME ones booking.py's SEED_SLOTS uses. If you add or rename a
department/doctor in one file, update the other.

FIXED IN THIS REVISION: the previous version had a real bug - the
general-medicine doctor's entry TITLE said "Dr. Ali" while the entry TEXT
said "Dr. Nguyen" (two different names for what was meant to be the same
person). Doctor names are also renamed throughout, kept identical to
booking.py's roster.

LIVEKIT COMPAT NOTE (this revision): added get_doctor_roster() at the
bottom. compat.py's get_doctor_roster() looks for a callable of that exact
name (or a DOCTORS/doctors attribute) on this module - without it, compat
silently falls back to its own built-in roster instead of this file's data.
The doctor "doctors" category entries below are the source of truth; this
function just reshapes them into {"name", "department", "bio"} dicts.
"""

_HARDCODED_KB = [
    {
        "id": "hours_general",
        "category": "hours",
        "title": "General hospital hours",
        "text": (
            "Riverside General's emergency department is open 24 hours a "
            "day, 7 days a week. Outpatient clinics and departments are "
            "open Monday through Saturday, 8 AM to 6 PM, and closed on "
            "Sundays except for emergencies."
        ),
    },
    {
        "id": "hours_pharmacy",
        "category": "hours",
        "title": "Pharmacy hours",
        "text": (
            "The Riverside General outpatient pharmacy is open Monday "
            "through Friday, 9 AM to 7 PM, and Saturday 9 AM to 2 PM. It is "
            "closed on Sundays. The inpatient pharmacy serving admitted "
            "patients operates 24 hours a day."
        ),
    },
    {
        "id": "dept_cardiology",
        "category": "departments",
        "title": "Cardiology department",
        "text": (
            "The Cardiology department at Riverside General handles heart "
            "and cardiovascular care, including checkups, ECGs, and "
            "follow-up care. Doctors: Dr. Imran Malik and Dr. Ayesha "
            "Siddiqui. Located on the 3rd floor of the main building."
        ),
    },
    {
        "id": "dept_general_medicine",
        "category": "departments",
        "title": "General medicine department",
        "text": (
            "The General Medicine department handles routine checkups, "
            "general illness, referrals to specialists, and ongoing care "
            "for adult patients. Doctor: Dr. Bilal Ahmed. Located on the "
            "1st floor of the main building."
        ),
    },
    {
        "id": "dept_pediatrics",
        "category": "departments",
        "title": "Pediatrics department",
        "text": (
            "The Pediatrics department handles care for infants, children, "
            "and teenagers, including checkups and vaccinations. Doctor: "
            "Dr. Sana Farooqi. Located on the 2nd floor of the main "
            "building."
        ),
    },
    # --- doctor bios -----------------------------------------------------
    {
        "id": "doctor_imran_malik",
        "category": "doctors",
        "title": "Dr. Imran Malik - Cardiology",
        "text": (
            "Dr. Imran Malik is a cardiologist at Riverside General "
            "specializing in general adult cardiology and preventive heart "
            "care. Fluent in English and Urdu. Sees patients Monday "
            "through Friday."
        ),
    },
    {
        "id": "doctor_ayesha_siddiqui",
        "category": "doctors",
        "title": "Dr. Ayesha Siddiqui - Cardiology",
        "text": (
            "Dr. Ayesha Siddiqui is a cardiologist at Riverside General "
            "with a focus on cardiac rehabilitation and follow-up care "
            "after cardiac events. Fluent in English and Punjabi. Sees "
            "patients Tuesday through Saturday."
        ),
    },
    {
        "id": "doctor_bilal_ahmed",
        "category": "doctors",
        "title": "Dr. Bilal Ahmed - General medicine",
        "text": (
            "Dr. Bilal Ahmed is a general medicine physician at Riverside "
            "General, seeing adult patients for routine checkups, illness, "
            "and referrals. Fluent in English and Urdu."
        ),
    },
    {
        "id": "doctor_sana_farooqi",
        "category": "doctors",
        "title": "Dr. Sana Farooqi - Pediatrics",
        "text": (
            "Dr. Sana Farooqi is a pediatrician at Riverside General, "
            "seeing patients from infancy through age 17, including "
            "vaccination schedules and routine wellness checks. Fluent in "
            "English and Sindhi."
        ),
    },
    # --- insurance / billing ----------------------------------------------
    {
        "id": "insurance_accepted",
        "category": "insurance",
        "title": "Accepted insurance",
        "text": (
            "Riverside General accepts most major insurance plans, "
            "including Blue Cross Blue Shield, Aetna, Cigna, and UnitedHealth. "
            "Patients without insurance can ask about the self-pay rate and "
            "financial assistance program when they arrive."
        ),
    },
    {
        "id": "billing_process",
        "category": "billing",
        "title": "Billing and payment",
        "text": (
            "Billing statements are mailed or emailed within two weeks of a "
            "visit. Payments can be made online, by phone, or in person at "
            "the billing office on the 1st floor. Payment plans are "
            "available on request."
        ),
    },
    # --- prescriptions -----------------------------------------------------
    {
        "id": "prescription_refill",
        "category": "prescriptions",
        "title": "Prescription refill process",
        "text": (
            "To request a prescription refill, patients can call the "
            "pharmacy directly or ask their doctor's office to send a refill "
            "request. Refills typically take 1 to 2 business days to "
            "process."
        ),
    },
    # --- visiting policy ---------------------------------------------------
    {
        "id": "visiting_policy",
        "category": "policy",
        "title": "Visitor policy",
        "text": (
            "Visiting hours for admitted patients are 10 AM to 8 PM daily. "
            "Each patient may have up to two visitors at a time. Children "
            "under 12 must be accompanied by an adult at all times."
        ),
    },
    # --- lab results ---------------------------------------------------
    {
        "id": "lab_results",
        "category": "lab",
        "title": "Getting lab results",
        "text": (
            "Lab results are typically available within 3 to 5 business "
            "days and can be accessed through the patient portal or by "
            "calling the department that ordered the test."
        ),
    },
    # --- parking / location -----------------------------------------------
    {
        "id": "parking",
        "category": "location",
        "title": "Parking and directions",
        "text": (
            "Riverside General has a visitor parking garage attached to the "
            "main building, with the first hour free and a flat daily rate "
            "after that. Valet parking is available at the main entrance "
            "on weekdays."
        ),
    },
]


# ---------------------------------------------------------------------------
# EDITABLE DATA LOADING
# ---------------------------------------------------------------------------
# HOSPITAL_KB is now loaded from hospital_data.json (an editable file the
# admin UI writes to) if that file exists, falling back to the hardcoded
# _HARDCODED_KB above if it's missing or unreadable. This is what makes the
# hospital data admin-editable at runtime WITHOUT touching Python code -
# the admin UI writes the JSON, and reload_kb() re-reads it.
#
# WHY A MODULE-LEVEL list that gets reassigned, not a function: tons of
# existing code (compat.py, rag.py) imports HOSPITAL_KB by name. Keeping it
# a module-level list means none of that has to change - reload_kb() mutates
# this same list object in place so even code holding a reference sees
# updates.

import json as _json
from pathlib import Path as _Path

_DATA_FILE = _Path(__file__).parent / "hospital_data.json"

HOSPITAL_KB = []  # populated by _load_kb() immediately below


def _load_kb() -> list:
    """Read the editable JSON store; fall back to hardcoded data if it's
    missing/corrupt so the system NEVER starts with no hospital data."""
    if _DATA_FILE.exists():
        try:
            with open(_DATA_FILE, "r", encoding="utf-8") as f:
                data = _json.load(f)
            entries = data.get("entries", [])
            if entries:
                return entries
        except (OSError, ValueError):
            pass  # corrupt/unreadable -> fall through to hardcoded
    return list(_HARDCODED_KB)


def reload_kb() -> list:
    """Re-read hospital_data.json and update HOSPITAL_KB IN PLACE (so any
    code holding a reference to the list sees the new data). The admin UI /
    rag.py call this after a save to pick up edits without a restart.
    Returns the refreshed list."""
    fresh = _load_kb()
    HOSPITAL_KB.clear()
    HOSPITAL_KB.extend(fresh)
    return HOSPITAL_KB


# populate at import time
reload_kb()


_REQUIRED_FIELDS = {"id", "category", "title", "text"}
_MIN_TEXT_LEN = 20
_MAX_TEXT_LEN = 400


def save_kb(entries: list) -> list:
    """Validate and write a new set of entries to hospital_data.json, then
    reload HOSPITAL_KB in place. Raises ValueError with a human-readable
    message if the data is invalid, so the admin UI can show the admin what
    to fix instead of silently writing broken data. Returns the problems
    list (empty = clean) after a successful write.

    NOTE: this validates and SAVES, but does NOT itself re-embed RAG - the
    caller (admin UI) is responsible for calling rag.reindex() afterward so
    semantic search reflects the change. Kept separate so this module stays
    free of the heavy sentence-transformers dependency."""
    problems = _validate_entries(entries)
    # Block only on STRUCTURAL problems (missing fields, duplicate ids) -
    # length/consistency warnings are surfaced but don't block a save, since
    # an admin mid-edit may legitimately have a short bio momentarily.
    blocking = [p for p in problems if "missing fields" in p or "duplicate id" in p]
    if blocking:
        raise ValueError(
            "Cannot save - fix these first:\n  - " + "\n  - ".join(blocking)
        )

    tmp = _DATA_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump({"entries": entries}, f, indent=2)
    tmp.replace(_DATA_FILE)  # atomic swap - never leaves a half-written file
    reload_kb()
    return problems


def _validate_entries(entries: list) -> list:
    """Validate ANY candidate list of entries (used by save_kb before
    writing, and by _validate_kb for the current data). Returns a list of
    human-readable problem strings; empty means clean."""
    problems = []
    seen_ids = set()

    for entry in entries:
        missing = _REQUIRED_FIELDS - entry.keys()
        if missing:
            problems.append(f"entry missing fields {missing}: {entry}")
            continue

        if entry["id"] in seen_ids:
            problems.append(f"duplicate id: {entry['id']}")
        seen_ids.add(entry["id"])

        text_len = len(entry["text"])
        if text_len < _MIN_TEXT_LEN:
            problems.append(f"{entry['id']}: text too short ({text_len} chars)")
        if text_len > _MAX_TEXT_LEN:
            problems.append(f"{entry['id']}: text too long ({text_len} chars)")

    # cross-check that doctor names mentioned in department entries actually
    # match a real doctor entry's title - this is exactly the class of bug
    # (Dr. Ali vs Dr. Nguyen) that slipped through before.
    dept_texts = " ".join(
        e["text"] for e in entries if e.get("category") == "departments"
    )
    for e in entries:
        if e.get("category") != "doctors":
            continue
        name = e["title"].split(" - ")[0].replace("Dr. ", "").strip()
        if name not in dept_texts:
            problems.append(
                f"{e['id']}: name '{name}' from title not found in any "
                f"department text - check for a title/text mismatch"
            )

    return problems


def _validate_kb():
    """Validate the currently-loaded HOSPITAL_KB (thin wrapper kept for the
    __main__ self-test below and any existing callers)."""
    return _validate_entries(HOSPITAL_KB)


def get_doctor_roster() -> list:
    """Reshape the 'doctors' category entries into the {"name",
    "department", "bio"} dicts compat.py / the frontend directory board
    expect. Derived from HOSPITAL_KB, not a separate list, so there is only
    ONE place to edit a doctor's name/department (this file's entries) -
    booking.py's SEED_SLOTS still has to be kept in sync manually, per this
    file's own consistency note above.
    """
    roster = []
    for entry in HOSPITAL_KB:
        if entry["category"] != "doctors":
            continue
        name, _, dept = entry["title"].partition(" - ")
        roster.append(
            {
                "name": name.strip(),
                "department": dept.strip().title(),  # normalizes casing, e.g.
                # "General medicine" -> "General Medicine", for display only -
                # does not touch the underlying text/data above.
                "bio": entry["text"],
            }
        )
    return roster


if __name__ == "__main__":
    problems = _validate_kb()

    categories = {}
    for entry in HOSPITAL_KB:
        categories.setdefault(entry["category"], []).append(entry["id"])

    print(f"{len(HOSPITAL_KB)} entries across {len(categories)} categories:")
    for cat, ids in sorted(categories.items()):
        print(f"  {cat}: {len(ids)} entries")

    roster = get_doctor_roster()
    print(f"\nget_doctor_roster() -> {len(roster)} doctors:")
    for doc in roster:
        print(f"  {doc['name']} ({doc['department']})")

    if problems:
        print(f"\n[FAIL] {len(problems)} problem(s) found:")
        for p in problems:
            print(f"  - {p}")
        print("\nDO NOT build retrieval on top of this until these are fixed.")
        exit(1)
    else:
        print(
            "\n[PASS] all entries well-formed, no duplicate ids, no length issues, "
            "no doctor name/title mismatches"
        )
