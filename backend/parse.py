import re

def parse_ocr_text(ocr_text):
    """
    Parses the given OCR text to extract three fields:
      1. State (as a two-letter abbreviation)
      2. Procedure/CPT code (5-digit code)
      3. Insurance (one of a predefined set)
    
    If a field is not found, returns an empty string for that field.
    
    Returns:
        (state, cpt_code, insurance)
    """
    # --- 1. Define known states (abbreviations) ---
    STATE_ABBREVIATIONS = {
        "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
        "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
        "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
        "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA",
        "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT",
        "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
        "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
        "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
        "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT",
        "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY"
    }
    valid_states = set(STATE_ABBREVIATIONS.values())
    state_full_to_abbr = {name.upper(): abbr for name, abbr in STATE_ABBREVIATIONS.items()}

    # --- 2. Define allowed insurers (if you only want to pick from a known set) ---
    ALLOWED_INSURERS = {
        "CIGNA": "CIGNA",
        "UNITED": "UNITED_HEALTHCARE",
        "BLUE CROSS": "BCBS_SOUTH_CAROLINA",
        "AETNA": "AETNA"
    }

    found_state = ""
    found_cpt = ""
    found_insurance = ""

    # Split text into lines for easier scanning
    lines = ocr_text.splitlines()

    # --- 3. Extract State ---
    for line in lines:
        words = line.split()
        for word in words:
            token = word.strip(",.").upper()
            # If it's already a known two-letter abbreviation
            if token in valid_states:
                found_state = token
                break
            # If it's a full state name
            elif token in state_full_to_abbr:
                found_state = state_full_to_abbr[token]
                break
        if found_state:
            break

    # --- 4. Extract CPT Code ---
    # Priority search: lines that mention "CPT"
    for line in lines:
        if "CPT" in line.upper():
            match = re.search(r"\b(\d{5})\b", line)
            if match:
                found_cpt = match.group(1)
                break
    # Fallback: if not found, search entire text for any 5-digit number
    if not found_cpt:
        cpt_match = re.search(r"\b(\d{5})\b", ocr_text)
        if cpt_match:
            found_cpt = cpt_match.group(1)

    # --- 5. Extract Insurance (from a predefined set) ---
    # Scan each line, see if it contains a known insurer keyword
    for line in lines:
        line_upper = line.upper()
        for keyword, insurer_name in ALLOWED_INSURERS.items():
            if keyword in line_upper:
                found_insurance = insurer_name
                break
        if found_insurance:
            break

    return (found_state, found_cpt, found_insurance)