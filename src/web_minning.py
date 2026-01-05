# Needed libraries 

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.key_input import KeyInput
from selenium.webdriver.common.actions.pointer_input import PointerInput

# System requiremenmts 
import time
import random

# Data Management 
import pandas as pd


import re
from dataclasses import dataclass
from typing import List, Tuple, Dict


### ---------------------------------------------------------------------------------------------
### Text patterns -------------------------------------------------------------------------------

# -----------------------------
# Data structure
# -----------------------------
@dataclass
class AddressMatch:
    text: str
    span: Tuple[int, int]
    score: float
    rule: str  # which pattern matched (for debugging)



LOCATION_EMOJI = r"(?:📍|📌|🗺️|🧭|🏠|🏢|🏬|🏪|🏨|🏫|🏥)"


# Lexicon / subpatterns (Spanish; tune for your city/country)

STREET_TYPES = r"""
(?:av(?:enida)?\.?|calle|c\.|blvd\.?|boulevard|paseo|calz\.?|calzada|circuito|privada|cerrada|
carretera|camino|andador|prol\.?|prolongaci[oó]n|eje\s+(?:central|vial)|anillo|perif[eé]rico)
""".strip()

# "Name-like" chunk (street names, multi-token, supports accents and numbers)
STREET_NAME = r"(?:[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-']*(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-']+)*)"

# IMPORTANT: Escape '#' because we use re.VERBOSE
NUM = r"(?:(?:\#\s*)?(?:\d{1,6})(?:\s*(?:bis|ter|a|b))?|s\/n|sin\s+n[uú]mero|sin\s+numero)"

NEIGHBORHOOD = r"(?:col\.?|colonia|fracc\.?|fraccionamiento|barrio|unidad\s+habitacional|u\.h\.)"
POSTAL = r"(?:c\.?p\.?\s*)?\b\d{5}\b"

# Add your city-specific tokens here if you want better precision
CITY_TOKENS = r"(?:cdmx|ciudad\s+de\s+m[eé]xico|mexico\s+d\.?f\.?|guadalajara|monterrey|puebla|quer[eé]taro)"

# Secondary: cross street / between
CROSS_WORDS = r"(?:esquina\s+con|esq\.?\s+con|entre)"

# Optional unit/floor/building cues
UNIT = r"(?:depto\.?|departamento|int\.?|interior|piso|torre|edif\.?|edificio|local|mz\.?|manzana|lote|km)\s*\w+"

# Stop expansion at non-address areas (tune)
STOP_RE = re.compile(
    r"(?:\n|\b(?:tel|telefono|whats|precio|horario|hrs|instagram|ig|facebook|link|env[ií]os)\b|https?://)",
    re.IGNORECASE
)


# Patterns 

# 1) Street + name + number + optional tails
CANDIDATE_RE = re.compile(
    rf"""
    (?P<cand>
        \b{STREET_TYPES}\s+{STREET_NAME}\s+
        (?:{NUM})
        (?:
            (?:\s*,\s*|\s+)
            (?:
                {NEIGHBORHOOD}\s+{STREET_NAME}
                |
                {POSTAL}
                |
                {CITY_TOKENS}
                |
                {UNIT}
                |
                [A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-']{{2,}}
            )
        ){{0,8}}
    )
    """,
    re.IGNORECASE | re.VERBOSE
)

# 2) Emoji + Street + name + number + optional tails
EMOJI_CANDIDATE_RE = re.compile(
    rf"""
    (?P<cand>
        {LOCATION_EMOJI}\s*
        \b{STREET_TYPES}\s+{STREET_NAME}\s+
        (?:{NUM})
        (?:
            (?:\s*,\s*|\s+)
            (?:
                {NEIGHBORHOOD}\s+{STREET_NAME}
                |
                {POSTAL}
                |
                {CITY_TOKENS}
                |
                {UNIT}
                |
                [A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-']{{2,}}
            )
        ){{0,8}}
    )
    """,
    re.IGNORECASE | re.VERBOSE
)

# 3) Emoji + general location line (when no street types are used)

EMOJI_SOFT_RE = re.compile(
    rf"""
    (?P<cand>
        {LOCATION_EMOJI}\s*
        [^\n;]{{10,180}}
    )
    """,
    re.IGNORECASE | re.VERBOSE
)

# 4) Cross-street/Between pattern (when no house number)
CROSS_RE = re.compile(
    rf"""
    (?P<cand>
        \b{CROSS_WORDS}\b
        [^\n;]{{10,160}}
    )
    """,
    re.IGNORECASE | re.VERBOSE
)


# Helpers

def expand_with_emoji(text: str, start: int, max_back: int = 12) -> int:
    """If a location emoji appears immediately before the match, include it."""
    back = max(0, start - max_back)
    prefix = text[back:start]
    m = re.search(rf"{LOCATION_EMOJI}\s*$", prefix)
    return back + m.start() if m else start

def _score(s: str) -> float:
    s_low = s.lower()
    score = 0.0

    # emoji boost
    if re.search(LOCATION_EMOJI, s):
        score += 2.5

    # structural cues
    if re.search(rf"\b{STREET_TYPES}\b", s_low):
        score += 3
    if re.search(rf"\b(?:{NUM})\b", s_low):
        score += 3
    if re.search(rf"\b{NEIGHBORHOOD}\b", s_low):
        score += 2
    if re.search(POSTAL, s_low):
        score += 2
    if re.search(CITY_TOKENS, s_low):
        score += 1
    if re.search(rf"\b{CROSS_WORDS}\b", s_low):
        score += 1
    if re.search(rf"\b(?:depto|departamento|int|interior|piso|torre|edif|edificio|local|mz|manzana|lote|km)\b", s_low):
        score += 0.5

    # length sanity
    L = len(s.strip())
    if L < 18:
        score -= 2
    elif L > 240:
        score -= 1

    return score

def _expand_forward(text: str, start: int, end: int, extra_chars: int = 140) -> Tuple[int, int]:
    """Expand forward to include trailing comma-separated tokens but stop at STOP_RE or max length."""
    max_end = min(len(text), end + extra_chars)
    tail = text[end:max_end]
    stop = STOP_RE.search(tail)
    if stop:
        max_end = end + stop.start()
    return start, max_end

def _dedupe(matches: List[AddressMatch]) -> List[AddressMatch]:
    """Dedupe by overlapping spans: keep higher score if overlaps heavily."""
    if not matches:
        return []

    matches = sorted(matches, key=lambda x: (x.span[0], -(x.span[1]-x.span[0]), -x.score))
    kept: List[AddressMatch] = []

    def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union else 0.0

    for m in matches:
        replaced = False
        for i, k in enumerate(kept):
            if overlap(m.span, k.span) >= 0.55:
                if m.score > k.score:
                    kept[i] = m
                replaced = True
                break
        if not replaced:
            kept.append(m)

    return sorted(kept, key=lambda x: x.score, reverse=True)


# Main API: simultaneous matching across patterns

def extract_addresses(text: str, top_k: int = 3, min_score: float = 5.0) -> List[AddressMatch]:
    """
    Simultaneously runs multiple patterns and returns top-k scored matches
    with original substring spans.
    """
    if not text or not text.strip():
        return []

    patterns: List[Tuple[str, re.Pattern]] = [
        ("street_number", CANDIDATE_RE),
        ("emoji_street_number", EMOJI_CANDIDATE_RE),
        ("cross_streets", CROSS_RE),
        ("emoji_soft", EMOJI_SOFT_RE),
    ]

    found: List[AddressMatch] = []

    for rule_name, rx in patterns:
        for m in rx.finditer(text):
            start, end = m.start("cand"), m.end("cand")

            # include emoji if immediately before match
            start = expand_with_emoji(text, start)

            # expand forward to catch trailing tokens (esp. after a strong street match)
            if rule_name in ("street_number", "emoji_street_number"):
                start, end = _expand_forward(text, start, end, extra_chars=160)
            else:
                start, end = _expand_forward(text, start, end, extra_chars=80)

            cand = text[start:end].strip(" ,.;:-")
            score = _score(cand)

            # Down-weight the soft emoji rule unless it contains strong cues
            if rule_name == "emoji_soft" and score < 6:
                score -= 1.5

            found.append(AddressMatch(text=cand, span=(start, end), score=score, rule=rule_name))

    # Dedupe and rank
    ranked = _dedupe(found)
    ranked = [m for m in ranked if m.score >= min_score][:top_k]
    return ranked


# place_detector.py
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class PlaceMatch:
    text: str
    span: Tuple[int, int]      # (start, end) in original string
    kind: str                 # "POI" | "CITY" | "PLACE"
    rule: str                 # which rule matched (debug)


# -----------------------------
# Configurable lexicons
# -----------------------------
DEFAULT_CITY_TOKENS = [
    # Mexico common
    "CDMX", "Ciudad de México", "México DF", "Mexico DF", "DF",
    "Guadalajara", "GDL", "Monterrey", "MTY", "Puebla", "Querétaro", "Queretaro",
    "Tijuana", "Mérida", "Merida", "Cancún", "Cancun", "Toluca",
]

# Keywords that strongly indicate a location mention nearby
LOCATION_CUES = [
    r"direcci[oó]n", r"ubicaci[oó]n", r"est[aá]\s+en", r"queda\s+en", r"estamos\s+en",
    r"nos\s+vemos\s+en", r"visita", r"ven\s+a", r"en\b", r"por\b", r"sobre\b"
]

# Emoji anchors often used in captions
LOCATION_EMOJI = r"(?:📍|📌|🗺️|🧭|🏠|🏢|🏬|🏪|🏨|🏫|🏥)"


# -----------------------------
# Helper: build safe city regex
# -----------------------------
def _build_city_regex(city_tokens: List[str]) -> re.Pattern:
    # Sort longer first to avoid partial matches (Ciudad de México vs México)
    toks = sorted(city_tokens, key=len, reverse=True)
    # Escape user tokens safely
    alts = "|".join(re.escape(t) for t in toks)
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


# -----------------------------
# Core patterns
# -----------------------------
# 1) "POI | CITY" or "POI - CITY" or "POI · CITY" or "POI, CITY"
# Example: "El Taco | CDMX"
POI_CITY_RE = re.compile(
    r"""
    (?P<poi>                                   # POI: allow Title-ish phrases
        (?:[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ\.\-&'’]+
        (?:\s+[A-ZÁÉÍÓÚÜÑ0-9][\wÁÉÍÓÚÜÑáéíóúüñ\.\-&'’]+){0,8})
    )
    \s*(?:\||-|—|–|·|,)\s*
    (?P<city>[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\. ]{2,40})
    """,
    re.VERBOSE
)

# 2) Cue-based: "en X", "dirección: X", "está en X", "visita X"
# We capture a "location phrase" after the cue up to punctuation/linebreak.
CUE_LOCATION_RE = re.compile(
    rf"""
    (?P<cue>
        {LOCATION_EMOJI}\s*|
        (?:{"|".join(LOCATION_CUES)})
    )
    \s*[:\-]?\s*
    (?P<loc>
        [^\n\.;!\?]{{3,120}}                   # capture until hard punctuation
    )
    """,
    re.IGNORECASE | re.VERBOSE
)

# 3) Hashtag / mention city: "#cdmx", "en CDMX"
HASHTAG_CITY_RE = re.compile(r"#(?P<city>[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,20})", re.IGNORECASE)


# -----------------------------
# Main API
# -----------------------------
def detect_places(
    text: str,
    city_tokens: Optional[List[str]] = None,
    max_results: int = 10
) -> List[PlaceMatch]:
    """
    Detect place-like substrings using:
      - POI|CITY title pattern ("El Taco | CDMX")
      - cue-based segments ("dirección:", "está en", "visita", "📍")
      - city token hits (optional, improves precision)
    Returns spans in original text.
    """
    if not text or not text.strip():
        return []

    city_tokens = city_tokens or DEFAULT_CITY_TOKENS
    city_re = _build_city_regex(city_tokens)

    matches: List[PlaceMatch] = []

    # Rule A: POI | CITY
    for m in POI_CITY_RE.finditer(text):
        poi = m.group("poi").strip(" -|,·—–")
        city_raw = m.group("city").strip(" -|,·—–")

        # Validate city with city lexicon if possible (otherwise accept)
        kind = "POI"
        if city_re.search(city_raw):
            # Use the matched city token span instead of full city_raw if you prefer
            kind = "POI"

        matches.append(PlaceMatch(
            text=f"{poi} | {city_raw}",
            span=(m.start(), m.end()),
            kind=kind,
            rule="poi_city"
        ))

    # Rule B: cue-based location phrases
    for m in CUE_LOCATION_RE.finditer(text):
        loc = m.group("loc").strip(" ,|-")
        span = (m.start("loc"), m.end("loc"))

        # If the captured phrase contains a known city token, mark as CITY/PLACE
        if city_re.search(loc):
            kind = "CITY"
        else:
            kind = "PLACE"

        matches.append(PlaceMatch(text=loc, span=span, kind=kind, rule="cue_location"))

    # Rule C: hashtag city (useful in reels captions)
    for m in HASHTAG_CITY_RE.finditer(text):
        tag = m.group("city")
        # only keep if it matches a known city token (or common short codes)
        if city_re.search(tag) or tag.lower() in {"cdmx", "gdl", "mty"}:
            matches.append(PlaceMatch(
                text="#" + tag,
                span=(m.start(), m.end()),
                kind="CITY",
                rule="hashtag_city"
            ))

    # Post-process: dedupe heavily overlapping spans, prefer earlier & longer
    matches = _dedupe_overlaps(matches)

    return matches[:max_results]


def _dedupe_overlaps(matches: List[PlaceMatch]) -> List[PlaceMatch]:
    if not matches:
        return []

    # Prefer longer spans, then earlier
    matches_sorted = sorted(matches, key=lambda x: (-(x.span[1] - x.span[0]), x.span[0]))
    kept: List[PlaceMatch] = []

    def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    for m in matches_sorted:
        if any(overlaps(m.span, k.span) for k in kept):
            continue
        kept.append(m)

    # Return in reading order
    return sorted(kept, key=lambda x: x.span[0])




### ---------------------------------------------------------------------------------------------
### Web Interactions ----------------------------------------------------------------------------

def scroll_until_no_new_items(
    driver,
    item_selector="div[id^='grid-item-container-']",
    pause=1.5,
    max_rounds=30
):
    last_count = 0

    for i in range(max_rounds):
        items = driver.find_elements(By.CSS_SELECTOR, item_selector)
        current_count = len(items)

        if current_count == last_count:
            break  # no new content loaded

        last_count = current_count

        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )
        time.sleep(pause)

    return driver.find_elements(By.CSS_SELECTOR, item_selector)

def scroll_through_elements(driver, elements, pause=0.5):
    for el in elements:
        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'})", el
        )
        time.sleep(pause)


    
def city_hashtags(city):
    city = city.lower()

    base = [
        f"#dondeir{city}",
        f"#quehacer{city}",
        f"#lugares{city}",
        f"#places #{city}",
        f"#places{city}",
        f"#turismo{city}",
    ]

    discovery = [
        f"#lugaresbonitos #{city}",
        f"#lugaressecretos #{city}",
        f"#lugaresimperdibles #{city}",
        f"#lugaresrecomendados #{city}",
        f"#experiencias #{city}",
        f"#experienciasunicas #{city}",
        f"#lugaresbonitos{city}",
        f"#lugaressecretos{city}",
    ]

    time_intent = [
        f"#findesemana #{city}",
        f"#findesemana{city}",
        f"#planesdefin #{city}",
        f"#planperfecto  #{city}",
        f"#escapada{city}",
    ]

    food_coffee = [
        f"#brunch #{city}",
        f"#cafes #{city}",
        f"#postres #{city}",
        f"#desayunos #{city}",
        f"#restaurantes #{city}",
        f"#brunch{city}",
        f"#cafes{city}",
        f"#comida{city}",
        f"#restaurantes{city}",
    ]

    events = [
        f"#evento #{city}",
        f"#eventos #{city}",
        f"#{city}", f"#eventos{city}", f"#agenda{city}",
        f"#rave #{city}", f"#rave{city}",
        f"#concierto #{city}", f"#concierto{city}",
        f"#festival #{city}", f"#festival{city}",
        f"#fiesta #{city}", f"#fiesta{city}",
        f"#findesemana #{city}", f"#findesemana{city}",
        f"#quehacer #{city}", f"#quehacer{city}",
        f"#dondeir #{city}", f"#dondeir{city}",
        f"#eventosgratis #{city}", f"#eventosgratis{city}",
        f"#bazar #{city}",f"#bazar{city}" 

    ]


    all_tags = (
        base
        + discovery
        + time_intent
        + food_coffee
        + events
    )

    # Deduplicate while preserving order
    seen = set()
    tags = [t for t in all_tags if not (t in seen or seen.add(t))]

    return tags




### ---------------------------------------------------------------------------------------------
### GeoCoding  ----------------------------------------------------------------------------------

import time
import requests
from typing import Optional, Dict, Any

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"

class GeocodingError(RuntimeError):
    pass

def geocode_osm(
    query: str,
    *,
    country_codes: str = "mx",          # ISO 3166-1 alpha-2, comma-separated allowed
    accept_language: str = "es",
    limit: int = 1,
    email: Optional[str] = None,        # optional but recommended for contact
    user_agent: str = "applied-intelligence-lab-geocoder/1.0 (contact: you@example.com)",
    timeout: int = 20,
    min_delay_s: float = 1.0,           # be polite; also helps avoid throttling
    bounded: bool = False,              # set True if you also provide viewbox
    viewbox: Optional[str] = None       # "minlon,minlat,maxlon,maxlat"
) -> Dict[str, Any]:
    """
    Forward geocode with OSM Nominatim: returns dict with lat/lon and match metadata.

    Returns keys (typical):
      - lat, lon (floats)
      - display_name (string)
      - class, type, importance, place_id, osm_type, osm_id
      - address (if addressdetails=1)
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")


    headers = {
    "User-Agent": (
        "AppliedIntelligenceLab-Geocoder/1.0 "
        "(contact: eddn.professional@proton.me)"
    ),
    "Accept-Language": "es",
}


    params = {
        "q": query,
        "format": "jsonv2",          # stable JSON format
        "addressdetails": 1,
        "limit": max(1, min(limit, 10)),
        "countrycodes": country_codes,
    }

    if email:
        params["email"] = email

    # Optional bounding box constraints (useful when you *know* the city)
    if viewbox:
        params["viewbox"] = viewbox
        params["bounded"] = 1 if bounded else 0

    # Politeness delay (recommended by usage policy spirit; also reduces blocking risk)
    time.sleep(max(0.0, float(min_delay_s)))

    r = requests.get(NOMINATIM_SEARCH_URL, params=params, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise GeocodingError(f"Nominatim HTTP {r.status_code}: {r.text[:300]}")

    data = r.json()
    if not data:
        return {
            "query": query,
            "found": False,
            "lat": None,
            "lon": None,
            "raw": [],
        }

    best = data[0]
    return {
        "query": query,
        "found": True,
        "lat": float(best["lat"]),
        "lon": float(best["lon"]),
        "display_name": best.get("display_name"),
        "class": best.get("class"),
        "type": best.get("type"),
        "importance": best.get("importance"),
        "place_id": best.get("place_id"),
        "osm_type": best.get("osm_type"),
        "osm_id": best.get("osm_id"),
        "address": best.get("address", {}),
        "raw": best,
    }
