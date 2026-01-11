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
from typing import List, Tuple, Dict, Optional


### ---------------------------------------------------------------------------------------------
### Text patterns -------------------------------------------------------------------------------


import logging
import re
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


# ---- Public API -------------------------------------------------------------

@dataclass(frozen=True)
class EventOccurrence:
    """Detected schedule/time occurrence inside free-form Spanish text."""
    start: Optional[datetime]
    end: Optional[datetime]
    date_text: Optional[str]
    time_text: Optional[str]
    span: Tuple[int, int]
    matched_text: str
    confidence: float
    labels: Tuple[str, ...]


def extract_event_occurrences(
    text: str,
    *,
    reference_dt: Optional[datetime] = None,
    timezone: str = "America/Mexico_City",
    max_results: int = 50,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Production-grade extractor for Spanish schedule/time occurrences in long text.

    What it detects (examples):
      - Relative dates: "hoy", "mañana", "pasado mañana"
      - Weekdays: "este viernes", "próximo lunes"
      - Explicit dates: "12/01", "12-01-2026", "12 de enero", "12 ene 2026"
      - Times: "8", "8 pm", "20:30", "a las 7:15", "7pm", "19 hrs", "mediodía"
      - Ranges: "de 8 a 10", "8-10 pm", "19:00 a 22:00"
      - Icons: ⏰ 🕒 ⌚ 📅 🗓️

    Returns:
      List of dicts (serializable) with start/end datetimes (timezone-aware when possible),
      plus raw matched text, span, confidence, and labels.

    Notes:
      - If a date is found without a time, start/end are set to midnight (00:00) with low confidence.
      - If a time is found without a date, date defaults to reference_dt.date() with low confidence.
      - Parsing is heuristic but robust; designed for social/event descriptions in Spanish.
    """
    log = logger or logging.getLogger(__name__)
    if not isinstance(text, str) or not text.strip():
        return []

    ref = reference_dt or datetime.now(_get_tz(timezone))
    tz = _get_tz(timezone)

    norm_text = _normalize_text(text)
    candidates = _find_candidate_spans(text, norm_text)

    occurrences: List[EventOccurrence] = []
    for (s, e, labels) in candidates:
        snippet = text[s:e]
        snippet_norm = norm_text[s:e]

        date_dt, date_text, date_conf = _parse_date(snippet_norm, ref, tz)
        t_start, t_end, time_text, time_conf = _parse_time_range(snippet_norm, ref, tz)

        start_dt, end_dt, conf, out_labels = _compose_datetimes(
            ref=ref,
            tz=tz,
            date_dt=date_dt,
            date_text=date_text,
            date_conf=date_conf,
            t_start=t_start,
            t_end=t_end,
            time_text=time_text,
            time_conf=time_conf,
            labels=labels,
        )

        if start_dt is None and end_dt is None and not (date_text or time_text):
            continue

        occurrences.append(
            EventOccurrence(
                start=start_dt,
                end=end_dt,
                date_text=date_text,
                time_text=time_text,
                span=(s, e),
                matched_text=snippet.strip(),
                confidence=conf,
                labels=tuple(sorted(out_labels)),
            )
        )

        if len(occurrences) >= max_results:
            log.info("Reached max_results=%s while extracting occurrences.", max_results)
            break

    # De-duplicate by (start,end,matched_text simplified)
    occurrences = _dedupe_occurrences(occurrences)

    # Sort by start datetime (unknowns last)
    occurrences.sort(key=lambda o: (o.start is None, o.start or ref, o.confidence), reverse=False)

    return [asdict(o) for o in occurrences]


# ---- Internals --------------------------------------------------------------

_TIME_ICONS = r"[⏰🕒⌚🕰️⏱️📅🗓️🗒️]"
_PLACE_ICONS = r"[📍🗺️]"
_EVENT_HINTS = (
    "evento", "concierto", "rave", "fiesta", "festival", "show", "presentacion", "presentación",
    "funcion", "función", "taller", "charla", "meetup", "reunion", "reunión", "clase",
    "boletos", "tickets", "cover", "entrada", "abrimos", "apertura", "closing", "lineup",
)
_DATE_HINTS = (
    "hoy", "manana", "mañana", "pasado manana", "pasado mañana", "este", "esta", "proximo",
    "próximo", "siguiente", "viernes", "sabado", "sábado", "domingo", "lunes", "martes",
    "miercoles", "miércoles", "jueves",
    "ene", "enero", "feb", "febrero", "mar", "marzo", "abr", "abril", "may", "mayo",
    "jun", "junio", "jul", "julio", "ago", "agosto", "sep", "sept", "septiembre",
    "oct", "octubre", "nov", "noviembre", "dic", "diciembre",
)
_TIME_HINTS = (
    "a las", "alas", "hora", "horas", "hrs", "hr", "h", "pm", "am", "medio dia", "mediodia",
    "medianoche",
)


_MONTHS = {
    "ene": 1, "enero": 1,
    "feb": 2, "febrero": 2,
    "mar": 3, "marzo": 3,
    "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "junio": 6,
    "jul": 7, "julio": 7,
    "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "septiembre": 9,
    "oct": 10, "octubre": 10,
    "nov": 11, "noviembre": 11,
    "dic": 12, "diciembre": 12,
}

_WEEKDAYS = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2, "miércoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5, "sábado": 5,
    "domingo": 6,
}


def _get_tz(tz_name: str):
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return None


def _normalize_text(text: str) -> str:
    # Keep original length alignment as much as possible:
    # - Lowercase
    # - Normalize accents to ASCII while preserving string length by replacing accented letters
    #   with their base form (best-effort). This is good enough for regex anchoring spans.
    lowered = text.lower()

    # Replace common multi-codepoint emoji variants to single placeholders (length may change slightly,
    # but we don't use normalized spans for returning; we keep spans from original).
    lowered = lowered.replace("\uFE0F", "")

    # Remove accents
    nfkd = unicodedata.normalize("NFKD", lowered)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def _find_candidate_spans(text: str, norm_text: str) -> List[Tuple[int, int, Tuple[str, ...]]]:
    """
    Identify likely spans containing schedule/time info.
    Returns spans in original text indices.
    """
    spans: List[Tuple[int, int, Tuple[str, ...]]] = []

    # 1) Explicit icon-driven spans (usually near the relevant info)
    icon_pat = re.compile(rf"({_TIME_ICONS}|{_PLACE_ICONS})")
    for m in icon_pat.finditer(text):
        s = max(0, m.start() - 80)
        e = min(len(text), m.end() + 120)
        spans.append((s, e, ("icon",)))

    # 2) Date/time formats and keywords
    keyword_pat = re.compile(
        r"("
        r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b"                  # 12/01, 12-01-2026
        r"|"
        r"\b\d{1,2}\s+de\s+[a-z]{3,10}(\s+\d{4})?\b"             # 12 de enero 2026
        r"|"
        r"\b(hoy|manana|mañana|pasado\s+manana|pasado\s+mañana)\b"
        r"|"
        r"\b(este|esta|proximo|próximo|siguiente)\s+(lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado|domingo)\b"
        r"|"
        r"\b(lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado|domingo)\b"
        r"|"
        r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b"                       # 8, 8pm, 20:30
        r"|"
        r"\b(a\s*las\s*)?\d{1,2}(:\d{2})?\s*(hrs|hr|h)\b"        # a las 19 hrs
        r"|"
        r"\b(mediodia|medio\s+dia|medianoche)\b"
        r")"
    )
    for m in keyword_pat.finditer(norm_text):
        s = max(0, m.start() - 90)
        e = min(len(text), m.end() + 140)
        spans.append((s, e, ("keyword",)))

    # 3) Event-hints (broadened window; schedule often appears near them)
    hints = "|".join(sorted(set(_EVENT_HINTS), key=len, reverse=True))
    if hints:
        event_pat = re.compile(rf"\b({hints})\b", re.IGNORECASE)
        for m in event_pat.finditer(norm_text):
            s = max(0, m.start() - 120)
            e = min(len(text), m.end() + 220)
            spans.append((s, e, ("event_hint",)))

    # Merge overlapping spans
    spans.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int, Tuple[str, ...]]] = []
    for s, e, labels in spans:
        if not merged:
            merged.append((s, e, labels))
            continue
        ps, pe, pl = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e), tuple(sorted(set(pl + labels))))
        else:
            merged.append((s, e, labels))

    return merged


def _parse_date(snippet_norm: str, ref: datetime, tz) -> Tuple[Optional[datetime], Optional[str], float]:
    """
    Returns (date_dt at 00:00, date_text, confidence).
    """
    s = snippet_norm

    # Relative: hoy/mañana/pasado mañana
    rel_pat = re.compile(r"\b(hoy|manana|mañana|pasado\s+manana|pasado\s+mañana)\b")
    m = rel_pat.search(s)
    if m:
        key = m.group(1).replace("mañana", "manana")
        if key == "hoy":
            d = ref.date()
            return _make_dt(d.year, d.month, d.day, 0, 0, tz), m.group(1), 0.78
        if key == "manana":
            d = (ref + timedelta(days=1)).date()
            return _make_dt(d.year, d.month, d.day, 0, 0, tz), m.group(1), 0.75
        if key == "pasado manana":
            d = (ref + timedelta(days=2)).date()
            return _make_dt(d.year, d.month, d.day, 0, 0, tz), m.group(1), 0.75

    # Weekday with modifiers
    wd_mod_pat = re.compile(
        r"\b(este|esta|proximo|próximo|siguiente)\s+"
        r"(lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado|domingo)\b"
    )
    m = wd_mod_pat.search(s)
    if m:
        target = m.group(2)
        dt = _next_weekday(ref, _weekday_index(target), force_next=True)
        return _make_dt(dt.year, dt.month, dt.day, 0, 0, tz), m.group(0), 0.72

    # Plain weekday (assume next occurrence, including today if time hasn't passed is unknown -> choose next)
    wd_pat = re.compile(r"\b(lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado|domingo)\b")
    m = wd_pat.search(s)
    if m:
        target = m.group(1)
        dt = _next_weekday(ref, _weekday_index(target), force_next=False)
        return _make_dt(dt.year, dt.month, dt.day, 0, 0, tz), m.group(1), 0.62

    # dd/mm(/yyyy) or dd-mm(/yyyy)
    num_date_pat = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")
    m = num_date_pat.search(s)
    if m:
        d = int(m.group(1))
        mo = int(m.group(2))
        y_raw = m.group(3)
        y = _coerce_year(y_raw, ref.year)
        if _valid_ymd(y, mo, d):
            return _make_dt(y, mo, d, 0, 0, tz), m.group(0), 0.90

    # "12 de enero (2026)"
    long_date_pat = re.compile(r"\b(\d{1,2})\s+de\s+([a-z]{3,10})(?:\s+(\d{4}))?\b")
    m = long_date_pat.search(s)
    if m:
        d = int(m.group(1))
        mo_key = m.group(2)
        y = int(m.group(3)) if m.group(3) else ref.year
        mo = _MONTHS.get(mo_key[:4], _MONTHS.get(mo_key, 0))
        if mo and _valid_ymd(y, mo, d):
            return _make_dt(y, mo, d, 0, 0, tz), m.group(0), 0.92

    return None, None, 0.0


def _parse_time_range(snippet_norm: str, ref: datetime, tz) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[str], float]:
    """
    Returns (start_hm, end_hm, time_text, confidence).
    start_hm/end_hm are (hour, minute) tuples.
    """
    s = snippet_norm

    # "mediodia / medianoche"
    m = re.search(r"\b(mediodia|medio\s+dia|medianoche)\b", s)
    if m:
        if "medianoche" in m.group(1):
            return (0, 0), None, m.group(1), 0.70
        return (12, 0), None, m.group(1), 0.70

    # Time range "de 8 a 10", "8-10 pm", "19:00 a 22:00"
    range_pat = re.compile(
        r"\b(?:de\s+)?"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*"
        r"(?:a|hasta|-|–)\s*"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b"
    )
    m = range_pat.search(s)
    if m:
        h1, mi1, ap1 = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        h2, mi2, ap2 = int(m.group(4)), int(m.group(5) or 0), m.group(6)

        # If only one am/pm is provided, propagate.
        if ap1 and not ap2:
            ap2 = ap1
        if ap2 and not ap1:
            ap1 = ap2

        h1 = _apply_ampm(h1, ap1)
        h2 = _apply_ampm(h2, ap2)

        if _valid_hm(h1, mi1) and _valid_hm(h2, mi2):
            return (h1, mi1), (h2, mi2), m.group(0), 0.92

    # Single time: "a las 7:15", "20:30", "8pm", "19 hrs"
    single_pat = re.compile(
        r"\b(?:a\s*las\s*)?"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:hrs|hr|h)?\b"
    )
    m = single_pat.search(s)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        h = _apply_ampm(h, ap)
        if _valid_hm(h, mi):
            conf = 0.85 if (":" in m.group(0) or ap) else 0.62  # "8" alone is ambiguous
            return (h, mi), None, m.group(0), conf

    return None, None, None, 0.0


def _compose_datetimes(
    *,
    ref: datetime,
    tz,
    date_dt: Optional[datetime],
    date_text: Optional[str],
    date_conf: float,
    t_start: Optional[Tuple[int, int]],
    t_end: Optional[Tuple[int, int]],
    time_text: Optional[str],
    time_conf: float,
    labels: Tuple[str, ...],
) -> Tuple[Optional[datetime], Optional[datetime], float, Tuple[str, ...]]:
    out_labels = set(labels)

    # If neither date nor time: no usable schedule.
    if date_dt is None and t_start is None:
        return None, None, 0.0, tuple(out_labels)

    # Default missing date to ref.date (low confidence)
    base_date = (date_dt.date() if date_dt else ref.date())
    if date_dt is None:
        out_labels.add("date_imputed")

    # Default missing time to 00:00 (very low confidence)
    if t_start is None:
        out_labels.add("time_missing")
        start_dt = _make_dt(base_date.year, base_date.month, base_date.day, 0, 0, tz)
        end_dt = None
        conf = 0.35 + 0.40 * date_conf  # mostly date-driven
        return start_dt, end_dt, min(conf, 0.75), tuple(out_labels)

    start_dt = _make_dt(base_date.year, base_date.month, base_date.day, t_start[0], t_start[1], tz)
    end_dt = None

    if t_end is not None:
        end_dt = _make_dt(base_date.year, base_date.month, base_date.day, t_end[0], t_end[1], tz)
        # If end < start, assume crosses midnight -> add 1 day
        if end_dt < start_dt:
            end_dt = end_dt + timedelta(days=1)
            out_labels.add("crosses_midnight")

    # Confidence fusion
    conf = 0.25 + 0.45 * max(date_conf, 0.25 if date_dt else 0.10) + 0.45 * time_conf
    conf = max(0.05, min(conf, 0.99))

    # Label if explicit icon present (helps)
    if "icon" in out_labels:
        conf = min(0.99, conf + 0.03)

    return start_dt, end_dt, conf, tuple(out_labels)


def _dedupe_occurrences(items: List[EventOccurrence]) -> List[EventOccurrence]:
    seen = set()
    out: List[EventOccurrence] = []
    for o in items:
        key = (
            o.start.isoformat() if o.start else None,
            o.end.isoformat() if o.end else None,
            _normalize_text(o.matched_text)[:180],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def _make_dt(y: int, mo: int, d: int, h: int, mi: int, tz):
    dt = datetime(y, mo, d, h, mi)
    if tz is not None:
        return dt.replace(tzinfo=tz)
    return dt  # naive fallback


def _weekday_index(wd: str) -> int:
    return _WEEKDAYS[wd]


def _next_weekday(ref: datetime, target_weekday: int, *, force_next: bool) -> datetime:
    """
    If force_next: always returns next week's instance if same weekday.
    Otherwise returns same weekday if it's today (useful when time exists elsewhere),
    else next occurrence.
    """
    current = ref.weekday()
    delta = (target_weekday - current) % 7
    if force_next and delta == 0:
        delta = 7
    if not force_next and delta == 0:
        return ref
    return ref + timedelta(days=delta)


def _coerce_year(y_raw: Optional[str], default_year: int) -> int:
    if not y_raw:
        return default_year
    y = int(y_raw)
    if y < 100:
        # 2-digit years: assume 2000-2099
        return 2000 + y
    return y


def _apply_ampm(hour: int, ampm: Optional[str]) -> int:
    if ampm is None:
        return hour
    ap = ampm.lower()
    if ap == "am":
        if hour == 12:
            return 0
        return hour
    if ap == "pm":
        if hour < 12:
            return hour + 12
        return hour
    return hour


def _valid_hm(h: int, m: int) -> bool:
    return 0 <= h <= 23 and 0 <= m <= 59


def _valid_ymd(y: int, mo: int, d: int) -> bool:
    try:
        datetime(y, mo, d)
        return True
    except ValueError:
        return False



### ---------------------------------------------------------------------------------------------
### Text patterns - Geo-indexing ----------------------------------------------------------------


import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple


# =========================
# Data models
# =========================

@dataclass(frozen=True)
class AddressMatch:
    text: str
    span: Tuple[int, int]
    score: float
    rule: str


@dataclass(frozen=True)
class PlaceMatch:
    text: str
    span: Tuple[int, int]
    kind: str
    rule: str


@dataclass(frozen=True)
class LocationLabelResult:
    label: str
    kind: str  # "ADDRESS" | "PLACE" | "NONE"
    score: float
    rule: str
    span: Tuple[int, int]


# =========================
# Language configuration
# =========================

@dataclass(frozen=True)
class LanguageConfig:
    """
    Provide language-specific regex subpatterns and lexicons.
    All fields are regex *fragments* unless explicitly noted.
    """
    language: str

    # Emoji / icon anchors
    location_emoji: str

    # Address structure
    street_types: str
    street_name: str
    num: str
    neighborhood: str
    postal: str
    city_tokens: str
    cross_words: str
    unit: str

    # Place detection (string tokens, not regex fragments)
    city_token_list: Tuple[str, ...]
    location_cues: Tuple[str, ...]

    # Stop conditions for expansion
    stop_re: Pattern


def default_spanish_config() -> LanguageConfig:
    location_emoji = r"(?:📍|📌|🗺️|🧭|🏠|🏢|🏬|🏪|🏨|🏫|🏥)"

    street_types = r"""
    (?:av(?:enida)?\.?|calle|c\.|blvd\.?|boulevard|paseo|calz\.?|calzada|circuito|privada|cerrada|
    carretera|camino|andador|prol\.?|prolongaci[oó]n|eje\s+(?:central|vial)|anillo|perif[eé]rico)
    """.strip()

    street_name = (
        r"(?:[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]"
        r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-']*"
        r"(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-']+)*)"
    )

    # IMPORTANT: Escape '#' because we use re.VERBOSE
    num = r"(?:(?:\#\s*)?(?:\d{1,6})(?:\s*(?:bis|ter|a|b))?|s\/n|sin\s+n[uú]mero|sin\s+numero)"
    neighborhood = r"(?:col\.?|colonia|fracc\.?|fraccionamiento|barrio|unidad\s+habitacional|u\.h\.)"
    postal = r"(?:c\.?p\.?\s*)?\b\d{5}\b"
    city_tokens = r"(?:cdmx|ciudad\s+de\s+m[eé]xico|mexico\s+d\.?f\.?|guadalajara|monterrey|puebla|quer[eé]taro)"
    cross_words = r"(?:esquina\s+con|esq\.?\s+con|entre)"
    unit = r"(?:depto\.?|departamento|int\.?|interior|piso|torre|edif\.?|edificio|local|mz\.?|manzana|lote|km)\s*\w+"

    stop_re = re.compile(
    r"(?:"
    r"\n"
    r"|\b(?:tel|telefono|whats|precio|horario|hrs|instagram|ig|facebook|link|env[ií]os)\b"
    r"|https?://"
    r")",
    re.IGNORECASE,
    )


    city_token_list = (
        "CDMX", "Ciudad de México", "México DF", "Mexico DF", "DF",
        "Guadalajara", "GDL", "Monterrey", "MTY", "Puebla", "Querétaro", "Queretaro",
        "Tijuana", "Mérida", "Merida", "Cancún", "Cancun", "Toluca",
    )

    location_cues = (
        r"direcci[oó]n", r"ubicaci[oó]n", r"est[aá]\s+en", r"queda\s+en", r"estamos\s+en",
        r"nos\s+vemos\s+en", r"visita", r"ven\s+a", r"en\b", r"por\b", r"sobre\b",
        r"explora(?:\s+la)",
        r"explora(?:\s+el)",
        r"descubre",
        r"direcci[oó]n",
        r"ubicaci[oó]n",
        r"domicilio",
        r"vis[ií]tanos(?:\s+en)?",
        r"visitenos(?:\s+en)?",
        r"te\s+esperamos(?:\s+en)?",
        r"nos\s+ubicamos(?:\s+en)?",
        r"nos\s+encontramos(?:\s+en)?",
        r"estamos(?:\s+en)?",
        r"est[aá](?:\s+en)?",
        r"queda(?:\s+en)?",
        r"encu[eé]ntranos(?:\s+en)?",
        r"encu[eé]ntra(?:\s+en)?",
        r"ub[ií]canos(?:\s+en)?",
        r"c[oó]mo\s+llegar(?:\s+a)?",
        r"c[oó]mo\s+llegar(?:\s+al)?",
        r"como\s+llegar(?:\s+al)?",
        r"como\s+llegar(?:\s+a)?",
        r"c[oó]mo\s+llego(?:\s+a)?",
        r"como\s+llego(?:\s+a)?",
        r"ruta(?:\s+a)?",
        r"google\s+maps",
        r"maps",
        r"waze",
        r"sucursal(?:\s+en)?",
        r"nos\s+vemos\s+en",
        r"visita",
        r"ven\s+a",
        r"pasa\s+por",
        r"donde\s+est[aá]mos",
        r"d[oó]nde\s+queda",
        r"en\b",
        r"por\b",
        r"sobre\b",
        r"cerca\s+de",
        r"frente\s+a",
        r"a\s+un\s+lado\s+de",
        r"junto\s+a",
        r"entre",
        r"explora",
        r"descubre",
        r"parque",
        r"patio",
        r"jardin",
        r"bosque",
        r"lago",
        r"laguna",
        r"cascada",
        r"playa",
        r"selva",
        r"reserva",
        r"reserva\s+natural",
        r"sendero",
        r"mirador",
        )

    return LanguageConfig(
        language="es",
        location_emoji=location_emoji,
        street_types=street_types,
        street_name=street_name,
        num=num,
        neighborhood=neighborhood,
        postal=postal,
        city_tokens=city_tokens,
        cross_words=cross_words,
        unit=unit,
        city_token_list=city_token_list,
        location_cues=location_cues,
        stop_re=stop_re,
    )


# =========================
# Compiler helpers
# =========================

@dataclass
class _CompiledPatterns:
    # address patterns
    street_number: Pattern
    emoji_street_number: Pattern
    cross_streets: Pattern
    emoji_soft: Pattern

    # place patterns
    poi_city: Pattern
    cue_location: Pattern
    hashtag_city: Pattern

    # city recognizer
    city_re: Pattern


_COMPILED_CACHE: Dict[str, _CompiledPatterns] = {}


def _build_city_regex(city_tokens: List[str]) -> Pattern:
    toks = sorted(city_tokens, key=len, reverse=True)
    alts = "|".join(re.escape(t) for t in toks)
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


def _compile_patterns(cfg: LanguageConfig) -> _CompiledPatterns:
    if cfg.language in _COMPILED_CACHE:
        return _COMPILED_CACHE[cfg.language]

    # 1) Street + name + number + optional tails
    street_number = re.compile(
        rf"""
        (?P<cand>
            \b{cfg.street_types}\s+{cfg.street_name}\s+
            (?:{cfg.num})
            (?:
                (?:\s*,\s*|\s+)
                (?:
                    {cfg.neighborhood}\s+{cfg.street_name}
                    |
                    {cfg.postal}
                    |
                    {cfg.city_tokens}
                    |
                    {cfg.unit}
                    |
                    [A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-']{{2,}}
                )
            ){{0,8}}
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # 2) Emoji + Street + name + number + optional tails
    emoji_street_number = re.compile(
        rf"""
        (?P<cand>
            {cfg.location_emoji}\s*
            \b{cfg.street_types}\s+{cfg.street_name}\s+
            (?:{cfg.num})
            (?:
                (?:\s*,\s*|\s+)
                (?:
                    {cfg.neighborhood}\s+{cfg.street_name}
                    |
                    {cfg.postal}
                    |
                    {cfg.city_tokens}
                    |
                    {cfg.unit}
                    |
                    [A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-']{{2,}}
                )
            ){{0,8}}
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # 3) Emoji + general location line
    emoji_soft = re.compile(
        rf"""
        (?P<cand>
            {cfg.location_emoji}\s*
            [^\n;]{{10,180}}
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # 4) Cross-street / Between pattern
    cross_streets = re.compile(
        rf"""
        (?P<cand>
            \b{cfg.cross_words}\b
            [^\n;]{{10,160}}
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # Place: "POI | CITY"
    poi_city = re.compile(
        r"""
        (?P<poi>
            (?:[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ\.\-&'’]+
            (?:\s+[A-ZÁÉÍÓÚÜÑ0-9][\wÁÉÍÓÚÜÑáéíóúüñ\.\-&'’]+){0,8})
        )
        \s*(?:\||-|—|–|·|,)\s*
        (?P<city>[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\. ]{2,40})
        """,
        re.VERBOSE,
    )

    cue_location = re.compile(
        rf"""
        (?P<cue>
            {cfg.location_emoji}\s*|
            (?:{"|".join(cfg.location_cues)})
        )
        \s*[:\-]?\s*
        (?P<loc>
            [^\n\.;!\?\|]{{3,120}}
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    hashtag_city = re.compile(r"#(?P<city>[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,20})", re.IGNORECASE)

    compiled = _CompiledPatterns(
        street_number=street_number,
        emoji_street_number=emoji_street_number,
        cross_streets=cross_streets,
        emoji_soft=emoji_soft,
        poi_city=poi_city,
        cue_location=cue_location,
        hashtag_city=hashtag_city,
        city_re=_build_city_regex(list(cfg.city_token_list)),
    )
    _COMPILED_CACHE[cfg.language] = compiled
    return compiled


# =========================
# Address extraction internals
# =========================

def _expand_with_emoji(text: str, start: int, location_emoji: str, max_back: int = 12) -> int:
    back = max(0, start - max_back)
    prefix = text[back:start]
    m = re.search(rf"{location_emoji}\s*$", prefix)
    return back + m.start() if m else start


def _expand_forward(text: str, start: int, end: int, stop_re: Pattern, extra_chars: int) -> Tuple[int, int]:
    max_end = min(len(text), end + extra_chars)
    tail = text[end:max_end]
    stop = stop_re.search(tail)
    if stop:
        max_end = end + stop.start()
    return start, max_end


def _address_score(s: str, cfg: LanguageConfig) -> float:
    s_low = s.lower()
    score = 0.0

    cues_alt = "|".join(f"(?:{cue})" for cue in cfg.location_cues)
    LOCATION_CUES_RE =  re.compile(rf"(?:{cues_alt})", re.IGNORECASE)

    if re.search(cfg.location_emoji, s):
        score += 2.5
    if re.search(LOCATION_CUES_RE, s):
        occurrences = sum(1 for _ in LOCATION_CUES_RE.finditer(s))
        score += 2.0 #* min(occurrences, 3)
    if re.search(LOCATION_CUES_RE, s_low):
        occurrences = sum(1 for _ in LOCATION_CUES_RE.finditer(s_low))
        score += 2.0 #* min(occurrences, 3)
    if re.search(rf"\b{cfg.street_types}\b", s_low):
        score += 3.0
    if re.search(rf"\b(?:{cfg.num})\b", s_low):
        score += 3.0
    if re.search(rf"\b{cfg.neighborhood}\b", s_low):
        score += 2.0
    if re.search(cfg.postal, s_low):
        score += 2.0
    if re.search(cfg.city_tokens, s_low):
        score += 1.0
    if re.search(rf"\b{cfg.cross_words}\b", s_low):
        score += 1.0
    if re.search(r"\b(?:depto|departamento|int|interior|piso|torre|edif|edificio|local|mz|manzana|lote|km)\b", s_low):
        score += 0.5

    L = len(s.strip())
    if L < 10:
        score -= 3.0
    if L >= 10 and L<18 :
        score -= 2.0
    elif L > 240:
        score -= 3.0

    return score


def _dedupe_address(matches: List[AddressMatch]) -> List[AddressMatch]:
    if not matches:
        return []

    matches = sorted(matches, key=lambda x: (x.span[0], -(x.span[1] - x.span[0]), -x.score))
    kept: List[AddressMatch] = []

    def overlap_ratio(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union else 0.0

    for m in matches:
        replaced = False
        for i, k in enumerate(kept):
            if overlap_ratio(m.span, k.span) >= 0.55:
                if m.score > k.score:
                    kept[i] = m
                replaced = True
                break
        if not replaced:
            kept.append(m)

    return sorted(kept, key=lambda x: x.score, reverse=True)


def extract_addresses(
    text: str,
    cfg: LanguageConfig,
    top_k: int = 3,
    min_score: float = 5.0,
) -> List[AddressMatch]:
    if not text or not text.strip():
        return []

    rx = _compile_patterns(cfg)

    patterns: List[Tuple[str, Pattern]] = [
        ("emoji_street_number", rx.emoji_street_number),
        ("street_number", rx.street_number),
        ("cross_streets", rx.cross_streets),
        ("emoji_soft", rx.emoji_soft),
    ]

    found: List[AddressMatch] = []

    for rule_name, pat in patterns:
        for m in pat.finditer(text):
            start, end = m.start("cand"), m.end("cand")

            start = _expand_with_emoji(text, start, cfg.location_emoji)

            if rule_name in ("street_number", "emoji_street_number"):
                start, end = _expand_forward(text, start, end, cfg.stop_re, extra_chars=160)
            else:
                start, end = _expand_forward(text, start, end, cfg.stop_re, extra_chars=80)

            cand = text[start:end].strip(" ,.;:-")
            score = _address_score(cand, cfg)

            # Down-weight the soft emoji rule unless it contains strong cues
            if rule_name == "emoji_soft" and score < 6.0:
                score -= 1.5

            found.append(AddressMatch(text=cand, span=(start, end), score=score, rule=rule_name))

    ranked = _dedupe_address(found)
    return [m for m in ranked if m.score >= min_score][:top_k]


# =========================
# Place detection internals
# =========================

def _dedupe_place_overlaps(matches: List[PlaceMatch]) -> List[PlaceMatch]:
    if not matches:
        return []

    matches_sorted = sorted(matches, key=lambda x: (-(x.span[1] - x.span[0]), x.span[0]))
    kept: List[PlaceMatch] = []

    def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    for m in matches_sorted:
        if any(overlaps(m.span, k.span) for k in kept):
            continue
        kept.append(m)

    # Keep POI/PLACE/CITY before weaker ones; then in reading order
    kind_rank = {"POI": 0, "PLACE": 1, "CITY": 2}
    return sorted(kept, key=lambda x: (kind_rank.get(x.kind, 9), x.span[0]))


def detect_places(
    text: str,
    cfg: LanguageConfig,
    max_results: int = 10,
) -> List[PlaceMatch]:
    if not text or not text.strip():
        return []

    rx = _compile_patterns(cfg)
    matches: List[PlaceMatch] = []

    # Rule 1: cue-based location phrases
    for m in rx.cue_location.finditer(text):
        loc = m.group("loc").strip(" ,|-")
        span = (m.start("loc"), m.end("loc"))
        kind = "CITY" if rx.city_re.search(loc) else "PLACE"
        matches.append(PlaceMatch(text=loc, span=span, kind=kind, rule="cue_location"))

    # Rule 2: POI | CITY
    for m in rx.poi_city.finditer(text):
        poi = m.group("poi").strip(" -|,·—–")
        city_raw = m.group("city").strip(" -|,·—–")
        # Validate city lightly if possible; keep POI regardless
        _ = rx.city_re.search(city_raw)  # intentional side-effect-free validation
        matches.append(PlaceMatch(
            text=poi,
            span=(m.start(), m.end()),
            kind="POI",
            rule="poi_city",
        ))

    # Rule 3: hashtag city
    for m in rx.hashtag_city.finditer(text):
        tag = m.group("city")
        if rx.city_re.search(tag) or tag.lower() in {"cdmx", "gdl", "mty"}:
            matches.append(PlaceMatch(
                text="#" + tag,
                span=(m.start(), m.end()),
                kind="CITY",
                rule="hashtag_city",
            ))

    matches = _dedupe_place_overlaps(matches)
    return matches[:max_results]


def _place_score(pm: PlaceMatch, cfg: LanguageConfig) -> float:
    """
    Place scoring is intentionally weaker than address scoring.
    We only allow place to win when address evidence is insufficient.
    """
    score = 0.0
    s = pm.text.strip()
    s_low = s.lower()

    cues_alt = "|".join(f"(?:{cue})" for cue in cfg.location_cues)
    LOCATION_CUES_RE =  re.compile(rf"(?:{cues_alt})", re.IGNORECASE)

    # Reward patter importance 
    if re.search(cfg.location_emoji, s):
        score += 3.0
    
    if re.search(LOCATION_CUES_RE, s):
        occurrences = sum(1 for _ in LOCATION_CUES_RE.finditer(s))
        score += 2.0 #* min(occurrences, 3)

    if re.search(LOCATION_CUES_RE, s_low):
        occurrences = sum(1 for _ in LOCATION_CUES_RE.finditer(s_low))
        score += 2.0 #* min(occurrences, 3)


    if pm.kind == "POI":
        score += 3.0
    elif pm.kind == "CITY":
        score += 2.0
    else:
        score += 1.0

    L = len(s)
    if L < 4:
        score -= 1.5
    elif L > 160:
        score -= 0.5

    # Penalize strings that look like marketing fragments rather than locations
    if re.search(r"\b(?:promo|gratis|descuento|reserv(a|as)|boletos|entradas)\b", s.lower()):
        score -= 0.75

    return score


# =========================
# Unified public API
# =========================

def best_location_label_from_text(
    text: str,
    cfg: Optional[LanguageConfig] = None,
     *,
    min_address_score: float = 7.0,
    min_place_score: float = 3.0,
) -> LocationLabelResult:
    """
    Priority order:
      1) Strong address formats (emoji + street + number; street + number)
      2) Secondary address (cross-streets; emoji_soft)
      3) Place-name patterns (POI|CITY; cue-based; hashtag city)

    `min_address_score` is set higher than the internal extractor default to enforce
    "strong priority" to address structure.
    """
    cfg = cfg or default_spanish_config()

    text = text.lower()

    if not text or not text.strip():
        return LocationLabelResult(label="", kind="NONE", score=0.0, rule="empty", span=(0, 0))

    # 1) Addresses (strongly preferred)
    addr = extract_addresses(text, cfg, top_k=5, min_score=0.0)  # filter here with custom logic
    if addr:
        best_addr = addr[0]
        if best_addr.score >= min_address_score:
            return LocationLabelResult(
                label=best_addr.text,
                kind="ADDRESS",
                score=best_addr.score,
                rule=best_addr.rule,
                span=best_addr.span,
            )

    # 2) If address exists but is weak, we still prefer it unless *very* weak
    if addr:
        best_addr = addr[0]
        # If it has street+num evidence, keep it even if slightly below threshold
        has_struct = (
            re.search(rf"\b{cfg.street_types}\b", best_addr.text.lower()) is not None
            and re.search(rf"\b(?:{cfg.num})\b", best_addr.text.lower()) is not None
        )
        if has_struct and best_addr.score >= (min_address_score - 1.5):
            return LocationLabelResult(
                label=best_addr.text,
                kind="ADDRESS",
                score=best_addr.score,
                rule=best_addr.rule,
                span=best_addr.span,
            )

    # 3) Places as fallback
    places = detect_places(text, cfg, max_results=10)
    if places:
        scored = [(p, _place_score(p, cfg)) for p in places]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_place, p_score = scored[0]
        if p_score >= min_place_score:
            return LocationLabelResult(
                label=best_place.text,
                kind="PLACE",
                score=p_score,
                rule=best_place.rule,
                span=best_place.span,
            )

    # 4) If nothing passes thresholds, return best available weak signal
    if addr:
        best_addr = addr[0]
        return LocationLabelResult(
            label=best_addr.text,
            kind="ADDRESS",
            score=best_addr.score,
            rule=best_addr.rule,
            span=best_addr.span,
        )

    if places:
        best_place = places[0]
        return LocationLabelResult(
            label=best_place.text,
            kind="PLACE",
            score=_place_score(best_place, cfg),
            rule=best_place.rule,
            span=best_place.span,
        )

    return LocationLabelResult(label="", kind="NONE", score=0.0, rule="no_match", span=(0, 0))





# =========================
# Optional: config for future languages
# =========================

def build_language_config(
    language: str,
    *,
    location_emoji: str,
    street_types: str,
    street_name: str,
    num: str,
    neighborhood: str,
    postal: str,
    city_tokens: str,
    cross_words: str,
    unit: str,
    city_token_list: List[str],
    location_cues: List[str],
    stop_re: Optional[Pattern] = None,
) -> LanguageConfig:
    return LanguageConfig(
        language=language,
        location_emoji=location_emoji,
        street_types=street_types,
        street_name=street_name,
        num=num,
        neighborhood=neighborhood,
        postal=postal,
        city_tokens=city_tokens,
        cross_words=cross_words,
        unit=unit,
        city_token_list=tuple(city_token_list),
        location_cues=tuple(location_cues),
        stop_re=stop_re
        or re.compile(
            r"(?:\n|\b(?:tel|phone|whats|price|hours|instagram|ig|facebook|link|shipping)\b|https?://)",
            re.IGNORECASE,
        ),
    )



import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional, Pattern, Sequence


@dataclass(frozen=True)
class TextFormatConfig:
    """
    Configuration for robust formatting/normalization of an address/place label.
    All patterns are applied in the order documented in `format_place_or_address()`.
    """
    language: str = "es"

    # Tokens that are typically connective/noise when isolated (caption-like text).
    # Use short, high-frequency words; keep them conservative to avoid deleting meaning.
    connectors: Sequence[str] = (
        "en", "de", "del", "la", "las", "el", "los", "y", "e", "a", "al", "por",
        "con", "sin", "para", "sobre", "entre", "frente", "cerca", "hacia", "desde",
    )

    # Extra “junk” tokens often seen in scraped captions / CTAs.
    # Keep conservative and only remove when they appear as standalone words.
    # Disclaimer and hard coded captions in web displays 
    junk_tokens: Sequence[str] = (
        "link", "bio", "dm", "inbox", "whatsapp", "wa", "ig", "instagram", "facebook",
        "maps", "google", "ubicacion", "ubicación", "direccion", "dirección",
        "este es un resumen del contenido generado con ia y no busca ofrecer un contexto basado en hechos", 
        "si crees que puede contener algún error, informanos en: tomentarios y ayuda: tiktok",
    )

    junk_description_tokens: Sequence[str] = (
        "link", "bio", "dm", "inbox", "whatsapp", "wa", "ig", "instagram", "facebook",
        r"(?:bonit[oa]s?|lind[oa]s?|hermos[oa]s?|preci[oó]s[oa]s?|bell[oa]s?|encantador(?:a|es)?|espectacular(?:es)?|incre[ií]ble(?:s)?|maravillos[oa]s?|impresionante(?:s)?|perfect[oa]s?|m[aá]gic[oa]s?|grande(?:s)?|enorme(?:s)?|gigante(?:s)?|inmens[oa]s?|peque[nñ][oa]s?|ampli[oa]s?|extens[oa]s?|el\s+m[aá]s\s+grande\s+de|el\s+m[aá]s\s+importante\s+de|uno\s+de\s+los\s+m[aá]s\s+grandes|mejor(?:es)?|excelente(?:s)?|imperdible(?:s)?|recomendado(?:s)?|favorito(?:s)?|top|ic[oó]nic[oa]s?|emblem[aá]tic[oa]s?|legendari[oa]s?|famos[oa]s?|inolvidable(?:s)?|[uú]nic[oa]s?|especial(?:es)?|so[nñ]ad[oa]s?|rom[aá]ntic[oa]s?|relajante(?:s)?|nuevo(?:s)?|reci[eé]n\s+abierto(?:s)?|moderno(?:s)?|tradicional(?:es)?|hist[oó]rico(?:s)?|cl[aá]sic[oa]s?|antiguo(?:s)?|de\s+moda|tendencia|viral(?:es)?|instagrameable(?:s)?|perfecto\s+para\s+fotos|ideal\s+para|el\s+lugar\s+perfecto\s+para|incre[ií]blemente|sumamente|realmente|muy|s[uú]per|mega|ultra)",
        "mas", "menos", "mejor", "peor", "mayor", "menor", "el peor de", "mas bonito de", "mas grande de",
        "más bonito de", "más grande de","más"
  
    )

    # If True: remove leading connectors (e.g., "en ...", "de ...") repeatedly.
    strip_leading_connectors: bool = True

    # If True: remove trailing connectors (rare but can occur) repeatedly.
    strip_trailing_connectors: bool = True

    # If True: drop isolated single-letter tokens (e.g. "A", "B") unless whitelisted.
    drop_isolated_letters: bool = True

    # Keep letters that can be meaningful in addresses (e.g. "Int A", "Torre B", "#12A")
    isolated_letter_whitelist: Sequence[str] = ("a", "b", "c")  # conservative default

    # Unicode normalization
    normalize_unicode_form: str = "NFKC"  # stable for user-facing text

    # If True: collapse whitespace aggressively.
    collapse_whitespace: bool = True

    # If True: normalize punctuation spacing and dedupe repeated punctuation.
    normalize_punctuation: bool = True

    # If True: title-case words (keeps acronyms/codes). Not always desired; default False.
    title_case: bool = False

    # If True: preserve accents as-is (recommended). If False: strip accents.
    keep_accents: bool = True

def unique_words_preserve_case(s: str) -> list[str]:
    seen = set()
    out = []
    for w in re.findall(r"\b\w+\b", s):
        key = w.lower()
        if key not in seen:
            seen.add(key)
            out.append(w)
    return " ".join(out)


def remove_connectors_and_junk(text: str, cfg) -> str:
    """
    Removes ALL occurrences of connectors and junk tokens (incl. regex fragments) from `text`,
    then normalizes whitespace/punctuation for place/address cleaning.

    Assumes:
      - cfg.connectors is a tuple/list of literal tokens (strings)
      - cfg.junk_tokens is a tuple/list of regex fragments (strings), as in your current design
    """
    if not text:
        return ""

    s = str(text)

    # Basic normalize: whitespace + common wrappers
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s{2,}", " ", s).strip().strip(" ,;:.-“”\"'()[]{}")

    # Build alternations
    junk_alt = "|".join(f"(?:{t})" for t in cfg.junk_description_tokens if t and t.strip())  # already regex

    # Remove ALL occurrences (not just leading/trailing)
    if junk_alt:
        s = re.sub(rf"(?i)\b(?:{junk_alt})\b", " ", s)

        # Hashtags: remove #<junk> plus optionally orphan '#'
        s = re.sub(rf"(?i)(?:^|\s)#(?:{junk_alt})\b", " ", s)
        s = re.sub(r"(?i)(?:^|\s)#(?=\s|$)", " ", s)

    # Cleanup punctuation & whitespace after removals
    s = re.sub(r"\s+([,;:.])", r"\1", s)
    s = re.sub(r"([,;:.])\s*", r"\1 ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;:.-“”\"'()[]{}")

    # Optional: if you also want to drop leftover empty separators like ", ,"
    s = re.sub(r"(?:\s*,\s*){2,}", ", ", s).strip(" ,")

    return s



def format_place_or_address(
    text: str,
    kind: str, 
    city:str,
    cfg: Optional[TextFormatConfig] = None,
) -> str:
    """
    Production-grade formatter for strings that are expected to represent a place or address.

    Goals:
      - Normalize unicode and spacing
      - Remove caption-like connectors (language-specific) at edges
      - Remove standalone junk tokens (e.g., "link", "bio", "ubicación") as standalone words
      - Drop isolated letters that are likely noise (with an allowlist)
      - Clean punctuation and separators without destroying address semantics

    This is intentionally conservative: it tries not to remove informative content.
    """
    if text is None:
        return ""

    cfg = cfg or TextFormatConfig()

    s = str(text)

    # 1) Unicode normalization (also fixes “full-width” variants, compatibility chars)
    s = unicodedata.normalize(cfg.normalize_unicode_form, s)

    # 2) Strip control characters and normalize newlines/tabs to spaces
    #    Keep only printable-ish characters; remove Cc/Cf categories.
    s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cc", "Cf"})
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # 3) Trim outer quotes / bullets / common wrappers
    s = s.strip().strip("“”\"'`´•·-–—|:")

    if not s:
        return ""

    # 4) Optional accent stripping (generally not recommended for addresses in ES)
    if not cfg.keep_accents:
        s = "".join(
            ch for ch in unicodedata.normalize("NFD", s)
            if unicodedata.category(ch) != "Mn"
        )

    # 5) Normalize common separators/spaces around punctuation
    #    Convert multiple separators to a single standardized form.
    if cfg.normalize_punctuation:
        # Normalize fancy dashes to hyphen, pipes/bullets to commas
        s = s.replace("–", "-").replace("—", "-").replace("·", ",").replace("|", ",")
        # Remove spaces before punctuation, standardize after punctuation
        s = re.sub(r"\s+([,;:.])", r"\1", s)
        s = re.sub(r"([,;:.])\s*", r"\1 ", s)
        # Collapse repeated punctuation
        s = re.sub(r"([,;:.])(?:\s*\1)+", r"\1", s)
        # Trim trailing punctuation leftovers
        s = s.strip(" ,;:.")

    # 6) Collapse whitespace
    if cfg.collapse_whitespace:
        s = re.sub(r"\s{2,}", " ", s).strip()

    if not s:
        return ""

    # Build reusable word-boundary patterns
    def _word_alt(tokens: Sequence[str]) -> str:
        # Escape literal tokens; allow accents already embedded.
        toks = [re.escape(t) for t in tokens if t and t.strip()]
        return "|".join(toks) if toks else r"(?!x)x"

    connectors_alt = _word_alt(cfg.connectors)
    junk_alt = _word_alt(cfg.junk_tokens)

    # 7) Remove standalone junk tokens (only when they appear as whole words)
    if cfg.junk_tokens:
        s = re.sub(rf"(?i)\b(?:{junk_alt})\b", "", s)
        s = re.sub(r"\s{2,}", " ", s).strip(" ,;:.-")
        # Remove any hashtag token (#word, #word123)
        s = re.sub(r"(?i)(?:^|\s)#\w+", "", s)


    # 8) Remove leading/trailing connectors repeatedly (e.g. "en", "de", "del", etc.)
    if cfg.strip_leading_connectors and cfg.connectors:
        # Leading: "en la", "de", etc. Repeat until stable.
        prev = None
        while prev != s:
            prev = s
            s = re.sub(rf"(?i)^(?:\b(?:{connectors_alt})\b\s+)+", "", s).strip()

    if cfg.strip_trailing_connectors and cfg.connectors:
        prev = None
        while prev != s:
            prev = s
            s = re.sub(rf"(?i)\s+(?:\b(?:{connectors_alt})\b)+$", "", s).strip()

    # 9) Token-level cleanup: drop isolated letters when likely noise
    if cfg.drop_isolated_letters:
        whitelist = {w.lower() for w in (cfg.isolated_letter_whitelist or [])}

        tokens = s.split(" ")
        cleaned = []
        for i, tok in enumerate(tokens):
            t = tok.strip()

            # Keep empty tokens out
            if not t:
                continue

            # Detect single-letter token (alphabetic) like "A" or "b"
            if len(t) == 1 and t.isalpha():
                if t.lower() in whitelist:
                    cleaned.append(t)
                    continue

                # Heuristic: keep if next token suggests it's meaningful (e.g., "Torre B", "Int A")
                # If previous token is a unit-ish cue, keep it.
                prev_tok = cleaned[-1].lower() if cleaned else ""
                next_tok = tokens[i + 1].lower() if i + 1 < len(tokens) else ""

                if prev_tok in {"int", "interior", "depto", "departamento", "torre", "edif", "edificio", "local", "bloque"}:
                    cleaned.append(t)
                    continue
                if next_tok and re.fullmatch(r"\d{1,6}", next_tok):
                    # "A 12" is usually noise, drop "A"
                    continue

                # Default: drop isolated letter
                continue

            cleaned.append(t)

        s = " ".join(cleaned).strip()

    # 10) Final punctuation/space normalization pass
    if cfg.normalize_punctuation:
        s = re.sub(r"\s+([,;:.])", r"\1", s)
        s = re.sub(r"([,;:.])\s*", r"\1 ", s)
        s = re.sub(r"\s{2,}", " ", s).strip(" ,;:.")

    # 11) Optional title-casing (keeps all-caps acronyms / short codes)
    if cfg.title_case and s:
        def smart_title(word: str) -> str:
            if re.fullmatch(r"[A-Z0-9]{2,}", word):  # acronyms / codes
                return word
            # Keep tokens containing digits as-is except first char
            if any(ch.isdigit() for ch in word):
                return word[:1].upper() + word[1:]
            return word[:1].upper() + word[1:].lower()

        s = " ".join(smart_title(w) for w in s.split())

    # Mantain unique words for address formatting 
    s = unique_words_preserve_case(s)

    # Place formal address for city if not address 
    if kind == "PLACE":
        s = s.replace(city,"") + f", {city}"
    
    return s




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
        f"#eventos{city}", f"#agenda{city}",
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
