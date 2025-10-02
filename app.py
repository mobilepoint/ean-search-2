# -*- coding: utf-8 -*-
import re, time, hashlib
import requests
import streamlit as st
from collections import defaultdict

# ===== Supabase client =====
try:
    from supabase import create_client
except Exception:
    create_client = None

st.set_page_config(page_title="GTIN/EAN Finder pe Supabase", layout="wide")
st.title("GTIN/EAN Finder (Google CSE) → scriere directă în Supabase")

# ===== Secrete =====
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_CSE_CX  = st.secrets.get("GOOGLE_CSE_CX")

SUPABASE_URL = st.secrets.get("supabase", {}).get("url")
SUPABASE_KEY = st.secrets.get("supabase", {}).get("anon_key") or st.secrets.get("supabase", {}).get("service_key")
SUPA_OK = bool(SUPABASE_URL and SUPABASE_KEY and create_client)
supa = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPA_OK else None
TBL = "ean_progress"

# ===== Sidebar: conexiuni =====
st.sidebar.header("Conexiuni")
st.sidebar.write("Google API:", "ON" if (GOOGLE_API_KEY and GOOGLE_CSE_CX) else "OFF")
if SUPA_OK:
    try:
        supa.table(TBL).select("id").limit(1).execute()
        st.sidebar.write("Supabase:", "ON")
    except Exception as e:
        st.sidebar.write("Supabase:", f"Eroare: {e}")
else:
    st.sidebar.write("Supabase:", "OFF")

if "request_count" not in st.session_state:
    st.session_state["request_count"] = 0

# ===== EAN utils =====
def ean13_check_digit(d12: str) -> int:
    if not re.fullmatch(r"\d{12}", d12):
        raise ValueError("d12 trebuie 12 cifre")
    s = 0
    for i, ch in enumerate(d12, start=1):
        d = ord(ch) - 48
        s += d * (3 if i % 2 == 0 else 1)
    return (10 - (s % 10)) % 10

def make_ean13(d12: str) -> str:
    return d12 + str(ean13_check_digit(d12))

def is_valid_ean13(code: str) -> bool:
    return bool(re.fullmatch(r"\d{13}", code)) and int(code[-1]) == ean13_check_digit(code[:12])

def upc12_to_gtin13(upc12: str):
    if not re.fullmatch(r"\d{12}", upc12):
        return None
    return make_ean13("0" + upc12)

def clean_digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

# seed → EAN-13 valid, prefix fix (ex: 594)
def ean13_with_prefix(prefix3: str, payload9: str) -> str:
    if not re.fullmatch(r"\d{3}", prefix3):
        raise ValueError("prefix3 trebuie 3 cifre")
    if not re.fullmatch(r"\d{9}", payload9):
        raise ValueError("payload9 trebuie 9 cifre")
    core12 = prefix3 + payload9
    return make_ean13(core12)

def ean13_from_seed_prefix(seed: str, used: set, prefix3: str = "594") -> str:
    base = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % (10**9)
    for i in range(1000000):
        payload9 = str((base + i) % (10**9)).zfill(9)
        e = ean13_with_prefix(prefix3, payload9)
        if e not in used:
            return e
    # fallback extrem
    return ean13_with_prefix(prefix3, "0"*9)

# ===== Google CSE (nemodificat) =====
def google_search(query: str, num: int = 5):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_CX:
        return []
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_CX, "q": query, "num": min(num, 10), "safe": "off"}
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=12)
    st.session_state["request_count"] += 1
    if r.status_code != 200:
        return []
    return r.json().get("items", []) or []

def fetch_url_text(url: str, timeout: int = 12) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200: return ""
        txt = re.sub(r"<[^>]+>", " ", r.text, flags=re.DOTALL)
        return re.sub(r"\s+", " ", txt)
    except Exception:
        return ""

# ===== Supabase helpers =====
def supa_fetch_all_rows(page_size: int = 1000):
    out, start = [], 0
    while True:
        q = (supa.table(TBL)
             .select("id,sku,name,ean,ean_note")
             .order("id", desc=False)
             .range(start, start + page_size - 1))
        res = q.execute()
        chunk = res.data or []
        if not chunk: break
        out.extend(chunk)
        if len(chunk) < page_size: break
        start += page_size
    return out

# ===== Deduplicare =====
def dedup_eans():
    rows = supa_fetch_all_rows()
    if not rows:
        return {"groups": 0, "changed": 0}

    groups = defaultdict(list)
    used = set()

    for r in rows:
        e = str(r.get("ean") or "").strip()
        if e and e.upper() != "NOT_FOUND":
            groups[e].append(r)
            used.add(e)

    def prio(note: str) -> int:
        note = (note or "").lower()
        if note == "found": return 3
        if note.startswith("synthetic"): return 2
        if note == "not_found": return 1
        return 0

    dup_groups = 0
    changed = 0

    for _, grp in groups.items():
        if len(grp) <= 1: continue
        dup_groups += 1
        grp_sorted = sorted(grp, key=lambda r: (-prio(r.get("ean_note")), int(r.get("id") or 0)))
        keeper = grp_sorted[0]  # păstrat

        for r in grp_sorted[1:]:
            rid, sku, name = r["id"], r.get("sku"), r.get("name")
            seed = f"dedup|{rid}|{sku}|{name}"
            cand = ean13_from_seed_prefix(seed, used, prefix3="594")
            supa.table(TBL).update({
                "sku": sku if sku else None,
                "name": name,
                "ean": cand,
                "ean_note": "synthetic_dedup"
            }).eq("id", rid).execute()
            used.add(cand); changed += 1

    return {"groups": dup_groups, "changed": changed}

# ===== Normalizare & verificare EAN (existent) =====
def normalize_and_fix_eans():
    rows = supa_fetch_all_rows()
    fixed, total = 0, 0
    used = set()

    # colectează deja folosite (curente)
    for r in rows:
        e = str(r.get("ean") or "").strip()
        if e and e.upper() != "NOT_FOUND":
            used.add(e)

    for r in rows:
        rid  = r["id"]
        raw  = str(r.get("ean") or "").strip()
        note = r.get("ean_note") or ""

        if not raw or raw.upper() == "NOT_FOUND":
            continue

        total += 1
        digits = clean_digits(raw)
        new_val = None
        base_seed = f"fix|{rid}|{r.get('sku')}|{r.get('name')}"

        if len(digits) == 13 and is_valid_ean13(digits):
            new_val = digits
        elif len(digits) == 13:
            new_val = make_ean13(digits[:12])
        elif len(digits) == 12:
            new_val = make_ean13(digits)
        else:
            new_val = ean13_from_seed_prefix(base_seed, used, "594")

        # unicități
        i = 0
        while new_val in used:
            i += 1
            new_val = ean13_from_seed_prefix(f"{base_seed}|{i}", used, "594")

        if new_val != raw:
            supa.table(TBL).update({
                "ean": new_val,
                "ean_note": note if note else "fixed"
            }).eq("id", rid).execute()
            fixed += 1
        used.add(new_val)

    return {"checked": total, "fixed": fixed}

# ===== NOU: Re-codare globală cu prefix 594, fără dubluri =====
def recode_all_to_prefix_594():
    """
    Parcurge TOT tabelul și rescrie *toate* EAN-urile la prefix 594.
    Generare stabilă din seed (id|sku|name), EAN-13 valid, fără duplicate globale.
    Setează ean_note = 'synthetic_594' pentru rândurile modificate.
    Returnează: total, updated, dup_after, invalid_after.
    """
    rows = supa_fetch_all_rows()
    used = set()
    total = len(rows)
    updated = 0

    # construiește setul 'used' cu EAN-urile țintă pe măsură ce generăm noi valori,
    # ca să garantăm unicitatea globală în noul spațiu 594.
    for r in rows:
        rid = r["id"]
        sku = r.get("sku")
        name = r.get("name")
        seed = f"recode594|{rid}|{sku}|{name}"
        new_ean = ean13_from_seed_prefix(seed, used, prefix3="594")
        old_ean = str(r.get("ean") or "").strip()

        if new_ean != old_ean:
            supa.table(TBL).update({
                "ean": new_ean,
                "ean_note": "synthetic_594"
            }).eq("id", rid).execute()
            updated += 1
        used.add(new_ean)

    # verificare după recodare
    rows2 = supa_fetch_all_rows()
    invalid_after = sum(1 for r in rows2
                        if str(r.get("ean") or "").strip()
                        and not is_valid_ean13(str(r.get("ean")).strip()))
    # duplicate count
    from collections import Counter
    eans = [str(r.get("ean") or "").strip() for r in rows2 if str(r.get("ean") or "").strip()]
    dup_after = sum(1 for _, c in Counter(eans).items() if c > 1)

    return {
        "total": total,
        "updated": updated,
        "invalid_after": invalid_after,
        "dup_groups_after": dup_after
    }

# ===== Sidebar: statistici + dedup + fix + recode 594 =====
st.sidebar.markdown("### Operațiuni pe tabel")

if st.sidebar.button("Detectează și repară duplicatele"):
    with st.spinner("Rulez deduplicarea..."):
        s = dedup_eans()
    st.sidebar.success(f"Grupuri duplicate: {s['groups']}. Rânduri rescrise: {s['changed']}")

if st.sidebar.button("Normalizează & verifică toate EAN"):
    with st.spinner("Rulez normalizarea și verificarea..."):
        s = normalize_and_fix_eans()
    st.sidebar.success(f"Verificate: {s['checked']}, Reparări: {s['fixed']}")

# ► Butonul nou:
if st.sidebar.button("Recodează TOT la prefix 594"):
    if not SUPA_OK:
        st.sidebar.error("Supabase indisponibil.")
    else:
        with st.spinner("Rescriu toate EAN-urile cu prefix 594 și elimin dublurile..."):
            s = recode_all_to_prefix_594()
        st.sidebar.success(
            f"Total: {s['total']}, Actualizate: {s['updated']}, "
            f"Invalid după: {s['invalid_after']}, Grupuri duplicate după: {s['dup_groups_after']}"
        )
