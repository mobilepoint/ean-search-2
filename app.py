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
    gt = "0" + upc12
    return gt if is_valid_ean13(gt) else None

def ean13_from_seed(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    core12 = str(int(h, 16) % (10**12)).zfill(12)
    return make_ean13(core12)

def clean_digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

EAN_RE = re.compile(r"\b(?:\d[ \t\-]?){12,14}\b")

def find_eans_in_text(text: str):
    out, seen = [], set()
    if not text: return out
    for m in EAN_RE.finditer(text):
        d = clean_digits(m.group(0))
        if len(d) == 13 and is_valid_ean13(d):
            if d not in seen: out.append(d); seen.add(d)
        elif len(d) == 12:
            gt = upc12_to_gtin13(d)
            if gt and gt not in seen: out.append(gt); seen.add(gt)
    return out

def choose_best_ean(texts_with_weights):
    scores = {}
    for text, w in texts_with_weights:
        for c in find_eans_in_text(text):
            base = 1.0
            if re.search(r"(ean|gtin|barcode|cod\s*ean|ean-13)", (text or "").lower()):
                base += 1.0
            scores[c] = scores.get(c, 0.0) + base * w
    return max(scores.items(), key=lambda kv: kv[1])[0] if scores else None

# ===== Google CSE =====
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

def lookup(mode: str, sku: str, name: str, query_status, max_urls: int = 5):
    if mode == "Doar SKU":
        queries = [f'"{sku}" ean', f'"{sku}" gtin']
    else:
        base = re.sub(r"\s+", " ", name or "").strip()
        queries = [f'"{base}" ean', f'"{base}" gtin']
    texts = []
    for q in queries:
        query_status.write(f"Query: {q}")
        items = google_search(q, num=max_urls)
        for rank, it in enumerate(items):
            w = 1.0 + (max_urls - rank) * 0.1
            texts.append((it.get("snippet",""), w))
            link = it.get("link")
            if link:
                page = fetch_url_text(link)
                if page: texts.append((page, w+0.5))
        best = choose_best_ean(texts)
        if best: return best
    return choose_best_ean(texts)

# ===== Supabase helpers =====
def count_total():
    res = supa.table(TBL).select("id", count="exact").execute()
    return res.count or 0

def count_pending():
    res = supa.table(TBL).select("id", count="exact").or_("ean.is.null,ean.eq.").execute()
    return res.count or 0

def count_status(val: str):
    res = supa.table(TBL).select("id", count="exact").eq("ean_note", val).execute()
    return res.count or 0

def fetch_pending_batch(limit: int, offset: int = 0):
    res = (supa.table(TBL).select("id,sku,name")
           .or_("ean.is.null,ean.eq.")
           .order("id", desc=False)
           .range(offset, offset + limit - 1)
           .execute())
    return res.data or []

def write_result(id_val, sku, name, ean_value, note_value):
    payload = {
        "sku": sku if sku else None,  # sku poate fi NULL
        "name": name,
        "ean": ean_value,
        "ean_note": note_value
    }
    supa.table(TBL).update(payload).eq("id", id_val).execute()

def supa_fetch_all_with_ean():
    return (supa.table(TBL)
            .select("id,sku,name,ean,ean_note")
            .not_("ean","is","null").neq("ean","")
            .execute().data or [])

def dedup_eans():
    rows = supa_fetch_all_with_ean()
    if not rows:
        return {"groups": 0, "changed": 0}

    by_ean = defaultdict(list)
    for r in rows:
        e = str(r.get("ean") or "")
        if is_valid_ean13(e):
            by_ean[e].append(r)

    existing = set(by_ean.keys())

    def prio(note):
        note = (note or "").lower()
        if note == "found": return 3
        if note.startswith("synthetic"): return 2
        if note == "not_found": return 1
        return 0

    groups = 0
    changed = 0

    for ean_val, group in by_ean.items():
        if len(group) <= 1:
            continue
        groups += 1

        group_sorted = sorted(group, key=lambda r: (-prio(r.get("ean_note")), int(r.get("id") or 0)))
        # păstrăm primul (prioritate mai mare); rescriem restul
        for r in group_sorted[1:]:
            rid   = r["id"]
            sku   = r.get("sku")
            name  = r.get("name")
            note0 = (r.get("ean_note") or "").lower()

            seed = f"dedup|{rid}|{sku}|{name}"
            new_ean = ean13_from_seed(seed)
            tries = 0
            while (new_ean in existing) and tries < 5000:
                tries += 1
                new_ean = ean13_from_seed(f"{seed}|{tries}")

            new_note = "synthetic_dedup" if note0 == "found" else "synthetic"

            supa.table(TBL).update({
                "sku": sku if sku else None,
                "name": name,
                "ean": new_ean,
                "ean_note": new_note
            }).eq("id", rid).execute()

            existing.add(new_ean)
            changed += 1

    return {"groups": groups, "changed": changed}

# ===== Sidebar: statistici + dedup =====
if SUPA_OK:
    total = count_total()
    pending = count_pending()
    with_ean = total - pending
    st.sidebar.header("Statistici")
    st.sidebar.write("Total:", total)
    st.sidebar.write("Cu EAN:", with_ean)
    st.sidebar.write("Fără EAN:", pending)
    st.sidebar.write("found:", count_status("found"))
    st.sidebar.write("synthetic:", count_status("synthetic"))
    st.sidebar.write("not_found:", count_status("not_found"))
    st.sidebar.write("gen_error:", count_status("gen_error"))
    st.sidebar.write("Google requests (sesiune):", st.session_state["request_count"])

st.sidebar.markdown("### Deduplicare EAN")
if st.sidebar.button("Detectează și repară duplicatele"):
    if not SUPA_OK:
        st.sidebar.error("Supabase indisponibil.")
    else:
        with st.spinner("Rulez deduplicarea..."):
            s = dedup_eans()
        st.sidebar.success(f"Grupuri: {s['groups']}, rescrise: {s['changed']}")

# ===== Controale rulare =====
mode = st.radio("Caută EAN după:", ["Doar SKU", "Doar Nume"])
synth_mode = st.radio("Dacă nu găsește, generează EAN sintetic:", ["Nu", "Da"])
how_many = st.radio("Procesează:", ["Primele N fără EAN", "Toate fără EAN"])
N = st.number_input("N", min_value=1, max_value=1_000_000, value=500) if how_many == "Primele N fără EAN" else None

if st.button("Pornește procesarea"):
    if not SUPA_OK:
        st.error("Supabase indisponibil.")
        st.stop()

    total_pending = count_pending()
    target = total_pending if N is None else min(total_pending, int(N))

    processed = 0
    page = 200
    bar = st.progress(0)
    status = st.empty()
    qstat = st.empty()
    offset = 0

    while processed < target:
        take = min(page, target - processed)
        batch = fetch_pending_batch(limit=take, offset=offset)
        if not batch:
            break

        for rec in batch:
            id_val = rec.get("id")
            sku    = (rec.get("sku") or "").strip()
            name   = (rec.get("name") or "").strip()

            if id_val is None:
                processed += 1
                bar.progress(int(processed * 100 / max(1, target)))
                continue

            found = lookup(mode, sku, name, qstat)

            if found and is_valid_ean13(found):
                write_result(id_val, sku, name, found, "found")
            else:
                if synth_mode == "Da":
                    seed = f"{sku}|{name}"
                    code = ean13_from_seed(seed)
                    write_result(id_val, sku, name, code, "synthetic")
                else:
                    write_result(id_val, sku, name, "NOT_FOUND", "not_found")

            processed += 1
            if processed % 10 == 0:
                status.write(f"Procesate: {processed}/{target}")
            bar.progress(int(processed * 100 / max(1, target)))
            time.sleep(0.1)

        offset += take

    st.success(f"Final: procesate {processed} din {target}.")
    if SUPA_OK:
        st.sidebar.write("—")
        st.sidebar.write("Cu EAN (după run):", count_total() - count_pending())
