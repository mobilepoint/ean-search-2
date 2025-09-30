# -*- coding: utf-8 -*-
import re, time, hashlib
import pandas as pd
import streamlit as st
import requests
from io import BytesIO

# ---- Supabase client ----
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

st.set_page_config(page_title="GTIN/EAN Finder (Excel) + Supabase", layout="wide")
st.title("GTIN/EAN Finder (Google CSE) + Persistență Supabase")

# ---- Secrete ----
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_CSE_CX  = st.secrets.get("GOOGLE_CSE_CX")

SUPABASE_URL = st.secrets.get("supabase", {}).get("url")
SUPABASE_KEY = st.secrets.get("supabase", {}).get("anon_key") or st.secrets.get("supabase", {}).get("service_key")
SUPA_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY and create_client)
supa: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPA_ENABLED else None
SUPA_TABLE = "ean_progress"

# ---- Sidebar status ----
st.sidebar.header("Status")
st.sidebar.write("Google API:", "ON" if (GOOGLE_API_KEY and GOOGLE_CSE_CX) else "OFF")
def ping_supabase():
    if not SUPA_ENABLED: return ("OFF", "chei lipsă sau client indisponibil")
    try:
        supa.table(SUPA_TABLE).select("*").limit(1).execute()
        return ("ON", f"Conectat la '{SUPA_TABLE}'")
    except Exception as e:
        return ("ON", f"Conectat, dar acces tabel: {e}")
SUPA_STATUS, SUPA_MSG = ping_supabase()
st.sidebar.write("Supabase:", SUPA_STATUS)
st.sidebar.caption(SUPA_MSG)

if "request_count" not in st.session_state:
    st.session_state["request_count"] = 0

# ---- EAN utils ----
def ean13_check_digit(d12: str) -> int:
    if not re.fullmatch(r"\d{12}", d12):
        raise ValueError("d12 trebuie să aibă 12 cifre")
    s = 0
    for i, ch in enumerate(d12, start=1):
        digit = ord(ch) - 48
        s += digit * (3 if i % 2 == 0 else 1)
    return (10 - (s % 10)) % 10

def make_ean13(d12: str) -> str:
    return d12 + str(ean13_check_digit(d12))

def is_valid_ean13(code: str) -> bool:
    return bool(re.fullmatch(r"\d{13}", code)) and int(code[-1]) == ean13_check_digit(code[:12])

def upc12_to_gtin13(upc12: str) -> str | None:
    if not re.fullmatch(r"\d{12}", upc12):
        return None
    gtin13 = "0" + upc12
    return gtin13 if is_valid_ean13(gtin13) else None

def ean13_from_seed(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    core12 = str(int(h, 16) % (10**12)).zfill(12)
    return make_ean13(core12)

def clean_digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

EAN_RE = re.compile(r"\b(?:\d[ \t\-]?){12,14}\b")

def find_eans_in_text(text: str):
    out = []
    for m in EAN_RE.finditer(text or ""):
        d = clean_digits(m.group(0))
        if len(d) == 13 and is_valid_ean13(d):
            out.append(d)
        elif len(d) == 12:
            gt = upc12_to_gtin13(d)
            if gt: out.append(gt)
    seen, res = set(), []
    for x in out:
        if x not in seen:
            res.append(x); seen.add(x)
    return res

def choose_best_ean(texts_with_weights):
    scores = {}
    for text, w in texts_with_weights:
        for c in find_eans_in_text(text):
            base = 1.0
            if re.search(r"(ean|gtin|barcode|cod\s*ean|ean-13)", text.lower()):
                base += 1.0
            scores[c] = scores.get(c, 0.0) + base * w
    return max(scores.items(), key=lambda kv: kv[1])[0] if scores else None

def google_search(query: str, num: int = 5):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_CX:
        return []
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_CX, "q": query, "num": min(num, 10), "safe": "off"}
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=12)
    st.session_state["request_count"] += 1
    if r.status_code != 200: return []
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
        queries = [f'"{name}" ean', f'"{name}" gtin']
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

def to_excel_bytes(df: pd.DataFrame, col_target: str, note_col: str) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="EANs")
    return output.getvalue()

# ---- Supabase helpers (tabel: id, sku, name, ean, ean_note) ----
def supa_upsert(id_val, sku, name, ean, ean_note):
    if not supa: return
    payload = {
        "id": None if (id_val is None or (isinstance(id_val, float) and pd.isna(id_val))) else int(id_val),
        "sku": sku,
        "name": name,
        "ean": ean,
        "ean_note": ean_note,
    }
    try:
        supa.table(SUPA_TABLE).upsert(payload, on_conflict="id,sku").execute()
    except Exception as e:
        st.sidebar.error(f"Supabase upsert fail (id={id_val}, sku={sku}): {e}")

def supa_fetch_all() -> dict[tuple, dict]:
    if not supa: return {}
    try:
        data = supa.table(SUPA_TABLE).select("id,sku,name,ean,ean_note").execute().data or []
        return {(r.get("id"), r.get("sku")): r for r in data}
    except Exception as e:
        st.sidebar.error(f"Supabase fetch fail: {e}")
        return {}

# ---- UI principal ----
st.sidebar.subheader("Quota Google")
st.sidebar.write("Requests în sesiune:", st.session_state["request_count"])

uploaded = st.file_uploader("Încarcă fișier Excel", type=["xls", "xlsx"])
if uploaded:
    try:
        df = pd.read_excel(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"Eroare la citirea Excel: {e}")
        st.stop()

    st.write("Previzualizare:", df.head(10))
    cols = list(df.columns)

    def idx_of(name, fallback=0):
        try: return cols.index(name)
        except ValueError: return fallback

    col_id    = st.selectbox("Coloană ID", cols, index=idx_of("ID", 0))
    col_sku   = st.selectbox("Coloană SKU", cols, index=idx_of("SKU", 1 if len(cols)>1 else 0))
    col_name  = st.selectbox("Coloană Nume", cols, index=idx_of("Name", 2 if len(cols)>2 else 0))
    col_target= st.selectbox("Coloană EAN", cols, index=idx_of("GTIN, UPC, EAN, or ISBN", len(cols)-1))

    mode = st.radio("Caută EAN după:", ["Doar SKU", "Doar Nume"])
    synth_mode = st.radio("Dacă nu găsește, generează EAN sintetic:", ["Nu", "Da"])

    note_col = "EAN_NOTE"
    if note_col not in df.columns:
        df[note_col] = ""

    mode_rows = st.radio("Rânduri procesate:", ["Primele N", "Toate"])
    max_rows = st.number_input("N=", 1, len(df), min(50,len(df))) if mode_rows=="Primele N" else len(df)

    # preluare ce e deja salvat (evită recăutarea)
    existing = supa_fetch_all()
    for idx in df.index:
        key = (None if pd.isna(df.at[idx, col_id]) else int(df.at[idx, col_id]), str(df.at[idx, col_sku]))
        rec = existing.get(key)
        if rec:
            if rec.get("ean"):      df.at[idx, col_target] = rec["ean"]
            if rec.get("ean_note"): df.at[idx, note_col]   = rec["ean_note"]
            if rec.get("name"):     df.at[idx, col_name]   = rec["name"]

    used_eans = set(x for x in df[col_target].astype(str).tolist() if is_valid_ean13(str(x)))

    if st.button("Pornește"):
        done = 0; bar = st.progress(0); status = st.empty(); qstat = st.empty()
        iterable = df.head(int(max_rows)).iterrows() if mode_rows=="Primele N" else df.iterrows()
        total = int(max_rows) if mode_rows=="Primele N" else len(df)

        for idx, row in iterable:
            id_val = row.get(col_id, None)
            sku    = str(row.get(col_sku, "")).strip()
            name   = str(row.get(col_name, "")).strip()
            current= str(row.get(col_target, "")).strip()
            note   = str(row.get(note_col, "")).strip().lower()

            # dacă avem deja în DB, sari
            key = (None if pd.isna(id_val) else int(id_val), sku)
            if key in existing and existing[key].get("ean"):
                done += 1; bar.progress(int(done*100/total)); continue

            # dacă EAN în fișier e valid sau marcat, scrie și sari
            if (current and is_valid_ean13(current)) or current.upper()=="NOT_FOUND" or note=="synthetic":
                supa_upsert(id_val, sku, name, current if current else None, note or ("not_found" if current.upper()=="NOT_FOUND" else "found"))
                done += 1; bar.progress(int(done*100/total)); continue

            found = lookup(mode, sku, name, qstat)

            if found and is_valid_ean13(found):
                df.at[idx, col_target] = found
                df.at[idx, note_col]   = "found"
                used_eans.add(found)
                supa_upsert(id_val, sku, name, found, "found")
            else:
                if synth_mode == "Da":
                    seed = f"{sku}|{name}"
                    code = ean13_from_seed(seed)
                    tries = 0
                    while (code in used_eans) and tries < 1000:
                        tries += 1
                        code = ean13_from_seed(f"{seed}|{tries}")
                    if is_valid_ean13(code) and code not in used_eans:
                        df.at[idx, col_target] = code
                        df.at[idx, note_col]   = "synthetic"
                        used_eans.add(code)
                        supa_upsert(id_val, sku, name, code, "synthetic")
                    else:
                        df.at[idx, col_target] = "NOT_FOUND"
                        df.at[idx, note_col]   = "gen_error"
                        supa_upsert(id_val, sku, name, None, "gen_error")
                else:
                    df.at[idx, col_target] = "NOT_FOUND"
                    df.at[idx, note_col]   = "not_found"
                    supa_upsert(id_val, sku, name, None, "not_found")

            done += 1
            if done % 5 == 0: status.write(f"Procesate: {done}/{total}")
            bar.progress(int(done*100/total)); time.sleep(0.15)

        st.success(f"Terminat. Rânduri procesate: {done}.")
        excel_data = to_excel_bytes(df, col_target, note_col)
        st.download_button("Descarcă Excel", data=excel_data, file_name="output_ean.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
