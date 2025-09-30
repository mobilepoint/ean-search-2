# -*- coding: utf-8 -*-
import re, time, hashlib
import pandas as pd
import streamlit as st
import requests
from io import BytesIO

try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

st.set_page_config(page_title="GTIN/EAN Finder via Google CSE (Excel)", layout="wide")
st.title("GTIN/EAN Finder via Google CSE (Excel)")

# --- Google ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_CSE_CX  = st.secrets.get("GOOGLE_CSE_CX")

# --- Supabase (din [supabase] în secrets.toml) ---
SUPABASE_URL = None
SUPABASE_KEY = None
if "supabase" in st.secrets:
    SUPABASE_URL = st.secrets["supabase"].get("url")
    SUPABASE_KEY = st.secrets["supabase"].get("anon_key") or st.secrets["supabase"].get("service_key")

SUPA_TABLE = "ean_progress"
SUPA_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY and create_client)
supa: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPA_ENABLED else None

# --- Sidebar: status ---
st.sidebar.header("Config")
st.sidebar.write("Google API:", "ON" if (GOOGLE_API_KEY and GOOGLE_CSE_CX) else "OFF")

def check_supabase_connection() -> tuple[str, str]:
    if not SUPA_ENABLED:
        return ("OFF", "Client inactiv sau chei lipsă")
    try:
        # probing ușor: dacă tabela există, returnează count
        resp = supa.table(SUPA_TABLE).select("id", count="exact").limit(1).execute()
        cnt = resp.count if hasattr(resp, "count") else (len(resp.data) if resp and resp.data else 0)
        return ("ON", f"Conectat. Tabel '{SUPA_TABLE}': ok, {cnt} rânduri+")
    except Exception as e:
        # conexiune OK dar tabel lipsă sau RLS/policy eroare
        try:
            # ping minimal pe schema: obținem timpul serverului via funcție standard (dacă există)
            _ = supa.auth.get_user()  # forțează cerere
            return ("ON", f"Conectat. Dar acces la '{SUPA_TABLE}' a eșuat: {e}")
        except Exception as e2:
            return ("OFF", f"Eroare conexiune: {e2}")

SUPA_STATUS, SUPA_MSG = check_supabase_connection()
st.sidebar.write("Supabase:", SUPA_STATUS)
st.sidebar.caption(SUPA_MSG)

if "request_count" not in st.session_state:
    st.session_state["request_count"] = 0
if "synth_counter" not in st.session_state:
    st.session_state["synth_counter"] = 1

DAILY_LIMIT = 100

# ==============================
# CALCULATOR EAN-13
# ==============================
def ean13_check_digit(d12: str) -> int:
    if not re.fullmatch(r"\d{12}", d12):
        raise ValueError("d12 trebuie să aibă exact 12 cifre")
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
# ==============================

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
        st.warning("Cheile Google lipsesc.")
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
        query_status.write(f"Query trimis: {q}")
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

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="EANs")
    return output.getvalue()

# ---------- Supabase helpers ----------
def supa_upsert_row(file_id: str, row_index: int, sku: str, name: str, target_col: str, ean: str | None, note: str, status: str):
    if not supa: return
    payload = {
        "file_id": file_id,
        "row_index": int(row_index),
        "sku": sku,
        "name": name,
        "target_col": target_col,
        "ean": ean,
        "note": note,
        "status": status,
    }
    try:
        supa.table(SUPA_TABLE).upsert(payload, on_conflict="file_id,row_index").execute()
    except Exception as e:
        st.sidebar.error(f"Supabase upsert fail idx {row_index}: {e}")

def supa_fetch_progress(file_id: str) -> dict[int, dict]:
    if not supa: return {}
    try:
        data = supa.table(SUPA_TABLE).select("row_index,ean,note,status").eq("file_id", file_id).execute().data or []
        return {int(r["row_index"]): {"ean": r.get("ean"), "note": r.get("note"), "status": r.get("status")} for r in data}
    except Exception as e:
        st.sidebar.error(f"Supabase fetch fail: {e}")
        return {}

def file_hash_id(uploaded_file_bytes: bytes) -> str:
    return hashlib.sha1(uploaded_file_bytes).hexdigest()

# ===== UI principal =====
st.sidebar.header("Quota Google API")
st.sidebar.write("Requests in session:", st.session_state["request_count"])
st.sidebar.write("Estimated daily free limit:", DAILY_LIMIT)

uploaded = st.file_uploader("Încarcă fișier Excel", type=["xls", "xlsx"])
if uploaded:
    file_bytes = uploaded.getvalue()
    file_id = file_hash_id(file_bytes)
    st.sidebar.write("File ID:", file_id[:12])

    try:
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    except Exception as e:
        st.error(f"Eroare la citirea Excel: {e}")
        st.stop()

    st.write("Previzualizare:", df.head(10))
    cols = list(df.columns)
    col_sku    = st.selectbox("Coloană SKU", cols, index=0)
    col_name   = st.selectbox("Coloană Denumire", cols, index=1 if len(cols)>1 else 0)
    col_target = st.selectbox("Coloană țintă pentru EAN-13", cols, index=len(cols)-1)
    mode       = st.radio("Cum cauți EAN?", ["Doar SKU", "Doar Nume"])

    synth_mode = st.radio("Completează cu EAN sintetic dacă nu găsește prin Google?", ["Nu", "Da"])

    note_col = "EAN_NOTE"
    if note_col not in df.columns:
        df[note_col] = ""

    mode_rows = st.radio("Ce rânduri procesezi?", ["Primele N rânduri", "Toate rândurile"])
    if mode_rows == "Primele N rânduri":
        max_rows = st.number_input("N rânduri de procesat", 1, len(df), min(50, len(df)))
    else:
        max_rows = len(df)

    # progres din Supabase
    progress = supa_fetch_progress(file_id)
    for idx, rec in progress.items():
        if idx in df.index:
            if rec.get("ean"):  df.at[idx, col_target] = rec["ean"]
            if rec.get("note"): df.at[idx, note_col]   = rec["note"]

    used_eans = set(x for x in df[col_target].astype(str).tolist() if is_valid_ean13(str(x)))

    if st.button("Pornește căutarea EAN"):
        done = 0; bar = st.progress(0); status = st.empty(); query_status = st.empty()
        iterable = df.head(int(max_rows)).iterrows() if mode_rows == "Primele N rânduri" else df.iterrows()
        total = int(max_rows) if mode_rows == "Primele N rânduri" else len(df)

        for idx, row in iterable:
            sku  = str(row.get(col_sku,"")).strip()
            name = str(row.get(col_name,"")).strip()
            current = str(row.get(col_target,"")).strip()
            note    = str(row.get(note_col,"")).strip().lower()

            prev = progress.get(int(idx), {})
            if prev.get("status") in {"done","skipped"}:
                done += 1; bar.progress(int(done*100/total)); continue

            if (current and is_valid_ean13(current)) or current.upper()=="NOT_FOUND" or note=="synthetic":
                supa_upsert_row(file_id, idx, sku, name, col_target, current if current else None, note or ("not_found" if current.upper()=="NOT_FOUND" else "found"), "skipped")
                done += 1; bar.progress(int(done*100/total)); continue

            found = lookup(mode, sku, name, query_status)

            if found and is_valid_ean13(found):
                df.at[idx, col_target] = found
                df.at[idx, note_col]   = "found"
                used_eans.add(found)
                supa_upsert_row(file_id, idx, sku, name, col_target, found, "found", "done")
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
                        supa_upsert_row(file_id, idx, sku, name, col_target, code, "synthetic", "done")
                    else:
                        df.at[idx, col_target] = "NOT_FOUND"
                        df.at[idx, note_col]   = "gen_error"
                        supa_upsert_row(file_id, idx, sku, name, col_target, None, "gen_error", "error")
                else:
                    df.at[idx, col_target] = "NOT_FOUND"
                    df.at[idx, note_col]   = "not_found"
                    supa_upsert_row(file_id, idx, sku, name, col_target, None, "not_found", "done")

            done += 1
            if done % 5 == 0:
                status.write(f"Procesate: {done}/{total}")
            bar.progress(int(done*100/total)); time.sleep(0.15)

        st.success(f"Terminat. Rânduri procesate: {done}.")
        excel_data = to_excel_bytes(df)
        st.download_button(
            "Descarcă Excel completat",
            data=excel_data,
            file_name="output_ean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download-ean"
        )
