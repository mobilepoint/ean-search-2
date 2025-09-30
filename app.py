# -*- coding: utf-8 -*-
import re, time
from collections import Counter
import pandas as pd
import streamlit as st
import requests
from io import BytesIO

st.set_page_config(page_title="GTIN/EAN Finder via Google CSE (Excel)", layout="wide")
st.title("GTIN/EAN Finder via Google CSE (Excel)")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_CSE_CX = st.secrets.get("GOOGLE_CSE_CX")

# Sidebar info
st.sidebar.header("Config check")
st.sidebar.write("API Key loaded:", bool(GOOGLE_API_KEY))
st.sidebar.write("CX loaded:", bool(GOOGLE_CSE_CX))

if "request_count" not in st.session_state:
    st.session_state["request_count"] = 0
if "synth_counter" not in st.session_state:
    st.session_state["synth_counter"] = 1
DAILY_LIMIT = 100

# ---------- EAN helpers ----------
def clean_digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

def ean13_check_digit(d12: str) -> int:
    s = sum((ord(ch)-48) * (3 if i%2 else 1) for i, ch in enumerate(d12))
    return (10 - (s % 10)) % 10

def is_valid_ean13(code: str) -> bool:
    d = clean_digits(code)
    return len(d) == 13 and ean13_check_digit(d[:12]) == int(d[-1])

def upc12_to_gtin13(upc: str):
    d = clean_digits(upc)
    if len(d) != 12: return None
    cand = "0" + d
    return cand if is_valid_ean13(cand) else None

EAN_RE = re.compile(r"\b(?:\d[ \t\-]?){12,14}\b")

def find_eans_in_text(text: str):
    out = []
    for m in EAN_RE.finditer(text or ""):
        d = clean_digits(m.group(0))
        if len(d) == 13 and is_valid_ean13(d): out.append(d)
        elif len(d) == 12:
            gt = upc12_to_gtin13(d)
            if gt: out.append(gt)
    return list(dict.fromkeys(out))

def choose_best_ean(texts_with_weights):
    scores = {}
    for text, w in texts_with_weights:
        for c in find_eans_in_text(text):
            base = 1.0
            if re.search(r"(ean|gtin|barcode|cod\s*ean|ean-13)", text.lower()):
                base += 1.0
            scores[c] = scores.get(c, 0.0) + base * w
    return max(scores.items(), key=lambda kv: kv[1])[0] if scores else None

# ---------- Google CSE ----------
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
    except Exception: return ""

def lookup(mode: str, sku: str, name: str, query_status, preferred_site: str | None, max_urls: int = 5):
    queries_base = ( [f'"{sku}" ean', f'"{sku}" gtin'] if mode == "Doar SKU"
                     else [f'"{name}" ean', f'"{name}" gtin'] )
    queries = []
    if preferred_site:
        for q in queries_base:
            queries.append(f'site:{preferred_site} {q}')
    queries += queries_base  # fallback general

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

# ---------- Synthetic EAN ----------
def most_common_prefix(eans: list[str], length: int = 7) -> str | None:
    pref = [e[:length] for e in eans if len(e) >= length]
    if not pref: return None
    return Counter(pref).most_common(1)[0][0]

def generate_synthetic_ean(prefix: str, seq: int) -> str:
    # completează la 12 cifre, apoi calculează cifra de control
    core = f"{prefix}{seq:0{12-len(prefix)}d}"[:12]
    cd = ean13_check_digit(core)
    return core + str(cd)

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="EANs")
    return output.getvalue()

# ---------- UI ----------
st.sidebar.header("Quota Google API")
st.sidebar.write("Requests in session:", st.session_state["request_count"])
st.sidebar.write("Estimated daily free limit:", DAILY_LIMIT)

uploaded = st.file_uploader("Încarcă fișier Excel", type=["xls", "xlsx"])
if uploaded:
    try:
        df = pd.read_excel(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"Eroare la citirea Excel: {e}")
        st.stop()

    st.write("Previzualizare:", df.head(10))

    cols = list(df.columns)
    col_sku = st.selectbox("Coloană SKU", cols, index=0)
    col_name = st.selectbox("Coloană Denumire", cols, index=1 if len(cols)>1 else 0)
    col_target = st.selectbox("Coloană țintă pentru EAN-13", cols, index=len(cols)-1)
    mode = st.radio("Cum cauți EAN?", ["Doar SKU", "Doar Nume"])

    preferred_site = st.text_input("Site preferat (ex: gsmok.com), opțional", value="").strip() or None

    synth_mode = st.radio(
        "Completează automat când nu găsește?",
        ["Nu, lasă NOT_FOUND", "Da, generează EAN sintetic din prefixul găsit"]
    )

    mode_rows = st.radio("Ce rânduri procesezi?", ["Primele N rânduri", "Toate rândurile"])
    if mode_rows == "Primele N rânduri":
        max_rows = st.number_input("N rânduri de procesat", 1, len(df), min(50,len(df)))
    else:
        max_rows = len(df)

    # pregătește coloana de notițe
    note_col = "EAN_NOTE"
    if note_col not in df.columns:
        df[note_col] = ""

    # prefix cel mai frecvent (pentru generare)
    existing_valid = [str(v) for v in df[col_target].astype(str).tolist() if is_valid_ean13(str(v))]
    default_prefix = most_common_prefix(existing_valid, length=7) or "2000000"  # 20xxxxxx ca fallback intern

    if st.button("Pornește căutarea EAN"):
        done = 0; bar = st.progress(0); status = st.empty(); query_status = st.empty()

        for idx, row in df.head(int(max_rows)).iterrows():
            sku, name = str(row.get(col_sku,"")).strip(), str(row.get(col_name,"")).strip()
            current = str(row.get(col_target,"")).strip()
            note = str(row.get(note_col,"")).strip().lower()

            # Skip dacă are EAN valid, NOT_FOUND, sau e marcat synthetic
            if (current and is_valid_ean13(current)) or current.upper()=="NOT_FOUND" or note=="synthetic":
                done+=1; bar.progress(int(done*100/max_rows)); continue

            found = lookup(mode, sku, name, query_status, preferred_site)

            if found and is_valid_ean13(found):
                df.at[idx, col_target] = found
                df.at[idx, note_col] = "found"
            else:
                if synth_mode.endswith("sintetic"):
                    seq = st.session_state["synth_counter"]
                    st.session_state["synth_counter"] += 1
                    code = generate_synthetic_ean(default_prefix, seq)
                    df.at[idx, col_target] = code
                    df.at[idx, note_col] = "synthetic"
                else:
                    df.at[idx, col_target] = "NOT_FOUND"
                    df.at[idx, note_col] = "not_found"

            done+=1
            if done%5==0: status.write(f"Procesate: {done}/{int(max_rows)}")
            bar.progress(int(done*100/max_rows)); time.sleep(0.15)

        st.success(f"Terminat. Rânduri procesate: {done}.")
        excel_data = to_excel_bytes(df)
        st.download_button(
            "Descarcă Excel completat",
            data=excel_data,
            file_name="output_ean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download-ean"
        )
        st.markdown(
            """
            <script>
            const btn = window.parent.document.querySelector('button[data-testid="stDownloadButton-download-ean"]');
            if (btn) { btn.click(); }
            </script>
            """,
            unsafe_allow_html=True
        )
