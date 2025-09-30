# -*- coding: utf-8 -*-
import re, time, hashlib
import pandas as pd
import streamlit as st
import requests
from io import BytesIO

st.set_page_config(page_title="GTIN/EAN Finder via Google CSE (Excel)", layout="wide")
st.title("GTIN/EAN Finder via Google CSE (Excel)")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_CSE_CX = st.secrets.get("GOOGLE_CSE_CX")

# sidebar info
st.sidebar.header("Config check")
st.sidebar.write("API Key loaded:", bool(GOOGLE_API_KEY))
st.sidebar.write("CX loaded:", bool(GOOGLE_CSE_CX))

if "request_count" not in st.session_state:
    st.session_state["request_count"] = 0
DAILY_LIMIT = 100

# ==============================
# CALCULATOR EAN-13
# ==============================
def ean13_check_digit(d12: str) -> int:
    """
    d12 = exact 12 cifre (string). Returnează cifra de control EAN-13.
    Regulă: poziții impare *1, poziții pare *3 (numerotare de la stânga, 1..12).
    """
    if not re.fullmatch(r"\d{12}", d12):
        raise ValueError("d12 trebuie să aibă exact 12 cifre")
    s = 0
    for i, ch in enumerate(d12, start=1):
        digit = ord(ch) - 48
        s += digit * (3 if i % 2 == 0 else 1)
    return (10 - (s % 10)) % 10

def make_ean13(d12: str) -> str:
    """Construiește EAN-13 complet din primele 12 cifre."""
    return d12 + str(ean13_check_digit(d12))

def is_valid_ean13(code: str) -> bool:
    """Verifică lungime 13 și cifră de control corectă."""
    return bool(re.fullmatch(r"\d{13}", code)) and int(code[-1]) == ean13_check_digit(code[:12])

def upc12_to_gtin13(upc12: str) -> str:
    """Transformă UPC-A (12 cifre) în GTIN-13 prin prefix 0 și recalcul check digit."""
    if not re.fullmatch(r"\d{12}", upc12):
        raise ValueError("UPC trebuie să aibă exact 12 cifre")
    return make_ean13("0" + upc12)

def ean13_from_seed(seed: str) -> str:
    """
    Generează un EAN-13 VALID dintr-un seed text:
    - ia 12 cifre din SHA-256(seed)
    - calculează cifra de control
    """
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
            try:
                gt = upc12_to_gtin13(d)
                if gt: out.append(gt)
            except: pass
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

# ===== UI principal =====
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

    if st.button("Testează generatorul EAN"):
        seed = "SKU123|Ecran Samsung A04s"
        st.write("Exemplu synthetic:", ean13_from_seed(seed))

    if st.button("Pornește căutarea EAN"):
        done = 0; bar = st.progress(0); status = st.empty(); query_status = st.empty()
        for idx, row in df.iterrows():
            sku, name = str(row.get(col_sku,"")).strip(), str(row.get(col_name,"")).strip()
            current = str(row.get(col_target,"")).strip()
            if current and is_valid_ean13(current):
                continue
            found = lookup(mode, sku, name, query_status)
            if found and is_valid_ean13(found):
                df.at[idx, col_target] = found
        st.success("Terminat")
        st.download_button(
            "Descarcă Excel completat",
            data=to_excel_bytes(df),
            file_name="output_ean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
