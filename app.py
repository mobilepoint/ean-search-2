# -*- coding: utf-8 -*-
import re, time, hashlib
import pandas as pd
import streamlit as st
import requests
from io import BytesIO
from collections import Counter

st.set_page_config(page_title="GTIN/EAN Finder via Google CSE (Excel)", layout="wide")
st.title("GTIN/EAN Finder via Google CSE (Excel)")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_CSE_CX  = st.secrets.get("GOOGLE_CSE_CX")

# sidebar info
st.sidebar.header("Config check")
st.sidebar.write("API Key loaded:", bool(GOOGLE_API_KEY))
st.sidebar.write("CX loaded:", bool(GOOGLE_CSE_CX))

if "request_count" not in st.session_state:
    st.session_state["request_count"] = 0

DAILY_LIMIT = 100

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

# ---------- EAN sintetic corect ----------
def most_common_prefix(eans, min_len=6, max_len=9):
    vals = [clean_digits(str(e)) for e in eans if is_valid_ean13(str(e))]
    if not vals: return None
    # testează lungimi 9..6 și ia cea mai frecventă
    best = None; best_cnt = -1
    for L in range(max_len, min_len-1, -1):
        prefs = [v[:L] for v in vals]
        if not prefs: continue
        pref, cnt = Counter(prefs).most_common(1)[0]
        if cnt > best_cnt:
            best = pref; best_cnt = cnt
    return best

def hash_ref(s: str, digits: int) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h, 16) % (10**digits)

def build_synthetic_ean(sku: str, name: str, base_prefix: str | None, used: set[str]) -> str:
    # 1) prefix realist
    if base_prefix:
        prefix = clean_digits(base_prefix)[:11]
    else:
        # derivă un prefix stabil din date dacă nu există în fișier
        seed = (sku or "") + "|" + (name or "")
        d = str(hash_ref(seed, 7)).rjust(7, "2")  # 7 cifre, evită multe zerouri
        prefix = d
    if len(prefix) < 6:
        prefix = (prefix + "201234")[:6]
    if len(prefix) > 11:
        prefix = prefix[:11]

    # 2) lungimea referinței
    item_len = 12 - len(prefix)
    if item_len <= 0:
        prefix = prefix[:7]
        item_len = 5  # 7 + 5 = 12

    # 3) referință din hash(SKU|Name) pentru a evita zerourile
    seed = f"{sku}|{name}"
    ref = hash_ref(seed, item_len)

    # 4) căutare a unui cod liber în caz de coliziune
    space = 10**item_len
    for i in range(min(space, 10000)):
        core = f"{prefix}{(ref + i) % space:0{item_len}d}"
        cd = ean13_check_digit(core)
        ean = core + str(cd)
        if ean not in used:
            return ean

    # fallback ultimă instanță
    core = f"{prefix}{0:0{item_len}d}"
    return core + str(ean13_check_digit(core))

# Sidebar quota
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
    col_sku    = st.selectbox("Coloană SKU", cols, index=0)
    col_name   = st.selectbox("Coloană Denumire", cols, index=1 if len(cols)>1 else 0)
    col_target = st.selectbox("Coloană țintă pentru EAN-13", cols, index=len(cols)-1)
    mode       = st.radio("Cum cauți EAN?", ["Doar SKU", "Doar Nume"])

    synth_mode = st.radio(
        "Completează cu EAN sintetic dacă nu găsește prin Google?",
        ["Nu", "Da"]
    )

    note_col = "EAN_NOTE"
    if note_col not in df.columns:
        df[note_col] = ""

    mode_rows = st.radio("Ce rânduri procesezi?", ["Primele N rânduri", "Toate rândurile"])
    if mode_rows == "Primele N rânduri":
        max_rows = st.number_input("N rânduri de procesat", 1, len(df), min(50,len(df)))
    else:
        max_rows = len(df)

    # prefix de bază din fișier, dacă există
    existing_valid = [str(v) for v in df[col_target].astype(str).tolist() if is_valid_ean13(str(v))]
    base_prefix = most_common_prefix(existing_valid)  # None dacă nu există
    used_eans = set([clean_digits(v) for v in existing_valid])

    if st.button("Pornește căutarea EAN"):
        done = 0; bar = st.progress(0); status = st.empty(); query_status = st.empty()

        for idx, row in df.head(int(max_rows)).iterrows():
            sku  = str(row.get(col_sku,"")).strip()
            name = str(row.get(col_name,"")).strip()
            cur  = str(row.get(col_target,"")).strip()
            note = str(row.get(note_col,"")).strip().lower()

            # Skip dacă deja are EAN valid sau NOT_FOUND
            if (cur and is_valid_ean13(cur)) or cur.upper()=="NOT_FOUND" or note=="synthetic":
                done+=1; bar.progress(int(done*100/max_rows)); continue

            found = lookup(mode, sku, name, query_status)

            if found and is_valid_ean13(found):
                df.at[idx, col_target] = found
                df.at[idx, note_col]   = "found"
                used_eans.add(clean_digits(found))
            else:
                if synth_mode == "Da":
                    syn = build_synthetic_ean(sku, name, base_prefix, used_eans)
                    df.at[idx, col_target] = syn
                    df.at[idx, note_col]   = "synthetic"
                    used_eans.add(clean_digits(syn))
                else:
                    df.at[idx, col_target] = "NOT_FOUND"
                    df.at[idx, note_col]   = "not_found"

            done+=1
            if done%5==0: status.write(f"Procesate: {done}/{int(max_rows)}")
            bar.progress(int(done*100/max_rows)); time.sleep(0.2)

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
