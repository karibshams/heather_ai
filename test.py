"""
test.py — Local testing UI for PackingListAI
============================================
Run:  streamlit run test.py
"""

import json
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="PackingListAI Tester", page_icon="🧳", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0f1117; color: #e0e0e0; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_key    = st.text_input("OpenAI API Key", type="password", help="Leave blank to use .env")
    use_vision = st.toggle("Use Vision (send image to AI)", value=True)
    ocr_lang   = st.selectbox("OCR Language", ["eng", "deu", "fra", "spa"])
    st.divider()
    st.caption("Supported: PDF, JPG, PNG, BMP, TIFF, WebP")

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🧳 PackingListAI — Local Tester")

uploaded = st.file_uploader(
    "Upload a packing list (PDF or image)",
    type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
)

if uploaded:
    st.success(f"📄 **{uploaded.name}** — {len(uploaded.getvalue()) / 1024:.1f} KB")

    if st.button("▶ Run Extraction", type="primary", use_container_width=True):

        # Save upload to a temp file so DocumentLoader can read it
        import tempfile
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        try:
            from ai_core import PackingListAI
        except ImportError as e:
            st.error(f"Cannot import ai_core: {e}")
            st.stop()

        with st.spinner("Running OCR + AI extraction…"):
            try:
                t0        = time.time()
                extractor = PackingListAI(
                    api_key    = api_key or None,
                    use_vision = use_vision,
                    ocr_lang   = ocr_lang,
                )
                result  = extractor.process_file(tmp_path)
                elapsed = time.time() - t0
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()
            finally:
                tmp_path.unlink(missing_ok=True)

        # ── Metrics ──────────────────────────────────────────────────────
        categories   = result.get("categories", [])
        total_items  = sum(len(c.get("items", [])) for c in categories)
        required     = sum(1 for c in categories for i in c.get("items", []) if i.get("is_required"))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⏱ Time",        f"{elapsed:.1f}s")
        c2.metric("📂 Categories",  len(categories))
        c3.metric("📦 Total Items", total_items)
        c4.metric("✅ Required",    required)

        st.divider()

        # ── Two-column view ───────────────────────────────────────────────
        left, right = st.columns(2, gap="large")

        with left:
            st.subheader("Categories & Items")
            for cat in categories:
                with st.expander(f"**{cat['name']}** — {len(cat.get('items', []))} items"):
                    for item in cat.get("items", []):
                        qty  = f" × {item['quantity']}" if item.get("quantity") else ""
                        flag = "🔴" if item.get("is_required") else "🟡 optional"
                        note = f"\n  > _{item['note']}_" if item.get("note") else ""
                        st.markdown(f"- **{item['title']}**{qty} {flag}{note}")

        with right:
            st.subheader("JSON Output")
            json_str = json.dumps(result, indent=2, ensure_ascii=False)
            st.code(json_str, language="json")
            st.download_button(
                "⬇ Download JSON",
                data      = json_str,
                file_name = f"{Path(uploaded.name).stem}_extracted.json",
                mime      = "application/json",
                use_container_width=True,
            )
else:
    st.markdown("""
    <div style="text-align:center;padding:5rem 0;opacity:.4">
        <div style="font-size:5rem">🧳</div>
        <p style="font-size:1.1rem">Upload a packing list PDF or image above to begin.</p>
    </div>
    """, unsafe_allow_html=True)