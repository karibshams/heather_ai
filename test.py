"""
test.py — Local Testing Interface for PackingListAI
====================================================
Provides a Streamlit web UI for uploading documents and inspecting
JSON extraction results — intended for local VS Code / dev use only.

Run:
    streamlit run test.py
"""

import json
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PackingListAI — Dev Tester",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal CSS polish
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .stApp { background: #0f1117; color: #e0e0e0; }
        .block-container { padding-top: 2rem; }
        .metric-card {
            background: #1a1d27;
            border: 1px solid #2e3148;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            text-align: center;
        }
        pre { background: #1a1d27 !important; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Leave blank to use OPENAI_API_KEY from .env",
    )
    use_vision = st.toggle("Use Vision (multimodal)", value=True)
    ocr_lang = st.selectbox("OCR Language", ["eng", "deu", "fra", "spa"], index=0)
    st.divider()
    st.markdown(
        "**Supported formats**\n- PDF (multi-page)\n- JPG / JPEG\n- PNG\n- BMP / TIFF / WebP"
    )
    st.divider()
    st.caption("PackingListAI Dev Tester — local use only")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🧳 PackingListAI — Extraction Tester")
st.caption("Upload a packing list image or PDF and inspect the extracted JSON.")

uploaded = st.file_uploader(
    "Drop your file here",
    type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    col_file, col_run = st.columns([3, 1])
    with col_file:
        st.success(f"📄 **{uploaded.name}** — {len(uploaded.getvalue()) / 1024:.1f} KB")
    with col_run:
        run_btn = st.button("▶ Extract", use_container_width=True, type="primary")

    if run_btn:
        # Lazy import so the app loads even without heavy deps installed
        try:
            from ai_core import PackingListAI, ResponseFormatter
        except ImportError as e:
            st.error(f"Could not import ai_core: {e}")
            st.stop()

        with st.spinner("Running OCR + AI extraction…"):
            try:
                t0 = time.time()
                extractor = PackingListAI(
                    api_key=api_key or None,
                    use_vision=use_vision,
                    ocr_lang=ocr_lang,
                )
                result_dict = extractor.process_bytes(
                    data=uploaded.getvalue(),
                    filename=uploaded.name,
                )
                elapsed = time.time() - t0
            except Exception as exc:
                st.error(f"❌ Extraction failed: {exc}")
                st.stop()

        # ------------------------------------------------------------------
        # Metrics bar
        # ------------------------------------------------------------------

        categories = result_dict.get("categories", [])
        total_items = sum(len(c.get("items", [])) for c in categories)
        required_count = sum(
            1
            for c in categories
            for item in c.get("items", [])
            if item.get("is_required")
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⏱ Time", f"{elapsed:.1f}s")
        m2.metric("📂 Categories", len(categories))
        m3.metric("📦 Total Items", total_items)
        m4.metric("✅ Required", required_count)

        st.divider()

        # ------------------------------------------------------------------
        # Two-column layout — category cards + raw JSON
        # ------------------------------------------------------------------

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("Extracted Categories")
            if not categories:
                st.info("No categories found.")
            for cat in categories:
                with st.expander(f"**{cat['name']}** — {len(cat.get('items', []))} items"):
                    for item in cat.get("items", []):
                        qty_str = f" × {item['quantity']}" if item.get("quantity") else ""
                        req_badge = "🔴" if item.get("is_required") else "🟡 optional"
                        note_str = f"\n  > {item['note']}" if item.get("note") else ""
                        st.markdown(f"- **{item['title']}**{qty_str} {req_badge}{note_str}")

        with right:
            st.subheader("Raw JSON Output")
            json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)
            st.code(json_str, language="json")
            st.download_button(
                label="⬇ Download JSON",
                data=json_str,
                file_name=f"{Path(uploaded.name).stem}_extracted.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    # Placeholder / welcome screen
    st.markdown(
        """
        <div style="text-align:center;padding:4rem 0;opacity:.5">
            <div style="font-size:4rem">🧳</div>
            <p style="font-size:1.2rem;margin-top:.5rem">
                Upload a packing list PDF or image to get started.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
