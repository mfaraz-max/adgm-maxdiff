import streamlit as st
st.set_page_config(page_title="Augmented MaxDiff Imputer", page_icon="✅", layout="wide")

import pandas as pd
import numpy as np
import io, os
from dotenv import load_dotenv
load_dotenv()  # so OPENAI_API_KEY from .env is available

# --- AUTH ---
import streamlit_authenticator as stauth
from collections.abc import Mapping

def to_dict(obj):
    """Deep-convert Streamlit Secrets or other mappings/lists to plain Python dict/list."""
    if isinstance(obj, Mapping):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(x) for x in obj]
    else:
        return obj

# Validate secrets exist
if "credentials" not in st.secrets or "cookie" not in st.secrets:
    st.error("Authentication secrets not configured. Add [credentials] and [cookie] blocks in .streamlit/secrets.toml.")
    st.stop()

# Convert to mutable dicts (streamlit_authenticator may mutate inputs)
credentials = to_dict(st.secrets["credentials"])
cookie_cfg  = to_dict(st.secrets["cookie"])

authenticator = stauth.Authenticate(
    credentials,
    cookie_cfg.get("name", "adgm_auth"),
    cookie_cfg.get("key", "please-change-me"),
    int(cookie_cfg.get("expiry_days", 7)),
)

# NEW API: use 'fields' instead of positional form_name
login_fields = {
    "form_name": "Login",
    "username": "Username",
    "password": "Password",
}

# Render login form
st.write("")  # spacing
name, auth_status, username = authenticator.login(fields=login_fields, location="main")

if auth_status is False:
    st.error("Username/password incorrect.")
    st.stop()
elif auth_status is None:
    st.info("Please enter your username and password.")
    st.stop()

# Only after successful login, import the app logic
from augmented_maxdiff import (
    augment_maxdiff,                # heuristic-individual
    augment_maxdiff_global,         # heuristic-global
    augment_maxdiff_llm,            # llm-individual
    augment_maxdiff_llm_global      # llm-global
)

# If OPENAI_API_KEY is kept in Streamlit secrets (optional), expose to env for the LLM calls
if not os.environ.get("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- UI ---
CUSTOM_CSS = """
<style>
html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.app-header { display:flex; align-items:center; gap:12px; padding:8px 0 24px; border-bottom:1px solid #eee; margin-bottom:18px;}
.badge {padding:4px 8px; border-radius:999px; font-size:12px; background:#eef6ff; color:#1e66f5; border:1px solid #cfe3ff;}
.metric {border-radius:14px; padding:14px; background:#fafafa; border:1px dashed #ddd;}
.download-box {border:1px solid #16a34a33; background:#16a34a0d; border-radius:16px; padding:16px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(f"""
<div class="app-header">
  <h2 style="margin:0;">🧠 Augmented Data Generation for MaxDiff</h2>
  <span class="badge">Heuristic & AI · Individual & Global</span>
  <span class="badge">Logged in as: {name}</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    authenticator.logout("Logout", "sidebar")

with st.sidebar:
    st.header("① Upload Data")
    uploaded = st.file_uploader("Excel file (.xlsx) with stacked MaxDiff rows", type=["xlsx"])
    sheet_name = st.text_input("Sheet name or index (default = 0)", value="0")
    if sheet_name.isdigit(): sheet = int(sheet_name)
    else: sheet = sheet_name

    st.markdown("---")
    st.header("② Select Mode(s) (run any/all)")
    use_heur_ind = st.checkbox("Heuristic — Individual", value=True)
    use_heur_glb = st.checkbox("Heuristic — Global", value=False)
    use_llm_ind  = st.checkbox("AI (LLM) — Individual", value=False)
    use_llm_glb  = st.checkbox("AI (LLM) — Global", value=False)
    if use_llm_ind or use_llm_glb:
        has_key = bool(os.environ.get("OPENAI_API_KEY","") or st.secrets.get("OPENAI_API_KEY", ""))
        st.caption("OPENAI_API_KEY detected: " + ("✅" if has_key else "❌ (set in .env or Streamlit secrets)"))

    st.markdown("---")
    st.header("③ Run")
    run_btn = st.button("Run Imputation", type="primary")

    st.markdown("---")
    st.header("Options")
    heur_label = st.text_input("Heuristic method label", value="heuristic-maxdiff-v1")
    llm_model  = st.text_input("LLM model", value="gpt-4o-mini")

# --- Main body ---
if uploaded is not None:
    try:
        xl = pd.ExcelFile(uploaded)
        st.caption(f"Detected sheets: {', '.join(map(str, xl.sheet_names))}")
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")

    try:
        preview_df = pd.read_excel(uploaded, sheet_name=sheet if str(sheet_name).strip() else 0, nrows=50)
        st.subheader("Preview (first 50 rows)")
        st.dataframe(preview_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not preview selected sheet: {e}")

    if run_btn:
        if not any([use_heur_ind, use_heur_glb, use_llm_ind, use_llm_glb]):
            st.warning("Select at least one mode."); st.stop()

        with st.spinner("Imputing missing Best/Worst values..."):
            tmp_in = "tmp_input.xlsx"
            with open(tmp_in, "wb") as f:
                f.write(uploaded.getbuffer())

            results = []

            # Heuristic — Individual
            if use_heur_ind:
                try:
                    df_hi = augment_maxdiff(tmp_in, output_path=None, sheet_name=sheet, method_label=heur_label or "heuristic-maxdiff-v1")
                    results.append(("heuristic_individual", df_hi))
                except Exception as e:
                    st.error(f"Heuristic — Individual failed: {e}")

            # Heuristic — Global
            if use_heur_glb:
                try:
                    df_hg = augment_maxdiff_global(tmp_in, output_path=None, sheet_name=sheet, method_label="heuristic-global-maxdiff-v1")
                    results.append(("heuristic_global", df_hg))
                except Exception as e:
                    st.error(f"Heuristic — Global failed: {e}")

            # LLM — Individual
            if use_llm_ind:
                try:
                    df_li = augment_maxdiff_llm(tmp_in, output_path=None, sheet_name=sheet, model=llm_model or "gpt-4o-mini", method_label="llm-predictive-maxdiff-v1")
                    results.append(("llm_individual", df_li))
                except Exception as e:
                    st.error(f"LLM — Individual failed (check OPENAI_API_KEY): {e}")

            # LLM — Global
            if use_llm_glb:
                try:
                    df_lg = augment_maxdiff_llm_global(tmp_in, output_path=None, sheet_name=sheet, model=llm_model or "gpt-4o-mini", method_label="llm-global-predictive-maxdiff-v1")
                    results.append(("llm_global", df_lg))
                except Exception as e:
                    st.error(f"LLM — Global failed (check OPENAI_API_KEY): {e}")

            if not results:
                st.stop()

            # Show metrics per mode
            for label, df_res in results:
                total_rows = len(df_res)
                imputed_rows = int(df_res["imputed_flag"].sum()) if "imputed_flag" in df_res.columns else 0
                observed_rows = total_rows - imputed_rows
                st.subheader(f"{label.replace('_',' ').title()} — Result Metrics")
                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="metric"><b>Total rows</b><br/>{total_rows}</div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric"><b>Observed rows</b><br/>{observed_rows}</div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric"><b>Imputed rows</b><br/>{imputed_rows}</div>', unsafe_allow_html=True)
                st.dataframe(df_res.head(50), use_container_width=True)

            # Build one Excel with N sheets (one per selected mode)
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
                for label, df_res in results:
                    sheetname = label[:31]  # Excel sheet name limit
                    df_res.to_excel(writer, index=False, sheet_name=sheetname)
            towrite.seek(0)
            st.markdown('<div class="download-box">✅ Done. Download the Excel with one sheet per selected mode.</div>', unsafe_allow_html=True)
            st.download_button("⬇️ Download completed_maxdiff.xlsx", data=towrite,
                               file_name="completed_maxdiff.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload a stacked MaxDiff Excel to begin.")
