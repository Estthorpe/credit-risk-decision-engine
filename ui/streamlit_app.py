from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st


# -----------------------------
# Brand / UI configuration
# -----------------------------
@dataclass(frozen=True)
class Brand:
    app_name: str = "Credit Risk Decision Engine"
    tagline: str = "PD Scoring · Audit-friendly reason codes"
    primary: str = "#0B2E4A"
    accent: str = "#2F80ED"
    success: str = "#1F8A70"
    warning: str = "#F2994A"
    danger: str = "#EB5757"
    bg: str = "#F7F9FC"
    card: str = "#FFFFFF"
    text: str = "#111827"
    muted: str = "#6B7280"


BRAND = Brand()

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_SCORE_URL = f"{DEFAULT_API_BASE}/score"
DEFAULT_HEALTH_URL = f"{DEFAULT_API_BASE}/health"


# Optional (UI-only) reason descriptions
REASON_DESC: Dict[str, str] = {
    "DAYS_BIRTH_YOUNG": "Applicant is relatively younger (risk historically higher for younger segments).",
    "EXT_SOURCE_LOW": "External score is low.",
    "AMT_CREDIT_HIGH": "Requested credit amount is high relative to typical approvals.",
    # Add more mappings as your reason_codes evolve.
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {BRAND.bg};
                color: {BRAND.text};
            }}
            .brand-header {{
                padding: 18px 18px 8px 18px;
                background: {BRAND.card};
                border-radius: 16px;
                border: 1px solid rgba(17,24,39,0.08);
                box-shadow: 0 6px 18px rgba(17,24,39,0.06);
                margin-bottom: 16px;
            }}
            .brand-title {{
                font-size: 30px;
                font-weight: 850;
                color: {BRAND.primary};
                margin: 0;
                line-height: 1.15;
            }}
            .brand-subtitle {{
                font-size: 14px;
                color: {BRAND.muted};
                margin-top: 6px;
            }}
            .card {{
                padding: 16px;
                background: {BRAND.card};
                border-radius: 16px;
                border: 1px solid rgba(17,24,39,0.08);
                box-shadow: 0 6px 18px rgba(17,24,39,0.06);
                margin-bottom: 16px;
            }}
            .pill {{
                display: inline-block;
                padding: 7px 12px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 0.4px;
                margin-right: 8px;
            }}
            .pill-ok {{ background: rgba(31,138,112,0.12); color: {BRAND.success}; }}
            .pill-warn {{ background: rgba(242,153,74,0.12); color: {BRAND.warning}; }}
            .pill-bad {{ background: rgba(235,87,87,0.12); color: {BRAND.danger}; }}
            .small-muted {{
                font-size: 12px;
                color: {BRAND.muted};
            }}
            .reason {{
                padding: 10px 12px;
                border-radius: 12px;
                background: rgba(47,128,237,0.08);
                border: 1px solid rgba(47,128,237,0.15);
                margin: 6px 0;
                font-size: 13px;
            }}
            .reason-title {{
                font-weight: 800;
                color: {BRAND.primary};
                margin-bottom: 2px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Helpers
# -----------------------------
def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def decision_badge(decision: str) -> str:
    d = (decision or "").strip().lower()
    if d in {"approve", "approved"}:
        return f'<span class="pill pill-ok">APPROVE</span>'
    if d in {"decline", "rejected"}:
        return f'<span class="pill pill-bad">DECLINE</span>'
    # manual_review or anything else
    return f'<span class="pill pill-warn">MANUAL REVIEW</span>'


def call_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout_s: int = 15) -> Dict[str, Any]:
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s) if payload else requests.get(url, timeout=timeout_s)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError("API unreachable. Start FastAPI first (see sidebar tip).") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError("API timed out. Check server load or increase timeout.") from e

    if resp.status_code != 200:
        # 422 is common for schema validation errors in FastAPI
        if resp.status_code == 422:
            raise RuntimeError(f"API validation error (422). Likely schema mismatch:\n{resp.text}")
        raise RuntimeError(f"API request failed ({resp.status_code}): {resp.text}")

    return resp.json()


def score_application(score_url: str, features: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    payload = {"request_id": request_id or str(uuid.uuid4()), "features": features}
    return call_json(score_url, payload=payload, timeout_s=30)


# -----------------------------
# Demo presets
# -----------------------------
def preset_low_risk() -> Dict[str, Any]:
    return {
        "EXT_SOURCE_1": 0.85,
        "EXT_SOURCE_2": 0.80,
        "EXT_SOURCE_3": 0.78,
        "AMT_INCOME_TOTAL": 250000.0,
        "AMT_CREDIT": 300000.0,
        "AMT_ANNUITY": 18000.0,
        "DAYS_BIRTH": -16000,
        "DAYS_EMPLOYED": -5000,
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_INCOME_TYPE": "Working",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "Y",
    }


def preset_high_risk() -> Dict[str, Any]:
    return {
        "EXT_SOURCE_1": 0.15,
        "EXT_SOURCE_2": 0.20,
        "EXT_SOURCE_3": 0.18,
        "AMT_INCOME_TOTAL": 70000.0,
        "AMT_CREDIT": 900000.0,
        "AMT_ANNUITY": 45000.0,
        "DAYS_BIRTH": -9000,
        "DAYS_EMPLOYED": -300,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_INCOME_TYPE": "Unemployed",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
    }


def init_state() -> None:
    if "form" not in st.session_state:
        st.session_state.form = preset_low_risk()
    if "blank_ext" not in st.session_state:
        st.session_state.blank_ext = {"EXT_SOURCE_1": False, "EXT_SOURCE_2": False, "EXT_SOURCE_3": False}
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


# -----------------------------
# Streamlit App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=BRAND.app_name, page_icon="🏦", layout="wide")
    inject_css()
    init_state()

    # Header
    st.markdown(
        f"""
        <div class="brand-header">
            <div class="brand-title">{BRAND.app_name}</div>
            <div class="brand-subtitle">{BRAND.tagline}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        api_base = st.text_input("FastAPI base URL", value=DEFAULT_API_BASE)
        score_url = api_base.rstrip("/") + "/score"
        health_url = api_base.rstrip("/") + "/health"

        st.caption("Tip: Start API first:")
        st.code("python -m uvicorn credit_risk_decision_engine.serving.app:app --reload --host 0.0.0.0 --port 8000")

        st.divider()
        st.subheader("API Status")
        if st.button("🔎 Check /health", use_container_width=True):
            try:
                health = call_json(health_url, payload=None, timeout_s=10)
                st.success("API healthy ✅")
                st.json(health)
            except Exception as e:
                st.error(str(e))

        st.divider()
        st.subheader("Demo Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load low-risk", use_container_width=True):
                st.session_state.form = preset_low_risk()
                st.session_state.blank_ext = {"EXT_SOURCE_1": False, "EXT_SOURCE_2": False, "EXT_SOURCE_3": False}
        with c2:
            if st.button("Load high-risk", use_container_width=True):
                st.session_state.form = preset_high_risk()
                st.session_state.blank_ext = {"EXT_SOURCE_1": False, "EXT_SOURCE_2": False, "EXT_SOURCE_3": False}

        if st.button("Reset to defaults", use_container_width=True):
            st.session_state.form = preset_low_risk()
            st.session_state.blank_ext = {"EXT_SOURCE_1": False, "EXT_SOURCE_2": False, "EXT_SOURCE_3": False}
            st.session_state.last_result = None

        st.divider()
        st.subheader("About")
        st.write(
            "This UI sends a minimal set of applicant fields to the **Credit Risk API** and displays the returned "
            "**PD**, **decision**, and **reason codes** (audit-friendly)."
        )

    # Layout
    left, right = st.columns([1.05, 0.95], gap="large")

    # -----------------------------
    # Inputs
    # -----------------------------
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Applicant Inputs (Minimal Fields)")
        st.caption("Enter what you know. Missing fields are handled by the API’s preprocessing + imputers.")

        form = st.session_state.form

        c1, c2, c3 = st.columns(3)

        with c1:
            form["EXT_SOURCE_1"] = st.number_input(
                "EXT_SOURCE_1 (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=float(form["EXT_SOURCE_1"]) if form["EXT_SOURCE_1"] is not None else 0.5,
                step=0.01,
                help="External bureau score (0–1). If unknown, tick 'Blank' below and the API will impute.",
            )
            form["AMT_INCOME_TOTAL"] = st.number_input(
                "AMT_INCOME_TOTAL",
                min_value=0.0,
                value=float(form["AMT_INCOME_TOTAL"]),
                step=1000.0,
                help="Total annual income amount used by the model.",
            )
            form["DAYS_BIRTH"] = st.number_input(
                "DAYS_BIRTH (negative days)",
                value=int(form["DAYS_BIRTH"]),
                step=30,
                help="Dataset encoding: negative days since birth. Example: -12000 ≈ 33 years old.",
            )
            form["CODE_GENDER"] = st.selectbox("CODE_GENDER", options=["F", "M"], index=0 if form["CODE_GENDER"] == "F" else 1)

        with c2:
            form["EXT_SOURCE_2"] = st.number_input(
                "EXT_SOURCE_2 (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=float(form["EXT_SOURCE_2"]) if form["EXT_SOURCE_2"] is not None else 0.5,
                step=0.01,
                help="External bureau score (0–1). If unknown, tick 'Blank' below and the API will impute.",
            )
            form["AMT_CREDIT"] = st.number_input(
                "AMT_CREDIT",
                min_value=0.0,
                value=float(form["AMT_CREDIT"]),
                step=5000.0,
                help="Requested credit principal amount.",
            )
            form["DAYS_EMPLOYED"] = st.number_input(
                "DAYS_EMPLOYED (negative days)",
                value=int(form["DAYS_EMPLOYED"]),
                step=30,
                help="Dataset encoding: negative days since employment started. Values near 0 may indicate short/unstable employment.",
            )
            form["FLAG_OWN_CAR"] = st.selectbox("FLAG_OWN_CAR", options=["N", "Y"], index=0 if form["FLAG_OWN_CAR"] == "N" else 1)

        with c3:
            form["EXT_SOURCE_3"] = st.number_input(
                "EXT_SOURCE_3 (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=float(form["EXT_SOURCE_3"]) if form["EXT_SOURCE_3"] is not None else 0.5,
                step=0.01,
                help="External bureau score (0–1). If unknown, tick 'Blank' below and the API will impute.",
            )
            form["AMT_ANNUITY"] = st.number_input(
                "AMT_ANNUITY",
                min_value=0.0,
                value=float(form["AMT_ANNUITY"]),
                step=500.0,
                help="Monthly/periodic annuity amount.",
            )

            edu_opts = [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ]
            form["NAME_EDUCATION_TYPE"] = st.selectbox(
                "NAME_EDUCATION_TYPE",
                options=edu_opts,
                index=edu_opts.index(form["NAME_EDUCATION_TYPE"]) if form["NAME_EDUCATION_TYPE"] in edu_opts else 0,
            )

            income_opts = ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"]
            form["NAME_INCOME_TYPE"] = st.selectbox(
                "NAME_INCOME_TYPE",
                options=income_opts,
                index=income_opts.index(form["NAME_INCOME_TYPE"]) if form["NAME_INCOME_TYPE"] in income_opts else 0,
            )

        # Optional: let user clear EXT_SOURCE_* if unknown
        st.markdown("---")
        st.caption("If EXT_SOURCE_* scores are unknown, set them to blank (API will impute).")
        bx1, bx2, bx3 = st.columns(3)
        with bx1:
            st.session_state.blank_ext["EXT_SOURCE_1"] = st.checkbox("Blank EXT_SOURCE_1", value=st.session_state.blank_ext["EXT_SOURCE_1"])
        with bx2:
            st.session_state.blank_ext["EXT_SOURCE_2"] = st.checkbox("Blank EXT_SOURCE_2", value=st.session_state.blank_ext["EXT_SOURCE_2"])
        with bx3:
            st.session_state.blank_ext["EXT_SOURCE_3"] = st.checkbox("Blank EXT_SOURCE_3", value=st.session_state.blank_ext["EXT_SOURCE_3"])

        st.markdown("</div>", unsafe_allow_html=True)

        # Build features dict
        features: Dict[str, Any] = dict(form)
        for k in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            if st.session_state.blank_ext.get(k, False):
                features[k] = None

        score_btn = st.button("🏁 Score Application", type="primary", use_container_width=True)

        with st.expander("Why minimal fields? (to reduce schema mismatch risk)"):
            st.write(
                "This UI intentionally sends a **small, stable** set of fields. "
                "The API uses the trained bundle's preprocessing pipeline and `feature_columns.json` "
                "to align inputs, impute missing values, and ensure consistent inference."
            )

    # -----------------------------
    # Output
    # -----------------------------
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Prediction Output")
        st.caption("Produced by the FastAPI scoring endpoint using the versioned model bundle.")

        if score_btn:
            with st.spinner("Calling scoring API..."):
                try:
                    result = score_application(score_url=score_url, features=features)
                    st.session_state.last_result = result
                except Exception as e:
                    st.session_state.last_result = None
                    st.error(f"Scoring failed: {e}")

        result = st.session_state.last_result
        if result:
            pd_hat = result.get("pd")
            decision = result.get("decision")
            reasons = result.get("reason_codes", [])
            model_version = result.get("model_version", "unknown")
            schema_version = result.get("input_schema_version", "unknown")
            latency_ms = result.get("latency_ms", None)

            st.markdown(decision_badge(decision), unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            k1.metric("PD (Probability of Default)", f"{pd_hat:.4f}" if isinstance(pd_hat, (float, int)) else str(pd_hat))
            k2.metric("Model Version", str(model_version))
            k3.metric("Latency (ms)", f"{latency_ms:.1f}" if isinstance(latency_ms, (float, int)) else str(latency_ms))

            st.markdown(f"<div class='small-muted'>Input schema version: {schema_version}</div>", unsafe_allow_html=True)
            st.markdown("---")

            st.subheader("🧠 Reason Codes")
            if reasons:
                for r in reasons[:5]:
                    desc = REASON_DESC.get(r)
                    if desc:
                        st.markdown(
                            f"<div class='reason'><div class='reason-title'>• {r}</div><div class='small-muted'>{desc}</div></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"<div class='reason'>• {r}</div>", unsafe_allow_html=True)
            else:
                st.info("No reason codes returned.")

            with st.expander("View raw API response"):
                st.code(json.dumps(result, indent=2), language="json")
        else:
            st.info("Enter inputs and click **Score Application** to see results.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='small-muted'>Note: This UI calls the API (recommended) to avoid preprocessing mismatch. "
        "The API owns inference via the bundle and its feature contract.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
