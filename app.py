from __future__ import annotations

import pandas as pd
import streamlit as st

import config
import model_utils
import styles

# ----------------------------------------------------------------------
# Page setup — must run before any other Streamlit command
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
styles.inject_css()


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------

def _load_example(idx: int) -> None:
    example = config.SAMPLE_CUSTOMERS[idx]
    for field, value in example["values"].items():
        if field == "SeniorCitizen":
            value = "Yes" if value == 1 else "No"
        st.session_state[f"in_{field}"] = value
    st.session_state["active_example_label"] = example["label"]
    st.session_state["active_example_actual"] = example["actual_churn"]
    st.session_state["prediction_result"] = None


def _reset_form() -> None:
    defaults = {
        "in_gender": "Female", "in_SeniorCitizen": "No", "in_Partner": "No", "in_Dependents": "No",
        "in_tenure": config.TENURE_DEFAULT, "in_PhoneService": "Yes", "in_MultipleLines": "No",
        "in_InternetService": "Fiber optic", "in_OnlineSecurity": "No", "in_OnlineBackup": "No",
        "in_DeviceProtection": "No", "in_TechSupport": "No", "in_StreamingTV": "No",
        "in_StreamingMovies": "No", "in_Contract": "Month-to-month", "in_PaperlessBilling": "Yes",
        "in_PaymentMethod": "Electronic check", "in_MonthlyCharges": config.MONTHLY_CHARGES_DEFAULT,
        "in_TotalCharges": config.TOTAL_CHARGES_DEFAULT,
    }
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state["prediction_result"] = None
    st.session_state["active_example_label"] = None
    st.session_state["active_example_actual"] = None


def _run_prediction() -> None:
    model = st.session_state.get("loaded_model")
    raw_row = {col: st.session_state.get(f"in_{col}") for col in config.RAW_COLUMN_ORDER}
    raw_row["SeniorCitizen"] = 1 if st.session_state.get("in_SeniorCitizen") == "Yes" else 0
    st.session_state["last_raw_row"] = raw_row
    st.session_state["prediction_result"] = model_utils.predict(model, raw_row)


# ----------------------------------------------------------------------
# Model loading (once per session, unless replaced via upload)
# ----------------------------------------------------------------------
if "loaded_model" not in st.session_state:
    bundled = model_utils.load_model_from_path(str(config.DEFAULT_MODEL_PATH))
    st.session_state["loaded_model"] = bundled.model
    st.session_state["model_source"] = "bundled" if bundled.model is not None else None
    st.session_state["bundled_load_result"] = bundled


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="app-title" style="font-size:1.25rem; font-weight:700;">Churn Predictor</div>',
        unsafe_allow_html=True,
    )
    st.caption("Telecom customer retention risk")
    st.markdown("---")

    model = st.session_state.get("loaded_model")
    bundled_result: model_utils.LoadResult = st.session_state.get("bundled_load_result")

    if model is not None:
        st.markdown(styles.status_pill(f"{type(model).__name__} loaded", "on"), unsafe_allow_html=True)
        n_expected = getattr(model, "n_features_in_", None)
        if n_expected is not None:
            n_raw = len(config.RAW_COLUMN_ORDER)
            if n_expected == n_raw:
                st.caption(f"Expects {n_expected} features — matches this form's raw columns.")
            else:
                st.caption(
                    f"⚠ Expects {n_expected} features; this form sends {n_raw} raw columns. "
                    "See \u201cAdapting to your model\u201d in README.md."
                )
    else:
        st.markdown(styles.status_pill("No model loaded", "off"), unsafe_allow_html=True)
        if bundled_result and not bundled_result.file_missing and bundled_result.error:
            st.error(bundled_result.error)
            if bundled_result.hint:
                st.caption(bundled_result.hint)

    with st.expander("Upload model.pkl" if model is None else "Replace model", expanded=(model is None)):
        uploaded = st.file_uploader(
            "Model file", type=["pkl", "joblib"], label_visibility="collapsed", key="model_uploader"
        )
        if uploaded is not None:
            fingerprint = (uploaded.name, uploaded.size)
            if st.session_state.get("uploaded_fingerprint") != fingerprint:
                st.session_state["uploaded_fingerprint"] = fingerprint
                result = model_utils.load_model_from_upload(uploaded)
                if result.model is not None:
                    st.session_state["loaded_model"] = result.model
                    st.session_state["model_source"] = "uploaded"
                    st.session_state["prediction_result"] = None
                    st.rerun()
                else:
                    st.error(result.error)
                    if result.hint:
                        st.caption(result.hint)
                    if result.raw_exception:
                        with st.expander("Technical details"):
                            st.code(result.raw_exception)

    st.markdown("---")
    st.markdown("**Try an example**")
    labels = [c["label"] for c in config.SAMPLE_CUSTOMERS]
    st.selectbox("Example customer", labels, key="example_choice", label_visibility="collapsed")
    chosen_idx = labels.index(st.session_state["example_choice"])
    st.button("Load this example", use_container_width=True, on_click=_load_example, args=(chosen_idx,))

    st.markdown("---")
    with st.expander("About this app"):
        st.caption(
            "A portfolio front end for a Telco customer-churn model. "
            "Fill in a customer's details and it estimates their probability "
            "of churning, using whatever model.pkl is loaded."
        )
        st.caption(
            "Built by **[Sahil](https://www.linkedin.com/in/sahilpatil2171/)** — a final-year "
            "MCA (Data Science) student who builds end-to-end "
            "machine learning projects like this one. More work on "
            "[GitHub](https://github.com/Sahil2171)."
        )
    st.caption(
        f"Reference data: {config.TRAINING_ROWS:,} customers, "
        f"{config.TRAINING_CHURN_RATE * 100:.1f}% historical churn rate."
    )


# ----------------------------------------------------------------------
# Main — header
# ----------------------------------------------------------------------
st.markdown(
    '<div class="app-title" style="font-size:1.9rem; font-weight:700;">Customer Churn Predictor</div>',
    unsafe_allow_html=True,
)
st.caption("Estimate a telecom customer's likelihood of churning from their account details.")

model = st.session_state.get("loaded_model")
status_html = styles.status_pill(
    "Model loaded" if model is not None else "Model not loaded", "on" if model is not None else "off"
) + styles.status_pill(f"{config.TRAINING_ROWS:,} customers in reference data", "on")
st.markdown(f'<div class="status-row">{status_html}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Main — input form
# ----------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Profile", "Account", "Services"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Gender", config.CATEGORY_OPTIONS["gender"], key="in_gender")
        st.selectbox("Senior citizen", ["No", "Yes"], key="in_SeniorCitizen")
    with c2:
        st.selectbox("Has a partner", config.CATEGORY_OPTIONS["Partner"], key="in_Partner")
        st.selectbox("Has dependents", config.CATEGORY_OPTIONS["Dependents"], key="in_Dependents")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.slider(
            "Tenure (months)", config.TENURE_RANGE[0], config.TENURE_RANGE[1],
            config.TENURE_DEFAULT, key="in_tenure",
        )
        st.selectbox("Contract", config.CATEGORY_OPTIONS["Contract"], key="in_Contract")
        st.selectbox("Paperless billing", config.CATEGORY_OPTIONS["PaperlessBilling"], key="in_PaperlessBilling")
    with c2:
        st.selectbox("Payment method", config.CATEGORY_OPTIONS["PaymentMethod"], key="in_PaymentMethod")
        st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=500.0,
            value=config.MONTHLY_CHARGES_DEFAULT, step=0.5, key="in_MonthlyCharges",
        )
        st.number_input(
            "Total charges ($)", min_value=0.0, max_value=15000.0,
            value=config.TOTAL_CHARGES_DEFAULT, step=1.0, key="in_TotalCharges",
        )
        st.caption("Tip: usually roughly tenure × monthly charges.")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Phone service", config.CATEGORY_OPTIONS["PhoneService"], key="in_PhoneService")
        st.selectbox("Multiple lines", config.CATEGORY_OPTIONS["MultipleLines"], key="in_MultipleLines")
        st.selectbox("Internet service", config.CATEGORY_OPTIONS["InternetService"], key="in_InternetService")
        st.selectbox("Online security", config.CATEGORY_OPTIONS["OnlineSecurity"], key="in_OnlineSecurity")
        st.selectbox("Online backup", config.CATEGORY_OPTIONS["OnlineBackup"], key="in_OnlineBackup")
    with c2:
        st.selectbox("Device protection", config.CATEGORY_OPTIONS["DeviceProtection"], key="in_DeviceProtection")
        st.selectbox("Tech support", config.CATEGORY_OPTIONS["TechSupport"], key="in_TechSupport")
        st.selectbox("Streaming TV", config.CATEGORY_OPTIONS["StreamingTV"], key="in_StreamingTV")
        st.selectbox("Streaming movies", config.CATEGORY_OPTIONS["StreamingMovies"], key="in_StreamingMovies")

if st.session_state.get("active_example_label"):
    st.caption(f"Form pre-filled from example: {st.session_state['active_example_label']}")

btn_col1, btn_col2, _ = st.columns([2, 1, 3])
with btn_col1:
    st.button(
        "Run prediction", type="primary", use_container_width=True,
        disabled=(model is None), on_click=_run_prediction,
    )
with btn_col2:
    st.button("Reset", use_container_width=True, on_click=_reset_form)


# ----------------------------------------------------------------------
# Main — results
# ----------------------------------------------------------------------
result: model_utils.PredictionResult | None = st.session_state.get("prediction_result")

if result is None:
    st.markdown(
        '<div class="empty-state">Fill in the customer&rsquo;s details above and run a '
        'prediction to see their churn risk here.</div>',
        unsafe_allow_html=True,
    )
elif result.error:
    st.error(result.error)
    if result.hint:
        st.info(result.hint)
    if result.raw_exception:
        with st.expander("Technical details"):
            st.code(result.raw_exception)
else:
    prob = result.probability
    if prob is not None:
        level = "low" if prob < config.RISK_LOW_MAX else ("medium" if prob < config.RISK_MEDIUM_MAX else "high")
        pct_display = f"{prob * 100:.1f}%"
    else:
        label_str = str(result.predicted_label).strip().lower()
        level = "high" if label_str in ("yes", "1", "true", "churn") else "low"
        pct_display = "\u2014"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    rc1, rc2 = st.columns([1, 1])

    with rc1:
        st.markdown(f'<div class="hero-number">{pct_display}</div>', unsafe_allow_html=True)
        st.caption(
            "Estimated churn probability" if prob is not None
            else "Model doesn't expose calibrated probabilities — showing predicted class only"
        )
        st.markdown(styles.risk_badge_html(level), unsafe_allow_html=True)

        raw_row = st.session_state.get("last_raw_row", {})
        signals = model_utils.risk_signals(raw_row)
        if signals:
            st.markdown("**Risk factors observed**")
            st.caption("General churn correlates for this dataset — not the model's own attribution.")
            items_html = "".join(f"<li>{s}</li>" for s in signals)
            st.markdown(f'<ul class="factor-list">{items_html}</ul>', unsafe_allow_html=True)

        if level != "low":
            recs = model_utils.recommendations_for(raw_row)
            st.markdown("**Suggested next steps**")
            items_html = "".join(f"<li>{r}</li>" for r in recs)
            st.markdown(f'<ul class="rec-list">{items_html}</ul>', unsafe_allow_html=True)

    with rc2:
        if prob is not None:
            st.plotly_chart(
                styles.build_gauge_figure(prob * 100, level),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        importance = model_utils.get_global_feature_importance(model)
        if importance:
            st.markdown("**What the model leans on most, overall**")
            imp_df = pd.DataFrame(importance, columns=["Feature", "Importance"]).set_index("Feature")
            st.bar_chart(imp_df, height=200)

    if st.session_state.get("active_example_actual"):
        st.markdown(
            '<div class="demo-note">Demo record — actual historical outcome in the training data: '
            f'<strong>{st.session_state["active_example_actual"]}</strong></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Predictions are estimates from the loaded model, not guarantees.")
