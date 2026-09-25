from __future__ import annotations

import pickle
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import RAW_COLUMN_ORDER, LABEL_ENCODINGS

POSITIVE_LABELS = {"yes", "1", "true", "churn"}

_COMMON_MODEL_KEYS = ("model", "clf", "classifier", "estimator", "best_estimator_")


def _unwrap_model_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        for key in _COMMON_MODEL_KEYS:
            candidate = obj.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                return candidate
    return obj


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

@dataclass
class LoadResult:
    model: Any = None
    error: Optional[str] = None
    hint: Optional[str] = None
    raw_exception: Optional[str] = None
    file_missing: bool = False


def _friendly_error(exc: Exception) -> tuple[str, str]:
    """Translate a raw unpickling exception into a plain-language
    explanation plus a concrete fix, based on the two failure modes that
    most often break a scikit-learn pickle moved between environments."""
    msg = str(exc)
    name = type(exc).__name__

    if isinstance(exc, (ModuleNotFoundError, ImportError)) or (
        isinstance(exc, AttributeError) and "Can't get attribute" in msg
    ):
        return (
            "The model file references a class Python can't find here — "
            "often a custom transformer or helper class that was defined "
            "inline in a notebook.",
            "If your training notebook defines any custom classes (a "
            "custom transformer, a wrapper, etc.), copy that exact class "
            "definition into model_utils.py and import it before "
            "unpickling — pickle needs the class importable under the "
            "same module path it was saved with.",
        )

    version_markers = ("_xp", "incompatible", "version", "unsupported pickle protocol", "node array")
    if isinstance(exc, KeyError) or any(m in msg.lower() for m in version_markers):
        return (
            "This looks like a library-version mismatch between the "
            "environment the model was trained in and this one — a common "
            "cause is scikit-learn, scipy, or numpy drift between versions.",
            "Run `pip show scikit-learn scipy numpy` in your training "
            "environment (e.g. the Colab notebook) and pin those exact "
            "versions in requirements.txt, or simplest: re-run your "
            "`pickle.dump(...)` / `joblib.dump(...)` from that same "
            "environment right before you deploy.",
        )

    return (f"Something went wrong ({name}).", "Expand the details below for the full traceback.")


@st.cache_resource(show_spinner=False)
def load_model_from_path(path: str) -> LoadResult:
    try:
        import joblib
        model = _unwrap_model_dict(joblib.load(path))
        return LoadResult(model=model)
    except FileNotFoundError:
        return LoadResult(error="No model file found at that path.", file_missing=True)
    except Exception:  # noqa: BLE001 — deliberately broad; this is the loader boundary
        try:
            with open(path, "rb") as f:
                model = _unwrap_model_dict(pickle.load(f))
            return LoadResult(model=model)
        except FileNotFoundError:
            return LoadResult(error="No model file found at that path.", file_missing=True)
        except Exception as exc2:  # noqa: BLE001
            explanation, hint = _friendly_error(exc2)
            return LoadResult(error=explanation, hint=hint, raw_exception=traceback.format_exc())


def load_model_from_upload(uploaded_file) -> LoadResult:
    try:
        import joblib
        uploaded_file.seek(0)
        model = _unwrap_model_dict(joblib.load(uploaded_file))
        return LoadResult(model=model)
    except Exception:  # noqa: BLE001
        try:
            uploaded_file.seek(0)
            model = _unwrap_model_dict(pickle.load(uploaded_file))
            return LoadResult(model=model)
        except Exception as exc2:  # noqa: BLE001
            explanation, hint = _friendly_error(exc2)
            return LoadResult(error=explanation, hint=hint, raw_exception=traceback.format_exc())


# ----------------------------------------------------------------------
# Adapting raw form input to what the model expects
# ----------------------------------------------------------------------

def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the raw form input to match what customer_churn_model.pkl
    actually expects: a bare RandomForestClassifier (not a Pipeline) with
    feature_names_in_ equal to config.RAW_COLUMN_ORDER, so it needs the
    same 19 columns but with each categorical column label-encoded to an
    integer rather than left as text.

    There was no training notebook available to confirm the exact
    encoding, so it was reverse-engineered: config.LABEL_ENCODINGS maps
    each category to sorted(unique values in the training CSV).index(value)
    — the same mapping scikit-learn's LabelEncoder produces by default —
    and running the real model against the full training CSV with this
    mapping reproduces its predictions with 95.5% accuracy, balanced
    across both classes (vs. 65%, worse than always guessing "No", for a
    deliberately reversed mapping). That's a strong enough match to be
    the real encoding rather than a coincidence, but if you retrain this
    model with different preprocessing, update LABEL_ENCODINGS in
    config.py (or this function) to match.
    """
    df = df.copy()
    for col, mapping in LABEL_ENCODINGS.items():
        df[col] = df[col].map(mapping)
    return df


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------

def _positive_class_index(model) -> int:
    """Works out which column of predict_proba's output is "churned",
    regardless of whether the model was trained on "Yes"/"No" strings or
    0/1 integers."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1
    classes = list(classes)
    for i, c in enumerate(classes):
        if str(c).strip().lower() in POSITIVE_LABELS:
            return i
    return 1 if len(classes) > 1 else 0


@dataclass
class PredictionResult:
    probability: Optional[float] = None
    predicted_label: Any = None
    error: Optional[str] = None
    hint: Optional[str] = None
    raw_exception: Optional[str] = None
    n_features_expected: Optional[int] = None
    n_features_given: Optional[int] = None


def predict(model, raw_row: dict) -> PredictionResult:
    """raw_row: a dict keyed exactly like config.RAW_COLUMN_ORDER, in
    human-readable form (strings for categoricals, numbers for tenure /
    charges, SeniorCitizen as 0/1). Passed through prepare_model_input()
    above, which applies the verified label encoding — see there if you
    swap in a model with different preprocessing."""

    if model is None:
        return PredictionResult(error="No model is loaded.", hint="Upload a model.pkl file in the sidebar first.")

    df = pd.DataFrame([raw_row], columns=RAW_COLUMN_ORDER)
    df = prepare_model_input(df)

    n_expected = getattr(model, "n_features_in_", None)
    n_given = df.shape[1]

    try:
        predicted_label = model.predict(df)[0]
    except Exception as exc:  # noqa: BLE001
        explanation, hint = _friendly_error(exc)
        if n_expected is not None and n_expected != n_given:
            explanation = (
                f"The model expects {n_expected} input feature(s) but this app "
                f"sent {n_given}. {explanation}"
            )
            hint = (
                "This usually means the model was trained on pre-encoded or "
                "one-hot-expanded features rather than a Pipeline that accepts "
                "raw columns. Open prepare_model_input() in model_utils.py and "
                "replace it with the exact encoding steps from your training "
                "notebook. " + (hint or "")
            )
        return PredictionResult(
            error=explanation, hint=hint, raw_exception=traceback.format_exc(),
            n_features_expected=n_expected, n_features_given=n_given,
        )

    probability = None
    if hasattr(model, "predict_proba"):
        try:
            idx = _positive_class_index(model)
            probability = float(model.predict_proba(df)[0][idx])
        except Exception:  # noqa: BLE001 — probability is a bonus, never fatal
            probability = None

    return PredictionResult(
        probability=probability, predicted_label=predicted_label,
        n_features_expected=n_expected, n_features_given=n_given,
    )


# ----------------------------------------------------------------------
# Best-effort GLOBAL feature importance — not per-instance explainability
# ----------------------------------------------------------------------

def get_global_feature_importance(model, top_n: int = 8):
    """Returns [(feature_name, importance), ...] or None if the model
    doesn't expose anything usable. This reflects what the model leans on
    overall across all training data — not a per-customer explanation of
    one specific prediction. Keep that distinction in any UI copy."""
    try:
        final_estimator = model
        feature_names = None

        if hasattr(model, "steps"):  # sklearn Pipeline
            final_estimator = model.steps[-1][1]
            try:
                feature_names = list(model[:-1].get_feature_names_out())
            except Exception:  # noqa: BLE001
                feature_names = None
        elif hasattr(model, "feature_names_in_"):  # bare estimator fit on a DataFrame
            feature_names = list(model.feature_names_in_)

        importances = getattr(final_estimator, "feature_importances_", None)
        if importances is None:
            coef = getattr(final_estimator, "coef_", None)
            if coef is not None:
                importances = np.abs(np.asarray(coef)).ravel()

        if importances is None:
            return None

        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        pairs = sorted(zip(feature_names, importances), key=lambda p: -abs(p[1]))
        return pairs[:top_n]
    except Exception:  # noqa: BLE001 — explainability is a bonus, never fatal
        return None


# ----------------------------------------------------------------------
# Rule-based risk signals & recommendations — general, well-known Telco
# churn correlates, NOT the model's own reasoning. Kept visually and
# textually separate from model-derived output so the two are never
# confused with each other.
# ----------------------------------------------------------------------

def risk_signals(raw_row: dict) -> list[str]:
    signals = []
    if raw_row.get("Contract") == "Month-to-month":
        signals.append("Month-to-month contract — no commitment period")
    if raw_row.get("tenure", 99) <= 6:
        signals.append("Very new customer (6 months or less)")
    if (raw_row.get("InternetService") == "Fiber optic"
            and raw_row.get("OnlineSecurity") == "No"
            and raw_row.get("TechSupport") == "No"):
        signals.append("Fiber internet with no security or tech-support add-ons")
    if raw_row.get("PaymentMethod") == "Electronic check":
        signals.append("Pays by electronic check")
    if raw_row.get("PaperlessBilling") == "Yes" and raw_row.get("tenure", 99) <= 12:
        signals.append("Paperless billing on a newer account")
    if raw_row.get("MonthlyCharges", 0) >= 90:
        signals.append("High monthly charges")
    return signals


def recommendations_for(raw_row: dict) -> list[str]:
    recs = []
    if raw_row.get("Contract") == "Month-to-month":
        recs.append("Offer an incentive to move to a 1- or 2-year contract")
    if raw_row.get("OnlineSecurity") == "No" or raw_row.get("TechSupport") == "No":
        recs.append("Bundle in security / tech-support add-ons at a discount")
    if raw_row.get("PaymentMethod") == "Electronic check":
        recs.append("Encourage a switch to automatic payment")
    if not recs:
        recs.append("Flag for proactive retention outreach")
    return recs
