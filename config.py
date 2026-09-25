"""
config.py — single source of truth for the app's data schema and tunables.

If you ever need to point this app at a different dataset or adjust risk
bands, this is the only file you should need to touch.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
APP_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = APP_DIR / "model.pkl"  # drop your trained model here

# ----------------------------------------------------------------------
# Raw feature schema — mirrors the original Telco Customer Churn CSV
# columns exactly (customerID and Churn excluded), in the order a
# notebook typically ends up with after `df.drop(['customerID', 'Churn'],
# axis=1)`. This is the column set/order the app builds for the model —
# see the "Adapting to your model" section in README.md if your model
# expects something else (e.g. pre-encoded/one-hot columns).
# ----------------------------------------------------------------------
RAW_COLUMN_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Exact category strings, confirmed against the uploaded training CSV
# (7,043 rows) — not guessed, so selectboxes can never send the model a
# category it has never seen.
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

# Numeric ranges/defaults, sourced from the uploaded training CSV
TENURE_RANGE = (0, 72)
TENURE_DEFAULT = 12
MONTHLY_CHARGES_RANGE = (18.0, 119.0)
MONTHLY_CHARGES_DEFAULT = 65.0
TOTAL_CHARGES_DEFAULT = 780.0

TRAINING_ROWS = 7043
TRAINING_CHURN_RATE = 1869 / 7043  # matches the uploaded CSV exactly

# ----------------------------------------------------------------------
# Label encoding for customer_churn_model.pkl — reverse-engineered from
# the actual model + training CSV (no notebook access). Each mapping is
# sorted(unique values).index(value), i.e. what scikit-learn's
# LabelEncoder produces by default. See prepare_model_input() in
# model_utils.py for how this was verified (95.5% match against the
# model's real predictions on the training set).
# ----------------------------------------------------------------------
LABEL_ENCODINGS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
        "Electronic check": 2, "Mailed check": 3,
    },
}

# ----------------------------------------------------------------------
# Risk bands — these are reasonable starting cut points, not derived
# from your model's actual calibration. Tune them once you know how
# your model's probabilities are distributed.
# ----------------------------------------------------------------------
RISK_LOW_MAX = 0.35
RISK_MEDIUM_MAX = 0.65

# ----------------------------------------------------------------------
# Built-in demo customers — four real rows sampled from the uploaded
# training CSV, used only for the "Try an example" shortcut. Each one's
# `actual_churn` is that row's real historical label, shown after a
# prediction purely so you can sanity-check the model against a known
# answer. It is never implied for a genuine, unknown customer.
# ----------------------------------------------------------------------
SAMPLE_CUSTOMERS = [
    {
        "label": "New fiber customer, no add-ons",
        "actual_churn": "Yes",
        "values": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 70.70, "TotalCharges": 151.65,
        },
    },
    {
        "label": "Long-tenure family plan, fully loaded",
        "actual_churn": "No",
        "values": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 69, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 113.25, "TotalCharges": 7895.15,
        },
    },
    {
        "label": "Established fiber customer, no security/support",
        "actual_churn": "Yes",
        "values": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 47, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 99.35, "TotalCharges": 4749.15,
        },
    },
    {
        "label": "Mid-tenure DSL household, one-year contract",
        "actual_churn": "No",
        "values": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 35, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 62.15, "TotalCharges": 2215.45,
        },
    },
]
