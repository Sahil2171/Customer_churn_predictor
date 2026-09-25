# Customer Churn Predictor

A Streamlit front end for `customer_churn_model.pkl` — enter a telecom
customer's account details and get back a churn-risk estimate, built on
the Telco Customer Churn dataset.

## What's in here

```
app.py                  Main Streamlit app — layout, form, results panel
config.py                Feature schema, category options, colors, demo customers
model_utils.py            Model loading, prediction, encoding, explainability
styles.py                 CSS theme + gauge chart
requirements.txt          Pinned dependencies
.streamlit/config.toml    Streamlit theme (dark, teal accent)
model.pkl                 Your trained model (bundled)
```

## Quickstart

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`. `model.pkl` is already in this
folder, so predictions work immediately — no setup beyond `pip install`.

## About the model, and how its input encoding was figured out

`customer_churn_model.pkl` unpickles to a **dict** — `{"model": <a
RandomForestClassifier>, "features_name": [...]}` — not a scikit-learn
Pipeline. `model_utils.py` unwraps that automatically. The classifier
itself is bare, meaning it expects its 19 input columns **already
numeric** — categorical columns like `Contract` or `InternetService`
label-encoded to integers, not left as text.

There was no access to the training notebook (the Colab link requires
Google sign-in), so rather than guess, the exact encoding was verified
empirically: run the real model against the full training CSV with a
candidate encoding (each category mapped to
`sorted(unique values).index(value)` — what scikit-learn's
`LabelEncoder` produces by default) and check its predictions against
the real `Churn` column.

- **With this encoding: 95.5% accuracy**, balanced across both classes
  (97% precision/recall on "No", 91–92% on "Yes" — see the full
  classification report by running the snippet at the bottom of this
  section).
- **Control test, deliberately reversed encoding: 65.3% accuracy** —
  worse than just always guessing "No" (73.5%), confirming the mapping
  direction matters and 95.5% isn't a coincidence.

The resulting mapping lives in `config.LABEL_ENCODINGS` and is applied
in `model_utils.prepare_model_input()`. If you retrain this model later
with different preprocessing, that's the one place to update — and the
same verification trick (predict on the training CSV, compare to the
real labels) is the fastest way to re-confirm a new encoding without
digging through notebook cells.

<details>
<summary>Reproduce the verification yourself</summary>

```python
import pandas as pd, joblib
from config import RAW_COLUMN_ORDER, LABEL_ENCODINGS

d = joblib.load("model.pkl")
clf = d["model"]

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
X = df[RAW_COLUMN_ORDER].copy()
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
for col, mapping in LABEL_ENCODINGS.items():
    X[col] = X[col].map(mapping)

y_true = (df["Churn"] == "Yes").astype(int)
acc = (clf.predict(X) == y_true).mean()
print(acc)  # 0.9546
```
</details>

### Why scikit-learn is pinned to exactly 1.6.1

Loading the model under a mismatched scikit-learn version raises an
`InconsistentVersionWarning` naming the version it was actually trained
with — that's how the exact pin in `requirements.txt` was confirmed
(tested here: zero warnings under 1.6.1, a warning under 1.8.0). It
still *worked* under 1.8.0 in testing, just with a warning, but version
drift between training and serving is a real risk for pickled
scikit-learn models — worth keeping pinned rather than letting it float
to "latest" on a fresh install. If you ever retrain and re-save the
model, check the version the same way (`pip show scikit-learn`
immediately after training) and update the pin.

## If you swap in a different or retrained model

Drop the new file in as `model.pkl` (or upload it via the sidebar) and
re-run. Two things to check if predictions then fail or look wrong:

1. **Sidebar feature-count check** — if it shows a feature-count
   mismatch, your new model expects a different number of inputs than
   this form sends (19 raw columns). That usually means it wants
   one-hot-expanded features rather than label-encoded ones.
2. **`model_utils.prepare_model_input()`** — this is the one function
   that turns the form's raw values into whatever the model expects.
   Update `config.LABEL_ENCODINGS`, or replace the encoding step
   entirely (e.g. `pd.get_dummies(...)` if your model expects one-hot
   columns), to match your new model's actual training preprocessing.
   The verification trick above (predict on the training CSV, diff
   against real labels) is the fastest way to confirm you got it right.

If the model file itself fails to load, the app shows a plain-language
explanation for the two most common causes (a custom class pickle can't
find, or a library-version mismatch) with a "Technical details"
expander showing the full traceback underneath.

## Deploying

**Streamlit Community Cloud** (free, simplest):
1. Push this folder to a GitHub repo (`model.pkl` included — it's 22 MB,
   under GitHub's 100 MB file limit, so no Git LFS needed).
2. [share.streamlit.io](https://share.streamlit.io) → New app → point
   it at the repo, `app.py` as the entry point.
3. It installs `requirements.txt` and picks up `.streamlit/config.toml`
   automatically.

**Hugging Face Spaces** or **Render** work too (choose the Streamlit
SDK / a Python web service respectively) — the file structure here
needs no changes for either.

## Design notes

- Theme colors live in `.streamlit/config.toml` (Streamlit's own,
  stable theming mechanism); `styles.py` layers finer custom-component
  CSS and animation on top, targeting a handful of Streamlit's more
  stable CSS hooks plus this app's own markup. If a Streamlit upgrade
  ever changes how something looks, that file is the first place to
  check — pin `streamlit==1.38.0` (already in requirements.txt) to
  avoid that entirely.
- Animation is deliberately light-touch rather than built on
  `streamlit-lottie`: button/input transitions throughout, plus one
  orchestrated reveal (fade + rise) when a result appears, instead of
  scattered hover effects on every element. This also avoids adding a
  dependency that fetches an animation file over the network at
  runtime — one less thing that can break in a locked-down deployment
  environment. If you'd like a Lottie animation somewhere specific,
  `streamlit-lottie` + `streamlit_lottie.st_lottie(url_or_json)` is a
  two-line addition.
- The "What the model leans on most" chart and "Risk factors observed"
  list are deliberately kept visually and textually separate: the
  first is the model's actual (global, not per-prediction)
  `feature_importances_`; the second is a general, well-known set of
  churn correlates for this dataset (month-to-month contracts, low
  tenure, fiber-without-add-ons, electronic check) applied as simple
  rules against this customer's inputs — not the model's own reasoning
  about this specific prediction. True per-prediction explainability
  (SHAP/LIME) was left out to keep the dependency footprint light; it's
  a reasonable next addition if you want it.

## Known limitations

- Risk-band cutoffs (Low/Medium/High, in `config.py`) are reasonable
  starting points, not derived from this model's actual probability
  calibration — worth revisiting once you have a sense of how its
  scores are distributed.
- The four "Try an example" customers are real rows from the training
  CSV, not held-out data — the model has seen them before, so treat
  them as a sanity check / demo, not a test of generalization.
