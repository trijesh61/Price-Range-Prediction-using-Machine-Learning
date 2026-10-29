
from pathlib import Path

import joblib
import numbers
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CodeX Beverage: Price Prediction",
    page_icon="🥤",
    layout="wide"
)


# =========================================================
# 2. LOAD SAVED MODEL AND PREPROCESSING FILES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

try:
    model = joblib.load(MODEL_DIR / "best_price_range_model.pkl")
    le = joblib.load(MODEL_DIR / "target_label_encoder.pkl")
    ohe = joblib.load(MODEL_DIR / "onehot_encoder.pkl")
    ordinal_mappings = joblib.load(MODEL_DIR / "ordinal_mappings.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.info("Check that all five .pkl files are inside the model folder.")
    st.stop()


def normalize_encoder_categories(encoder):
    """Normalize saved OneHotEncoder category arrays to consistent dtypes.

    Some saved encoders contain numeric categories stored in object arrays.
    Recent NumPy/scikit-learn combinations can fail when checking those values.
    Keep text categories as strings and numeric categories as numeric arrays.
    """
    normalized_categories = []

    for categories in encoder.categories_:
        values = list(categories)

        if all(isinstance(value, (str, np.str_)) for value in values):
            normalized_categories.append(np.asarray(values, dtype=str))
        elif all(
            isinstance(value, numbers.Number) and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            normalized_categories.append(np.asarray(values, dtype=np.float64))
        else:
            # Preserve categories that are not a simple homogeneous string/numeric set.
            normalized_categories.append(np.asarray(values))

    encoder.categories_ = normalized_categories
    return encoder


ohe = normalize_encoder_categories(ohe)



# =========================================================
# 3. CUSTOM CSS - ADAPTS TO STREAMLIT THEME
# =========================================================

st.markdown(
    """
    <style>
    /* Follow Streamlit's active theme */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1 {
        text-align: center;
        color: var(--text-color);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: var(--text-color);
        opacity: 0.75;
        margin-bottom: 2rem;
    }

    h2, h3, label, p {
        color: var(--text-color);
    }

    /* Theme-compatible buttons */
    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Theme-compatible input fields */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 7px;
    }

    /* Prediction confidence metric */
    [data-testid="stMetricValue"] {
        font-weight: 700;
    }

    /* Keep the layout usable on smaller screens */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 1.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. HEADER
# =========================================================

st.title("CodeX Beverage: Price Prediction")


# =========================================================
# 5. FEATURE ENGINEERING
#    Matches the formulas in the training notebook
# =========================================================

def engineer_features(raw):
    data = raw.copy()

    # Standardize category spelling as done in the notebook
    data["zone"] = data["zone"].replace({
        "Metor": "Metro",
        "urbna": "Urban"
    })

    data["current_brand"] = data["current_brand"].replace({
        "Establishd": "Established",
        "newcomer": "Newcomer"
    })

    # Age group
    def age_groups(age):
        if 18 <= age <= 25:
            return "18-25"
        elif 26 <= age <= 35:
            return "26-35"
        elif 36 <= age <= 45:
            return "36-45"
        elif 46 <= age <= 55:
            return "46-55"
        elif 56 <= age <= 70:
            return "56-70"
        elif age > 70:
            return "70+"
        return np.nan

    data["age_group"] = data["age"].apply(age_groups)
    data = data.drop(columns=["age"])

    # Frequency score
    data["consume_frequency(weekly)_score"] = (
        data["consume_frequency(weekly)"].map({
            "0-2 times": 1,
            "3-4 times": 2,
            "5-7 times": 3
        })
    )

    # Awareness score
    data["awareness_of_other_brands_score"] = (
        data["awareness_of_other_brands"].map({
            "0 to 1": 1,
            "2 to 4": 2,
            "above 4": 3
        })
    )

    # Exact cf_ab_score formula from notebook
    data["cf_ab_score"] = (
        data["consume_frequency(weekly)_score"]
        /
        (
            data["awareness_of_other_brands_score"]
            + data["consume_frequency(weekly)_score"]
        )
    ).round(2)
    data["cf_ab_score"] = pd.to_numeric(
        data["cf_ab_score"], errors="coerce"
    ).astype(np.float64)

    # Zone score
    data["zone_score"] = data["zone"].map({
        "Metro": 4,
        "Urban": 3,
        "Rural": 1,
        "Semi-Urban": 2
    })

    # Income score
    data["income_levels_score"] = data["income_levels"].map({
        "<10L": 1,
        "10L - 15L": 2,
        "16L - 25L": 3,
        "26L - 35L": 4,
        "> 35L": 5,
        "Not Reported": 0
    })

    # Exact zas_score formula
    data["zas_score"] = (
        data["zone_score"] * data["income_levels_score"]
    )

    # Exact bsi formula
    data["bsi"] = (
        (data["current_brand"] != "Established")
        &
        data["reasons_for_choosing_brands"].isin(
            ["Price", "Quality"]
        )
    ).astype(int)

    # Remove the columns excluded from X in the notebook
    data = data.drop(
        columns=[
            "respondent_id",
            "price_range",
            "awareness_of_other_brands_score",
            "consume_frequency(weekly)_score",
            "income_levels_score",
            "zone_score"
        ],
        errors="ignore"
    )

    # Apply the same ordinal mappings as training
    for column, mapping in ordinal_mappings.items():
        if column not in data.columns:
            raise ValueError(
                f"Required ordinal feature '{column}' is missing."
            )

        original_values = data[column].copy()
        data[column] = data[column].map(mapping)

        if data[column].isna().any():
            invalid = original_values[data[column].isna()].tolist()
            raise ValueError(
                f"Unknown category in '{column}': {invalid}"
            )

    # Validate engineered values before one-hot encoding
    for column in ["cf_ab_score", "zas_score", "bsi"]:
        if column in data.columns and data[column].isna().any():
            raise ValueError(
                f"Feature '{column}' contains missing values. "
                "Check the input categories."
            )

    return data


# =========================================================
# 6. PREPROCESS INPUT FOR MODEL PREDICTION
# =========================================================

def preprocess_input(raw):
    data = engineer_features(raw)

    # Use the same categorical feature names as the fitted encoder
    categorical_columns = list(ohe.feature_names_in_)

    missing_columns = [
        col for col in categorical_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing categorical columns: {missing_columns}"
        )

    # Match each input column's dtype to the categories stored in the encoder.
    # This avoids object-dtype numeric categories causing np.isnan TypeErrors.
    for index, column in enumerate(categorical_columns):
        categories = ohe.categories_[index]

        if np.issubdtype(categories.dtype, np.number):
            data[column] = pd.to_numeric(data[column], errors="raise").astype(np.float64)
        elif np.issubdtype(categories.dtype, np.str_):
            data[column] = data[column].astype(str)

    # One-hot encode
    encoded_array = ohe.transform(data[categorical_columns])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=ohe.get_feature_names_out(categorical_columns),
        index=data.index
    )

    # Drop original categorical features and join encoded features
    data = data.drop(columns=categorical_columns)
    data = pd.concat([data, encoded_df], axis=1)

    # Match the exact training feature names and order
    missing_features = [
        col for col in feature_columns
        if col not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    data = data.reindex(columns=feature_columns)

    # Force numeric types; fail if a string cannot be converted
    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise"
        )

    data = data.astype(np.float64)

    if data.isna().any().any():
        bad_columns = data.columns[
            data.isna().any()
        ].tolist()

        raise ValueError(
            f"Missing values in processed input: {bad_columns}"
        )

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError(
            "Processed input contains infinite or invalid values."
        )

    return data


# =========================================================
# 7. INPUT FORM - FOUR COLUMNS
# =========================================================

with st.form("prediction_form"):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=25
        )

        income_levels = st.selectbox(
            "Income Level",
            [
                "<10L",
                "10L - 15L",
                "16L - 25L",
                "26L - 35L",
                "> 35L",
                "Not Reported"
            ]
        )

        awareness = st.selectbox(
            "Awareness of Other Brands",
            ["0 to 1", "2 to 4", "above 4"]
        )

        packaging = st.selectbox(
            "Packaging Preference",
            ["Plastic", "Glass", "Can", "Carton"]
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

        frequency = st.selectbox(
            "Consume Frequency (weekly)",
            ["0-2 times", "3-4 times", "5-7 times"]
        )

        reason = st.selectbox(
            "Reasons for Choosing Brands",
            ["Price", "Quality", "Brand Reputation", "Availability"]
        )

        health = st.selectbox(
            "Health Concerns",
            [
                "Low (Not very concerned)",
                "Medium (Moderately health-conscious)",
                "High (Very health-conscious)"
            ]
        )

    with col3:
        zone = st.selectbox(
            "Zone",
            ["Metro", "Urban", "Semi-Urban", "Rural"]
        )

        current_brand = st.selectbox(
            "Current Brand",
            ["Established", "Newcomer"]
        )

        flavor = st.selectbox(
            "Flavor Preference",
            ["Classic", "Fruity", "Spicy", "Sweet", "Other"]
        )

        situation = st.selectbox(
            "Typical Consumption Situations",
            ["At Home", "Work", "Travel", "Social Gatherings", "Other"]
        )

    with col4:
        occupation = st.selectbox(
            "Occupation",
            ["Student", "Salaried", "Business", "Self-Employed", "Other"]
        )

        consumption_size = st.selectbox(
            "Preferable Consumption Size",
            [
                "Small (250 ml)",
                "Medium (500 ml)",
                "Large (1 L)"
            ]
        )

        purchase_channel = st.selectbox(
            "Purchase Channel",
            ["Retail Store", "Supermarket", "Online", "Other"]
        )

    st.write("")

    submitted = st.form_submit_button(
        "Calculate Price Range",
        use_container_width=False
    )


# =========================================================
# 8. PREDICTION
# =========================================================

if submitted:
    raw_input = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "zone": zone,
        "occupation": occupation,
        "income_levels": income_levels,
        "consume_frequency(weekly)": frequency,
        "current_brand": current_brand,
        "preferable_consumption_size": consumption_size,
        "awareness_of_other_brands": awareness,
        "reasons_for_choosing_brands": reason,
        "flavor_preference": flavor,
        "purchase_channel": purchase_channel,
        "packaging_preference": packaging,
        "health_concerns": health,
        "typical_consumption_situations": situation
    }])

    try:
        processed_input = preprocess_input(raw_input)

        # Keep the same feature order used when training the model.
        prediction = model.predict(processed_input)
        predicted_label = le.inverse_transform(
            np.asarray(prediction).astype(int)
        )[0]

        st.success(f"Predicted Price Range: {predicted_label}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_input)[0]
            confidence = float(np.max(probabilities)) * 100
            st.metric("Model confidence", f"{confidence:.2f}%")

    except Exception as error:
        st.error("Prediction failed. Check the error details below.")
        st.exception(error)
