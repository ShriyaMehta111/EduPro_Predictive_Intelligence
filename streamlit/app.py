
# ============================================================
# EDUPRO PREDICTIVE INTELLIGENCE DASHBOARD
# Unified Mentor Internship Project
# ============================================================

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduPro Predictive Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is expected to be inside:
# EduPro_Predictive_Modeling/
# OR:
# EduPro_Predictive_Modeling/streamlit/
#
# This resolver supports both situations.

APP_DIR = Path(__file__).resolve().parent


def find_project_root():
    candidates = [
        APP_DIR,
        APP_DIR.parent,
        APP_DIR.parent.parent,
    ]

    for candidate in candidates:
        if (
            (candidate / "data").exists()
            or (candidate / "models").exists()
            or (candidate / "results").exists()
        ):
            return candidate

    return APP_DIR.parent


BASE_DIR = find_project_root()

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELING_DIR = PROCESSED_DIR / "modeling"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# FILE SEARCH HELPER
# ============================================================

def first_existing(paths):
    """
    Return the first existing file from a list of possible paths.
    """
    for path in paths:
        try:
            if path.exists() and path.is_file():
                return path
        except Exception:
            pass

    return None


# ============================================================
# DATASET PATHS
# ============================================================

COURSE_MASTER_PATH = first_existing(
    [
        PROCESSED_DIR / "course_master_dataset.csv",
        DATA_DIR / "course_master_dataset.csv",
        DATA_DIR / "final_course_dataset.csv",
        BASE_DIR / "data" / "final_course_dataset.csv",
    ]
)

FEATURE_DATA_PATH = first_existing(
    [
        PROCESSED_DIR / "feature_engineered_dataset.csv",
        DATA_DIR / "feature_engineered_dataset.csv",
    ]
)

ENROLLMENT_DATA_PATH = first_existing(
    [
        MODELING_DIR / "enrollment_modeling_dataset.csv",
        PROCESSED_DIR / "enrollment_modeling_dataset.csv",
        DATA_DIR / "enrollment_modeling_dataset.csv",
    ]
)

REVENUE_DATA_PATH = first_existing(
    [
        MODELING_DIR / "revenue_modeling_dataset.csv",
        PROCESSED_DIR / "revenue_modeling_dataset.csv",
        DATA_DIR / "revenue_modeling_dataset.csv",
    ]
)

CATEGORY_DATA_PATH = first_existing(
    [
        MODELING_DIR / "category_revenue_modeling_dataset.csv",
        PROCESSED_DIR / "category_revenue_modeling_dataset.csv",
        DATA_DIR / "category_revenue_modeling_dataset.csv",
    ]
)


# ============================================================
# MODEL PATH RESOLVER
# ============================================================

def resolve_model_path(model_type):
    """
    Resolve model filenames used by different stages of the project.

    The project has used different filenames during development.
    The dashboard therefore checks all known valid filenames.
    """

    if model_type == "enrollment":

        candidates = [
            MODELS_DIR / "final_enrollment_model.pkl",
            MODELS_DIR / "enrollment_model.pkl",
            MODELS_DIR / "random_forest_enrollment_model.pkl",
            MODELS_DIR / "final_time_series_model.pkl",
        ]

    elif model_type == "revenue":

        candidates = [
            MODELS_DIR / "final_revenue_model.pkl",
            MODELS_DIR / "revenue_model.pkl",
            MODELS_DIR / "random_forest_revenue_model.pkl",
            MODELS_DIR / "final_time_series_model.pkl",
        ]

    elif model_type == "category":

        candidates = [
            MODELS_DIR / "final_category_revenue_model.pkl",
            MODELS_DIR / "category_revenue_model.pkl",
            MODELS_DIR / "final_time_series_category_revenue_model.pkl",
        ]

    else:
        return None

    return first_existing(candidates)


ENROLLMENT_MODEL_PATH = resolve_model_path("enrollment")
REVENUE_MODEL_PATH = resolve_model_path("revenue")
CATEGORY_MODEL_PATH = resolve_model_path("category")


# ============================================================
# RESULT FILE PATHS
# ============================================================

ENROLLMENT_IMPORTANCE_PATH = first_existing(
    [
        RESULTS_DIR / "enrollment_feature_importance.csv",
        RESULTS_DIR / "random_forest_enrollment_feature_importance.csv",
    ]
)

REVENUE_IMPORTANCE_PATH = first_existing(
    [
        RESULTS_DIR / "revenue_feature_importance.csv",
        RESULTS_DIR / "random_forest_revenue_feature_importance.csv",
    ]
)

CATEGORY_IMPORTANCE_PATH = first_existing(
    [
        RESULTS_DIR / "category_revenue_feature_importance.csv",
        RESULTS_DIR / "category_feature_importance.csv",
    ]
)

ENROLLMENT_FORECAST_PATH = first_existing(
    [
        RESULTS_DIR / "enrollment_forecast_predictions.csv",
        RESULTS_DIR / "final_enrollment_forecast.csv",
    ]
)

REVENUE_FORECAST_PATH = first_existing(
    [
        RESULTS_DIR / "revenue_forecast_predictions.csv",
        RESULTS_DIR / "course_revenue_forecast.csv",
    ]
)

CATEGORY_FORECAST_PATH = first_existing(
    [
        RESULTS_DIR / "category_revenue_forecast_predictions.csv",
        RESULTS_DIR / "category_revenue_forecast.csv",
    ]
)


# ============================================================
# SAFE DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(path_string):

    if not path_string:
        return None

    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ============================================================
# SAFE MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(path_string):

    if not path_string:
        return None, "No compatible model file was found."

    path = Path(path_string)

    if not path.exists():
        return None, f"Model file not found: {path}"

    try:

        model = joblib.load(path)

        return model, None

    except Exception as exc:

        return (
            None,
            f"Could not load {path.name}: {exc}",
        )


# ============================================================
# LOAD ALL DATA
# ============================================================

@st.cache_data(show_spinner=False)
def get_data():

    return {

        "course_master": load_csv(
            str(COURSE_MASTER_PATH) if COURSE_MASTER_PATH else ""
        ),

        "feature": load_csv(
            str(FEATURE_DATA_PATH) if FEATURE_DATA_PATH else ""
        ),

        "enrollment": load_csv(
            str(ENROLLMENT_DATA_PATH) if ENROLLMENT_DATA_PATH else ""
        ),

        "revenue": load_csv(
            str(REVENUE_DATA_PATH) if REVENUE_DATA_PATH else ""
        ),

        "category": load_csv(
            str(CATEGORY_DATA_PATH) if CATEGORY_DATA_PATH else ""
        ),

        "enrollment_importance": load_csv(
            str(ENROLLMENT_IMPORTANCE_PATH)
            if ENROLLMENT_IMPORTANCE_PATH
            else ""
        ),

        "revenue_importance": load_csv(
            str(REVENUE_IMPORTANCE_PATH)
            if REVENUE_IMPORTANCE_PATH
            else ""
        ),

        "category_importance": load_csv(
            str(CATEGORY_IMPORTANCE_PATH)
            if CATEGORY_IMPORTANCE_PATH
            else ""
        ),

        "enrollment_forecast": load_csv(
            str(ENROLLMENT_FORECAST_PATH)
            if ENROLLMENT_FORECAST_PATH
            else ""
        ),

        "revenue_forecast": load_csv(
            str(REVENUE_FORECAST_PATH)
            if REVENUE_FORECAST_PATH
            else ""
        ),

        "category_forecast": load_csv(
            str(CATEGORY_FORECAST_PATH)
            if CATEGORY_FORECAST_PATH
            else ""
        ),
    }


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_models():

    enrollment_model, enrollment_error = load_model(
        str(ENROLLMENT_MODEL_PATH)
        if ENROLLMENT_MODEL_PATH
        else ""
    )

    revenue_model, revenue_error = load_model(
        str(REVENUE_MODEL_PATH)
        if REVENUE_MODEL_PATH
        else ""
    )

    category_model, category_error = load_model(
        str(CATEGORY_MODEL_PATH)
        if CATEGORY_MODEL_PATH
        else ""
    )

    return {

        "enrollment": enrollment_model,

        "revenue": revenue_model,

        "category": category_model,

        "errors": {

            "enrollment": enrollment_error,

            "revenue": revenue_error,

            "category": category_error,
        },

        "paths": {

            "enrollment": str(ENROLLMENT_MODEL_PATH)
            if ENROLLMENT_MODEL_PATH
            else None,

            "revenue": str(REVENUE_MODEL_PATH)
            if REVENUE_MODEL_PATH
            else None,

            "category": str(CATEGORY_MODEL_PATH)
            if CATEGORY_MODEL_PATH
            else None,
        },
    }


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_number(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def money(value):

    return f"₹{max(0, float(value)):,.2f}"


def integer(value):

    return f"{max(0, int(round(float(value)))):,}"


def next_month(year, month):

    year = int(year)
    month = int(month)

    if month == 12:
        return year + 1, 1

    return year, month + 1


def month_features(year, month):

    month = int(month)

    quarter = ((month - 1) // 3) + 1

    return {

        "Month": month,

        "Quarter": quarter,

        "IsQuarterStart": int(
            month in [1, 4, 7, 10]
        ),

        "IsQuarterEnd": int(
            month in [3, 6, 9, 12]
        ),

        "MonthSin": np.sin(
            2 * np.pi * month / 12
        ),

        "MonthCos": np.cos(
            2 * np.pi * month / 12
        ),
    }


# ============================================================
# MODEL FEATURE DISCOVERY
# ============================================================

def model_feature_names(
    model,
    fallback_df=None,
    target_names=None
):

    if model is None:
        return []

    names = getattr(
        model,
        "feature_names_in_",
        None
    )

    if names is not None:
        return list(names)

    if hasattr(model, "named_steps"):

        for step in model.named_steps.values():

            names = getattr(
                step,
                "feature_names_in_",
                None
            )

            if names is not None:
                return list(names)

    if hasattr(model, "steps"):

        for _, step in model.steps:

            names = getattr(
                step,
                "feature_names_in_",
                None
            )

            if names is not None:
                return list(names)

    if fallback_df is not None:

        excluded = set(
            target_names or []
        )

        return [
            col
            for col in fallback_df.columns
            if col not in excluded
        ]

    return []


# ============================================================
# FEATURE ALIGNMENT
# ============================================================

def align_features(
    df,
    model,
    target_names=None
):

    if model is None:
        raise ValueError(
            "Model is not available."
        )

    features = model_feature_names(
        model,
        df,
        target_names
    )

    if not features:

        raise ValueError(
            "Unable to determine the model's expected feature columns."
        )

    out = df.copy()

    for col in features:

        if col not in out.columns:

            out[col] = 0

    out = out[features]

    for col in out.columns:

        if (
            out[col].dtype == "object"
            or str(out[col].dtype).startswith("category")
        ):

            out[col] = pd.to_numeric(
                out[col],
                errors="coerce"
            )

        else:

            out[col] = pd.to_numeric(
                out[col],
                errors="coerce"
            )

        out[col] = out[col].fillna(0)

    return out


# ============================================================
# PREDICTION HELPER
# ============================================================

def prediction_value(
    model,
    row,
    target_names
):

    if model is None:

        raise ValueError(
            "Required model is not loaded."
        )

    x = align_features(
        pd.DataFrame([row]),
        model,
        target_names
    )

    prediction = model.predict(x)

    if isinstance(prediction, (list, tuple, np.ndarray)):

        prediction = prediction[0]

    return max(
        0.0,
        float(prediction)
    )


# ============================================================
# COURSE FEATURE PREPARATION
# ============================================================

def infer_next_course_row(
    feature_df,
    course_id
):

    if feature_df is None:
        raise ValueError(
            "Feature-engineered dataset is unavailable."
        )

    if feature_df.empty:

        raise ValueError(
            "Feature-engineered dataset is empty."
        )

    data = feature_df.copy()

    if "CourseID" not in data.columns:

        raise ValueError(
            "CourseID column is missing."
        )

    if "YearMonth" in data.columns:

        data["YearMonth"] = (
            data["YearMonth"]
            .astype(str)
        )

        data = data.sort_values(
            ["CourseID", "YearMonth"]
        )

    rows = data[
        data["CourseID"].astype(str)
        == str(course_id)
    ]

    if rows.empty:

        raise ValueError(
            f"Course {course_id} was not found."
        )

    row = rows.iloc[-1].copy()

    if (
        "Year" in row.index
        and "Month" in row.index
    ):

        year, month = next_month(
            row["Year"],
            row["Month"]
        )

        row["Year"] = year
        row["Month"] = month

        row["YearMonth"] = (
            f"{year:04d}-{month:02d}"
        )

        row.update(
            month_features(
                year,
                month
            )
        )

        if "TimeIndex" in row.index:

            row["TimeIndex"] = (
                clean_number(
                    row.get("TimeIndex", 0)
                ) + 1
            )

        if "YearProgress" in row.index:

            row["YearProgress"] = (
                (month - 1) / 12.0
            )

    return row


# ============================================================
# COURSE INPUT TRANSFORMATION
# ============================================================

def apply_course_inputs(
    row,
    price,
    duration,
    level,
    experience,
    rating
):

    row = row.copy()

    price = float(price)
    duration = float(duration)
    experience = float(experience)
    rating = float(rating)

    row["CoursePrice"] = price

    row["CourseDuration"] = duration

    row["CourseRating"] = rating

    row["AvgTeacherExperience"] = experience

    row["AvgTeacherRating"] = rating


    # --------------------------------------------------------
    # COURSE TYPE
    # --------------------------------------------------------

    if "CourseType" in row.index:

        row["CourseType"] = (
            "Free"
            if price <= 0
            else "Paid"
        )

    if "CourseTypeEncoded" in row.index:

        row["CourseTypeEncoded"] = int(
            price > 0
        )


    # --------------------------------------------------------
    # TEACHER QUALITY
    # --------------------------------------------------------

    if "TeacherRatingScore" in row.index:

        row["TeacherRatingScore"] = rating

    if "TeacherQualityScore" in row.index:

        row["TeacherQualityScore"] = (
            rating * experience
        )


    # --------------------------------------------------------
    # EXPERIENCE FEATURES
    # --------------------------------------------------------

    if "ExperienceBucket" in row.index:

        row["ExperienceBucket"] = (
            "High"
            if experience >= 8
            else (
                "Medium"
                if experience >= 4
                else "Low"
            )
        )

    if "Experience_High" in row.index:

        row["Experience_High"] = int(
            experience >= 8
        )


    # --------------------------------------------------------
    # RATING FEATURES
    # --------------------------------------------------------

    if "RatingTier" in row.index:

        row["RatingTier"] = (
            "Excellent"
            if rating >= 4.5
            else (
                "Good"
                if rating >= 3.5
                else (
                    "Medium"
                    if rating >= 2.5
                    else "Low"
                )
            )
        )

    rating_columns = [
        "Rating_Excellent",
        "Rating_Good",
        "Rating_Medium",
        "Rating_Low",
    ]

    for col in rating_columns:

        if col in row.index:
            row[col] = 0

    if "Rating_Excellent" in row.index:

        row["Rating_Excellent"] = int(
            rating >= 4.5
        )

    if "Rating_Good" in row.index:

        row["Rating_Good"] = int(
            3.5 <= rating < 4.5
        )

    if "Rating_Medium" in row.index:

        row["Rating_Medium"] = int(
            2.5 <= rating < 3.5
        )

    if "Rating_Low" in row.index:

        row["Rating_Low"] = int(
            rating < 2.5
        )


    # --------------------------------------------------------
    # PRICE BAND
    # --------------------------------------------------------

    if "PriceBand" in row.index:

        row["PriceBand"] = (
            "Free"
            if price <= 0
            else (
                "Low"
                if price <= 100
                else (
                    "Medium"
                    if price <= 300
                    else "High"
                )
            )
        )

    price_columns = [
        "PriceBand_Free",
        "PriceBand_Low",
        "PriceBand_Medium",
        "PriceBand_High",
    ]

    for col in price_columns:

        if col in row.index:
            row[col] = 0

    if "PriceBand_Free" in row.index:

        row["PriceBand_Free"] = int(
            price <= 0
        )

    if "PriceBand_Low" in row.index:

        row["PriceBand_Low"] = int(
            0 < price <= 100
        )

    if "PriceBand_Medium" in row.index:

        row["PriceBand_Medium"] = int(
            100 < price <= 300
        )

    if "PriceBand_High" in row.index:

        row["PriceBand_High"] = int(
            price > 300
        )


    # --------------------------------------------------------
    # DURATION GROUP
    # --------------------------------------------------------

    duration_group = (
        "Short"
        if duration <= 10
        else (
            "Medium"
            if duration <= 30
            else "Long"
        )
    )

    if "DurationGroup" in row.index:

        row["DurationGroup"] = duration_group

    if "DurationBucket" in row.index:

        row["DurationBucket"] = duration_group

    duration_columns = [
        "Duration_Short",
        "Duration_Medium",
        "Duration_Long",
    ]

    for col in duration_columns:

        if col in row.index:
            row[col] = 0

    if "Duration_Short" in row.index:

        row["Duration_Short"] = int(
            duration <= 10
        )

    if "Duration_Medium" in row.index:

        row["Duration_Medium"] = int(
            10 < duration <= 30
        )

    if "Duration_Long" in row.index:

        row["Duration_Long"] = int(
            duration > 30
        )


    # --------------------------------------------------------
    # COURSE LEVEL
    # --------------------------------------------------------

    level_map = {
        "Beginner": 0,
        "Intermediate": 1,
        "Advanced": 2,
    }

    if "CourseLevel" in row.index:

        row["CourseLevel"] = level

    if "CourseLevelEncoded" in row.index:

        row["CourseLevelEncoded"] = (
            level_map.get(level, 0)
        )


    # --------------------------------------------------------
    # RATING GROUP
    # --------------------------------------------------------

    rating_group = (
        "Low"
        if rating < 2
        else (
            "Medium"
            if rating < 3
            else (
                "Good"
                if rating < 4
                else "Excellent"
            )
        )
    )

    if "RatingGroup" in row.index:

        row["RatingGroup"] = rating_group


    return row


# ============================================================
# MODEL STATUS
# ============================================================

def display_model_status(models):

    st.markdown("### Model Status")

    cols = st.columns(3)

    statuses = [
        (
            "Enrollment Model",
            models["enrollment"],
            models["errors"]["enrollment"],
        ),

        (
            "Revenue Model",
            models["revenue"],
            models["errors"]["revenue"],
        ),

        (
            "Category Revenue Model",
            models["category"],
            models["errors"]["category"],
        ),
    ]

    for col, (name, model, error) in zip(
        cols,
        statuses
    ):

        with col:

            if model is not None:

                col.success(
                    f"✓ {name} loaded"
                )

            else:

                col.warning(
                    f"⚠ {name} unavailable"
                )

                if error:
                    col.caption(error)


# ============================================================
# CUSTOM CSS
# ============================================================

# ============================================================
# CUSTOM CSS
# ============================================================

def add_common_css():

    css_path = Path(__file__).resolve().parent / "style.css"

    if not css_path.exists():
        st.warning(
            f"style.css was not found at: {css_path}"
        )
        return

    try:
        with open(css_path, "r", encoding="utf-8") as css_file:
            css = css_file.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )

    except Exception as exc:
        st.warning(
            f"Could not load style.css: {exc}"
        )

# ============================================================
# OVERVIEW PAGE
# ============================================================

def home_page(data, models):

    st.markdown(
        '<div class="main-title">EduPro Predictive Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Predict future course demand, course revenue and '
        'category-level revenue performance using historical EduPro data.'
        '</div>',
        unsafe_allow_html=True,
    )

    display_model_status(models)

    master = data["course_master"]
    feature = data["feature"]
    category = data["category"]

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    if master is not None:

        courses = (
            master["CourseID"]
            .nunique()
            if "CourseID" in master.columns
            else 0
        )

        categories = (
            master["CourseCategory"]
            .nunique()
            if "CourseCategory" in master.columns
            else 0
        )

    else:

        courses = 0
        categories = 0

    c1.metric(
        "Courses",
        f"{courses:,}"
    )

    c2.metric(
        "Categories",
        f"{categories:,}"
    )

    c3.metric(
        "Course-month records",
        f"{len(feature):,}"
        if feature is not None
        else "—"
    )

    c4.metric(
        "Category-month records",
        f"{len(category):,}"
        if category is not None
        else "—"
    )

    st.markdown(
        "### What this dashboard supports"
    )

    cols = st.columns(4)

    cards = [

        (
            "Course Demand",
            "Estimate future enrollment demand "
            "using course and instructor characteristics."
        ),

        (
            "Course Revenue",
            "Estimate expected course revenue "
            "using historical and course-level features."
        ),

        (
            "Category Revenue",
            "Forecast future revenue for a selected "
            "course category."
        ),

        (
            "Business Decisions",
            "Use model outputs, feature importance "
            "and category comparisons to support decisions."
        ),
    ]

    for col, (title, text) in zip(
        cols,
        cards
    ):

        with col:

            st.markdown(
                f"**{title}**"
            )

            st.write(text)

    st.markdown(
        "### Project Workflow"
    )

    st.write(
        """
        Historical EduPro data → Data preparation →
        Feature engineering → Predictive modeling →
        Forecasting → Dashboard → Business insights
        """
    )

    st.info(
        "Use the prediction pages for scenario testing, "
        "then use feature importance, category comparisons "
        "and business insights to interpret the results."
    )


# ============================================================
# COURSE PREDICTION PAGE
# ============================================================

def course_prediction_page(
    data,
    models,
    target="enrollment"
):

    if target == "enrollment":

        title = "Course Demand Prediction"

        model = models["enrollment"]

    else:

        title = "Course Revenue Prediction"

        model = models["revenue"]

    feature = data["feature"]

    master = data["course_master"]

    st.title(title)

    st.write(
        "Use course and instructor characteristics "
        "to generate a next-period prediction."
    )

    if model is None:

        st.error(
            models["errors"].get(
                target,
                "Required model could not be loaded."
            )
            or
            "Required model could not be loaded."
        )

        st.info(
            "The application automatically searches the "
            "project's models folder for supported model filenames."
        )

        return

    if feature is None:

        st.error(
            "Feature-engineered dataset is unavailable."
        )

        return

    if master is None:

        st.error(
            "Course master dataset is unavailable."
        )

        return

    if "CourseID" not in master.columns:

        st.error(
            "CourseID column is missing from the course dataset."
        )

        return

    course_ids = (
        master["CourseID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    course_ids = sorted(course_ids)

    if not course_ids:

        st.error(
            "No courses are available."
        )

        return

    course_id = st.selectbox(
        "Select course",
        course_ids
    )

    selected_rows = master[
        master["CourseID"].astype(str)
        == course_id
    ]

    if selected_rows.empty:

        st.error(
            "Selected course could not be found."
        )

        return

    course_info = selected_rows.iloc[0]

    left, right = st.columns(2)

    with left:

        st.markdown(
            "#### Course Information"
        )

        st.write(
            f"**Course:** "
            f"{course_info.get('CourseName', course_id)}"
        )

        st.write(
            f"**Category:** "
            f"{course_info.get('CourseCategory', '—')}"
        )

        st.write(
            f"**Current Type:** "
            f"{course_info.get('CourseType', '—')}"
        )

        st.write(
            f"**Current Level:** "
            f"{course_info.get('CourseLevel', '—')}"
        )

    with right:

        st.markdown(
            "#### Scenario Inputs"
        )

        price = st.number_input(
            "Course Price",
            min_value=0.0,
            value=clean_number(
                course_info.get(
                    "CoursePrice",
                    0
                )
            ),
            step=10.0,
        )

        duration = st.number_input(
            "Course Duration",
            min_value=1.0,
            value=clean_number(
                course_info.get(
                    "CourseDuration",
                    10
                )
            ),
            step=1.0,
        )

        level_options = [
            "Beginner",
            "Intermediate",
            "Advanced",
        ]

        current_level = str(
            course_info.get(
                "CourseLevel",
                "Beginner"
            )
        )

        level = st.selectbox(
            "Course Level",
            level_options,
            index=(
                level_options.index(
                    current_level
                )
                if current_level
                in level_options
                else 0
            ),
        )

        experience = st.number_input(
            "Instructor Experience (years)",
            min_value=0.0,
            value=clean_number(
                course_info.get(
                    "AvgTeacherExperience",
                    3
                )
            ),
            step=0.5,
        )

        rating = st.slider(
            "Instructor / Course Rating",
            min_value=0.0,
            max_value=5.0,
            value=min(
                5.0,
                max(
                    0.0,
                    clean_number(
                        course_info.get(
                            "AvgTeacherRating",
                            4
                        )
                    )
                )
            ),
            step=0.1,
        )

    if st.button(
        "Generate Prediction",
        type="primary"
    ):

        try:

            row = infer_next_course_row(
                feature,
                course_id
            )

            row = apply_course_inputs(
                row,
                price,
                duration,
                level,
                experience,
                rating,
            )

            prediction = prediction_value(
                model,
                row,
                [
                    "EnrollmentTarget",
                    "RevenueTarget",
                ],
            )

            st.divider()

            st.markdown(
                "### Prediction Result"
            )

            if target == "enrollment":

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Predicted Next-Period Enrollment",
                        integer(prediction)
                    )

                baseline = clean_number(
                    row.get(
                        "PreviousMonthEnrollment",
                        row.get(
                            "MonthlyEnrollment",
                            0
                        )
                    )
                )

                with c2:

                    if baseline > 0:

                        change = (
                            (prediction - baseline)
                            / baseline
                        ) * 100

                        st.metric(
                            "Change vs Previous Period",
                            f"{change:+.1f}%"
                        )

                    else:

                        st.metric(
                            "Previous Enrollment",
                            integer(baseline)
                        )

            else:

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Predicted Next-Period Revenue",
                        money(prediction)
                    )

                baseline = clean_number(
                    row.get(
                        "PreviousMonthRevenue",
                        row.get(
                            "DailyRevenue",
                            0
                        )
                    )
                )

                with c2:

                    if baseline > 0:

                        change = (
                            (prediction - baseline)
                            / baseline
                        ) * 100

                        st.metric(
                            "Change vs Previous Period",
                            f"{change:+.1f}%"
                        )

                    else:

                        st.metric(
                            "Previous Revenue",
                            money(baseline)
                        )

            st.markdown(
                "### Business Interpretation"
            )

            insights = []

            if price <= 0:

                insights.append(
                    "The selected course is free. "
                    "A free offering can support learner acquisition, "
                    "while revenue depends on paid transactions."
                )

            elif price > 300:

                insights.append(
                    "The selected price is in the high-price band. "
                    "Compare alternative prices if predicted demand is weak."
                )

            else:

                insights.append(
                    "The selected price is within a paid price band. "
                    "Scenario testing can help compare demand and revenue outcomes."
                )

            if rating >= 4.5:

                insights.append(
                    "The instructor/course rating is strong "
                    "and provides a positive quality signal."
                )

            elif rating < 3.5:

                insights.append(
                    "The rating is relatively low. "
                    "Improving learner experience and course quality may help."
                )

            if experience >= 8:

                insights.append(
                    "Instructor experience is high "
                    "and provides a strong instructor-quality signal."
                )

            elif experience < 4:

                insights.append(
                    "Instructor experience is relatively low. "
                    "Mentoring or experienced co-instruction may be considered."
                )

            for insight in insights:

                st.markdown(
                    f"""
                    <div class="insight-box">
                    {insight}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                f"Forecast period: "
                f"{row.get('YearMonth', 'Next period')}"
            )

        except Exception as exc:

            st.error(
                f"Prediction could not be generated: {exc}"
            )


# ============================================================
# CATEGORY REVENUE FORECAST
# ============================================================

def category_revenue_page(
    data,
    models
):

    st.title(
        "Category Revenue Forecast"
    )

    model = models["category"]

    category_df = data["category"]

    if category_df is None:

        st.error(
            "Category revenue modeling dataset is unavailable."
        )

        return

    if category_df.empty:

        st.error(
            "Category revenue modeling dataset is empty."
        )

        return

    if model is None:

        st.warning(
            models["errors"]["category"]
            or
            "Category revenue model is unavailable."
        )

        st.info(
            "Historical category analysis remains available "
            "even when the trained category model is not present."
        )

    category_df = category_df.copy()

    if "CourseCategory" not in category_df.columns:

        st.error(
            "CourseCategory column is missing."
        )

        return

    if "YearMonth" in category_df.columns:

        category_df["YearMonth"] = (
            category_df["YearMonth"]
            .astype(str)
        )

    categories = sorted(
        category_df[
            "CourseCategory"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not categories:

        st.error(
            "No categories are available."
        )

        return

    selected = st.selectbox(
        "Select Course Category",
        categories
    )

    rows = category_df[
        category_df["CourseCategory"].astype(str)
        == selected
    ].copy()

    if rows.empty:

        st.warning(
            "No historical records found for this category."
        )

        return

    if "YearMonth" in rows.columns:

        rows = rows.sort_values(
            "YearMonth"
        )

    st.markdown(
        "### Historical Category Performance"
    )

    target_column = None

    for candidate in [
        "CategoryRevenueTarget",
        "CategoryRevenue",
        "Revenue",
        "ActualCategoryRevenue",
    ]:

        if candidate in rows.columns:

            target_column = candidate

            break

    if target_column is not None:

        chart_data = rows[
            ["YearMonth", target_column]
        ].copy()

        chart_data[target_column] = pd.to_numeric(
            chart_data[target_column],
            errors="coerce"
        ).fillna(0)

        chart_data = chart_data.set_index(
            "YearMonth"
        )

        chart_data.columns = [
            "Revenue"
        ]

        st.line_chart(
            chart_data
        )

        latest_actual = clean_number(
            rows.iloc[-1][target_column]
        )

    else:

        latest_actual = 0.0

        st.info(
            "Historical category revenue target "
            "column was not found."
        )

    if model is None:

        return

    if (
        "Year" not in rows.columns
        or "Month" not in rows.columns
    ):

        st.warning(
            "Year/Month fields required for future category forecasting are missing."
        )

        return

    latest = rows.iloc[-1].copy()

    year, month = next_month(
        latest["Year"],
        latest["Month"]
    )

    forecast = latest.copy()

    forecast["Year"] = year

    forecast["Month"] = month

    forecast["YearMonth"] = (
        f"{year:04d}-{month:02d}"
    )

    forecast.update(
        month_features(
            year,
            month
        )
    )

    if "TimeIndex" in forecast.index:

        forecast["TimeIndex"] = (
            clean_number(
                latest.get(
                    "TimeIndex",
                    0
                )
            ) + 1
        )

    if "YearProgress" in forecast.index:

        forecast["YearProgress"] = (
            (month - 1) / 12.0
        )

    if "PreviousMonthCategoryRevenue" in forecast.index:

        forecast[
            "PreviousMonthCategoryRevenue"
        ] = latest_actual

    if "HasPreviousCategoryRevenue" in forecast.index:

        forecast[
            "HasPreviousCategoryRevenue"
        ] = 1

    feature_names = model_feature_names(
        model,
        category_df,
        [
            "CategoryRevenueTarget"
        ]
    )

    for col in feature_names:

        if col.startswith(
            "Category_"
        ):

            forecast[col] = int(
                col
                == f"Category_{selected}"
            )

    try:

        prediction = prediction_value(
            model,
            forecast,
            [
                "CategoryRevenueTarget"
            ]
        )

    except Exception as exc:

        st.error(
            f"Category forecast could not be generated: {exc}"
        )

        return

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Forecast Month",
        forecast["YearMonth"]
    )

    c2.metric(
        "Predicted Category Revenue",
        money(prediction)
    )

    if latest_actual > 0:

        change = (
            (prediction - latest_actual)
            / latest_actual
        ) * 100

        c3.metric(
            "Change vs Latest Actual",
            f"{change:+.1f}%"
        )

    else:

        c3.metric(
            "Latest Actual Revenue",
            money(latest_actual)
        )

    st.markdown(
        "### Business Recommendation"
    )

    if (
        latest_actual > 0
        and prediction
        > latest_actual * 1.10
    ):

        st.success(
            "The forecast indicates strong category growth. "
            "Consider increasing content availability, "
            "instructor capacity and promotional support."
        )

    elif (
        latest_actual > 0
        and prediction
        < latest_actual * 0.90
    ):

        st.warning(
            "The forecast indicates a potential decline. "
            "Review pricing, promotion, course quality "
            "and instructor capacity."
        )

    else:

        st.info(
            "The forecast is relatively close to the latest "
            "actual level. Continue monitoring course-level performance."
        )


# ============================================================
# CATEGORY DEMAND COMPARISON
# ============================================================

def category_comparison_page(
    data,
    models
):

    st.title(
        "Category-Level Demand Comparison"
    )

    feature = data["feature"]

    model = models["enrollment"]

    if feature is None:

        st.error(
            "Feature-engineered data is unavailable."
        )

        return

    if model is None:

        st.warning(
            "Enrollment model is unavailable. "
            "Showing historical category demand instead."
        )

        historical = feature.copy()

        if (
            "CourseCategory" in historical.columns
            and "MonthlyEnrollment" in historical.columns
        ):

            summary = (
                historical
                .groupby(
                    "CourseCategory",
                    as_index=False
                )
                .agg(
                    Enrollment=(
                        "MonthlyEnrollment",
                        "sum"
                    )
                )
                .sort_values(
                    "Enrollment",
                    ascending=False
                )
            )

            st.dataframe(
                summary,
                use_container_width=True
            )

            st.bar_chart(
                summary.set_index(
                    "CourseCategory"
                )["Enrollment"]
            )

        return

    dataf = feature.copy()

    if "YearMonth" not in dataf.columns:

        st.error(
            "YearMonth column is missing."
        )

        return

    dataf["YearMonth"] = (
        dataf["YearMonth"]
        .astype(str)
    )

    latest_month = sorted(
        dataf["YearMonth"]
        .dropna()
        .unique()
    )[-1]

    latest = dataf[
        dataf["YearMonth"]
        == latest_month
    ].copy()

    if latest.empty:

        st.error(
            "Latest course-month data is unavailable."
        )

        return

    if (
        "Year" not in latest.columns
        or "Month" not in latest.columns
    ):

        st.error(
            "Year/Month columns are required."
        )

        return

    next_y, next_m = next_month(
        latest.iloc[0]["Year"],
        latest.iloc[0]["Month"]
    )

    predictions = []

    for _, source in latest.iterrows():

        row = source.copy()

        row["Year"] = next_y

        row["Month"] = next_m

        row["YearMonth"] = (
            f"{next_y:04d}-{next_m:02d}"
        )

        row.update(
            month_features(
                next_y,
                next_m
            )
        )

        if "TimeIndex" in row.index:

            row["TimeIndex"] = (
                clean_number(
                    source.get(
                        "TimeIndex",
                        0
                    )
                ) + 1
            )

        if "YearProgress" in row.index:

            row["YearProgress"] = (
                (next_m - 1) / 12.0
            )

        try:

            prediction = prediction_value(
                model,
                row,
                [
                    "EnrollmentTarget",
                    "RevenueTarget",
                ]
            )

        except Exception:

            prediction = np.nan

        predictions.append(
            prediction
        )

    latest[
        "PredictedNextMonthEnrollment"
    ] = predictions

    if "CourseCategory" not in latest.columns:

        st.error(
            "CourseCategory column is missing."
        )

        return

    enrollment_column = None

    for candidate in [
        "MonthlyEnrollment",
        "EnrollmentTarget",
        "EnrollmentCount",
    ]:

        if candidate in latest.columns:

            enrollment_column = candidate

            break

    if enrollment_column is None:

        st.error(
            "Enrollment column is unavailable."
        )

        return

    summary = (
        latest
        .groupby(
            "CourseCategory",
            as_index=False
        )
        .agg(
            CurrentEnrollment=(
                enrollment_column,
                "sum"
            ),

            PredictedEnrollment=(
                "PredictedNextMonthEnrollment",
                "sum"
            ),
        )
    )

    summary["GrowthPercent"] = np.where(
        summary["CurrentEnrollment"] > 0,

        (
            (
                summary["PredictedEnrollment"]
                - summary["CurrentEnrollment"]
            )
            / summary["CurrentEnrollment"]
        )
        * 100,

        np.nan,
    )

    summary = summary.sort_values(
        "PredictedEnrollment",
        ascending=False
    )

    st.caption(
        f"Latest available period: {latest_month}. "
        f"Forecast period: {next_y:04d}-{next_m:02d}."
    )

    st.dataframe(
        summary.style.format(
            {
                "CurrentEnrollment": "{:,.0f}",
                "PredictedEnrollment": "{:,.0f}",
                "GrowthPercent": "{:+.1f}%",
            }
        ),
        use_container_width=True,
    )

    st.markdown(
        "### Predicted Enrollment by Category"
    )

    st.bar_chart(
        summary.set_index(
            "CourseCategory"
        )[
            "PredictedEnrollment"
        ]
    )

    if not summary.empty:

        top = summary.iloc[0]

        st.info(
            f"**{top['CourseCategory']}** has the "
            f"highest projected enrollment at approximately "
            f"**{top['PredictedEnrollment']:,.0f}** "
            f"learners in the next period."
        )


# ============================================================
# FEATURE IMPORTANCE PAGE
# ============================================================

def importance_page(data):

    st.title(
        "Feature Importance Explorer"
    )

    choices = {

        "Enrollment Demand":
            data["enrollment_importance"],

        "Course Revenue":
            data["revenue_importance"],

        "Category Revenue":
            data["category_importance"],
    }

    selected = st.selectbox(
        "Select Target",
        list(choices.keys())
    )

    df = choices[selected]

    if df is None or df.empty:

        st.warning(
            "Feature-importance CSV is unavailable."
        )

        st.info(
            "The dashboard will never generate fake "
            "feature-importance values. It displays only "
            "real model-derived results."
        )

        return

    df = df.copy()

    feature_col = None

    for candidate in [
        "Feature",
        "feature",
        "FeatureName",
    ]:

        if candidate in df.columns:

            feature_col = candidate

            break

    if feature_col is None:

        feature_col = df.columns[0]

    value_col = None

    for candidate in [
        "Importance",
        "importance",
        "FeatureImportance",
    ]:

        if candidate in df.columns:

            value_col = candidate

            break

    if value_col is None:

        value_col = df.columns[-1]

    df[value_col] = pd.to_numeric(
        df[value_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[value_col]
    )

    df = df.sort_values(
        value_col,
        ascending=False
    )

    if df.empty:

        st.warning(
            "No valid feature-importance values were found."
        )

        return

    max_features = min(
        30,
        len(df)
    )

    default_features = min(
        10,
        max_features
    )

    if max_features >= 5:

        top_n = st.slider(
            "Number of features",
            min_value=5,
            max_value=max_features,
            value=default_features,
        )

    else:

        top_n = max_features

    top = df.head(top_n)

    st.dataframe(
        top,
        use_container_width=True
    )

    st.markdown(
        "### Feature Importance"
    )

    chart = (
        top
        .set_index(feature_col)[value_col]
        .sort_values()
    )

    st.bar_chart(
        chart
    )

    st.info(
        "Feature importance shows which variables the trained "
        "model relied on most. It represents model influence, "
        "not proof of direct causation."
    )


# ============================================================
# FORECAST VISUALIZATIONS
# ============================================================

def forecast_visualizations_page(
    data
):

    st.title(
        "Revenue & Forecast Visualizations"
    )

    monthly_category = load_csv(
        str(
            RESULTS_DIR
            / "monthly_category_revenue_forecast_summary.csv"
        )
    )

    monthly_revenue = load_csv(
        str(
            RESULTS_DIR
            / "monthly_revenue_forecast_summary.csv"
        )
    )

    monthly_enrollment = load_csv(
        str(
            RESULTS_DIR
            / "monthly_enrollment_forecast_summary.csv"
        )
    )

    revenue = data[
        "revenue_forecast"
    ]

    category = data[
        "category_forecast"
    ]

    tabs = st.tabs(
        [
            "Course Revenue",
            "Category Revenue",
            "Enrollment",
        ]
    )

    # --------------------------------------------------------
    # COURSE REVENUE
    # --------------------------------------------------------

    with tabs[0]:

        if (
            monthly_revenue is not None
            and not monthly_revenue.empty
        ):

            x = monthly_revenue.copy()

            if "YearMonth" in x.columns:

                x = x.set_index(
                    "YearMonth"
                )

            cols = [
                c
                for c in [
                    "ActualRevenue",
                    "PredictedRevenue",
                ]
                if c in x.columns
            ]

            if cols:

                st.line_chart(
                    x[cols]
                )

            st.dataframe(
                monthly_revenue,
                use_container_width=True
            )

            st.download_button(
                "Download Course Revenue Forecast CSV",
                monthly_revenue.to_csv(
                    index=False
                ),
                file_name="course_revenue_forecast.csv",
                mime="text/csv",
            )

        elif revenue is not None:

            st.dataframe(
                revenue,
                use_container_width=True
            )

            st.download_button(
                "Download Revenue Data",
                revenue.to_csv(
                    index=False
                ),
                file_name="revenue_forecast.csv",
                mime="text/csv",
            )

        else:

            st.info(
                "Course revenue forecast output is not available."
            )


    # --------------------------------------------------------
    # CATEGORY REVENUE
    # --------------------------------------------------------

    with tabs[1]:

        if (
            monthly_category is not None
            and not monthly_category.empty
        ):

            x = monthly_category.copy()

            if "YearMonth" in x.columns:

                x = x.set_index(
                    "YearMonth"
                )

            cols = [
                c
                for c in [
                    "ActualCategoryRevenue",
                    "PredictedCategoryRevenue",
                ]
                if c in x.columns
            ]

            if cols:

                st.line_chart(
                    x[cols]
                )

            st.dataframe(
                monthly_category,
                use_container_width=True
            )

            st.download_button(
                "Download Category Revenue Forecast CSV",
                monthly_category.to_csv(
                    index=False
                ),
                file_name="category_revenue_forecast.csv",
                mime="text/csv",
            )

        elif category is not None:

            st.dataframe(
                category,
                use_container_width=True
            )

        else:

            st.info(
                "Category revenue forecast output is not available."
            )


    # --------------------------------------------------------
    # ENROLLMENT
    # --------------------------------------------------------

    with tabs[2]:

        if (
            monthly_enrollment is not None
            and not monthly_enrollment.empty
        ):

            x = monthly_enrollment.copy()

            if "YearMonth" in x.columns:

                x = x.set_index(
                    "YearMonth"
                )

            cols = [
                c
                for c in [
                    "ActualEnrollment",
                    "PredictedEnrollment",
                ]
                if c in x.columns
            ]

            if cols:

                st.line_chart(
                    x[cols]
                )

            st.dataframe(
                monthly_enrollment,
                use_container_width=True
            )

            st.download_button(
                "Download Enrollment Forecast CSV",
                monthly_enrollment.to_csv(
                    index=False
                ),
                file_name="enrollment_forecast.csv",
                mime="text/csv",
            )

        else:

            st.info(
                "Enrollment forecast summary is not available."
            )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def business_insights_page(
    data
):

    st.title(
        "Business Insights & Recommendations"
    )

    master = data["course_master"]

    category = data["category"]

    if master is None:

        st.error(
            "Course master dataset is unavailable."
        )

        return

    # --------------------------------------------------------
    # PRICING
    # --------------------------------------------------------

    st.markdown(
        "### Pricing Strategy"
    )

    st.write(
        "Use course demand and revenue prediction "
        "pages to compare alternative pricing scenarios. "
        "Pricing decisions should consider both demand and revenue."
    )

    # --------------------------------------------------------
    # COURSE PORTFOLIO
    # --------------------------------------------------------

    st.markdown(
        "### Course Portfolio"
    )

    revenue_col = None

    for candidate in [
        "TotalRevenue",
        "Revenue",
    ]:

        if candidate in master.columns:

            revenue_col = candidate

            break

    if revenue_col is not None:

        columns = [
            c
            for c in [
                "CourseName",
                "CourseCategory",
                revenue_col,
            ]
            if c in master.columns
        ]

        top = (
            master
            .sort_values(
                revenue_col,
                ascending=False
            )
            .head(5)
        )

        st.dataframe(
            top[columns],
            use_container_width=True
        )

        st.write(
            "Prioritize high-performing courses for "
            "instructor capacity, content updates and promotion."
        )

    # --------------------------------------------------------
    # INSTRUCTOR STRATEGY
    # --------------------------------------------------------

    st.markdown(
        "### Instructor Strategy"
    )

    st.write(
        "Instructor experience and instructor rating are "
        "important predictive inputs. Courses with weaker "
        "forecasts can be reviewed for instructor quality, "
        "expertise alignment and learner experience."
    )

    # --------------------------------------------------------
    # CATEGORY STRATEGY
    # --------------------------------------------------------

    st.markdown(
        "### Category Strategy"
    )

    if (
        category is not None
        and not category.empty
        and "CourseCategory" in category.columns
    ):

        category_copy = category.copy()

        if "YearMonth" in category_copy.columns:

            category_copy = category_copy.sort_values(
                "YearMonth"
            )

        category_revenue_col = None

        for candidate in [
            "CategoryRevenueTarget",
            "CategoryRevenue",
            "Revenue",
        ]:

            if candidate in category_copy.columns:

                category_revenue_col = candidate

                break

        if category_revenue_col is not None:

            latest = (
                category_copy
                .groupby(
                    "CourseCategory"
                )
                .tail(1)
                .copy()
            )

            latest = latest.sort_values(
                category_revenue_col,
                ascending=False
            )

            display_columns = [
                c
                for c in [
                    "CourseCategory",
                    "YearMonth",
                    category_revenue_col,
                    "CategoryEnrollment",
                ]
                if c in latest.columns
            ]

            st.dataframe(
                latest[display_columns],
                use_container_width=True
            )

    st.write(
        "Categories showing sustained revenue and enrollment "
        "strength can receive additional content and promotional "
        "support, while weaker categories should be investigated "
        "before major expansion."
    )

    # --------------------------------------------------------
    # MODELING NOTE
    # --------------------------------------------------------

    st.markdown(
        "### Important Modeling Note"
    )

    st.info(
        "Predictions support decision-making but do not replace "
        "business judgment. Scenario testing is particularly "
        "important because historical datasets represent past behavior."
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

def dataset_explorer_page(
    data
):

    st.title(
        "Dataset Explorer"
    )

    datasets = {

        "Course Master Dataset":
            data["course_master"],

        "Feature Engineered Dataset":
            data["feature"],

        "Enrollment Modeling Dataset":
            data["enrollment"],

        "Revenue Modeling Dataset":
            data["revenue"],

        "Category Revenue Modeling Dataset":
            data["category"],
    }

    selected = st.selectbox(
        "Select dataset",
        list(datasets.keys())
    )

    df = datasets[selected]

    if df is None:

        st.warning(
            "This dataset is not available."
        )

        return

    st.write(
        f"Rows: **{len(df):,}**"
    )

    st.write(
        f"Columns: **{len(df.columns):,}**"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.download_button(
        "Download Dataset CSV",
        df.to_csv(
            index=False
        ),
        file_name=(
            selected
            .lower()
            .replace(" ", "_")
            + ".csv"
        ),
        mime="text/csv",
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_information_page(models):

    st.title("Model Information")

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    model_rows = []

    for name, key in [
        ("Enrollment", "enrollment"),
        ("Revenue", "revenue"),
        ("Category Revenue", "category"),
    ]:

        model = models[key]
        path = models["paths"][key]

        if model is not None:

            model_name = type(model).__name__

            if hasattr(model, "named_steps"):

                model_name = (
                    "Pipeline: "
                    + " → ".join(
                        [
                            type(step).__name__
                            for step in model.named_steps.values()
                        ]
                    )
                )

            model_rows.append(
                {
                    "Target": name,
                    "Status": "Loaded",
                    "Model": model_name,
                    "Path": path or "—",
                }
            )

        else:

            model_rows.append(
                {
                    "Target": name,
                    "Status": "Unavailable",
                    "Model": "—",
                    "Path": path or "No compatible file found",
                }
            )

    model_status_df = pd.DataFrame(model_rows)

    st.markdown("### Model Status")

    st.dataframe(
        model_status_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    st.markdown("### Model Evaluation")

    evaluation_path = None

    # First check common exact locations
    evaluation_candidates = [

        RESULTS_DIR / "model_evaluation_results.csv",

        DATA_DIR / "model_evaluation_results.csv",

        PROCESSED_DIR / "model_evaluation_results.csv",

        MODELING_DIR / "model_evaluation_results.csv",

        BASE_DIR / "model_evaluation_results.csv",

    ]

    evaluation_path = first_existing(
        evaluation_candidates
    )

    # --------------------------------------------------------
    # RECURSIVE SEARCH
    # --------------------------------------------------------

    if evaluation_path is None:

        try:

            possible_files = list(
                BASE_DIR.rglob("*.csv")
            )

            for file in possible_files:

                filename = file.name.lower()

                if (
                    "evaluation" in filename
                    or "model_evaluation" in filename
                    or "model_performance" in filename
                    or "model_metrics" in filename
                    or "evaluation_results" in filename
                ):

                    evaluation_path = file

                    break

        except Exception:
            evaluation_path = None

    # --------------------------------------------------------
    # DISPLAY EVALUATION
    # --------------------------------------------------------

    if evaluation_path is not None:

        evaluation = load_csv(
            str(evaluation_path)
        )

        if evaluation is not None and not evaluation.empty:

            st.success(
                f"Model evaluation loaded: "
                f"{evaluation_path.name}"
            )

            st.dataframe(
                evaluation,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Model Evaluation Results",
                evaluation.to_csv(
                    index=False
                ),
                file_name="model_evaluation_results.csv",
                mime="text/csv",
            )

        else:

            st.warning(
                "The evaluation file was found, "
                "but it could not be read or is empty."
            )

    else:

        # ----------------------------------------------------
        # CALCULATE EVALUATION FROM AVAILABLE DATA
        # ----------------------------------------------------

        st.info(
            "A saved model evaluation CSV was not found. "
            "The dashboard can still display model status and predictions."
        )

        st.caption(
            "This does not affect the prediction functionality."
        )

        # Look for other CSV files containing evaluation metrics
        metric_files = []

        try:

            for file in BASE_DIR.rglob("*.csv"):

                name = file.name.lower()

                if any(
                    word in name
                    for word in [
                        "evaluation",
                        "performance",
                        "metrics",
                        "comparison",
                        "model_results",
                    ]
                ):

                    metric_files.append(file)

        except Exception:
            metric_files = []

        if metric_files:

            st.markdown(
                "#### Other Model Result Files Found"
            )

            for file in metric_files[:10]:

                st.write(
                    f"• `{file.relative_to(BASE_DIR)}`"
                )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    add_common_css()

    data = get_data()

    models = get_models()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.title(
        "EduPro Dashboard"
    )

    st.sidebar.caption(
        "Predictive Modeling & Revenue Forecasting"
    )

    pages = [

        "Overview",

        "Course Demand Prediction",

        "Course Revenue Prediction",

        "Category Revenue Forecast",

        "Category-Level Demand Comparison",

        "Revenue & Forecast Visualizations",

        "Feature Importance Explorer",

        "Business Insights & Recommendations",

        "Dataset Explorer",

        "Model Information",
    ]

    page = st.sidebar.radio(
        "Navigate",
        pages
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        "### Project"
    )

    st.sidebar.caption(
        f"Root: {BASE_DIR}"
    )

    st.sidebar.caption(
        f"Models: {MODELS_DIR}"
    )

    st.sidebar.caption(
        f"Results: {RESULTS_DIR}"
    )

    st.sidebar.divider()

    if page == "Overview":

        home_page(
            data,
            models
        )

    elif page == "Course Demand Prediction":

        course_prediction_page(
            data,
            models,
            "enrollment"
        )

    elif page == "Course Revenue Prediction":

        course_prediction_page(
            data,
            models,
            "revenue"
        )

    elif page == "Category Revenue Forecast":

        category_revenue_page(
            data,
            models
        )

    elif page == "Category-Level Demand Comparison":

        category_comparison_page(
            data,
            models
        )

    elif page == "Revenue & Forecast Visualizations":

        forecast_visualizations_page(
            data
        )

    elif page == "Feature Importance Explorer":

        importance_page(
            data
        )

    elif page == "Business Insights & Recommendations":

        business_insights_page(
            data
        )

    elif page == "Dataset Explorer":

        dataset_explorer_page(
            data
        )

    elif page == "Model Information":

        model_information_page(
            models
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()

