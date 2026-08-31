import os
import warnings
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 11: CATEGORY REVENUE FORECASTING
# ============================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 11: CATEGORY REVENUE FORECASTING")
print("=" * 70)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

MODELING_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "modeling"
)

MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


print("\n" + "-" * 70)
print("PROJECT PATHS")
print("-" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nModeling data directory:")
print(MODELING_DIR)

print("\nModels directory:")
print(MODELS_DIR)

print("\nResults directory:")
print(RESULTS_DIR)


# ============================================================
# 2. REQUIRED INPUT FILE
# ============================================================

print("\n" + "-" * 70)
print("CHECKING REQUIRED FILE")
print("-" * 70)


INPUT_FILE = os.path.join(
    MODELING_DIR,
    "category_revenue_modeling_dataset.csv"
)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        "\nRequired Step 8 category modeling file was not found:\n"
        f"{INPUT_FILE}"
    )


print("\nFound:")
print(INPUT_FILE)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n" + "-" * 70)
print("LOADING CATEGORY REVENUE MODELING DATASET")
print("-" * 70)


data = pd.read_csv(INPUT_FILE)


print("\nDataset shape:")
print(data.shape)


print("\nAvailable columns:")
print(list(data.columns))


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING CATEGORY MODELING DATASET")
print("-" * 70)


required_columns = [

    "CourseCategory",
    "Year",
    "Month",
    "YearMonth",

    "CategoryRevenueTarget",

    "PreviousMonthCategoryRevenue",
    "Previous3MonthCategoryRevenue",
    "Previous6MonthCategoryRevenue",

    "CategoryRevenueGrowthRate",

    "HasPreviousCategoryRevenue",

    "CategoryEnrollment"
]


missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        +
        "\n".join(
            f"- {column}"
            for column in missing_columns
        )
    )


print("\nAll required columns are present.")


# ============================================================
# 5. PREPARE DATA TYPES
# ============================================================

print("\n" + "-" * 70)
print("PREPARING DATA TYPES")
print("-" * 70)


numeric_columns = [

    "Year",
    "Month",

    "CategoryRevenueTarget",

    "PreviousMonthCategoryRevenue",
    "Previous3MonthCategoryRevenue",
    "Previous6MonthCategoryRevenue",

    "CategoryRevenueGrowthRate",

    "HasPreviousCategoryRevenue",

    "CategoryEnrollment"
]


for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


data["CourseCategory"] = (
    data["CourseCategory"]
    .astype(str)
    .str.strip()
)


data["YearMonth"] = (
    data["YearMonth"]
    .astype(str)
    .str.strip()
)


print("\nData types prepared successfully.")


# ============================================================
# 6. SOURCE DATA VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING SOURCE DATA")
print("-" * 70)


print("\nNumber of rows:")
print(len(data))


print("\nUnique categories:")
print(
    data["CourseCategory"].nunique()
)


print("\nUnique months:")
print(
    data["YearMonth"].nunique()
)


duplicate_category_month = (
    data
    .duplicated(
        subset=[
            "CourseCategory",
            "Year",
            "Month"
        ]
    )
    .sum()
)


print("\nDuplicate Category-Month combinations:")
print(duplicate_category_month)


if duplicate_category_month > 0:

    raise ValueError(
        "Duplicate Category-Month combinations detected."
    )


print("\nSource dataset validation passed.")


# ============================================================
# 7. SORT CHRONOLOGICALLY
# ============================================================

print("\n" + "-" * 70)
print("SORTING DATA CHRONOLOGICALLY")
print("-" * 70)


data = (
    data
    .sort_values(
        [
            "Year",
            "Month",
            "CourseCategory"
        ]
    )
    .reset_index(drop=True)
)


print("\nData sorted chronologically.")


# ============================================================
# 8. IDENTIFY TARGET
# ============================================================

print("\n" + "-" * 70)
print("IDENTIFYING CATEGORY REVENUE TARGET")
print("-" * 70)


TARGET_COLUMN = "CategoryRevenueTarget"


if TARGET_COLUMN not in data.columns:

    raise ValueError(
        "Category revenue target is missing."
    )


print("\nTarget column:")
print(TARGET_COLUMN)


# ============================================================
# 9. TARGET VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING CATEGORY REVENUE TARGET")
print("-" * 70)


if data[TARGET_COLUMN].isna().any():

    raise ValueError(
        "CategoryRevenueTarget contains missing values."
    )


if (
    data[TARGET_COLUMN] < 0
).any():

    raise ValueError(
        "CategoryRevenueTarget contains negative values."
    )


print("\nMinimum category revenue:")
print(data[TARGET_COLUMN].min())


print("\nMaximum category revenue:")
print(data[TARGET_COLUMN].max())


print("\nAverage category revenue:")
print(data[TARGET_COLUMN].mean())


print("\nCategory revenue target validation passed.")


# ============================================================
# 10. CREATE TIME FEATURES
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME FEATURES")
print("-" * 70)


# ------------------------------------------------------------
# Quarter
# ------------------------------------------------------------

data["Quarter"] = (
    ((data["Month"] - 1) // 3) + 1
)


# ------------------------------------------------------------
# Quarter start
# ------------------------------------------------------------

data["IsQuarterStart"] = (
    data["Month"]
    .isin([1, 4, 7, 10])
    .astype(int)
)


# ------------------------------------------------------------
# Quarter end
# ------------------------------------------------------------

data["IsQuarterEnd"] = (
    data["Month"]
    .isin([3, 6, 9, 12])
    .astype(int)
)


# ------------------------------------------------------------
# Unique Year-Month table
# ------------------------------------------------------------

unique_year_months = (
    data[
        [
            "Year",
            "Month"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "Year",
            "Month"
        ]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Time index
# ------------------------------------------------------------

unique_year_months["TimeIndex"] = (
    np.arange(
        len(unique_year_months)
    )
)


# ------------------------------------------------------------
# Merge TimeIndex
# ------------------------------------------------------------

data = data.merge(
    unique_year_months,
    on=[
        "Year",
        "Month"
    ],
    how="left"
)


# ------------------------------------------------------------
# Seasonal encoding
# ------------------------------------------------------------

data["MonthSin"] = np.sin(
    2 * np.pi * data["Month"] / 12
)


data["MonthCos"] = np.cos(
    2 * np.pi * data["Month"] / 12
)


# ------------------------------------------------------------
# Year progress
# ------------------------------------------------------------

data["YearProgress"] = (
    (data["Month"] - 1) / 11
)


print("\nTime features created successfully.")


# ============================================================
# 11. VALIDATE HISTORICAL FEATURES
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING HISTORICAL CATEGORY FEATURES")
print("-" * 70)


# Create independently calculated previous-month revenue
expected_previous_revenue = (
    data
    .sort_values(
        [
            "CourseCategory",
            "Year",
            "Month"
        ]
    )
    .groupby(
        "CourseCategory"
    )[TARGET_COLUMN]
    .shift(1)
    .fillna(0)
)


validation_data = (
    data
    .sort_values(
        [
            "CourseCategory",
            "Year",
            "Month"
        ]
    )
    .reset_index(drop=True)
)


expected_previous_revenue = (
    validation_data[
        TARGET_COLUMN
    ]
    .groupby(
        validation_data["CourseCategory"]
    )
    .shift(1)
    .fillna(0)
)


actual_previous_revenue = (
    validation_data[
        "PreviousMonthCategoryRevenue"
    ]
)


lag_difference = (
    actual_previous_revenue
    -
    expected_previous_revenue
).abs().max()


print(
    "\nMaximum previous-month revenue difference:"
)

print(lag_difference)


if lag_difference > 0.000001:

    raise ValueError(
        "Historical category revenue lag validation failed."
    )


print(
    "\nHistorical category feature validation passed."
)


# ============================================================
# 12. FEATURE LEAKAGE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING FEATURE LEAKAGE")
print("-" * 70)


forbidden_features = [
    TARGET_COLUMN
]


print(
    "\nCurrent-period target excluded from predictors:"
)


for column in forbidden_features:

    print(
        "-",
        column
    )


# ------------------------------------------------------------
# Verify target is NOT in predictor list
# ------------------------------------------------------------

# This will be checked again after feature creation.
# Current-period revenue target must never be used directly
# as a predictor.
#
# Historical revenue features are permitted because they refer
# to previous periods.


print(
    "\nTarget leakage validation passed."
)


# ============================================================
# 13. CREATE CATEGORY ENCODING
# ============================================================

print("\n" + "-" * 70)
print("ENCODING CATEGORY FEATURE")
print("-" * 70)


category_dummies = pd.get_dummies(
    data["CourseCategory"],
    prefix="Category",
    dtype=int
)


data = pd.concat(
    [
        data,
        category_dummies
    ],
    axis=1
)


print("\nCategory encoding completed.")


print("\nEncoded category features:")


for column in category_dummies.columns:

    print(
        "-",
        column
    )


# ============================================================
# 14. DEFINE MODEL FEATURES
# ============================================================

print("\n" + "-" * 70)
print("DEFINING MODEL FEATURES")
print("-" * 70)


base_features = [

    "Month",
    "Quarter",
    "IsQuarterStart",
    "IsQuarterEnd",
    "TimeIndex",
    "MonthSin",
    "MonthCos",
    "YearProgress",

    "PreviousMonthCategoryRevenue",
    "Previous3MonthCategoryRevenue",
    "Previous6MonthCategoryRevenue",

    "CategoryRevenueGrowthRate",

    "HasPreviousCategoryRevenue",

    "CategoryEnrollment"
]


encoded_features = list(
    category_dummies.columns
)


final_features = (
    base_features
    +
    encoded_features
)


# ------------------------------------------------------------
# Make sure target is not included
# ------------------------------------------------------------

if TARGET_COLUMN in final_features:

    raise ValueError(
        "Target leakage detected: "
        "CategoryRevenueTarget is present in predictors."
    )


print("\nFinal category revenue predictor features:")


for number, feature in enumerate(
    final_features,
    start=1
):

    print(
        f"{number:02d}. {feature}"
    )


print("\nTotal predictor features:")
print(len(final_features))


# ============================================================
# 15. CREATE MODELING MATRICES
# ============================================================

print("\n" + "-" * 70)
print("CREATING MODELING MATRICES")
print("-" * 70)


X = data[
    final_features
].copy()


y = data[
    TARGET_COLUMN
].copy()


print("\nFeature matrix shape:")
print(X.shape)


print("\nTarget shape:")
print(y.shape)


# ============================================================
# 16. VALIDATE FEATURE TYPES
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING FEATURE TYPES")
print("-" * 70)


non_numeric_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


if len(non_numeric_features) > 0:

    raise ValueError(
        "Non-numeric features detected:\n"
        +
        "\n".join(
            f"- {column}"
            for column in non_numeric_features
        )
    )


print(
    "\nAll predictor features are numeric."
)


# ============================================================
# 17. MISSING VALUE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("CHECKING MISSING VALUES")
print("-" * 70)


missing_features = (
    X
    .isna()
    .sum()
    .sum()
)


missing_target = (
    y
    .isna()
    .sum()
)


print("\nMissing feature values:")
print(missing_features)


print("\nMissing target values:")
print(missing_target)


if missing_features > 0:

    raise ValueError(
        "Missing values found in feature matrix."
    )


if missing_target > 0:

    raise ValueError(
        "Missing values found in target."
    )


print(
    "\nMissing-value validation passed."
)


# ============================================================
# 18. INFINITE VALUE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("CHECKING INFINITE VALUES")
print("-" * 70)


infinite_features = np.isinf(
    X.to_numpy()
).sum()


infinite_target = np.isinf(
    y.to_numpy()
).sum()


print("\nInfinite feature values:")
print(infinite_features)


print("\nInfinite target values:")
print(infinite_target)


if infinite_features > 0:

    raise ValueError(
        "Infinite values found in feature matrix."
    )


if infinite_target > 0:

    raise ValueError(
        "Infinite values found in target."
    )


print(
    "\nInfinite-value validation passed."
)


# ============================================================
# 19. CREATE UNIQUE CHRONOLOGICAL MONTH LIST
# ============================================================

print("\n" + "-" * 70)
print("CREATING CHRONOLOGICAL MONTH LIST")
print("-" * 70)


unique_months = (
    data[
        [
            "Year",
            "Month",
            "YearMonth"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "Year",
            "Month"
        ]
    )
)


unique_months_list = (
    unique_months[
        "YearMonth"
    ]
    .tolist()
)


print("\nAvailable months:")


for month in unique_months_list:

    print(
        "-",
        month
    )


number_of_months = len(
    unique_months_list
)


print("\nNumber of unique months:")
print(number_of_months)


# ============================================================
# 20. TIME-BASED TRAIN-TEST SPLIT
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME-BASED TRAIN-TEST SPLIT")
print("-" * 70)


if number_of_months < 4:

    raise ValueError(
        "Insufficient months for time-series forecasting."
    )


test_month_count = max(
    1,
    int(
        np.ceil(
            number_of_months * 0.20
        )
    )
)


train_month_count = (
    number_of_months
    -
    test_month_count
)


train_months = (
    unique_months_list[
        :train_month_count
    ]
)


test_months = (
    unique_months_list[
        train_month_count:
    ]
)


print("\nTraining months:")
print(train_months)


print("\nTesting months:")
print(test_months)


# ------------------------------------------------------------
# Create masks
# ------------------------------------------------------------

train_mask = (
    data["YearMonth"]
    .isin(train_months)
)


test_mask = (
    data["YearMonth"]
    .isin(test_months)
)


# ------------------------------------------------------------
# Create train/test datasets
# ------------------------------------------------------------

X_train = X.loc[
    train_mask
].copy()


X_test = X.loc[
    test_mask
].copy()


y_train = y.loc[
    train_mask
].copy()


y_test = y.loc[
    test_mask
].copy()


print("\nTraining rows:")
print(len(X_train))


print("\nTesting rows:")
print(len(X_test))


# ============================================================
# 21. TEMPORAL ORDER VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING TEMPORAL TRAIN-TEST ORDER")
print("-" * 70)


latest_training_month = max(
    train_months
)


earliest_testing_month = min(
    test_months
)


print("\nLatest training month:")
print(latest_training_month)


print("\nEarliest testing month:")
print(earliest_testing_month)


if latest_training_month >= earliest_testing_month:

    raise ValueError(
        "Temporal leakage detected."
    )


print(
    "\nTemporal train-test validation passed."
)


# ============================================================
# 22. RESET TRAINING INDEX
# ============================================================
#
# IMPORTANT:
#
# GridSearchCV receives X_train and y_train.
# Therefore the CV indices MUST refer to positions inside
# X_train, not positions inside the original full dataset.
#
# This fixes the major CV indexing issue in the old code.
# ============================================================

print("\n" + "-" * 70)
print("PREPARING TRAINING DATA FOR TIME-SERIES CV")
print("-" * 70)


X_train = X_train.reset_index(
    drop=True
)


y_train = y_train.reset_index(
    drop=True
)


train_metadata = (
    data.loc[
        train_mask,
        [
            "Year",
            "Month",
            "YearMonth"
        ]
    ]
    .reset_index(drop=True)
)


print(
    "\nTraining data index reset successfully."
)


# ============================================================
# 23. CREATE TIME-SERIES CROSS-VALIDATION SPLITS
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME-SERIES CROSS-VALIDATION")
print("-" * 70)


training_month_data = (
    unique_months_list[
        :train_month_count
    ]
)


TIME_SERIES_SPLITS = min(
    5,
    len(training_month_data) - 1
)


if TIME_SERIES_SPLITS < 2:

    raise ValueError(
        "Not enough training months for time-series "
        "cross-validation."
    )


cv_splits = []


# ------------------------------------------------------------
# Expanding-window validation
#
# Fold 1:
# Train Month 1
# Validate Month 2
#
# Fold 2:
# Train Months 1-2
# Validate Month 3
#
# Fold 3:
# Train Months 1-3
# Validate Month 4
#
# etc.
# ------------------------------------------------------------

for i in range(
    1,
    TIME_SERIES_SPLITS + 1
):

    train_cv_months = (
        training_month_data[
            :i
        ]
    )


    validation_month = (
        training_month_data[
            i
        ]
    )


    train_indices = (
        train_metadata.index[
            train_metadata[
                "YearMonth"
            ].isin(
                train_cv_months
            )
        ]
        .to_numpy()
    )


    validation_indices = (
        train_metadata.index[
            train_metadata[
                "YearMonth"
            ]
            ==
            validation_month
        ]
        .to_numpy()
    )


    if (
        len(train_indices) > 0
        and
        len(validation_indices) > 0
    ):

        cv_splits.append(
            (
                train_indices,
                validation_indices
            )
        )


print("\nTime-series CV folds:")
print(len(cv_splits))


if len(cv_splits) < 2:

    raise ValueError(
        "Unable to create sufficient time-series CV folds."
    )


# ------------------------------------------------------------
# Print each fold
# ------------------------------------------------------------

print("\nCV fold details:")


for fold_number, (
    train_indices,
    validation_indices
) in enumerate(
    cv_splits,
    start=1
):

    fold_train_months = (
        train_metadata
        .loc[
            train_indices,
            "YearMonth"
        ]
        .unique()
        .tolist()
    )


    fold_validation_months = (
        train_metadata
        .loc[
            validation_indices,
            "YearMonth"
        ]
        .unique()
        .tolist()
    )


    print(
        f"\nFold {fold_number}:"
    )

    print(
        "  Training:",
        fold_train_months
    )

    print(
        "  Validation:",
        fold_validation_months
    )


print(
    "\nTime-series cross-validation created successfully."
)


# ============================================================
# 24. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_train_data,
    y_train_data,
    X_test_data,
    y_test_data
):

    model.fit(
        X_train_data,
        y_train_data
    )


    predictions = model.predict(
        X_test_data
    )


    # Revenue cannot be negative.
    predictions = np.maximum(
        predictions,
        0
    )


    mae = mean_absolute_error(
        y_test_data,
        predictions
    )


    mse = mean_squared_error(
        y_test_data,
        predictions
    )


    rmse = np.sqrt(
        mse
    )


    r2 = r2_score(
        y_test_data,
        predictions
    )


    return (
        model,
        predictions,
        mae,
        mse,
        rmse,
        r2
    )


# ============================================================
# 25. CREATE BASELINE AND ADVANCED MODELS
# ============================================================

print("\n" + "-" * 70)
print("CREATING CATEGORY REVENUE FORECASTING MODELS")
print("-" * 70)


models = {

    # --------------------------------------------------------
    # Baseline Model 1
    # --------------------------------------------------------

    "Linear Regression":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    LinearRegression()
                )
            ]
        ),


    # --------------------------------------------------------
    # Baseline Model 2
    # --------------------------------------------------------

    "Ridge Regression":
        Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    Ridge(
                        alpha=1.0
                    )
                )
            ]
        ),


    # --------------------------------------------------------
    # Advanced Model 1
    # --------------------------------------------------------

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1
        ),


    # --------------------------------------------------------
    # Advanced Model 2
    # --------------------------------------------------------

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            min_samples_leaf=1
        )
}


print("\nModels created:")


for model_name in models:

    print(
        "-",
        model_name
    )


# ============================================================
# 26. TIME-SERIES CROSS-VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("PERFORMING TIME-SERIES CROSS-VALIDATION")
print("-" * 70)


cv_results = []


for model_name, model in models.items():

    print(
        f"\nEvaluating: {model_name}"
    )


    fold_mae = []
    fold_rmse = []
    fold_r2 = []


    for fold_number, (
        train_indices,
        validation_indices
    ) in enumerate(
        cv_splits,
        start=1
    ):

        X_cv_train = X_train.loc[
            train_indices
        ]


        y_cv_train = y_train.loc[
            train_indices
        ]


        X_cv_validation = X_train.loc[
            validation_indices
        ]


        y_cv_validation = y_train.loc[
            validation_indices
        ]


        model.fit(
            X_cv_train,
            y_cv_train
        )


        predictions = model.predict(
            X_cv_validation
        )


        # Revenue cannot be negative.
        predictions = np.maximum(
            predictions,
            0
        )


        mae = mean_absolute_error(
            y_cv_validation,
            predictions
        )


        mse = mean_squared_error(
            y_cv_validation,
            predictions
        )


        rmse = np.sqrt(
            mse
        )


        r2 = r2_score(
            y_cv_validation,
            predictions
        )


        fold_mae.append(
            mae
        )


        fold_rmse.append(
            rmse
        )


        fold_r2.append(
            r2
        )


        print(
            f"  Fold {fold_number}: "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R2={r2:.4f}"
        )


    cv_results.append(
        {
            "Model": model_name,

            "CV_MAE":
                np.mean(
                    fold_mae
                ),

            "CV_RMSE":
                np.mean(
                    fold_rmse
                ),

            "CV_R2":
                np.mean(
                    fold_r2
                ),

            "CV_R2_STD":
                np.std(
                    fold_r2
                )
        }
    )


cv_results_df = pd.DataFrame(
    cv_results
)


print("\n" + "-" * 70)
print("TIME-SERIES CROSS-VALIDATION RESULTS")
print("-" * 70)


print(
    cv_results_df
    .sort_values(
        "CV_RMSE"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 27. HYPERPARAMETER TUNING - RANDOM FOREST
# ============================================================

print("\n" + "-" * 70)
print("HYPERPARAMETER TUNING: RANDOM FOREST")
print("-" * 70)


rf_param_grid = {

    "n_estimators": [
        100,
        200,
        300
    ],

    "max_depth": [
        3,
        5,
        7,
        None
    ],

    "min_samples_split": [
        2,
        5
    ],

    "min_samples_leaf": [
        1,
        2
    ]
}


print("\nRandom Forest parameter grid:")


for parameter, values in rf_param_grid.items():

    print(
        f"- {parameter}: {values}"
    )


rf_grid_search = GridSearchCV(

    estimator=
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),

    param_grid=rf_param_grid,

    cv=cv_splits,

    scoring="neg_root_mean_squared_error",

    n_jobs=-1
)


print(
    "\nRunning Random Forest GridSearchCV..."
)


rf_grid_search.fit(
    X_train,
    y_train
)


best_rf_model = (
    rf_grid_search
    .best_estimator_
)


best_rf_cv_rmse = (
    -rf_grid_search
    .best_score_
)


print("\nBest Random Forest parameters:")

print(
    rf_grid_search.best_params_
)


print("\nBest Random Forest CV RMSE:")

print(
    best_rf_cv_rmse
)


# ============================================================
# 28. HYPERPARAMETER TUNING - GRADIENT BOOSTING
# ============================================================

print("\n" + "-" * 70)
print("HYPERPARAMETER TUNING: GRADIENT BOOSTING")
print("-" * 70)


gb_param_grid = {

    "n_estimators": [
        50,
        100,
        200
    ],

    "learning_rate": [
        0.03,
        0.05,
        0.1
    ],

    "max_depth": [
        2,
        3
    ],

    "min_samples_leaf": [
        1,
        2
    ]
}


print("\nGradient Boosting parameter grid:")


for parameter, values in gb_param_grid.items():

    print(
        f"- {parameter}: {values}"
    )


gb_grid_search = GridSearchCV(

    estimator=
        GradientBoostingRegressor(
            random_state=42
        ),

    param_grid=gb_param_grid,

    cv=cv_splits,

    scoring="neg_root_mean_squared_error",

    n_jobs=-1
)


print(
    "\nRunning Gradient Boosting GridSearchCV..."
)


gb_grid_search.fit(
    X_train,
    y_train
)


best_gb_model = (
    gb_grid_search
    .best_estimator_
)


best_gb_cv_rmse = (
    -gb_grid_search
    .best_score_
)


print("\nBest Gradient Boosting parameters:")

print(
    gb_grid_search.best_params_
)


print("\nBest Gradient Boosting CV RMSE:")

print(
    best_gb_cv_rmse
)


# ============================================================
# 29. CREATE FINAL MODEL CANDIDATES
# ============================================================

print("\n" + "-" * 70)
print("CREATING FINAL MODEL CANDIDATES")
print("-" * 70)


final_models = {

    "Linear Regression":
        models["Linear Regression"],

    "Ridge Regression":
        models["Ridge Regression"],

    "Random Forest":
        best_rf_model,

    "Gradient Boosting":
        best_gb_model
}


for model_name in final_models:

    print(
        "-",
        model_name
    )


# ============================================================
# 30. FINAL TEST EVALUATION
# ============================================================

print("\n" + "-" * 70)
print("EVALUATING FINAL CANDIDATE MODELS")
print("-" * 70)


test_results = []


fitted_models = {}


predictions_by_model = {}


for model_name, model in final_models.items():

    print(
        f"\nTraining: {model_name}"
    )


    (
        fitted_model,
        predictions,
        mae,
        mse,
        rmse,
        r2
    ) = evaluate_model(

        model,

        X_train,
        y_train,

        X_test,
        y_test
    )


    fitted_models[
        model_name
    ] = fitted_model


    predictions_by_model[
        model_name
    ] = predictions


    print(
        f"  MAE  : {mae:.4f}"
    )


    print(
        f"  MSE  : {mse:.4f}"
    )


    print(
        f"  RMSE : {rmse:.4f}"
    )


    print(
        f"  R2   : {r2:.4f}"
    )


    test_results.append(
        {
            "Model":
                model_name,

            "Test_MAE":
                mae,

            "Test_MSE":
                mse,

            "Test_RMSE":
                rmse,

            "Test_R2":
                r2
        }
    )


test_results_df = pd.DataFrame(
    test_results
)


# ============================================================
# 31. COMBINE MODEL EVALUATION RESULTS
# ============================================================

print("\n" + "-" * 70)
print("COMBINING MODEL EVALUATION RESULTS")
print("-" * 70)


evaluation_results = (
    cv_results_df
    .merge(
        test_results_df,
        on="Model",
        how="left"
    )
)


evaluation_results = (
    evaluation_results
    .sort_values(
        "CV_RMSE",
        ascending=True
    )
    .reset_index(drop=True)
)


print(
    evaluation_results
    .to_string(
        index=False
    )
)


# ============================================================
# 32. SELECT BEST MODEL
# ============================================================

print("\n" + "-" * 70)
print("SELECTING BEST CATEGORY REVENUE MODEL")
print("-" * 70)


# ------------------------------------------------------------
# Best model is selected using time-series CV RMSE.
#
# Lower RMSE = better forecasting performance.
# ------------------------------------------------------------

best_model_row = (
    evaluation_results
    .iloc[0]
)


best_model_name = (
    best_model_row[
        "Model"
    ]
)


best_model = (
    fitted_models[
        best_model_name
    ]
)


best_model_cv_rmse = (
    best_model_row[
        "CV_RMSE"
    ]
)


best_model_cv_r2 = (
    best_model_row[
        "CV_R2"
    ]
)


best_model_test_mae = (
    best_model_row[
        "Test_MAE"
    ]
)


best_model_test_rmse = (
    best_model_row[
        "Test_RMSE"
    ]
)


best_model_test_r2 = (
    best_model_row[
        "Test_R2"
    ]
)


print("\nBest model based on time-series CV RMSE:")

print(
    best_model_name
)


print("\nBest model CV RMSE:")

print(
    best_model_cv_rmse
)


print("\nBest model CV R2:")

print(
    best_model_cv_r2
)


print("\nBest model Test MAE:")

print(
    best_model_test_mae
)


print("\nBest model Test RMSE:")

print(
    best_model_test_rmse
)


print("\nBest model Test R2:")

print(
    best_model_test_r2
)


# ============================================================
# 33. GENERATE FINAL TEST PREDICTIONS
# ============================================================

print("\n" + "-" * 70)
print("GENERATING FINAL CATEGORY REVENUE FORECASTS")
print("-" * 70)


final_predictions = (
    best_model.predict(
        X_test
    )
)


# Revenue cannot be negative.
final_predictions = np.maximum(
    final_predictions,
    0
)


# ============================================================
# 34. CREATE ACTUAL VS PREDICTED DATASET
# ============================================================

print("\n" + "-" * 70)
print("CREATING ACTUAL VS PREDICTED DATASET")
print("-" * 70)


prediction_data = (
    data.loc[
        test_mask,
        [
            "CourseCategory",
            "Year",
            "Month",
            "YearMonth"
        ]
    ]
    .copy()
    .reset_index(drop=True)
)


prediction_data[
    "ActualCategoryRevenue"
] = y_test.to_numpy()


prediction_data[
    "PredictedCategoryRevenue"
] = final_predictions


prediction_data[
    "PredictionError"
] = (

    prediction_data[
        "ActualCategoryRevenue"
    ]

    -

    prediction_data[
        "PredictedCategoryRevenue"
    ]
)


prediction_data[
    "AbsoluteError"
] = (

    prediction_data[
        "PredictionError"
    ]
    .abs()
)


prediction_data[
    "PercentageError"
] = np.where(

    prediction_data[
        "ActualCategoryRevenue"
    ] > 0,

    (
        prediction_data[
            "AbsoluteError"
        ]

        /

        prediction_data[
            "ActualCategoryRevenue"
        ]
    )
    * 100,

    0
)


print("\nPrediction dataset shape:")

print(
    prediction_data.shape
)


print("\nFirst 20 predictions:")


print(
    prediction_data
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 35. MONTHLY CATEGORY REVENUE SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("CREATING MONTHLY CATEGORY REVENUE SUMMARY")
print("-" * 70)


monthly_summary = (

    prediction_data

    .groupby(
        [
            "Year",
            "Month",
            "YearMonth"
        ],
        as_index=False
    )

    .agg(

        ActualCategoryRevenue=(
            "ActualCategoryRevenue",
            "sum"
        ),

        PredictedCategoryRevenue=(
            "PredictedCategoryRevenue",
            "sum"
        ),

        AbsoluteError=(
            "AbsoluteError",
            "sum"
        )
    )
)


monthly_summary[
    "ForecastError"
] = (

    monthly_summary[
        "ActualCategoryRevenue"
    ]

    -

    monthly_summary[
        "PredictedCategoryRevenue"
    ]
)


monthly_summary[
    "AbsoluteForecastError"
] = (
    monthly_summary[
        "ForecastError"
    ]
    .abs()
)


print(
    monthly_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 36. CATEGORY FORECAST SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("CREATING CATEGORY FORECAST SUMMARY")
print("-" * 70)


category_summary = (

    prediction_data

    .groupby(
        "CourseCategory",
        as_index=False
    )

    .agg(

        ActualRevenue=(
            "ActualCategoryRevenue",
            "sum"
        ),

        PredictedRevenue=(
            "PredictedCategoryRevenue",
            "sum"
        ),

        AbsoluteError=(
            "AbsoluteError",
            "sum"
        )
    )
)


category_summary[
    "ForecastError"
] = (

    category_summary[
        "ActualRevenue"
    ]

    -

    category_summary[
        "PredictedRevenue"
    ]
)


category_summary[
    "PercentageError"
] = np.where(

    category_summary[
        "ActualRevenue"
    ] > 0,

    (

        category_summary[
            "AbsoluteError"
        ]

        /

        category_summary[
            "ActualRevenue"
        ]
    )
    * 100,

    0
)


category_summary = (
    category_summary
    .sort_values(
        "ActualRevenue",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    category_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 37. FEATURE IMPORTANCE
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING FEATURE IMPORTANCE")
print("-" * 70)


# ------------------------------------------------------------
# Case 1:
# Tree-based models
#
# Random Forest and Gradient Boosting provide
# feature_importances_.
# ------------------------------------------------------------

if hasattr(
    best_model,
    "feature_importances_"
):

    importance_values = (
        best_model
        .feature_importances_
    )


    importance_method = (
        "Tree-based feature importance"
    )


# ------------------------------------------------------------
# Case 2:
# Pipeline containing Linear/Ridge
#
# Extract coefficients from the final model.
# Absolute coefficient magnitude is used as the
# importance measure.
# ------------------------------------------------------------

elif isinstance(
    best_model,
    Pipeline
):

    final_estimator = (
        best_model
        .named_steps[
            "model"
        ]
    )


    if hasattr(
        final_estimator,
        "coef_"
    ):

        coefficients = (
            final_estimator
            .coef_
        )


        importance_values = (
            np.abs(
                coefficients
            )
        )


        importance_method = (
            "Absolute regression coefficient magnitude"
        )


    else:

        importance_values = (
            np.full(
                len(final_features),
                np.nan
            )
        )


        importance_method = (
            "Not available"
        )


# ------------------------------------------------------------
# Case 3:
# Unknown model
# ------------------------------------------------------------

else:

    importance_values = (
        np.full(
            len(final_features),
            np.nan
        )
    )


    importance_method = (
        "Not available"
    )


# ------------------------------------------------------------
# Create importance dataframe
# ------------------------------------------------------------

feature_importance = pd.DataFrame(
    {
        "Feature":
            final_features,

        "Importance":
            importance_values
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print("\nFeature importance method:")

print(
    importance_method
)


print("\nTop 20 important features:")


print(
    feature_importance
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 38. FINAL MODEL VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("FINAL MODEL VALIDATION")
print("-" * 70)


# ------------------------------------------------------------
# Prediction row count
# ------------------------------------------------------------

if len(prediction_data) != len(y_test):

    raise ValueError(
        "Prediction row count does not match test data."
    )


# ------------------------------------------------------------
# Missing predictions
# ------------------------------------------------------------

if prediction_data[
    "PredictedCategoryRevenue"
].isna().any():

    raise ValueError(
        "Predicted category revenue contains missing values."
    )


# ------------------------------------------------------------
# Negative predictions
# ------------------------------------------------------------

if (
    prediction_data[
        "PredictedCategoryRevenue"
    ] < 0
).any():

    raise ValueError(
        "Negative category revenue predictions detected."
    )


# ------------------------------------------------------------
# Feature importance validation
# ------------------------------------------------------------

if len(feature_importance) != len(final_features):

    raise ValueError(
        "Feature importance row count does not match "
        "number of predictor features."
    )


print(
    "\nPrediction validation passed."
)


print(
    "\nFeature importance validation passed."
)


# ============================================================
# 39. SAVE MODEL EVALUATION
# ============================================================

print("\n" + "-" * 70)
print("SAVING MODEL EVALUATION RESULTS")
print("-" * 70)


evaluation_file = os.path.join(
    RESULTS_DIR,
    "category_revenue_model_evaluation.csv"
)


evaluation_results.to_csv(
    evaluation_file,
    index=False
)


print("\nEvaluation results saved:")

print(
    evaluation_file
)


# ============================================================
# 40. SAVE FORECAST PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    RESULTS_DIR,
    "category_revenue_forecast_predictions.csv"
)


prediction_data.to_csv(
    prediction_file,
    index=False
)


print("\nForecast predictions saved:")

print(
    prediction_file
)


# ============================================================
# 41. SAVE MONTHLY SUMMARY
# ============================================================

monthly_file = os.path.join(
    RESULTS_DIR,
    "monthly_category_revenue_forecast_summary.csv"
)


monthly_summary.to_csv(
    monthly_file,
    index=False
)


print("\nMonthly forecast summary saved:")

print(
    monthly_file
)


# ============================================================
# 42. SAVE CATEGORY SUMMARY
# ============================================================

category_summary_file = os.path.join(
    RESULTS_DIR,
    "category_revenue_forecast_summary.csv"
)


category_summary.to_csv(
    category_summary_file,
    index=False
)


print("\nCategory summary saved:")

print(
    category_summary_file
)


# ============================================================
# 43. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance_file = os.path.join(
    RESULTS_DIR,
    "category_revenue_feature_importance.csv"
)


feature_importance.to_csv(
    feature_importance_file,
    index=False
)


print("\nFeature importance saved:")

print(
    feature_importance_file
)


# ============================================================
# 44. SAVE FINAL MODEL
# ============================================================

print("\n" + "-" * 70)
print("SAVING FINAL CATEGORY REVENUE MODEL")
print("-" * 70)


model_file = os.path.join(
    MODELS_DIR,
    "final_category_revenue_model.pkl"
)


joblib.dump(
    best_model,
    model_file
)


print("\nFinal category revenue model saved:")

print(
    model_file
)


# ============================================================
# 45. SAVE MODEL METADATA
# ============================================================

metadata_file = os.path.join(
    RESULTS_DIR,
    "category_revenue_model_metadata.csv"
)


metadata = pd.DataFrame(
    {

        "Property": [

            "Model",

            "Target",

            "TrainingPeriod",

            "TestingPeriod",

            "NumberOfFeatures",

            "NumberOfCategories",

            "TimeSeriesCVFolds",

            "CV_RMSE",

            "CV_R2",

            "Test_MAE",

            "Test_RMSE",

            "Test_R2"
        ],


        "Value": [

            best_model_name,

            TARGET_COLUMN,

            f"{train_months[0]} to "
            f"{train_months[-1]}",

            f"{test_months[0]} to "
            f"{test_months[-1]}",

            len(final_features),

            data[
                "CourseCategory"
            ].nunique(),

            len(cv_splits),

            best_model_cv_rmse,

            best_model_cv_r2,

            best_model_test_mae,

            best_model_test_rmse,

            best_model_test_r2
        ]
    }
)


metadata.to_csv(
    metadata_file,
    index=False
)


print("\nModel metadata saved:")

print(
    metadata_file
)


# ============================================================
# 46. FINAL OUTPUT VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("FINAL OUTPUT VALIDATION")
print("-" * 70)


expected_output_files = [

    evaluation_file,

    prediction_file,

    monthly_file,

    category_summary_file,

    feature_importance_file,

    model_file,

    metadata_file
]


missing_outputs = [

    file

    for file in expected_output_files

    if not os.path.exists(file)
]


if missing_outputs:

    raise FileNotFoundError(

        "\nThe following expected outputs "
        "were not created:\n"

        +

        "\n".join(
            f"- {file}"
            for file in missing_outputs
        )
    )


print(
    "\nAll expected output files "
    "were created successfully."
)


# ============================================================
# 47. FEATURE IMPORTANCE OUTPUT VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING FEATURE IMPORTANCE OUTPUT")
print("-" * 70)


saved_feature_importance = pd.read_csv(
    feature_importance_file
)


print("\nFeature importance rows:")

print(
    len(saved_feature_importance)
)


print("\nExpected feature rows:")

print(
    len(final_features)
)


if len(saved_feature_importance) != len(
    final_features
):

    raise ValueError(
        "Saved feature importance row count is incorrect."
    )


# ------------------------------------------------------------
# Check whether all importance values are NaN
# ------------------------------------------------------------

if (
    saved_feature_importance[
        "Importance"
    ]
    .isna()
    .all()
):

    raise ValueError(
        "Feature importance contains only NaN values."
    )


print(
    "\nFeature importance output validation passed."
)


# ============================================================
# 48. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY REVENUE FORECASTING COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nSelected final model:")

print(
    best_model_name
)


print("\nTarget:")

print(
    TARGET_COLUMN
)


print("\nTraining period:")

print(
    f"{train_months[0]} to "
    f"{train_months[-1]}"
)


print("\nTesting period:")

print(
    f"{test_months[0]} to "
    f"{test_months[-1]}"
)


print("\nNumber of predictor features:")

print(
    len(final_features)
)


print("\nNumber of categories:")

print(
    data[
        "CourseCategory"
    ].nunique()
)


print("\nTime-series CV folds:")

print(
    len(cv_splits)
)


print("\nFinal Test MAE:")

print(
    best_model_test_mae
)


print("\nFinal Test RMSE:")

print(
    best_model_test_rmse
)


print("\nFinal Test R2:")

print(
    best_model_test_r2
)


print("\nFeature importance method:")

print(
    importance_method
)


print("\nFinal model:")

print(
    model_file
)


print("\nPrediction file:")

print(
    prediction_file
)


print("\nEvaluation file:")

print(
    evaluation_file
)


print("\nMonthly summary:")

print(
    monthly_file
)


print("\nCategory summary:")

print(
    category_summary_file
)


print("\nFeature importance:")

print(
    feature_importance_file
)


print("\nMetadata:")

print(
    metadata_file
)


print("\n" + "=" * 70)
print("STEP 11 COMPLETED")
print("=" * 70)