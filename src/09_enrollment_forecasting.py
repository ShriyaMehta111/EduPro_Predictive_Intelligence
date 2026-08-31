import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV
)

warnings.filterwarnings("ignore")


# ============================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 9: ENROLLMENT FORECASTING
# ============================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 9: ENROLLMENT FORECASTING")
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

PROCESSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed"
)

MODELING_DIR = os.path.join(
    PROCESSED_DIR,
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
# 2. REQUIRED FILES
# ============================================================

print("\n" + "-" * 70)
print("CHECKING REQUIRED FILES")
print("-" * 70)


modeling_file = os.path.join(
    MODELING_DIR,
    "enrollment_modeling_dataset.csv"
)

feature_file = os.path.join(
    MODELING_DIR,
    "final_model_features.csv"
)


required_files = [
    modeling_file,
    feature_file
]


for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired file was not found:\n{file_path}"
        )

    print("\nFound:")
    print(file_path)


# ============================================================
# 3. LOAD MODELING DATA
# ============================================================

print("\n" + "-" * 70)
print("LOADING ENROLLMENT MODELING DATASET")
print("-" * 70)


data = pd.read_csv(
    modeling_file
)


feature_list_data = pd.read_csv(
    feature_file
)


print("\nDataset shape:")
print(data.shape)

print("\nFeature list shape:")
print(feature_list_data.shape)


# ============================================================
# 4. IDENTIFY FEATURE LIST
# ============================================================

print("\n" + "-" * 70)
print("IDENTIFYING FINAL MODEL FEATURES")
print("-" * 70)


if "Feature" in feature_list_data.columns:

    feature_columns = (
        feature_list_data["Feature"]
        .dropna()
        .astype(str)
        .tolist()
    )

elif "FeatureName" in feature_list_data.columns:

    feature_columns = (
        feature_list_data["FeatureName"]
        .dropna()
        .astype(str)
        .tolist()
    )

else:

    feature_columns = (
        feature_list_data.iloc[:, 0]
        .dropna()
        .astype(str)
        .tolist()
    )


print("\nNumber of features from Step 8:")
print(len(feature_columns))


print("\nFeatures:")

for i, feature in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# 5. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING MODELING DATASET")
print("-" * 70)


required_columns = [
    "CourseID",
    "Year",
    "Month",
    "YearMonth",
    "EnrollmentTarget"
]


for column in required_columns:

    if column not in data.columns:

        raise KeyError(
            f"Required column missing: {column}"
        )


missing_features = [
    feature
    for feature in feature_columns
    if feature not in data.columns
]


if missing_features:

    print("\nMissing model features:")

    for feature in missing_features:

        print("-", feature)

    raise KeyError(
        "One or more features from Step 8 "
        "are missing from the enrollment dataset."
    )


print("\nAll required modeling columns are present.")


# ============================================================
# 6. PREPARE DATA TYPES
# ============================================================

print("\n" + "-" * 70)
print("PREPARING DATA TYPES")
print("-" * 70)


data["Year"] = pd.to_numeric(
    data["Year"],
    errors="coerce"
)

data["Month"] = pd.to_numeric(
    data["Month"],
    errors="coerce"
)

data["EnrollmentTarget"] = pd.to_numeric(
    data["EnrollmentTarget"],
    errors="coerce"
)


for feature in feature_columns:

    data[feature] = pd.to_numeric(
        data[feature],
        errors="coerce"
    )


print("\nData types prepared successfully.")


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
            "CourseID"
        ]
    )
    .reset_index(drop=True)
)


print("\nData sorted by Year, Month and CourseID.")


# ============================================================
# 8. SOURCE DATA VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING SOURCE DATA")
print("-" * 70)


print("\nNumber of rows:")
print(len(data))


print("\nUnique courses:")
print(
    data["CourseID"].nunique()
)


print("\nUnique months:")
print(
    data["YearMonth"].nunique()
)


duplicate_course_month = (
    data
    .duplicated(
        subset=[
            "CourseID",
            "YearMonth"
        ]
    )
    .sum()
)


print("\nDuplicate Course-Month combinations:")
print(duplicate_course_month)


if duplicate_course_month > 0:

    raise ValueError(
        "Duplicate Course-Month combinations found."
    )


# ============================================================
# 9. TARGET VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING ENROLLMENT TARGET")
print("-" * 70)


if data["EnrollmentTarget"].isna().sum() > 0:

    raise ValueError(
        "EnrollmentTarget contains missing values."
    )


if (
    data["EnrollmentTarget"] < 0
).any():

    raise ValueError(
        "EnrollmentTarget contains negative values."
    )


if "MonthlyEnrollment" in data.columns:

    target_difference = (
        data["EnrollmentTarget"]
        -
        data["MonthlyEnrollment"]
    ).abs().max()

    print("\nMaximum target difference:")
    print(target_difference)

    if target_difference > 0.000001:

        raise ValueError(
            "EnrollmentTarget does not match "
            "MonthlyEnrollment."
        )


print("\nEnrollment target validation passed.")


# ============================================================
# 10. FEATURE MISSING VALUE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING FEATURE VALUES")
print("-" * 70)


feature_missing = (
    data[feature_columns]
    .isna()
    .sum()
    .sum()
)


print("\nTotal missing feature values:")
print(feature_missing)


if feature_missing > 0:

    raise ValueError(
        "Model features contain missing values."
    )


# ============================================================
# 11. INFINITE VALUE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("CHECKING INFINITE VALUES")
print("-" * 70)


infinite_count = np.isinf(
    data[feature_columns]
    .to_numpy(dtype=float)
).sum()


print("\nInfinite feature values:")
print(infinite_count)


if infinite_count > 0:

    raise ValueError(
        "Model features contain infinite values."
    )


print("\nInfinite-value validation passed.")


# ============================================================
# 12. DEFINE MODELING MATRIX
# ============================================================

print("\n" + "-" * 70)
print("CREATING MODELING MATRICES")
print("-" * 70)


X = data[
    feature_columns
].copy()


y = data[
    "EnrollmentTarget"
].copy()


print("\nFeature matrix shape:")
print(X.shape)


print("\nTarget shape:")
print(y.shape)


# ============================================================
# 13. IDENTIFY TRAINING AND TESTING MONTHS
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME-BASED TRAIN-TEST SPLIT")
print("-" * 70)


available_months = (
    data["YearMonth"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


print("\nAvailable months:")

for month in available_months:

    print("-", month)


if len(available_months) < 6:

    raise ValueError(
        "Not enough monthly observations "
        "for time-based forecasting."
    )


# ------------------------------------------------------------
# Use the same 75% / 25% temporal structure created in Step 8.
# With 12 months this gives:
# Training = first 9 months
# Testing  = last 3 months
# ------------------------------------------------------------

test_month_count = 3


if len(available_months) <= test_month_count:

    raise ValueError(
        "Not enough months to create "
        "a 3-month testing period."
    )


training_months = (
    available_months[
        :-test_month_count
    ]
)


testing_months = (
    available_months[
        -test_month_count:
    ]
)


train_mask = (
    data["YearMonth"]
    .isin(training_months)
)


test_mask = (
    data["YearMonth"]
    .isin(testing_months)
)


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


print("\nTraining months:")
print(training_months)


print("\nTesting months:")
print(testing_months)


print("\nTraining rows:")
print(len(X_train))


print("\nTesting rows:")
print(len(X_test))


# ============================================================
# 14. TEMPORAL ORDER VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING TEMPORAL TRAIN-TEST ORDER")
print("-" * 70)


latest_training_month = (
    training_months[-1]
)


earliest_testing_month = (
    testing_months[0]
)


print("\nLatest training month:")
print(latest_training_month)


print("\nEarliest testing month:")
print(earliest_testing_month)


if latest_training_month >= earliest_testing_month:

    raise ValueError(
        "Temporal leakage detected: "
        "training period overlaps testing period."
    )


print("\nTemporal train-test validation passed.")


# ============================================================
# 15. CREATE TIME SERIES CROSS-VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME-SERIES CROSS-VALIDATION")
print("-" * 70)


n_splits = 5


if len(X_train) <= n_splits:

    raise ValueError(
        "Training dataset is too small "
        "for 5-fold TimeSeriesSplit."
    )


time_series_cv = TimeSeriesSplit(
    n_splits=n_splits
)


print("\nTimeSeriesSplit folds:")
print(n_splits)


print("\nTime-series cross-validation object created.")


# ============================================================
# 16. MODEL DEFINITIONS
# ============================================================

print("\n" + "-" * 70)
print("CREATING BASELINE AND FORECASTING MODELS")
print("-" * 70)


models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


for model_name in models:

    print(
        f"- {model_name}"
    )


# ============================================================
# 17. MODEL EVALUATION FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 18. CROSS-VALIDATION EVALUATION
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
        train_index,
        validation_index
    ) in enumerate(
        time_series_cv.split(X_train),
        start=1
    ):


        X_cv_train = X_train.iloc[
            train_index
        ]

        X_cv_validation = X_train.iloc[
            validation_index
        ]


        y_cv_train = y_train.iloc[
            train_index
        ]

        y_cv_validation = y_train.iloc[
            validation_index
        ]


        cv_model = clone(
            model
        )


        cv_model.fit(
            X_cv_train,
            y_cv_train
        )


        cv_prediction = (
            cv_model.predict(
                X_cv_validation
            )
        )


        metrics = calculate_metrics(
            y_cv_validation,
            cv_prediction
        )


        fold_mae.append(
            metrics["MAE"]
        )

        fold_rmse.append(
            metrics["RMSE"]
        )

        fold_r2.append(
            metrics["R2"]
        )


        print(
            f"  Fold {fold_number}: "
            f"MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"R2={metrics['R2']:.4f}"
        )


    cv_results.append({

        "Model":
            model_name,

        "CV_MAE":
            np.mean(fold_mae),

        "CV_RMSE":
            np.mean(fold_rmse),

        "CV_R2":
            np.mean(fold_r2),

        "CV_R2_STD":
            np.std(fold_r2)

    })


cv_results_df = pd.DataFrame(
    cv_results
)


# ============================================================
# 19. DISPLAY CROSS-VALIDATION RESULTS
# ============================================================

print("\n" + "-" * 70)
print("TIME-SERIES CROSS-VALIDATION RESULTS")
print("-" * 70)


print(
    cv_results_df.to_string(
        index=False
    )
)


# ============================================================
# 20. HYPERPARAMETER TUNING - RANDOM FOREST
# ============================================================

print("\n" + "-" * 70)
print("HYPERPARAMETER TUNING: RANDOM FOREST")
print("-" * 70)


random_forest = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


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
    estimator=random_forest,
    param_grid=rf_param_grid,
    cv=time_series_cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    refit=True
)


print("\nRunning Random Forest GridSearchCV...")


rf_grid_search.fit(
    X_train,
    y_train
)


best_rf_model = (
    rf_grid_search.best_estimator_
)


best_rf_parameters = (
    rf_grid_search.best_params_
)


best_rf_cv_rmse = (
    -rf_grid_search.best_score_
)


print("\nBest Random Forest parameters:")
print(best_rf_parameters)


print("\nBest Random Forest CV RMSE:")
print(best_rf_cv_rmse)


# ============================================================
# 21. HYPERPARAMETER TUNING - GRADIENT BOOSTING
# ============================================================

print("\n" + "-" * 70)
print("HYPERPARAMETER TUNING: GRADIENT BOOSTING")
print("-" * 70)


gradient_boosting = (
    GradientBoostingRegressor(
        random_state=42
    )
)


gb_param_grid = {

    "n_estimators": [
        50,
        100,
        200
    ],

    "learning_rate": [
        0.03,
        0.05,
        0.10
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
    estimator=gradient_boosting,
    param_grid=gb_param_grid,
    cv=time_series_cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    refit=True
)


print("\nRunning Gradient Boosting GridSearchCV...")


gb_grid_search.fit(
    X_train,
    y_train
)


best_gb_model = (
    gb_grid_search.best_estimator_
)


best_gb_parameters = (
    gb_grid_search.best_params_
)


best_gb_cv_rmse = (
    -gb_grid_search.best_score_
)


print("\nBest Gradient Boosting parameters:")
print(best_gb_parameters)


print("\nBest Gradient Boosting CV RMSE:")
print(best_gb_cv_rmse)


# ============================================================
# 22. ADD TUNED MODELS TO MODEL COLLECTION
# ============================================================

tuned_models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(
            random_state=42
        ),

    "Random Forest":
        best_rf_model,

    "Gradient Boosting":
        best_gb_model
}


# ============================================================
# 23. TRAIN MODELS ON TRAINING DATA
# ============================================================

print("\n" + "-" * 70)
print("TRAINING FINAL CANDIDATE MODELS")
print("-" * 70)


test_results = []

prediction_dictionary = {}


for model_name, model in tuned_models.items():

    print(
        f"\nTraining: {model_name}"
    )


    model.fit(
        X_train,
        y_train
    )


    prediction = (
        model.predict(
            X_test
        )
    )


    metrics = calculate_metrics(
        y_test,
        prediction
    )


    test_results.append({

        "Model":
            model_name,

        "Test_MAE":
            metrics["MAE"],

        "Test_MSE":
            metrics["MSE"],

        "Test_RMSE":
            metrics["RMSE"],

        "Test_R2":
            metrics["R2"]

    })


    prediction_dictionary[
        model_name
    ] = prediction


    print(
        f"  MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"  MSE  : {metrics['MSE']:.4f}"
    )

    print(
        f"  RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"  R2   : {metrics['R2']:.4f}"
    )


test_results_df = pd.DataFrame(
    test_results
)


# ============================================================
# 24. MERGE CV AND TEST RESULTS
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


print(
    evaluation_results.to_string(
        index=False
    )
)


# ============================================================
# 25. SELECT BEST MODEL
# ============================================================

print("\n" + "-" * 70)
print("SELECTING BEST ENROLLMENT FORECASTING MODEL")
print("-" * 70)


# ------------------------------------------------------------
# Primary selection criterion:
# Cross-validation RMSE
#
# Lower RMSE = better forecasting performance.
# ------------------------------------------------------------

best_model_name = (
    evaluation_results
    .sort_values(
        "CV_RMSE",
        ascending=True
    )
    .iloc[0]["Model"]
)


best_model = (
    tuned_models[
        best_model_name
    ]
)


print("\nBest model based on Time-Series CV RMSE:")
print(best_model_name)


best_model_cv_row = (
    evaluation_results[
        evaluation_results["Model"]
        ==
        best_model_name
    ]
    .iloc[0]
)


print("\nBest model CV RMSE:")
print(
    best_model_cv_row["CV_RMSE"]
)


print("\nBest model CV R2:")
print(
    best_model_cv_row["CV_R2"]
)


print("\nBest model test RMSE:")
print(
    best_model_cv_row["Test_RMSE"]
)


print("\nBest model test R2:")
print(
    best_model_cv_row["Test_R2"]
)


# ============================================================
# 26. FINAL MODEL RETRAINING
# ============================================================

print("\n" + "-" * 70)
print("RETRAINING SELECTED FINAL MODEL")
print("-" * 70)


final_model = clone(
    best_model
)


final_model.fit(
    X_train,
    y_train
)


print("\nFinal enrollment forecasting model trained.")


# ============================================================
# 27. FINAL TEST PREDICTIONS
# ============================================================

print("\n" + "-" * 70)
print("GENERATING FINAL ENROLLMENT FORECASTS")
print("-" * 70)


final_predictions = (
    final_model.predict(
        X_test
    )
)


final_predictions = np.maximum(
    final_predictions,
    0
)


final_metrics = calculate_metrics(
    y_test,
    final_predictions
)


print("\nFinal model:")
print(best_model_name)


print("\nFinal test MAE:")
print(final_metrics["MAE"])


print("\nFinal test MSE:")
print(final_metrics["MSE"])


print("\nFinal test RMSE:")
print(final_metrics["RMSE"])


print("\nFinal test R2:")
print(final_metrics["R2"])


# ============================================================
# 28. CREATE PREDICTION DATASET
# ============================================================

print("\n" + "-" * 70)
print("CREATING ACTUAL VS PREDICTED DATASET")
print("-" * 70)


prediction_output = data.loc[
    test_mask,
    [
        "CourseID",
        "Year",
        "Month",
        "YearMonth"
    ]
].copy()


prediction_output[
    "ActualEnrollment"
] = y_test.values


prediction_output[
    "PredictedEnrollment"
] = final_predictions


prediction_output[
    "PredictionError"
] = (
    prediction_output[
        "ActualEnrollment"
    ]
    -
    prediction_output[
        "PredictedEnrollment"
    ]
)


prediction_output[
    "AbsoluteError"
] = (
    prediction_output[
        "PredictionError"
    ]
    .abs()
)


prediction_output[
    "PercentageError"
] = np.where(

    prediction_output[
        "ActualEnrollment"
    ] != 0,

    (
        prediction_output[
            "AbsoluteError"
        ]
        /
        prediction_output[
            "ActualEnrollment"
        ]
    )
    * 100,

    0
)


prediction_output = (
    prediction_output
    .sort_values(
        [
            "Year",
            "Month",
            "CourseID"
        ]
    )
    .reset_index(drop=True)
)


print("\nPrediction dataset shape:")
print(
    prediction_output.shape
)


print("\nFirst 20 predictions:")

print(
    prediction_output
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 29. MONTHLY FORECAST SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("MONTHLY ENROLLMENT FORECAST SUMMARY")
print("-" * 70)


monthly_forecast_summary = (
    prediction_output
    .groupby(
        [
            "Year",
            "Month",
            "YearMonth"
        ]
    )
    .agg(

        ActualEnrollment=(
            "ActualEnrollment",
            "sum"
        ),

        PredictedEnrollment=(
            "PredictedEnrollment",
            "sum"
        ),

        AbsoluteError=(
            "AbsoluteError",
            "sum"
        )

    )
    .reset_index()
)


monthly_forecast_summary[
    "ForecastError"
] = (
    monthly_forecast_summary[
        "ActualEnrollment"
    ]
    -
    monthly_forecast_summary[
        "PredictedEnrollment"
    ]
)


print(
    monthly_forecast_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 30. FEATURE IMPORTANCE
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING FEATURE IMPORTANCE")
print("-" * 70)


feature_importance = None


if hasattr(
    final_model,
    "feature_importances_"
):

    feature_importance = pd.DataFrame({

        "Feature":
            feature_columns,

        "Importance":
            final_model.feature_importances_

    })


elif hasattr(
    final_model,
    "coef_"
):

    coefficients = (
        np.asarray(
            final_model.coef_
        )
        .reshape(-1)
    )


    feature_importance = pd.DataFrame({

        "Feature":
            feature_columns,

        "Coefficient":
            coefficients,

        "AbsoluteCoefficient":
            np.abs(coefficients)

    })


if feature_importance is not None:

    if (
        "Importance"
        in feature_importance.columns
    ):

        feature_importance = (
            feature_importance
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        print(
            "\nTop 20 important features:"
        )

        print(
            feature_importance
            .head(20)
            .to_string(
                index=False
            )
        )

    else:

        feature_importance = (
            feature_importance
            .sort_values(
                "AbsoluteCoefficient",
                ascending=False
            )
            .reset_index(drop=True)
        )

        print(
            "\nTop 20 features by absolute coefficient:"
        )

        print(
            feature_importance
            .head(20)
            .to_string(
                index=False
            )
        )

else:

    print(
        "\nFeature importance is not available "
        "for the selected model."
    )


# ============================================================
# 31. MODEL VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("FINAL MODEL VALIDATION")
print("-" * 70)


if len(prediction_output) != len(
    testing_months
) * data["CourseID"].nunique():

    raise ValueError(
        "Prediction output does not contain "
        "the expected number of test observations."
    )


if prediction_output[
    "PredictedEnrollment"
].isna().sum() > 0:

    raise ValueError(
        "Predicted enrollment contains missing values."
    )


if np.isinf(
    prediction_output[
        "PredictedEnrollment"
    ].to_numpy()
).sum() > 0:

    raise ValueError(
        "Predicted enrollment contains infinite values."
    )


if (
    prediction_output[
        "PredictedEnrollment"
    ] < 0
).any():

    raise ValueError(
        "Predicted enrollment contains negative values."
    )


print(
    "\nPrediction validation passed."
)


# ============================================================
# 32. SAVE EVALUATION RESULTS
# ============================================================

print("\n" + "-" * 70)
print("SAVING MODEL EVALUATION RESULTS")
print("-" * 70)


evaluation_file = os.path.join(
    RESULTS_DIR,
    "enrollment_model_evaluation.csv"
)


evaluation_results.to_csv(
    evaluation_file,
    index=False
)


print("\nEvaluation results saved:")
print(evaluation_file)


# ============================================================
# 33. SAVE PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    RESULTS_DIR,
    "enrollment_forecast_predictions.csv"
)


prediction_output.to_csv(
    prediction_file,
    index=False
)


print("\nForecast predictions saved:")
print(prediction_file)


# ============================================================
# 34. SAVE MONTHLY SUMMARY
# ============================================================

monthly_summary_file = os.path.join(
    RESULTS_DIR,
    "monthly_enrollment_forecast_summary.csv"
)


monthly_forecast_summary.to_csv(
    monthly_summary_file,
    index=False
)


print("\nMonthly forecast summary saved:")
print(monthly_summary_file)


# ============================================================
# 35. SAVE FEATURE IMPORTANCE
# ============================================================

if feature_importance is not None:

    feature_importance_file = os.path.join(
        RESULTS_DIR,
        "enrollment_feature_importance.csv"
    )


    feature_importance.to_csv(
        feature_importance_file,
        index=False
    )


    print(
        "\nFeature importance saved:"
    )

    print(
        feature_importance_file
    )


# ============================================================
# 36. SAVE FINAL MODEL
# ============================================================

print("\n" + "-" * 70)
print("SAVING FINAL ENROLLMENT MODEL")
print("-" * 70)


final_model_file = os.path.join(
    MODELS_DIR,
    "final_enrollment_model.pkl"
)


joblib.dump(
    final_model,
    final_model_file
)


if not os.path.exists(
    final_model_file
):

    raise FileNotFoundError(
        "Final enrollment model was not saved."
    )


print("\nFinal model saved:")
print(final_model_file)


# ============================================================
# 37. SAVE MODEL METADATA
# ============================================================

model_metadata = pd.DataFrame({

    "Property": [

        "Model",
        "TrainingStartMonth",
        "TrainingEndMonth",
        "TestingStartMonth",
        "TestingEndMonth",
        "TrainingRows",
        "TestingRows",
        "NumberOfFeatures",
        "TimeSeriesCVFolds",
        "CV_RMSE",
        "CV_R2",
        "Test_MAE",
        "Test_MSE",
        "Test_RMSE",
        "Test_R2"

    ],

    "Value": [

        best_model_name,

        training_months[0],

        training_months[-1],

        testing_months[0],

        testing_months[-1],

        len(X_train),

        len(X_test),

        len(feature_columns),

        n_splits,

        best_model_cv_row[
            "CV_RMSE"
        ],

        best_model_cv_row[
            "CV_R2"
        ],

        final_metrics[
            "MAE"
        ],

        final_metrics[
            "MSE"
        ],

        final_metrics[
            "RMSE"
        ],

        final_metrics[
            "R2"
        ]

    ]

})


metadata_file = os.path.join(
    RESULTS_DIR,
    "enrollment_model_metadata.csv"
)


model_metadata.to_csv(
    metadata_file,
    index=False
)


print("\nModel metadata saved:")
print(metadata_file)


# ============================================================
# 38. FINAL OUTPUT VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("FINAL OUTPUT VALIDATION")
print("-" * 70)


output_files = [
    evaluation_file,
    prediction_file,
    monthly_summary_file,
    final_model_file,
    metadata_file
]


if feature_importance is not None:

    output_files.append(
        feature_importance_file
    )


for output_file in output_files:

    if not os.path.exists(
        output_file
    ):

        raise FileNotFoundError(
            f"Expected output was not created:\n"
            f"{output_file}"
        )


print(
    "\nAll expected outputs were created successfully."
)


# ============================================================
# 39. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ENROLLMENT FORECASTING COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nSelected final model:")
print(best_model_name)


print("\nTraining period:")
print(
    f"{training_months[0]} "
    f"to "
    f"{training_months[-1]}"
)


print("\nTesting period:")
print(
    f"{testing_months[0]} "
    f"to "
    f"{testing_months[-1]}"
)


print("\nNumber of predictor features:")
print(len(feature_columns))


print("\nTime-series CV folds:")
print(n_splits)


print("\nFinal Test MAE:")
print(final_metrics["MAE"])


print("\nFinal Test MSE:")
print(final_metrics["MSE"])


print("\nFinal Test RMSE:")
print(final_metrics["RMSE"])


print("\nFinal Test R2:")
print(final_metrics["R2"])


print("\nFinal model:")
print(final_model_file)


print("\nPrediction file:")
print(prediction_file)


print("\nEvaluation file:")
print(evaluation_file)


print("\nMonthly summary:")
print(monthly_summary_file)


print("\n" + "=" * 70)
print("STEP 9 COMPLETED")
print("=" * 70)