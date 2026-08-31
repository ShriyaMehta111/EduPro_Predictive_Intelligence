import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


warnings.filterwarnings("ignore")


# ============================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 10: COURSE REVENUE FORECASTING
# ============================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 10: COURSE REVENUE FORECASTING")
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
# 2. REQUIRED FILES
# ============================================================

print("\n" + "-" * 70)
print("CHECKING REQUIRED FILES")
print("-" * 70)


MODEL_DATA_FILE = os.path.join(
    MODELING_DIR,
    "revenue_modeling_dataset.csv"
)

FEATURE_FILE = os.path.join(
    MODELING_DIR,
    "final_model_features.csv"
)


if not os.path.exists(MODEL_DATA_FILE):

    raise FileNotFoundError(
        f"\nRequired file was not found:\n{MODEL_DATA_FILE}"
    )

print("\nFound:")
print(MODEL_DATA_FILE)


if not os.path.exists(FEATURE_FILE):

    raise FileNotFoundError(
        f"\nRequired file was not found:\n{FEATURE_FILE}"
    )

print("\nFound:")
print(FEATURE_FILE)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "-" * 70)
print("LOADING REVENUE MODELING DATASET")
print("-" * 70)


data = pd.read_csv(
    MODEL_DATA_FILE
)

feature_list = pd.read_csv(
    FEATURE_FILE
)


print("\nDataset shape:")
print(data.shape)

print("\nFeature list shape:")
print(feature_list.shape)


# ============================================================
# 4. IDENTIFY FINAL MODEL FEATURES
# ============================================================

print("\n" + "-" * 70)
print("IDENTIFYING FINAL MODEL FEATURES")
print("-" * 70)


if "Feature" not in feature_list.columns:

    raise KeyError(
        "The final_model_features.csv file must contain "
        "a column named 'Feature'."
    )


model_features = (
    feature_list["Feature"]
    .dropna()
    .astype(str)
    .tolist()
)


print("\nNumber of features from Step 8:")
print(len(model_features))


print("\nFeatures:")

for index, feature in enumerate(
    model_features,
    start=1
):

    print(
        f"{index:02d}. {feature}"
    )


# ============================================================
# 5. REQUIRED MODELING COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING MODELING DATASET")
print("-" * 70)


required_columns = [
    "CourseID",
    "Year",
    "Month",
    "YearMonth",
    "RevenueTarget"
]


for column in required_columns:

    if column not in data.columns:

        raise KeyError(
            f"Missing required modeling column: {column}"
        )


missing_features = [
    feature
    for feature in model_features
    if feature not in data.columns
]


if missing_features:

    raise KeyError(
        "The following model features are missing:\n"
        +
        "\n".join(missing_features)
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

data["RevenueTarget"] = pd.to_numeric(
    data["RevenueTarget"],
    errors="coerce"
)

data["YearMonth"] = data["YearMonth"].astype(str)


for feature in model_features:

    data[feature] = pd.to_numeric(
        data[feature],
        errors="coerce"
    )


print("\nData types prepared successfully.")


# ============================================================
# 7. VALIDATE SOURCE DATA
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
            "Year",
            "Month"
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


if data[
    [
        "Year",
        "Month",
        "RevenueTarget"
    ]
].isna().any().any():

    raise ValueError(
        "Required source columns contain missing values."
    )


print("\nSource dataset validation passed.")


# ============================================================
# 8. SORT CHRONOLOGICALLY
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
# 9. VALIDATE REVENUE TARGET
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING REVENUE TARGET")
print("-" * 70)


if "MonthlyRevenue" in data.columns:

    revenue_difference = (
        data["RevenueTarget"]
        -
        data["MonthlyRevenue"]
    ).abs().max()

    print("\nMaximum Revenue target difference:")
    print(revenue_difference)

    if revenue_difference > 0.000001:

        raise ValueError(
            "RevenueTarget does not match MonthlyRevenue."
        )


if (
    data["RevenueTarget"] < 0
).any():

    raise ValueError(
        "Revenue target contains negative values."
    )


print("\nRevenue target validation passed.")


# ============================================================
# 10. VALIDATE FEATURE VALUES
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING FEATURE VALUES")
print("-" * 70)


feature_missing = (
    data[model_features]
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


print("\nMissing-value validation passed.")


# ============================================================
# 11. CHECK INFINITE VALUES
# ============================================================

print("\n" + "-" * 70)
print("CHECKING INFINITE VALUES")
print("-" * 70)


infinite_values = np.isinf(
    data[model_features]
    .to_numpy(dtype=float)
).sum()


print("\nInfinite feature values:")
print(infinite_values)


if infinite_values > 0:

    raise ValueError(
        "Infinite values found in model features."
    )


print("\nInfinite-value validation passed.")


# ============================================================
# 12. TARGET LEAKAGE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING TARGET LEAKAGE")
print("-" * 70)


for forbidden_column in [
    "RevenueTarget",
    "MonthlyRevenue"
]:

    if forbidden_column in model_features:

        raise ValueError(
            f"Target leakage detected: "
            f"{forbidden_column} is present in model features."
        )


current_period_revenue_features = [
    "MonthlyRevenue",
    "MonthlyRevenuePerEnrollment",
    "RevenueEfficiency"
]


leakage_features_present = [
    feature
    for feature in current_period_revenue_features
    if feature in model_features
]


if leakage_features_present:

    raise ValueError(
        "Current-period revenue leakage detected:\n"
        +
        "\n".join(leakage_features_present)
    )


print(
    "\nNo current-period revenue target leakage detected."
)


# ============================================================
# 13. CREATE MODELING MATRICES
# ============================================================

print("\n" + "-" * 70)
print("CREATING MODELING MATRICES")
print("-" * 70)


X = data[
    model_features
].copy()


y = data[
    "RevenueTarget"
].copy()


print("\nFeature matrix shape:")
print(X.shape)


print("\nTarget shape:")
print(y.shape)


# ============================================================
# 14. CREATE TIME-BASED TRAIN/TEST SPLIT
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME-BASED TRAIN-TEST SPLIT")
print("-" * 70)


available_months = (
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


available_month_strings = (
    available_months[
        "YearMonth"
    ]
    .tolist()
)


print("\nAvailable months:")

for month in available_month_strings:

    print("-", month)


if len(available_month_strings) < 6:

    raise ValueError(
        "At least 6 unique months are required "
        "for time-based forecasting."
    )


test_month_count = 3


if len(available_month_strings) <= test_month_count:

    raise ValueError(
        "Not enough months available for "
        "training and testing."
    )


training_months = (
    available_month_strings[
        :-test_month_count
    ]
)


testing_months = (
    available_month_strings[
        -test_month_count:
    ]
)


print("\nTraining months:")
print(training_months)


print("\nTesting months:")
print(testing_months)


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


print("\nTraining rows:")
print(len(X_train))


print("\nTesting rows:")
print(len(X_test))


# ============================================================
# 15. VALIDATE TEMPORAL TRAIN-TEST ORDER
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING TEMPORAL TRAIN-TEST ORDER")
print("-" * 70)


latest_training_month = max(
    training_months
)

earliest_testing_month = min(
    testing_months
)


print("\nLatest training month:")
print(latest_training_month)


print("\nEarliest testing month:")
print(earliest_testing_month)


if latest_training_month >= earliest_testing_month:

    raise ValueError(
        "Temporal leakage detected: "
        "testing period overlaps training period."
    )


print("\nTemporal train-test validation passed.")


# ============================================================
# 16. CREATE TIME-SERIES CROSS-VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME-SERIES CROSS-VALIDATION")
print("-" * 70)


N_SPLITS = 5


tscv = TimeSeriesSplit(
    n_splits=N_SPLITS
)


print("\nTimeSeriesSplit folds:")
print(N_SPLITS)


print("\nTime-series cross-validation object created.")


# ============================================================
# 17. DEFINE BASELINE AND FORECASTING MODELS
# ============================================================

print("\n" + "-" * 70)
print("CREATING BASELINE AND FORECASTING MODELS")
print("-" * 70)


models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(
            alpha=1.0
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=5,
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

    print("-", model_name)


# ============================================================
# 18. TIME-SERIES CROSS-VALIDATION
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
        tscv.split(X_train),
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


        current_model = clone(
            model
        )


        current_model.fit(
            X_cv_train,
            y_cv_train
        )


        predictions = (
            current_model
            .predict(
                X_cv_validation
            )
        )


        mae = mean_absolute_error(
            y_cv_validation,
            predictions
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_cv_validation,
                predictions
            )
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
        }
    )


cv_results_df = pd.DataFrame(
    cv_results
)


print("\n" + "-" * 70)
print("TIME-SERIES CROSS-VALIDATION RESULTS")
print("-" * 70)


print(
    cv_results_df.to_string(
        index=False
    )
)


# ============================================================
# 19. HYPERPARAMETER TUNING - RANDOM FOREST
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


print(
    "\nRunning Random Forest GridSearchCV..."
)


rf_grid = GridSearchCV(

    estimator=RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ),

    param_grid=rf_param_grid,

    cv=tscv,

    scoring="neg_root_mean_squared_error",

    n_jobs=-1,

    refit=True
)


rf_grid.fit(
    X_train,
    y_train
)


best_rf = rf_grid.best_estimator_


best_rf_cv_rmse = (
    -rf_grid.best_score_
)


print("\nBest Random Forest parameters:")
print(
    rf_grid.best_params_
)


print("\nBest Random Forest CV RMSE:")
print(
    best_rf_cv_rmse
)


# ============================================================
# 20. HYPERPARAMETER TUNING - GRADIENT BOOSTING
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


print(
    "\nRunning Gradient Boosting GridSearchCV..."
)


gb_grid = GridSearchCV(

    estimator=GradientBoostingRegressor(
        random_state=42
    ),

    param_grid=gb_param_grid,

    cv=tscv,

    scoring="neg_root_mean_squared_error",

    n_jobs=-1,

    refit=True
)


gb_grid.fit(
    X_train,
    y_train
)


best_gb = gb_grid.best_estimator_


best_gb_cv_rmse = (
    -gb_grid.best_score_
)


print("\nBest Gradient Boosting parameters:")
print(
    gb_grid.best_params_
)


print("\nBest Gradient Boosting CV RMSE:")
print(
    best_gb_cv_rmse
)


# ============================================================
# 21. TRAIN FINAL CANDIDATE MODELS
# ============================================================

print("\n" + "-" * 70)
print("TRAINING FINAL CANDIDATE MODELS")
print("-" * 70)


candidate_models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(
            alpha=1.0
        ),

    "Random Forest":
        best_rf,

    "Gradient Boosting":
        best_gb
}


test_results = []


trained_models = {}


for model_name, model in candidate_models.items():

    print(
        f"\nTraining: {model_name}"
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    mse = mean_squared_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mse
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    trained_models[
        model_name
    ] = model


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


test_results_df = pd.DataFrame(
    test_results
)


# ============================================================
# 22. COMBINE MODEL EVALUATION RESULTS
# ============================================================

print("\n" + "-" * 70)
print("COMBINING MODEL EVALUATION RESULTS")
print("-" * 70)


evaluation_results = cv_results_df.merge(
    test_results_df,
    on="Model",
    how="left",
    validate="one_to_one"
)


print(
    evaluation_results.to_string(
        index=False
    )
)


# ============================================================
# 23. SELECT BEST MODEL
# ============================================================

print("\n" + "-" * 70)
print("SELECTING BEST REVENUE FORECASTING MODEL")
print("-" * 70)


best_model_name = (
    evaluation_results
    .sort_values(
        "CV_RMSE"
    )
    .iloc[0]["Model"]
)


best_model_cv_rmse = (
    evaluation_results
    .loc[
        evaluation_results["Model"]
        ==
        best_model_name,
        "CV_RMSE"
    ]
    .iloc[0]
)


best_model_cv_r2 = (
    evaluation_results
    .loc[
        evaluation_results["Model"]
        ==
        best_model_name,
        "CV_R2"
    ]
    .iloc[0]
)


best_model_test_rmse = (
    evaluation_results
    .loc[
        evaluation_results["Model"]
        ==
        best_model_name,
        "Test_RMSE"
    ]
    .iloc[0]
)


best_model_test_r2 = (
    evaluation_results
    .loc[
        evaluation_results["Model"]
        ==
        best_model_name,
        "Test_R2"
    ]
    .iloc[0]
)


print("\nBest model based on Time-Series CV RMSE:")
print(best_model_name)


print("\nBest model CV RMSE:")
print(best_model_cv_rmse)


print("\nBest model CV R2:")
print(best_model_cv_r2)


print("\nBest model test RMSE:")
print(best_model_test_rmse)


print("\nBest model test R2:")
print(best_model_test_r2)


# ============================================================
# 24. RETRAIN SELECTED FINAL MODEL
# ============================================================

print("\n" + "-" * 70)
print("RETRAINING SELECTED FINAL MODEL")
print("-" * 70)


final_model = trained_models[
    best_model_name
]


final_model.fit(
    X_train,
    y_train
)


print("\nFinal revenue forecasting model trained.")


# ============================================================
# 25. GENERATE FINAL REVENUE FORECASTS
# ============================================================

print("\n" + "-" * 70)
print("GENERATING FINAL REVENUE FORECASTS")
print("-" * 70)


final_predictions = (
    final_model
    .predict(
        X_test
    )
)


final_mae = mean_absolute_error(
    y_test,
    final_predictions
)


final_mse = mean_squared_error(
    y_test,
    final_predictions
)


final_rmse = np.sqrt(
    final_mse
)


final_r2 = r2_score(
    y_test,
    final_predictions
)


print("\nFinal model:")
print(best_model_name)


print("\nFinal test MAE:")
print(final_mae)


print("\nFinal test MSE:")
print(final_mse)


print("\nFinal test RMSE:")
print(final_rmse)


print("\nFinal test R2:")
print(final_r2)


# ============================================================
# 26. CREATE ACTUAL VS PREDICTED DATASET
# ============================================================

print("\n" + "-" * 70)
print("CREATING ACTUAL VS PREDICTED DATASET")
print("-" * 70)


prediction_data = data.loc[
    test_mask,
    [
        "CourseID",
        "Year",
        "Month",
        "YearMonth"
    ]
].copy()


prediction_data[
    "ActualRevenue"
] = y_test.values


prediction_data[
    "PredictedRevenue"
] = final_predictions


prediction_data[
    "PredictionError"
] = (
    prediction_data["ActualRevenue"]
    -
    prediction_data["PredictedRevenue"]
)


prediction_data[
    "AbsoluteError"
] = (
    prediction_data["PredictionError"]
    .abs()
)


prediction_data[
    "PercentageError"
] = np.where(

    prediction_data[
        "ActualRevenue"
    ] != 0,

    (
        prediction_data[
            "AbsoluteError"
        ]
        /
        prediction_data[
            "ActualRevenue"
        ]
    )
    * 100,

    0
)


prediction_data = (
    prediction_data
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
# 27. MONTHLY REVENUE FORECAST SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("MONTHLY REVENUE FORECAST SUMMARY")
print("-" * 70)


monthly_forecast_summary = (
    prediction_data
    .groupby(
        [
            "Year",
            "Month",
            "YearMonth"
        ]
    )
    .agg(

        ActualRevenue=(
            "ActualRevenue",
            "sum"
        ),

        PredictedRevenue=(
            "PredictedRevenue",
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
        "ActualRevenue"
    ]
    -
    monthly_forecast_summary[
        "PredictedRevenue"
    ]
)


print(
    monthly_forecast_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 28. CALCULATE FEATURE IMPORTANCE
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING FEATURE IMPORTANCE")
print("-" * 70)


if hasattr(
    final_model,
    "feature_importances_"
):

    importance_values = (
        final_model
        .feature_importances_
    )


elif hasattr(
    final_model,
    "coef_"
):

    importance_values = np.abs(
        final_model.coef_
    )


else:

    importance_values = np.zeros(
        len(model_features)
    )


feature_importance = pd.DataFrame(
    {
        "Feature":
            model_features,

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


print("\nTop 20 important features:")


print(
    feature_importance
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 29. FINAL MODEL VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("FINAL MODEL VALIDATION")
print("-" * 70)


if len(final_predictions) != len(
    y_test
):

    raise ValueError(
        "Prediction count does not match test target count."
    )


if not np.isfinite(
    final_predictions
).all():

    raise ValueError(
        "Final model generated invalid "
        "infinite or NaN predictions."
    )


if prediction_data[
    "ActualRevenue"
].isna().any():

    raise ValueError(
        "Actual revenue contains missing values."
    )


if prediction_data[
    "PredictedRevenue"
].isna().any():

    raise ValueError(
        "Predicted revenue contains missing values."
    )


print("\nPrediction validation passed.")


# ============================================================
# 30. SAVE MODEL EVALUATION RESULTS
# ============================================================

print("\n" + "-" * 70)
print("SAVING MODEL EVALUATION RESULTS")
print("-" * 70)


evaluation_file = os.path.join(
    RESULTS_DIR,
    "revenue_model_evaluation.csv"
)


evaluation_results.to_csv(
    evaluation_file,
    index=False
)


print("\nEvaluation results saved:")
print(evaluation_file)


# ============================================================
# 31. SAVE FORECAST PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    RESULTS_DIR,
    "revenue_forecast_predictions.csv"
)


prediction_data.to_csv(
    prediction_file,
    index=False
)


print("\nForecast predictions saved:")
print(prediction_file)


# ============================================================
# 32. SAVE MONTHLY SUMMARY
# ============================================================

monthly_summary_file = os.path.join(
    RESULTS_DIR,
    "monthly_revenue_forecast_summary.csv"
)


monthly_forecast_summary.to_csv(
    monthly_summary_file,
    index=False
)


print("\nMonthly forecast summary saved:")
print(monthly_summary_file)


# ============================================================
# 33. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance_file = os.path.join(
    RESULTS_DIR,
    "revenue_feature_importance.csv"
)


feature_importance.to_csv(
    feature_importance_file,
    index=False
)


print("\nFeature importance saved:")
print(feature_importance_file)


# ============================================================
# 34. SAVE FINAL REVENUE MODEL
# ============================================================

print("\n" + "-" * 70)
print("SAVING FINAL REVENUE MODEL")
print("-" * 70)


final_model_file = os.path.join(
    MODELS_DIR,
    "final_revenue_model.pkl"
)


joblib.dump(
    final_model,
    final_model_file
)


print("\nFinal model saved:")
print(final_model_file)


# ============================================================
# 35. SAVE MODEL METADATA
# ============================================================

metadata_file = os.path.join(
    RESULTS_DIR,
    "revenue_model_metadata.csv"
)


metadata = pd.DataFrame(
    [
        {
            "ModelType":
                "Course Revenue Forecasting",

            "SelectedModel":
                best_model_name,

            "TrainingStart":
                training_months[0],

            "TrainingEnd":
                training_months[-1],

            "TestingStart":
                testing_months[0],

            "TestingEnd":
                testing_months[-1],

            "NumberOfFeatures":
                len(model_features),

            "TimeSeriesCVFolds":
                N_SPLITS,

            "CV_MAE":
                best_model_cv_rmse
                if False
                else
                evaluation_results.loc[
                    evaluation_results["Model"]
                    ==
                    best_model_name,
                    "CV_MAE"
                ].iloc[0],

            "CV_RMSE":
                best_model_cv_rmse,

            "CV_R2":
                best_model_cv_r2,

            "Test_MAE":
                final_mae,

            "Test_MSE":
                final_mse,

            "Test_RMSE":
                final_rmse,

            "Test_R2":
                final_r2
        }
    ]
)


metadata.to_csv(
    metadata_file,
    index=False
)


print("\nModel metadata saved:")
print(metadata_file)


# ============================================================
# 36. FINAL OUTPUT VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("FINAL OUTPUT VALIDATION")
print("-" * 70)


expected_outputs = [
    evaluation_file,
    prediction_file,
    monthly_summary_file,
    feature_importance_file,
    final_model_file,
    metadata_file
]


missing_outputs = [
    file_path
    for file_path in expected_outputs
    if not os.path.exists(file_path)
]


if missing_outputs:

    raise FileNotFoundError(
        "The following expected outputs were not created:\n"
        +
        "\n".join(missing_outputs)
    )


print(
    "\nAll expected outputs were created successfully."
)


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REVENUE FORECASTING COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nSelected final model:")
print(best_model_name)


print("\nTraining period:")
print(
    f"{training_months[0]} to {training_months[-1]}"
)


print("\nTesting period:")
print(
    f"{testing_months[0]} to {testing_months[-1]}"
)


print("\nNumber of predictor features:")
print(
    len(model_features)
)


print("\nTime-series CV folds:")
print(N_SPLITS)


print("\nFinal Test MAE:")
print(final_mae)


print("\nFinal Test MSE:")
print(final_mse)


print("\nFinal Test RMSE:")
print(final_rmse)


print("\nFinal Test R2:")
print(final_r2)


print("\nFinal model:")
print(final_model_file)


print("\nPrediction file:")
print(prediction_file)


print("\nEvaluation file:")
print(evaluation_file)


print("\nMonthly summary:")
print(monthly_summary_file)


print("\nFeature importance:")
print(feature_importance_file)


print("\n" + "=" * 70)
print("STEP 10 COMPLETED")
print("=" * 70)