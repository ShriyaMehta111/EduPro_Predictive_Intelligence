import os
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")


# ============================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 8: MODEL PREPARATION & FEATURE SELECTION
# ============================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 8: MODEL PREPARATION & FEATURE SELECTION")
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


os.makedirs(
    MODELING_DIR,
    exist_ok=True
)


print("\n" + "-" * 70)
print("PROJECT PATH")
print("-" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nProcessed data directory:")
print(PROCESSED_DIR)

print("\nModeling data directory:")
print(MODELING_DIR)


# ============================================================
# 2. REQUIRED FILE
# ============================================================

print("\n" + "-" * 70)
print("CHECKING REQUIRED FILE")
print("-" * 70)


INPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "feature_engineered_dataset.csv"
)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"\nRequired file was not found:\n{INPUT_FILE}"
    )


print("\nFound:")
print(INPUT_FILE)


# ============================================================
# 3. LOAD FEATURE ENGINEERED DATA
# ============================================================

print("\n" + "-" * 70)
print("LOADING FEATURE ENGINEERED DATASET")
print("-" * 70)


data = pd.read_csv(
    INPUT_FILE
)


print("\nDataset shape:")
print(data.shape)


print("\nNumber of rows:")
print(len(data))


print("\nNumber of columns:")
print(len(data.columns))


# ============================================================
# 4. REQUIRED COLUMN VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING REQUIRED COLUMNS")
print("-" * 70)


required_columns = [
    "CourseID",
    "Year",
    "Month",
    "YearMonth",
    "MonthlyEnrollment",
    "MonthlyRevenue",
    "MonthlyTransactionCount",
    "AverageTransactionAmount",
    "CourseCategory",
    "CourseType",
    "CourseLevel",
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "TeacherCount",
    "AvgTeacherExperience",
    "AvgTeacherRating",
    "ExpertiseMatchRate",
    "Quarter",
    "TimeIndex",
    "PreviousMonthEnrollment",
    "PreviousMonthRevenue",
    "PreviousMonthTransactionCount",
    "PreviousMonthAverageTransactionAmount",
    "Previous3MonthEnrollmentSum",
    "Previous3MonthRevenueSum",
    "Previous6MonthEnrollment",
    "Previous6MonthRevenue",
    "EnrollmentGrowthRate",
    "RevenueGrowthRate",
    "HasPreviousMonthData",
    "MonthSin",
    "MonthCos",
    "YearProgress",
    "EnrollmentTarget",
    "RevenueTarget"
]


missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing_columns:

    raise KeyError(
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


data["Year"] = pd.to_numeric(
    data["Year"],
    errors="coerce"
)

data["Month"] = pd.to_numeric(
    data["Month"],
    errors="coerce"
)

data["TimeIndex"] = pd.to_numeric(
    data["TimeIndex"],
    errors="coerce"
)


numeric_columns = [
    "MonthlyEnrollment",
    "MonthlyRevenue",
    "MonthlyTransactionCount",
    "AverageTransactionAmount",
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "TeacherCount",
    "AvgTeacherExperience",
    "AvgTeacherRating",
    "ExpertiseMatchRate",
    "PreviousMonthEnrollment",
    "PreviousMonthRevenue",
    "PreviousMonthTransactionCount",
    "PreviousMonthAverageTransactionAmount",
    "Previous3MonthEnrollmentSum",
    "Previous3MonthRevenueSum",
    "Previous6MonthEnrollment",
    "Previous6MonthRevenue",
    "EnrollmentGrowthRate",
    "RevenueGrowthRate",
    "HasPreviousMonthData",
    "MonthSin",
    "MonthCos",
    "YearProgress",
    "EnrollmentTarget",
    "RevenueTarget"
]


for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


print("\nData types prepared successfully.")


# ============================================================
# 6. SOURCE DATA VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING SOURCE DATA")
print("-" * 70)


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
        "Duplicate Course-Month combinations detected."
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
            "CourseID"
        ]
    )
    .reset_index(drop=True)
)


print("\nData sorted chronologically.")


# ============================================================
# 8. TARGET DEFINITIONS
# ============================================================

print("\n" + "-" * 70)
print("DEFINING MODELING TARGETS")
print("-" * 70)


enrollment_target = "EnrollmentTarget"

revenue_target = "RevenueTarget"


print("\nEnrollment forecasting target:")
print(enrollment_target)

print("\nRevenue forecasting target:")
print(revenue_target)


# ============================================================
# 9. TARGET VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING TARGET VARIABLES")
print("-" * 70)


enrollment_difference = (
    data["EnrollmentTarget"]
    -
    data["MonthlyEnrollment"]
).abs().max()


revenue_difference = (
    data["RevenueTarget"]
    -
    data["MonthlyRevenue"]
).abs().max()


print("\nMaximum Enrollment target difference:")
print(enrollment_difference)


print("\nMaximum Revenue target difference:")
print(revenue_difference)


if enrollment_difference > 0:

    raise ValueError(
        "EnrollmentTarget does not match MonthlyEnrollment."
    )


if revenue_difference > 0.000001:

    raise ValueError(
        "RevenueTarget does not match MonthlyRevenue."
    )


print("\nTarget validation passed.")


# ============================================================
# 10. CREATE CATEGORY REVENUE DATASET
# ============================================================

print("\n" + "-" * 70)
print("CREATING CATEGORY REVENUE TARGET")
print("-" * 70)


category_monthly_revenue = (
    data
    .groupby(
        [
            "CourseCategory",
            "Year",
            "Month",
            "YearMonth"
        ],
        as_index=False
    )
    .agg(
        CategoryRevenueTarget=(
            "MonthlyRevenue",
            "sum"
        ),

        CategoryEnrollment=(
            "MonthlyEnrollment",
            "sum"
        )
    )
)


print("\nCategory-month dataset shape:")
print(
    category_monthly_revenue.shape
)


print("\nUnique categories:")
print(
    category_monthly_revenue[
        "CourseCategory"
    ].nunique()
)


print("\nUnique months:")
print(
    category_monthly_revenue[
        "YearMonth"
    ].nunique()
)


# ============================================================
# 11. CATEGORY REVENUE RECONCILIATION
# ============================================================

print("\n" + "-" * 70)
print("CATEGORY REVENUE RECONCILIATION")
print("-" * 70)


category_total_revenue = (
    category_monthly_revenue[
        "CategoryRevenueTarget"
    ]
    .sum()
)


original_total_revenue = (
    data["MonthlyRevenue"]
    .sum()
)


category_revenue_difference = abs(
    category_total_revenue
    -
    original_total_revenue
)


print("\nOriginal total revenue:")
print(original_total_revenue)


print("\nCategory total revenue:")
print(category_total_revenue)


print("\nRevenue difference:")
print(category_revenue_difference)


if category_revenue_difference > 0.000001:

    raise ValueError(
        "Category revenue does not reconcile with "
        "the original monthly revenue."
    )


print("\nCategory revenue reconciliation passed.")


# ============================================================
# 12. CREATE CATEGORY REVENUE FORECASTING DATASET
# ============================================================

print("\n" + "-" * 70)
print("CREATING CATEGORY REVENUE FORECASTING DATASET")
print("-" * 70)


category_data = category_monthly_revenue.copy()


category_data = (
    category_data
    .sort_values(
        [
            "CourseCategory",
            "Year",
            "Month"
        ]
    )
    .reset_index(drop=True)
)


# Previous month category revenue
category_data[
    "PreviousMonthCategoryRevenue"
] = (
    category_data
    .groupby("CourseCategory")[
        "CategoryRevenueTarget"
    ]
    .shift(1)
)


# Previous 3-month category revenue
category_data[
    "Previous3MonthCategoryRevenue"
] = (
    category_data
    .groupby("CourseCategory")[
        "CategoryRevenueTarget"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=3,
            min_periods=1
        )
        .sum()
    )
)


# Previous 6-month category revenue
category_data[
    "Previous6MonthCategoryRevenue"
] = (
    category_data
    .groupby("CourseCategory")[
        "CategoryRevenueTarget"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=6,
            min_periods=1
        )
        .sum()
    )
)


# Category revenue growth
previous_category_revenue = (
    category_data
    .groupby("CourseCategory")[
        "CategoryRevenueTarget"
    ]
    .shift(1)
)


category_data[
    "CategoryRevenueGrowthRate"
] = np.where(
    previous_category_revenue > 0,
    (
        category_data[
            "CategoryRevenueTarget"
        ]
        -
        previous_category_revenue
    )
    /
    previous_category_revenue,
    0
)


# Category revenue availability
category_data[
    "HasPreviousCategoryRevenue"
] = (
    previous_category_revenue
    .notna()
    .astype(int)
)


# Fill only historical values that are unavailable
category_historical_columns = [
    "PreviousMonthCategoryRevenue",
    "Previous3MonthCategoryRevenue",
    "Previous6MonthCategoryRevenue"
]


for column in category_historical_columns:

    category_data[column] = (
        category_data[column]
        .fillna(0)
    )


category_data[
    "CategoryRevenueGrowthRate"
] = (
    category_data[
        "CategoryRevenueGrowthRate"
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        0
    )
    .fillna(0)
)


print("\nCategory revenue forecasting features created.")


# ============================================================
# 13. CATEGORY REVENUE LEAKAGE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("CATEGORY REVENUE LEAKAGE VALIDATION")
print("-" * 70)


category_check = (
    category_data
    .sort_values(
        [
            "CourseCategory",
            "Year",
            "Month"
        ]
    )
    .copy()
)


category_check[
    "ExpectedPreviousRevenue"
] = (
    category_check
    .groupby("CourseCategory")[
        "CategoryRevenueTarget"
    ]
    .shift(1)
    .fillna(0)
)


category_lag_difference = (
    category_check[
        "PreviousMonthCategoryRevenue"
    ]
    -
    category_check[
        "ExpectedPreviousRevenue"
    ]
).abs().max()


print(
    "\nMaximum PreviousMonthCategoryRevenue difference:"
)

print(category_lag_difference)


if category_lag_difference > 0.000001:

    raise ValueError(
        "Category revenue historical feature leakage detected."
    )


print("\nCategory historical leakage validation passed.")


# ============================================================
# 14. IDENTIFY CURRENT-PERIOD TARGET COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("IDENTIFYING TARGET LEAKAGE COLUMNS")
print("-" * 70)


target_columns = [
    "EnrollmentTarget",
    "RevenueTarget",
    "MonthlyEnrollment",
    "MonthlyRevenue"
]


print("\nColumns excluded from predictor features:")

for column in target_columns:

    print("-", column)


# ============================================================
# 15. IDENTIFY IDENTIFIER / TEXT COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("IDENTIFYING NON-PREDICTIVE IDENTIFIER COLUMNS")
print("-" * 70)


identifier_columns = [
    "CourseID",
    "CourseName",
    "MonthName",
    "YearMonth",
    "CourseCategory",
    "CourseType",
    "CourseLevel",
    "PriceBand",
    "DurationBucket",
    "RatingTier",
    "ExperienceBucket"
]


print("\nIdentifier/categorical text columns excluded from "
      "raw numeric predictor matrix:")

for column in identifier_columns:

    print("-", column)


# ============================================================
# 16. IDENTIFY ONE-HOT ENCODED COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("IDENTIFYING ENCODED CATEGORICAL FEATURES")
print("-" * 70)


encoded_columns = [
    column
    for column in data.columns
    if (
        column.startswith("Category_")
        or
        column.startswith("PriceBand_")
        or
        column.startswith("Duration_")
        or
        column.startswith("Rating_")
        or
        column.startswith("Experience_")
    )
]


print("\nEncoded categorical columns:")

for column in encoded_columns:

    print("-", column)


# ============================================================
# 17. DEFINE BASE NUMERIC FEATURES
# ============================================================

print("\n" + "-" * 70)
print("DEFINING BASE NUMERIC FEATURES")
print("-" * 70)


excluded_columns = (
    target_columns
    +
    identifier_columns
)


candidate_features = [
    column
    for column in data.columns
    if (
        column not in excluded_columns
        and
        pd.api.types.is_numeric_dtype(
            data[column]
        )
    )
]


print("\nCandidate numeric features:")

for column in candidate_features:

    print("-", column)


print("\nNumber of candidate features:")
print(len(candidate_features))


# ============================================================
# 18. REMOVE CURRENT-PERIOD OBSERVED PERFORMANCE FEATURES
# ============================================================

print("\n" + "-" * 70)
print("REMOVING CURRENT-PERIOD PERFORMANCE LEAKAGE")
print("-" * 70)


current_period_performance_columns = [
    "MonthlyTransactionCount",
    "AverageTransactionAmount"
]


print(
    "\nThe following current-period performance variables "
    "are excluded because they are not known before forecasting:"
)


for column in current_period_performance_columns:

    print("-", column)


candidate_features = [
    column
    for column in candidate_features
    if column not in current_period_performance_columns
]


# ============================================================
# 19. REMOVE TARGET-DERIVED CURRENT-PERIOD FEATURES
# ============================================================

print("\n" + "-" * 70)
print("REMOVING TARGET-DERIVED CURRENT-PERIOD FEATURES")
print("-" * 70)


target_derived_current_features = [
    "MonthlyRevenuePerEnrollment",
    "RevenueEfficiency"
]


for column in target_derived_current_features:

    if column in candidate_features:

        print("Removing:", column)

        candidate_features.remove(
            column
        )


print(
    "\nTarget-derived current-period features removed."
)


# ============================================================
# 20. VERIFY NUMERIC FEATURE MATRIX
# ============================================================

print("\n" + "-" * 70)
print("CREATING NUMERIC FEATURE MATRIX")
print("-" * 70)


X_all = data[
    candidate_features
].copy()


print("\nFeature matrix shape:")
print(X_all.shape)


# ============================================================
# 21. MISSING VALUE CHECK
# ============================================================

print("\n" + "-" * 70)
print("CHECKING FEATURE MISSING VALUES")
print("-" * 70)


missing_features = (
    X_all
    .isna()
    .sum()
)


missing_features = (
    missing_features[
        missing_features > 0
    ]
)


if len(missing_features) > 0:

    print("\nMissing values found:")

    print(missing_features)

    raise ValueError(
        "Feature matrix contains missing values."
    )


print("\nNo missing feature values found.")


# ============================================================
# 22. INFINITE VALUE CHECK
# ============================================================

print("\n" + "-" * 70)
print("CHECKING INFINITE VALUES")
print("-" * 70)


infinite_count = np.isinf(
    X_all
    .select_dtypes(
        include=np.number
    )
    .to_numpy()
).sum()


print("\nInfinite numeric values:")
print(infinite_count)


if infinite_count > 0:

    raise ValueError(
        "Infinite numeric values found in feature matrix."
    )


print("\nInfinite-value validation passed.")


# ============================================================
# 23. ZERO VARIANCE FEATURE ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("ANALYZING ZERO-VARIANCE FEATURES")
print("-" * 70)


variance_selector = VarianceThreshold(
    threshold=0
)


variance_selector.fit(
    X_all
)


variance_values = (
    X_all
    .var()
)


zero_variance_features = (
    variance_values[
        variance_values == 0
    ]
    .index
    .tolist()
)


print("\nZero-variance features:")

if zero_variance_features:

    for column in zero_variance_features:

        print("-", column)

else:

    print("None")


# Remove zero-variance features
if zero_variance_features:

    X_all = X_all.drop(
        columns=zero_variance_features
    )

    candidate_features = [
        column
        for column in candidate_features
        if column not in zero_variance_features
    ]


print("\nFeatures remaining after zero-variance removal:")
print(len(candidate_features))


# ============================================================
# 24. CORRELATION ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("PERFORMING FEATURE CORRELATION ANALYSIS")
print("-" * 70)


correlation_matrix = (
    X_all
    .corr()
)


CORRELATION_THRESHOLD = 0.90


high_correlation_pairs = []


columns = correlation_matrix.columns


for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        correlation_value = (
            correlation_matrix.iloc[
                i,
                j
            ]
        )

        if (
            abs(correlation_value)
            >=
            CORRELATION_THRESHOLD
        ):

            high_correlation_pairs.append(
                {
                    "Feature1": columns[i],
                    "Feature2": columns[j],
                    "Correlation": correlation_value
                }
            )


correlation_report = pd.DataFrame(
    high_correlation_pairs
)


print(
    "\nCorrelation threshold:"
)

print(
    CORRELATION_THRESHOLD
)


print(
    "\nNumber of highly correlated feature pairs:"
)

print(
    len(correlation_report)
)


if len(correlation_report) > 0:

    print(
        "\nHighly correlated feature pairs:"
    )

    print(
        correlation_report
        .sort_values(
            "Correlation",
            key=lambda x: x.abs(),
            ascending=False
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nNo feature pairs exceeded the "
        "correlation threshold."
    )


# ============================================================
# 25. SAVE CORRELATION REPORT
# ============================================================

CORRELATION_FILE = os.path.join(
    MODELING_DIR,
    "feature_correlation_report.csv"
)


correlation_report.to_csv(
    CORRELATION_FILE,
    index=False
)


print("\nCorrelation report saved:")
print(CORRELATION_FILE)


# ============================================================
# 26. REMOVE REDUNDANT FEATURES
# ============================================================

print("\n" + "-" * 70)
print("REMOVING REDUNDANT FEATURES")
print("-" * 70)


# We remove only obvious redundant variables.
# Historical features are retained because they represent
# different forecasting windows and are useful for models.

redundant_features = []


# Do not automatically remove highly correlated historical
# features solely based on correlation. They have different
# forecasting meanings and will be evaluated by the models.


print(
    "\nNo forecasting features were automatically removed "
    "solely because of high correlation."
)


print(
    "\nReason:"
)

print(
    "Highly correlated lag/rolling features can still carry "
    "different business meaning and will be evaluated during "
    "model training."
)


# ============================================================
# 27. FINAL FEATURE LIST
# ============================================================

final_features = [
    column
    for column in candidate_features
    if column not in redundant_features
]


print("\n" + "-" * 70)
print("FINAL MODELING FEATURES")
print("-" * 70)


for number, column in enumerate(
    final_features,
    start=1
):

    print(
        f"{number:02d}. {column}"
    )


print("\nTotal final features:")
print(len(final_features))


# ============================================================
# 28. CREATE ENROLLMENT MODELING DATASET
# ============================================================

print("\n" + "-" * 70)
print("CREATING ENROLLMENT MODELING DATASET")
print("-" * 70)


enrollment_model_data = data[
    [
        "CourseID",
        "Year",
        "Month",
        "YearMonth"
    ]
    +
    final_features
    +
    [
        enrollment_target
    ]
].copy()


print("\nEnrollment modeling dataset shape:")
print(
    enrollment_model_data.shape
)


# ============================================================
# 29. CREATE REVENUE MODELING DATASET
# ============================================================

print("\n" + "-" * 70)
print("CREATING REVENUE MODELING DATASET")
print("-" * 70)


revenue_model_data = data[
    [
        "CourseID",
        "Year",
        "Month",
        "YearMonth"
    ]
    +
    final_features
    +
    [
        revenue_target
    ]
].copy()


print("\nRevenue modeling dataset shape:")
print(
    revenue_model_data.shape
)


# ============================================================
# 30. CREATE CATEGORY REVENUE MODELING DATASET
# ============================================================

print("\n" + "-" * 70)
print("CREATING CATEGORY REVENUE MODELING DATASET")
print("-" * 70)


category_feature_columns = [
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


category_model_data = category_data[
    [
        "CourseCategory"
    ]
    +
    category_feature_columns
].copy()


print("\nCategory revenue modeling dataset shape:")
print(
    category_model_data.shape
)


# ============================================================
# 31. CREATE TIME-BASED TRAIN/TEST SPLIT INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME-BASED TRAIN/TEST SPLIT")
print("-" * 70)


unique_months = (
    sorted(
        data[
            "YearMonth"
        ]
        .unique()
    )
)


print("\nAvailable months:")

for month in unique_months:

    print("-", month)


number_of_months = len(
    unique_months
)


if number_of_months < 4:

    raise ValueError(
        "Insufficient time periods for time-series validation."
    )


# Last 20% of chronological months are reserved for testing.
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
    unique_months[
        :train_month_count
    ]
)


test_months = (
    unique_months[
        train_month_count:
    ]
)


print("\nTraining months:")

print(
    train_months
)


print("\nTesting months:")

print(
    test_months
)


enrollment_train = (
    enrollment_model_data[
        enrollment_model_data[
            "YearMonth"
        ].isin(train_months)
    ]
    .copy()
)


enrollment_test = (
    enrollment_model_data[
        enrollment_model_data[
            "YearMonth"
        ].isin(test_months)
    ]
    .copy()
)


revenue_train = (
    revenue_model_data[
        revenue_model_data[
            "YearMonth"
        ].isin(train_months)
    ]
    .copy()
)


revenue_test = (
    revenue_model_data[
        revenue_model_data[
            "YearMonth"
        ].isin(test_months)
    ]
    .copy()
)


category_train = (
    category_model_data[
        category_model_data[
            "YearMonth"
        ].isin(train_months)
    ]
    .copy()
)


category_test = (
    category_model_data[
        category_model_data[
            "YearMonth"
        ].isin(test_months)
    ]
    .copy()
)


print("\nEnrollment training rows:")
print(len(enrollment_train))


print("\nEnrollment testing rows:")
print(len(enrollment_test))


print("\nRevenue training rows:")
print(len(revenue_train))


print("\nRevenue testing rows:")
print(len(revenue_test))


print("\nCategory revenue training rows:")
print(len(category_train))


print("\nCategory revenue testing rows:")
print(len(category_test))


# ============================================================
# 32. VERIFY TEMPORAL ORDER
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING TEMPORAL TRAIN-TEST ORDER")
print("-" * 70)


latest_train_month = (
    max(train_months)
)


earliest_test_month = (
    min(test_months)
)


print("\nLatest training month:")
print(latest_train_month)


print("\nEarliest testing month:")
print(earliest_test_month)


if latest_train_month >= earliest_test_month:

    raise ValueError(
        "Temporal leakage detected: test period is not "
        "strictly after training period."
    )


print("\nTemporal train-test validation passed.")


# ============================================================
# 33. CREATE TIME SERIES CROSS-VALIDATION OBJECT
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME-SERIES CROSS-VALIDATION")
print("-" * 70)


TIME_SERIES_SPLITS = 5


if train_month_count <= TIME_SERIES_SPLITS:

    TIME_SERIES_SPLITS = max(
        2,
        train_month_count - 1
    )


time_series_cv = TimeSeriesSplit(
    n_splits=TIME_SERIES_SPLITS
)


print("\nTimeSeriesSplit folds:")
print(TIME_SERIES_SPLITS)


print(
    "\nTime-series cross-validation object created."
)


# ============================================================
# 34. SAVE FINAL FEATURE LIST
# ============================================================

print("\n" + "-" * 70)
print("SAVING FINAL FEATURE LIST")
print("-" * 70)


feature_list_file = os.path.join(
    MODELING_DIR,
    "final_model_features.csv"
)


feature_list_df = pd.DataFrame(
    {
        "Feature": final_features
    }
)


feature_list_df.to_csv(
    feature_list_file,
    index=False
)


print("\nFeature list saved:")
print(feature_list_file)


# ============================================================
# 35. SAVE ENROLLMENT MODELING DATA
# ============================================================

ENROLLMENT_FILE = os.path.join(
    MODELING_DIR,
    "enrollment_modeling_dataset.csv"
)


enrollment_model_data.to_csv(
    ENROLLMENT_FILE,
    index=False
)


print("\nEnrollment modeling dataset saved:")
print(ENROLLMENT_FILE)


# ============================================================
# 36. SAVE REVENUE MODELING DATA
# ============================================================

REVENUE_FILE = os.path.join(
    MODELING_DIR,
    "revenue_modeling_dataset.csv"
)


revenue_model_data.to_csv(
    REVENUE_FILE,
    index=False
)


print("\nRevenue modeling dataset saved:")
print(REVENUE_FILE)


# ============================================================
# 37. SAVE CATEGORY REVENUE MODELING DATA
# ============================================================

CATEGORY_FILE = os.path.join(
    MODELING_DIR,
    "category_revenue_modeling_dataset.csv"
)


category_model_data.to_csv(
    CATEGORY_FILE,
    index=False
)


print("\nCategory revenue modeling dataset saved:")
print(CATEGORY_FILE)


# ============================================================
# 38. SAVE TRAIN-TEST DATASETS
# ============================================================

enrollment_train.to_csv(
    os.path.join(
        MODELING_DIR,
        "enrollment_train.csv"
    ),
    index=False
)


enrollment_test.to_csv(
    os.path.join(
        MODELING_DIR,
        "enrollment_test.csv"
    ),
    index=False
)


revenue_train.to_csv(
    os.path.join(
        MODELING_DIR,
        "revenue_train.csv"
    ),
    index=False
)


revenue_test.to_csv(
    os.path.join(
        MODELING_DIR,
        "revenue_test.csv"
    ),
    index=False
)


category_train.to_csv(
    os.path.join(
        MODELING_DIR,
        "category_revenue_train.csv"
    ),
    index=False
)


category_test.to_csv(
    os.path.join(
        MODELING_DIR,
        "category_revenue_test.csv"
    ),
    index=False
)


print("\nTrain-test datasets saved successfully.")


# ============================================================
# 39. FINAL VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("FINAL MODEL PREPARATION VALIDATION")
print("-" * 70)


print("\nOriginal feature-engineered rows:")
print(len(data))


print("\nEnrollment modeling rows:")
print(len(enrollment_model_data))


print("\nRevenue modeling rows:")
print(len(revenue_model_data))


print("\nCategory revenue modeling rows:")
print(len(category_model_data))


print("\nFinal predictor count:")
print(len(final_features))


# Missing validation
final_missing = (
    enrollment_model_data
    .isna()
    .sum()
    .sum()
)


if final_missing > 0:

    raise ValueError(
        "Missing values found in enrollment modeling dataset."
    )


final_missing_revenue = (
    revenue_model_data
    .isna()
    .sum()
    .sum()
)


if final_missing_revenue > 0:

    raise ValueError(
        "Missing values found in revenue modeling dataset."
    )


final_missing_category = (
    category_model_data
    .isna()
    .sum()
    .sum()
)


if final_missing_category > 0:

    raise ValueError(
        "Missing values found in category revenue "
        "modeling dataset."
    )


print("\nMissing-value validation passed.")


# Target leakage validation
for forbidden_column in target_columns:

    if forbidden_column in final_features:

        raise ValueError(
            f"Target leakage detected. "
            f"{forbidden_column} is present in predictors."
        )


print("\nTarget leakage validation passed.")


# ============================================================
# 40. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nInput file:")
print(INPUT_FILE)


print("\nEnrollment modeling dataset:")
print(ENROLLMENT_FILE)


print("\nRevenue modeling dataset:")
print(REVENUE_FILE)


print("\nCategory revenue modeling dataset:")
print(CATEGORY_FILE)


print("\nFinal number of predictor features:")
print(len(final_features))


print("\nTraining months:")
print(
    f"{train_months[0]} to {train_months[-1]}"
)


print("\nTesting months:")
print(
    f"{test_months[0]} to {test_months[-1]}"
)


print("\nTime-series CV folds:")
print(TIME_SERIES_SPLITS)


print("\n" + "=" * 70)
print("STEP 8 COMPLETED")
print("=" * 70)