import os
import numpy as np
import pandas as pd


# ======================================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 7: FEATURE ENGINEERING
# ======================================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 7: FEATURE ENGINEERING")
print("=" * 70)


# ======================================================================
# 1. PROJECT PATHS
# ======================================================================

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


print("\n" + "-" * 70)
print("PROJECT PATH")
print("-" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nProcessed data directory:")
print(PROCESSED_DIR)


# ======================================================================
# 2. REQUIRED FILE
# ======================================================================

INPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "monthly_forecasting_dataset.csv"
)


print("\n" + "-" * 70)
print("CHECKING REQUIRED FILE")
print("-" * 70)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"\nRequired file was not found:\n{INPUT_FILE}"
    )


print("\nFound:")
print(INPUT_FILE)


# ======================================================================
# 3. LOAD DATA
# ======================================================================

print("\n" + "-" * 70)
print("LOADING MONTHLY FORECASTING DATASET")
print("-" * 70)


data = pd.read_csv(
    INPUT_FILE
)


print("\nDataset shape:")
print(data.shape)


print("\nDataset columns:")

for column in data.columns:
    print("-", column)


# ======================================================================
# 4. REQUIRED COLUMN VALIDATION
# ======================================================================

print("\n" + "-" * 70)
print("VALIDATING REQUIRED COLUMNS")
print("-" * 70)


required_columns = [
    "CourseID",
    "Year",
    "Month",
    "MonthName",
    "YearMonth",

    "MonthlyEnrollment",
    "MonthlyRevenue",
    "MonthlyTransactionCount",
    "AverageTransactionAmount",

    "CourseName",
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
    "IsQuarterStart",
    "IsQuarterEnd",
    "TimeIndex"
]


missing_required_columns = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing_required_columns:

    raise KeyError(
        "The following required columns are missing:\n"
        + "\n".join(missing_required_columns)
    )


print("\nAll required columns are present.")


# ======================================================================
# 5. PREPARE DATA TYPES
# ======================================================================

print("\n" + "-" * 70)
print("PREPARING DATA TYPES")
print("-" * 70)


numeric_columns = [
    "Year",
    "Month",
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
    "Quarter",
    "IsQuarterStart",
    "IsQuarterEnd",
    "TimeIndex"
]


for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


data["CourseID"] = data["CourseID"].astype(str)

data["CourseCategory"] = (
    data["CourseCategory"]
    .astype(str)
    .str.strip()
)

data["CourseType"] = (
    data["CourseType"]
    .astype(str)
    .str.strip()
)

data["CourseLevel"] = (
    data["CourseLevel"]
    .astype(str)
    .str.strip()
)


# Create proper monthly date
data["Date"] = pd.to_datetime(
    data["YearMonth"] + "-01",
    errors="coerce"
)


if data["Date"].isna().any():

    raise ValueError(
        "Invalid YearMonth values were found."
    )


print("\nData types prepared successfully.")


# ======================================================================
# 6. SOURCE DATA VALIDATION
# ======================================================================

print("\n" + "-" * 70)
print("VALIDATING SOURCE DATA")
print("-" * 70)


unique_courses = data["CourseID"].nunique()

unique_months = data["YearMonth"].nunique()

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


print("\nUnique courses:")
print(unique_courses)

print("\nUnique months:")
print(unique_months)

print("\nDuplicate Course-Month combinations:")
print(duplicate_course_month)


if unique_courses != 60:

    raise ValueError(
        f"Expected 60 courses but found {unique_courses}."
    )


if unique_months != 12:

    raise ValueError(
        f"Expected 12 months but found {unique_months}."
    )


if duplicate_course_month > 0:

    raise ValueError(
        "Duplicate Course-Month combinations found."
    )


# ======================================================================
# 7. SORT DATA
# ======================================================================

print("\n" + "-" * 70)
print("SORTING DATA CHRONOLOGICALLY")
print("-" * 70)


data = (
    data
    .sort_values(
        [
            "CourseID",
            "Date"
        ]
    )
    .reset_index(drop=True)
)


print("\nData sorted by CourseID and month.")


# ======================================================================
# 8. COURSE PRICE BANDS
# ======================================================================

print("\n" + "-" * 70)
print("CREATING COURSE PRICE BANDS")
print("-" * 70)


def create_price_band(price):

    if price == 0:
        return "Free"

    elif price <= 100:
        return "Low"

    elif price <= 300:
        return "Medium"

    else:
        return "High"


data["PriceBand"] = (
    data["CoursePrice"]
    .apply(create_price_band)
)


price_band_course_distribution = (
    data[
        [
            "CourseID",
            "PriceBand"
        ]
    ]
    .drop_duplicates()
    ["PriceBand"]
    .value_counts()
    .sort_index()
)


print("\nPrice band distribution across unique courses:")

print(
    price_band_course_distribution
)


# ======================================================================
# 9. COURSE DURATION BUCKETS
# ======================================================================

print("\n" + "-" * 70)
print("CREATING COURSE DURATION BUCKETS")
print("-" * 70)


def create_duration_bucket(duration):

    if duration <= 10:
        return "Short"

    elif duration <= 30:
        return "Medium"

    else:
        return "Long"


data["DurationBucket"] = (
    data["CourseDuration"]
    .apply(create_duration_bucket)
)


duration_course_distribution = (
    data[
        [
            "CourseID",
            "DurationBucket"
        ]
    ]
    .drop_duplicates()
    ["DurationBucket"]
    .value_counts()
    .sort_index()
)


print("\nDuration bucket distribution across unique courses:")

print(
    duration_course_distribution
)


# ======================================================================
# 10. COURSE RATING TIERS
# ======================================================================

print("\n" + "-" * 70)
print("CREATING COURSE RATING TIERS")
print("-" * 70)


def create_rating_tier(rating):

    if rating < 2:
        return "Low"

    elif rating < 3:
        return "Medium"

    elif rating < 4:
        return "Good"

    else:
        return "Excellent"


data["RatingTier"] = (
    data["CourseRating"]
    .apply(create_rating_tier)
)


rating_course_distribution = (
    data[
        [
            "CourseID",
            "RatingTier"
        ]
    ]
    .drop_duplicates()
    ["RatingTier"]
    .value_counts()
    .sort_index()
)


print("\nRating tier distribution across unique courses:")

print(
    rating_course_distribution
)


# ======================================================================
# 11. COURSE TYPE ENCODING
# ======================================================================

print("\n" + "-" * 70)
print("ENCODING COURSE TYPE")
print("-" * 70)


course_type_mapping = {
    "Free": 0,
    "Paid": 1
}


unexpected_course_types = set(
    data["CourseType"].unique()
) - set(
    course_type_mapping.keys()
)


if unexpected_course_types:

    raise ValueError(
        "Unexpected CourseType values found: "
        + str(unexpected_course_types)
    )


data["CourseTypeEncoded"] = (
    data["CourseType"]
    .map(course_type_mapping)
    .astype(int)
)


print("\nCourse type encoding:")

print(
    data[
        [
            "CourseType",
            "CourseTypeEncoded"
        ]
    ]
    .drop_duplicates()
    .sort_values("CourseTypeEncoded")
    .to_string(index=False)
)


# ======================================================================
# 12. COURSE LEVEL ENCODING
# ======================================================================

print("\n" + "-" * 70)
print("ENCODING COURSE LEVEL")
print("-" * 70)


course_level_mapping = {
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2
}


unexpected_course_levels = set(
    data["CourseLevel"].unique()
) - set(
    course_level_mapping.keys()
)


if unexpected_course_levels:

    raise ValueError(
        "Unexpected CourseLevel values found: "
        + str(unexpected_course_levels)
    )


data["CourseLevelEncoded"] = (
    data["CourseLevel"]
    .map(course_level_mapping)
    .astype(int)
)


print("\nCourse level encoding:")

print(
    data[
        [
            "CourseLevel",
            "CourseLevelEncoded"
        ]
    ]
    .drop_duplicates()
    .sort_values("CourseLevelEncoded")
    .to_string(index=False)
)


# ======================================================================
# 13. TEACHER EXPERIENCE BUCKETS
# ======================================================================

print("\n" + "-" * 70)
print("CREATING TEACHER EXPERIENCE BUCKETS")
print("-" * 70)


def create_experience_bucket(experience):

    if experience <= 3:
        return "Low"

    elif experience <= 7:
        return "Medium"

    else:
        return "High"


data["ExperienceBucket"] = (
    data["AvgTeacherExperience"]
    .apply(create_experience_bucket)
)


experience_course_distribution = (
    data[
        [
            "CourseID",
            "ExperienceBucket"
        ]
    ]
    .drop_duplicates()
    ["ExperienceBucket"]
    .value_counts()
    .sort_index()
)


print(
    "\nExperience bucket distribution across unique courses:"
)

print(
    experience_course_distribution
)


# ======================================================================
# 14. TEACHER RATING SCORE
# ======================================================================

print("\n" + "-" * 70)
print("CREATING TEACHER RATING SCORE")
print("-" * 70)


data["TeacherRatingScore"] = (
    data["AvgTeacherRating"]
)


print("\nTeacherRatingScore created.")


# ======================================================================
# 15. TEACHER QUALITY SCORE
# ======================================================================

print("\n" + "-" * 70)
print("CREATING TEACHER QUALITY SCORE")
print("-" * 70)


data["TeacherQualityScore"] = (
    data["AvgTeacherRating"]
    *
    data["AvgTeacherExperience"]
)


print("\nTeacherQualityScore created.")


# ======================================================================
# 16. EXPERTISE-CATEGORY MATCH SCORE
# ======================================================================

print("\n" + "-" * 70)
print("CREATING EXPERTISE-CATEGORY MATCH SCORE")
print("-" * 70)


data["ExpertiseMatchScore"] = (
    data["ExpertiseMatchRate"]
)


print("\nExpertiseMatchScore created.")


# ======================================================================
# 17. REVENUE EFFICIENCY
# ======================================================================

print("\n" + "-" * 70)
print("CREATING REVENUE EFFICIENCY")
print("-" * 70)


data["RevenueEfficiency"] = np.where(
    data["CoursePrice"] > 0,
    data["MonthlyRevenue"] / data["CoursePrice"],
    0
)


print("\nRevenueEfficiency created.")


# ======================================================================
# 18. MONTHLY REVENUE PER ENROLLMENT
# ======================================================================

print("\n" + "-" * 70)
print("CREATING MONTHLY REVENUE PER ENROLLMENT")
print("-" * 70)


data["MonthlyRevenuePerEnrollment"] = np.where(
    data["MonthlyEnrollment"] > 0,
    data["MonthlyRevenue"]
    /
    data["MonthlyEnrollment"],
    0
)


print("\nMonthlyRevenuePerEnrollment created.")


# ======================================================================
# 19. HISTORICAL LAG FEATURES
# ======================================================================

print("\n" + "-" * 70)
print("CREATING HISTORICAL LAG FEATURES")
print("-" * 70)


grouped = data.groupby(
    "CourseID",
    group_keys=False
)


data["PreviousMonthEnrollment"] = (
    grouped["MonthlyEnrollment"]
    .shift(1)
)


data["PreviousMonthRevenue"] = (
    grouped["MonthlyRevenue"]
    .shift(1)
)


data["PreviousMonthTransactionCount"] = (
    grouped["MonthlyTransactionCount"]
    .shift(1)
)


data["PreviousMonthAverageTransactionAmount"] = (
    grouped["AverageTransactionAmount"]
    .shift(1)
)


print("\nHistorical lag features created.")


# ======================================================================
# 20. PREVIOUS 3-MONTH ROLLING FEATURES
# ======================================================================

print("\n" + "-" * 70)
print("CREATING PREVIOUS 3-MONTH ROLLING FEATURES")
print("-" * 70)


data["Previous3MonthEnrollmentSum"] = (
    data
    .groupby("CourseID")["MonthlyEnrollment"]
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


data["Previous3MonthRevenueSum"] = (
    data
    .groupby("CourseID")["MonthlyRevenue"]
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


# Keep explicit aliases for clarity
data["Previous3MonthEnrollment"] = (
    data["Previous3MonthEnrollmentSum"]
)


data["Previous3MonthRevenue"] = (
    data["Previous3MonthRevenueSum"]
)


print("\nPrevious 3-month rolling sums created.")


# ======================================================================
# 21. PREVIOUS 6-MONTH ROLLING FEATURES
# ======================================================================

print("\n" + "-" * 70)
print("CREATING PREVIOUS 6-MONTH ROLLING FEATURES")
print("-" * 70)


data["Previous6MonthEnrollment"] = (
    data
    .groupby("CourseID")["MonthlyEnrollment"]
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


data["Previous6MonthRevenue"] = (
    data
    .groupby("CourseID")["MonthlyRevenue"]
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


print("\nPrevious 6-month rolling features created.")


# ======================================================================
# 22. GROWTH FEATURES
# ======================================================================

print("\n" + "-" * 70)
print("CREATING GROWTH FEATURES")
print("-" * 70)


data["EnrollmentGrowthRate"] = (
    data
    .groupby("CourseID")["MonthlyEnrollment"]
    .pct_change(
        periods=1,
        fill_method=None
    )
)


data["RevenueGrowthRate"] = (
    data
    .groupby("CourseID")["MonthlyRevenue"]
    .pct_change(
        periods=1,
        fill_method=None
    )
)


# Replace infinite growth values caused by zero previous values
data["EnrollmentGrowthRate"] = (
    data["EnrollmentGrowthRate"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


data["RevenueGrowthRate"] = (
    data["RevenueGrowthRate"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


print("\nGrowth features created.")


# ======================================================================
# 23. HISTORICAL AVAILABILITY INDICATOR
# ======================================================================

print("\n" + "-" * 70)
print("CREATING HISTORICAL AVAILABILITY INDICATOR")
print("-" * 70)


data["HasPreviousMonthData"] = (
    data["PreviousMonthEnrollment"]
    .notna()
    .astype(int)
)


print("\nHasPreviousMonthData created.")


# ======================================================================
# 24. HANDLE INITIAL HISTORICAL VALUES
# ======================================================================

print("\n" + "-" * 70)
print("HANDLING INITIAL HISTORICAL VALUES")
print("-" * 70)


historical_columns = [
    "PreviousMonthEnrollment",
    "PreviousMonthRevenue",
    "PreviousMonthTransactionCount",
    "PreviousMonthAverageTransactionAmount",

    "Previous3MonthEnrollment",
    "Previous3MonthRevenue",
    "Previous3MonthEnrollmentSum",
    "Previous3MonthRevenueSum",

    "Previous6MonthEnrollment",
    "Previous6MonthRevenue",

    "EnrollmentGrowthRate",
    "RevenueGrowthRate"
]


for column in historical_columns:

    data[column] = (
        data[column]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


print("\nInitial historical values handled.")


# ======================================================================
# 25. ADDITIONAL TIME FEATURES
# ======================================================================

print("\n" + "-" * 70)
print("CREATING ADDITIONAL TIME FEATURES")
print("-" * 70)


data["MonthSin"] = (
    np.sin(
        2 * np.pi * data["Month"] / 12
    )
)


data["MonthCos"] = (
    np.cos(
        2 * np.pi * data["Month"] / 12
    )
)


data["YearProgress"] = (
    (data["Month"] - 1)
    /
    11
)


print("\nAdditional time features created.")


# ======================================================================
# 26. ONE-HOT ENCODE COURSE CATEGORY
# ======================================================================

print("\n" + "-" * 70)
print("ENCODING COURSE CATEGORY")
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


print("\nCourse category encoded using one-hot encoding.")

print("\nCategory columns created:")

for column in category_dummies.columns:

    print("-", column)


# ======================================================================
# 27. ONE-HOT ENCODE PRICE BAND
# ======================================================================

print("\n" + "-" * 70)
print("ENCODING PRICE BAND")
print("-" * 70)


price_dummies = pd.get_dummies(
    data["PriceBand"],
    prefix="PriceBand",
    dtype=int
)


data = pd.concat(
    [
        data,
        price_dummies
    ],
    axis=1
)


print("\nPriceBand one-hot encoding completed.")


# ======================================================================
# 28. ONE-HOT ENCODE DURATION BUCKET
# ======================================================================

print("\n" + "-" * 70)
print("ENCODING DURATION BUCKET")
print("-" * 70)


duration_dummies = pd.get_dummies(
    data["DurationBucket"],
    prefix="Duration",
    dtype=int
)


data = pd.concat(
    [
        data,
        duration_dummies
    ],
    axis=1
)


print("\nDuration bucket one-hot encoding completed.")


# ======================================================================
# 29. ONE-HOT ENCODE RATING TIER
# ======================================================================

print("\n" + "-" * 70)
print("ENCODING RATING TIER")
print("-" * 70)


rating_dummies = pd.get_dummies(
    data["RatingTier"],
    prefix="Rating",
    dtype=int
)


data = pd.concat(
    [
        data,
        rating_dummies
    ],
    axis=1
)


print("\nRating tier one-hot encoding completed.")


# ======================================================================
# 30. ONE-HOT ENCODE EXPERIENCE BUCKET
# ======================================================================

print("\n" + "-" * 70)
print("ENCODING EXPERIENCE BUCKET")
print("-" * 70)


experience_dummies = pd.get_dummies(
    data["ExperienceBucket"],
    prefix="Experience",
    dtype=int
)


data = pd.concat(
    [
        data,
        experience_dummies
    ],
    axis=1
)


print("\nExperience bucket one-hot encoding completed.")


# ======================================================================
# 31. TARGET VARIABLES
# ======================================================================

print("\n" + "-" * 70)
print("DEFINING PREDICTION TARGETS")
print("-" * 70)


data["EnrollmentTarget"] = (
    data["MonthlyEnrollment"]
)


data["RevenueTarget"] = (
    data["MonthlyRevenue"]
)


print("\nEnrollmentTarget:")
print("MonthlyEnrollment")


print("\nRevenueTarget:")
print("MonthlyRevenue")


# ======================================================================
# 32. TARGET RECONCILIATION
# ======================================================================

print("\n" + "-" * 70)
print("TARGET RECONCILIATION")
print("-" * 70)


enrollment_target_difference = abs(
    data["EnrollmentTarget"].sum()
    -
    data["MonthlyEnrollment"].sum()
)


revenue_target_difference = abs(
    data["RevenueTarget"].sum()
    -
    data["MonthlyRevenue"].sum()
)


print("\nEnrollment target difference:")
print(enrollment_target_difference)


print("\nRevenue target difference:")
print(revenue_target_difference)


if enrollment_target_difference != 0:

    raise ValueError(
        "Enrollment target reconciliation failed."
    )


if revenue_target_difference > 0.000001:

    raise ValueError(
        "Revenue target reconciliation failed."
    )


print("\nTarget reconciliation passed.")


# ======================================================================
# 33. NUMERICAL VALIDATION
# ======================================================================

print("\n" + "-" * 70)
print("FINAL NUMERICAL VALIDATION")
print("-" * 70)


numeric_data = data.select_dtypes(
    include=[np.number]
)


infinite_values = (
    np.isinf(
        numeric_data
    )
    .sum()
    .sum()
)


missing_values = (
    data.isna()
    .sum()
    .sum()
)


print("\nInfinite numeric values:")
print(infinite_values)


print("\nRemaining missing values:")
print(missing_values)


if infinite_values > 0:

    raise ValueError(
        "Infinite numeric values remain."
    )


if missing_values > 0:

    missing_columns = (
        data.isna()
        .sum()
    )

    missing_columns = (
        missing_columns[
            missing_columns > 0
        ]
    )

    print("\nColumns containing missing values:")
    print(missing_columns)

    raise ValueError(
        "Missing values remain in the feature-engineered dataset."
    )


# ======================================================================
# 34. FEATURE ENGINEERING SUMMARY
# ======================================================================

print("\n" + "-" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("-" * 70)


print("\nCourse Features:")

course_features = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "CourseTypeEncoded",
    "CourseLevelEncoded",
    "PriceBand",
    "DurationBucket",
    "RatingTier"
]


for feature in course_features:

    print("  -", feature)


print("\nTeacher Features:")

teacher_features = [
    "TeacherCount",
    "AvgTeacherExperience",
    "AvgTeacherRating",
    "ExperienceBucket",
    "TeacherRatingScore",
    "TeacherQualityScore",
    "ExpertiseMatchRate",
    "ExpertiseMatchScore"
]


for feature in teacher_features:

    print("  -", feature)


print("\nHistorical Features:")

historical_features_summary = [
    "PreviousMonthEnrollment",
    "PreviousMonthRevenue",
    "PreviousMonthTransactionCount",
    "PreviousMonthAverageTransactionAmount",
    "Previous3MonthEnrollment",
    "Previous3MonthRevenue",
    "Previous3MonthEnrollmentSum",
    "Previous3MonthRevenueSum",
    "Previous6MonthEnrollment",
    "Previous6MonthRevenue",
    "EnrollmentGrowthRate",
    "RevenueGrowthRate",
    "HasPreviousMonthData"
]


for feature in historical_features_summary:

    print("  -", feature)


print("\nTime Features:")

time_features = [
    "Year",
    "Month",
    "Quarter",
    "IsQuarterStart",
    "IsQuarterEnd",
    "TimeIndex",
    "MonthSin",
    "MonthCos",
    "YearProgress"
]


for feature in time_features:

    print("  -", feature)


print("\nTarget Variables:")

print("  - EnrollmentTarget")
print("  - RevenueTarget")


# ======================================================================
# 35. FEATURE ENGINEERED DATASET VALIDATION
# ======================================================================

print("\n" + "-" * 70)
print("FEATURE ENGINEERED DATASET VALIDATION")
print("-" * 70)


print("\nNumber of rows:")
print(len(data))


print("\nNumber of columns:")
print(len(data.columns))


print("\nUnique courses:")
print(data["CourseID"].nunique())


print("\nUnique months:")
print(data["YearMonth"].nunique())


print("\nTotal monthly enrollment:")
print(data["MonthlyEnrollment"].sum())


print("\nTotal monthly revenue:")
print(data["MonthlyRevenue"].sum())


expected_rows = (
    unique_courses
    *
    unique_months
)


if len(data) != expected_rows:

    raise ValueError(
        f"Expected {expected_rows} rows but found {len(data)}."
    )


# ======================================================================
# 36. VERIFY HISTORICAL FEATURES DO NOT USE CURRENT/FUTURE DATA
# ======================================================================

print("\n" + "-" * 70)
print("HISTORICAL FEATURE LEAKAGE VALIDATION")
print("-" * 70)


# Verify previous-month enrollment equals the actual prior month
expected_previous_enrollment = (
    data
    .groupby("CourseID")["MonthlyEnrollment"]
    .shift(1)
    .fillna(0)
)


previous_enrollment_difference = (
    (
        data["PreviousMonthEnrollment"]
        -
        expected_previous_enrollment
    )
    .abs()
    .max()
)


print("\nMaximum PreviousMonthEnrollment difference:")
print(previous_enrollment_difference)


if previous_enrollment_difference > 0.000001:

    raise ValueError(
        "PreviousMonthEnrollment does not match historical data."
    )


# Verify previous-month revenue
expected_previous_revenue = (
    data
    .groupby("CourseID")["MonthlyRevenue"]
    .shift(1)
    .fillna(0)
)


previous_revenue_difference = (
    (
        data["PreviousMonthRevenue"]
        -
        expected_previous_revenue
    )
    .abs()
    .max()
)


print("\nMaximum PreviousMonthRevenue difference:")
print(previous_revenue_difference)


if previous_revenue_difference > 0.000001:

    raise ValueError(
        "PreviousMonthRevenue does not match historical data."
    )


print(
    "\nHistorical feature leakage validation passed."
)


# ======================================================================
# 37. REMOVE TEMPORARY DATE COLUMN
# ======================================================================

print("\n" + "-" * 70)
print("REMOVING TEMPORARY COLUMNS")
print("-" * 70)


data = data.drop(
    columns=["Date"]
)


print("\nTemporary date column removed.")


# ======================================================================
# 38. FINAL COLUMN LIST
# ======================================================================

print("\n" + "-" * 70)
print("FINAL FEATURE ENGINEERED DATASET COLUMNS")
print("-" * 70)


for index, column in enumerate(
    data.columns,
    start=1
):

    print(
        f"{index:02d}. {column}"
    )


# ======================================================================
# 39. SAVE FEATURE ENGINEERED DATASET
# ======================================================================

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "feature_engineered_dataset.csv"
)


print("\n" + "-" * 70)
print("SAVING FEATURE ENGINEERED DATASET")
print("-" * 70)


data.to_csv(
    OUTPUT_FILE,
    index=False
)


if not os.path.exists(OUTPUT_FILE):

    raise FileNotFoundError(
        "Feature engineered dataset was not created."
    )


print("\nFeature engineered dataset saved successfully.")


# ======================================================================
# 40. FINAL OUTPUT SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nOutput file:")
print(OUTPUT_FILE)


print("\nFinal shape:")
print(data.shape)


print("\nUnique courses:")
print(data["CourseID"].nunique())


print("\nUnique months:")
print(data["YearMonth"].nunique())


print("\nTotal enrollment:")
print(data["MonthlyEnrollment"].sum())


print("\nTotal revenue:")
print(data["MonthlyRevenue"].sum())


print("\nTotal engineered columns:")
print(len(data.columns))


print("\n" + "=" * 70)
print("STEP 7 COMPLETED")
print("=" * 70)