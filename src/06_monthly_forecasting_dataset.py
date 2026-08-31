import os
import pandas as pd


# ============================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 5B: MONTHLY FORECASTING DATASET
# ============================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 5B: MONTHLY FORECASTING DATASET")
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


print("\n" + "-" * 70)
print("PROJECT PATH")
print("-" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nProcessed data directory:")
print(PROCESSED_DIR)


# ============================================================
# 2. REQUIRED FILES
# ============================================================

print("\n" + "-" * 70)
print("CHECKING REQUIRED FILES")
print("-" * 70)


required_files = [
    "cleaned_courses.csv",
    "cleaned_teachers.csv",
    "cleaned_transactions.csv"
]


for file_name in required_files:

    file_path = os.path.join(
        PROCESSED_DIR,
        file_name
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired file was not found:\n{file_path}"
        )

    print(f"\nFound: {file_name}")


# ============================================================
# 3. LOAD CLEANED DATA
# ============================================================

print("\n" + "-" * 70)
print("LOADING CLEANED DATA")
print("-" * 70)


courses = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "cleaned_courses.csv"
    )
)


teachers = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "cleaned_teachers.csv"
    )
)


transactions = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "cleaned_transactions.csv"
    ),
    parse_dates=["TransactionDate"]
)


print("\nCourses:")
print(courses.shape)

print("\nTeachers:")
print(teachers.shape)

print("\nTransactions:")
print(transactions.shape)


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING REQUIRED COLUMNS")
print("-" * 70)


required_course_columns = [
    "CourseID",
    "CourseCategory",
    "CourseType",
    "CourseLevel",
    "CoursePrice",
    "CourseDuration",
    "CourseRating"
]


required_teacher_columns = [
    "TeacherID",
    "Expertise",
    "YearsOfExperience",
    "TeacherRating"
]


required_transaction_columns = [
    "TransactionID",
    "CourseID",
    "TransactionDate",
    "Amount",
    "TeacherID"
]


for column in required_course_columns:

    if column not in courses.columns:

        raise KeyError(
            f"Missing column in Courses dataset: {column}"
        )


for column in required_teacher_columns:

    if column not in teachers.columns:

        raise KeyError(
            f"Missing column in Teachers dataset: {column}"
        )


for column in required_transaction_columns:

    if column not in transactions.columns:

        raise KeyError(
            f"Missing column in Transactions dataset: {column}"
        )


print("\nAll required columns are present.")


# ============================================================
# 5. CREATE COURSE-TEACHER INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("CREATING COURSE-TEACHER INFORMATION")
print("-" * 70)


course_teacher_data = transactions.merge(
    courses[
        [
            "CourseID",
            "CourseCategory"
        ]
    ],
    on="CourseID",
    how="left",
    validate="many_to_one"
)


course_teacher_data = course_teacher_data.merge(
    teachers[
        [
            "TeacherID",
            "Expertise",
            "YearsOfExperience",
            "TeacherRating"
        ]
    ],
    on="TeacherID",
    how="left",
    validate="many_to_one"
)


print("\nCourse-teacher transaction dataset:")
print(course_teacher_data.shape)


# ============================================================
# 6. VALIDATE COURSE-TEACHER MERGE
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING COURSE-TEACHER MERGE")
print("-" * 70)


missing_course_category = (
    course_teacher_data["CourseCategory"]
    .isna()
    .sum()
)


missing_teacher_expertise = (
    course_teacher_data["Expertise"]
    .isna()
    .sum()
)


missing_teacher_experience = (
    course_teacher_data["YearsOfExperience"]
    .isna()
    .sum()
)


missing_teacher_rating = (
    course_teacher_data["TeacherRating"]
    .isna()
    .sum()
)


print("\nMissing CourseCategory:")
print(missing_course_category)

print("\nMissing Teacher Expertise:")
print(missing_teacher_expertise)

print("\nMissing Teacher Experience:")
print(missing_teacher_experience)

print("\nMissing Teacher Rating:")
print(missing_teacher_rating)


if (
    missing_course_category > 0
    or
    missing_teacher_expertise > 0
    or
    missing_teacher_experience > 0
    or
    missing_teacher_rating > 0
):

    raise ValueError(
        "Course-teacher merge contains missing information."
    )


print("\nCourse-teacher merge validated successfully.")


# ============================================================
# 7. CALCULATE EXPERTISE MATCH
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING EXPERTISE MATCH")
print("-" * 70)


course_teacher_data["ExpertiseMatch"] = (
    course_teacher_data["CourseCategory"]
    ==
    course_teacher_data["Expertise"]
)


# ============================================================
# 8. CALCULATE COURSE-LEVEL TEACHER FEATURES
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING COURSE-LEVEL TEACHER FEATURES")
print("-" * 70)


teacher_features = (
    course_teacher_data
    .groupby("CourseID")
    .agg(
        TeacherCount=(
            "TeacherID",
            "nunique"
        ),

        AvgTeacherExperience=(
            "YearsOfExperience",
            "mean"
        ),

        AvgTeacherRating=(
            "TeacherRating",
            "mean"
        ),

        ExpertiseMatchRate=(
            "ExpertiseMatch",
            "mean"
        )
    )
    .reset_index()
)


print("\nTeacher features calculated.")

print("\nNumber of courses:")
print(
    teacher_features["CourseID"].nunique()
)


# ============================================================
# 9. CREATE MONTH COLUMN
# ============================================================

print("\n" + "-" * 70)
print("CREATING MONTHLY TIME PERIOD")
print("-" * 70)


transactions["Year"] = (
    transactions["TransactionDate"]
    .dt.year
)


transactions["Month"] = (
    transactions["TransactionDate"]
    .dt.month
)


transactions["MonthName"] = (
    transactions["TransactionDate"]
    .dt.strftime("%B")
)


transactions["YearMonth"] = (
    transactions["TransactionDate"]
    .dt.to_period("M")
    .astype(str)
)


print("\nEarliest month:")
print(
    transactions["YearMonth"].min()
)


print("\nLatest month:")
print(
    transactions["YearMonth"].max()
)


print("\nNumber of unique months:")
print(
    transactions["YearMonth"].nunique()
)


# ============================================================
# 10. AGGREGATE MONTHLY COURSE PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("AGGREGATING MONTHLY COURSE PERFORMANCE")
print("-" * 70)


monthly_performance = (
    transactions
    .groupby(
        [
            "CourseID",
            "Year",
            "Month",
            "MonthName",
            "YearMonth"
        ]
    )
    .agg(
        MonthlyEnrollment=(
            "UserID",
            "nunique"
        ),

        MonthlyRevenue=(
            "Amount",
            "sum"
        ),

        MonthlyTransactionCount=(
            "TransactionID",
            "count"
        ),

        AverageTransactionAmount=(
            "Amount",
            "mean"
        )
    )
    .reset_index()
)


print("\nMonthly performance shape:")
print(
    monthly_performance.shape
)


# ============================================================
# 11. CREATE COMPLETE COURSE-MONTH GRID
# ============================================================

print("\n" + "-" * 70)
print("CREATING COMPLETE COURSE-MONTH GRID")
print("-" * 70)


unique_courses = (
    courses["CourseID"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


unique_periods = (
    transactions[
        [
            "Year",
            "Month",
            "MonthName",
            "YearMonth"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["Year", "Month"]
    )
)


course_period_grid = pd.MultiIndex.from_product(
    [
        unique_courses,
        unique_periods.itertuples(
            index=False,
            name=None
        )
    ],
    names=[
        "CourseID",
        "Period"
    ]
)


course_period_grid = (
    course_period_grid
    .to_frame(index=False)
)


period_columns = [
    "Year",
    "Month",
    "MonthName",
    "YearMonth"
]


course_period_grid[
    period_columns
] = pd.DataFrame(
    course_period_grid["Period"].tolist(),
    index=course_period_grid.index
)


course_period_grid = (
    course_period_grid
    .drop(columns=["Period"])
)


print("\nComplete course-month grid shape:")
print(
    course_period_grid.shape
)


# ============================================================
# 12. MERGE MONTHLY PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("MERGING MONTHLY PERFORMANCE WITH COMPLETE GRID")
print("-" * 70)


monthly_data = course_period_grid.merge(
    monthly_performance,
    on=[
        "CourseID",
        "Year",
        "Month",
        "MonthName",
        "YearMonth"
    ],
    how="left",
    validate="one_to_one"
)


# ============================================================
# 13. FILL MONTHS WITH NO TRANSACTIONS
# ============================================================

print("\n" + "-" * 70)
print("HANDLING MONTHS WITHOUT TRANSACTIONS")
print("-" * 70)


monthly_numeric_columns = [
    "MonthlyEnrollment",
    "MonthlyRevenue",
    "MonthlyTransactionCount",
    "AverageTransactionAmount"
]


for column in monthly_numeric_columns:

    monthly_data[column] = (
        monthly_data[column]
        .fillna(0)
    )


print("\nMissing monthly performance values replaced with 0.")


# ============================================================
# 14. ADD COURSE INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("ADDING COURSE INFORMATION")
print("-" * 70)


course_information = courses[
    [
        "CourseID",
        "CourseName",
        "CourseCategory",
        "CourseType",
        "CourseLevel",
        "CoursePrice",
        "CourseDuration",
        "CourseRating"
    ]
]


monthly_data = monthly_data.merge(
    course_information,
    on="CourseID",
    how="left",
    validate="many_to_one"
)


# ============================================================
# 15. ADD TEACHER INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("ADDING TEACHER INFORMATION")
print("-" * 70)


monthly_data = monthly_data.merge(
    teacher_features,
    on="CourseID",
    how="left",
    validate="many_to_one"
)


# ============================================================
# 16. VALIDATE FINAL MERGES
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING FINAL MONTHLY DATASET")
print("-" * 70)


print("\nFinal monthly dataset shape:")
print(
    monthly_data.shape
)


missing_values = (
    monthly_data.isna()
    .sum()
)


print("\nMissing values by column:")

print(
    missing_values[
        missing_values > 0
    ]
)


if missing_values.sum() > 0:

    raise ValueError(
        "Final monthly dataset contains missing values."
    )


# ============================================================
# 17. SORT DATA CHRONOLOGICALLY
# ============================================================

print("\n" + "-" * 70)
print("SORTING DATA CHRONOLOGICALLY")
print("-" * 70)


monthly_data = (
    monthly_data
    .sort_values(
        [
            "CourseID",
            "Year",
            "Month"
        ]
    )
    .reset_index(drop=True)
)


print("\nDataset sorted by CourseID and date.")


# ============================================================
# 18. CREATE TIME FEATURES
# ============================================================

print("\n" + "-" * 70)
print("CREATING TIME FEATURES")
print("-" * 70)


monthly_data["Quarter"] = (
    ((monthly_data["Month"] - 1) // 3) + 1
)


monthly_data["IsQuarterStart"] = (
    monthly_data["Month"]
    .isin([1, 4, 7, 10])
    .astype(int)
)


monthly_data["IsQuarterEnd"] = (
    monthly_data["Month"]
    .isin([3, 6, 9, 12])
    .astype(int)
)


monthly_data["TimeIndex"] = (
    monthly_data["Year"] * 12
    +
    monthly_data["Month"]
)


print("\nTime features created.")


# ============================================================
# 19. DATASET VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("MONTHLY DATASET VALIDATION")
print("-" * 70)


expected_rows = (
    len(unique_courses)
    *
    len(unique_periods)
)


actual_rows = len(monthly_data)


print("\nExpected rows:")
print(expected_rows)


print("\nActual rows:")
print(actual_rows)


if actual_rows != expected_rows:

    raise ValueError(
        "Monthly dataset does not contain the expected "
        "course-month combinations."
    )


duplicate_course_month = (
    monthly_data
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


print("\nUnique courses:")
print(
    monthly_data["CourseID"].nunique()
)


print("\nUnique months:")
print(
    monthly_data["YearMonth"].nunique()
)


# ============================================================
# 20. RECONCILE TOTAL ENROLLMENTS
# ============================================================

print("\n" + "-" * 70)
print("ENROLLMENT RECONCILIATION")
print("-" * 70)


raw_total_enrollment = (
    transactions["UserID"]
    .count()
)


monthly_total_enrollment = (
    monthly_data["MonthlyEnrollment"]
    .sum()
)


print("\nRaw transaction enrollment count:")
print(raw_total_enrollment)


print("\nMonthly enrollment total:")
print(monthly_total_enrollment)


if raw_total_enrollment != monthly_total_enrollment:

    raise ValueError(
        "Monthly enrollment total does not match "
        "the raw transaction count."
    )


print("\nEnrollment reconciliation passed.")


# ============================================================
# 21. RECONCILE TOTAL REVENUE
# ============================================================

print("\n" + "-" * 70)
print("REVENUE RECONCILIATION")
print("-" * 70)


raw_total_revenue = (
    transactions["Amount"]
    .sum()
)


monthly_total_revenue = (
    monthly_data["MonthlyRevenue"]
    .sum()
)


revenue_difference = abs(
    raw_total_revenue
    -
    monthly_total_revenue
)


print("\nRaw transaction revenue:")
print(raw_total_revenue)


print("\nMonthly revenue:")
print(monthly_total_revenue)


print("\nRevenue difference:")
print(revenue_difference)


if revenue_difference > 0.000001:

    raise ValueError(
        "Monthly revenue does not match "
        "raw transaction revenue."
    )


print("\nRevenue reconciliation passed.")


# ============================================================
# 22. MONTHLY SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("MONTHLY SUMMARY")
print("-" * 70)


monthly_summary = (
    monthly_data
    .groupby(
        [
            "Year",
            "Month",
            "MonthName",
            "YearMonth"
        ]
    )
    .agg(
        TotalEnrollment=(
            "MonthlyEnrollment",
            "sum"
        ),

        TotalRevenue=(
            "MonthlyRevenue",
            "sum"
        ),

        ActiveCourses=(
            "CourseID",
            lambda x: (
                monthly_data.loc[
                    x.index,
                    "MonthlyEnrollment"
                ]
                .gt(0)
                .sum()
            )
        )
    )
    .reset_index()
)


print(
    monthly_summary.to_string(
        index=False
    )
)


# ============================================================
# 23. FINAL COLUMN LIST
# ============================================================

print("\n" + "-" * 70)
print("FINAL MONTHLY DATASET COLUMNS")
print("-" * 70)


for column in monthly_data.columns:

    print("-", column)


# ============================================================
# 24. SAVE DATASET
# ============================================================

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "monthly_forecasting_dataset.csv"
)


monthly_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 25. FINAL OUTPUT CHECK
# ============================================================

if not os.path.exists(OUTPUT_FILE):

    raise FileNotFoundError(
        "Monthly forecasting dataset was not created."
    )


print("\n" + "=" * 70)
print("MONTHLY FORECASTING DATASET CREATED SUCCESSFULLY")
print("=" * 70)


print("\nOutput file:")
print(OUTPUT_FILE)


print("\nFinal shape:")
print(
    monthly_data.shape
)


print("\nUnique courses:")
print(
    monthly_data["CourseID"].nunique()
)


print("\nUnique months:")
print(
    monthly_data["YearMonth"].nunique()
)


print("\nTotal enrollment:")
print(
    monthly_data["MonthlyEnrollment"].sum()
)


print("\nTotal revenue:")
print(
    monthly_data["MonthlyRevenue"].sum()
)


print("\n" + "=" * 70)
print("STEP 5B COMPLETED")
print("=" * 70)