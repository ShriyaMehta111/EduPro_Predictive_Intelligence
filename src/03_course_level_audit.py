import os
import pandas as pd


# ============================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 3: COURSE-LEVEL BUSINESS AUDIT
# ============================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 3: COURSE-LEVEL BUSINESS AUDIT")
print("=" * 70)


# ============================================================
# 1. PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

DATA_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "raw",
    "EduPro_Dataset.xlsx"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("\n" + "-" * 70)
print("PROJECT PATH")
print("-" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nDataset:")
print(DATA_FILE)

print("\nProcessed data directory:")
print(OUTPUT_DIR)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "-" * 70)
print("LOADING DATA")
print("-" * 70)

users = pd.read_excel(
    DATA_FILE,
    sheet_name="Users"
)

teachers = pd.read_excel(
    DATA_FILE,
    sheet_name="Teachers"
)

courses = pd.read_excel(
    DATA_FILE,
    sheet_name="Courses"
)

transactions = pd.read_excel(
    DATA_FILE,
    sheet_name="Transactions"
)

transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"],
    errors="coerce"
)

print("\nAll datasets loaded successfully.")


# ============================================================
# 3. CREATE COURSE-TRANSACTION DATA
# ============================================================

print("\n" + "-" * 70)
print("MERGING COURSES WITH TRANSACTIONS")
print("-" * 70)

course_transactions = transactions.merge(
    courses,
    on="CourseID",
    how="left",
    validate="many_to_one"
)

print("\nMerged dataset shape:")
print(course_transactions.shape)


# ============================================================
# 4. CREATE TRANSACTION-TEACHER DATA
# ============================================================

print("\n" + "-" * 70)
print("MERGING TRANSACTIONS WITH TEACHERS")
print("-" * 70)

course_transactions = course_transactions.merge(
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

print("\nFinal transaction-level analytical dataset shape:")
print(course_transactions.shape)


# ============================================================
# 5. VERIFY MERGES
# ============================================================

print("\n" + "-" * 70)
print("MERGE VALIDATION")
print("-" * 70)

missing_course_names = (
    course_transactions["CourseName"]
    .isna()
    .sum()
)

missing_teacher_experience = (
    course_transactions["YearsOfExperience"]
    .isna()
    .sum()
)

print("\nRows without Course information:")
print(missing_course_names)

print("\nRows without Teacher information:")
print(missing_teacher_experience)


# ============================================================
# 6. ENROLLMENT COUNT
# ============================================================

print("\n" + "-" * 70)
print("ENROLLMENT CALCULATION")
print("-" * 70)

enrollment_data = (
    transactions
    .groupby("CourseID")["UserID"]
    .nunique()
    .reset_index(
        name="EnrollmentCount"
    )
)

print("\nTotal courses with enrollment:")
print(len(enrollment_data))

print("\nTotal enrollments:")
print(
    enrollment_data["EnrollmentCount"].sum()
)


# ============================================================
# 7. REVENUE METRICS
# ============================================================

print("\n" + "-" * 70)
print("REVENUE CALCULATION")
print("-" * 70)

revenue_data = (
    transactions
    .groupby("CourseID")["Amount"]
    .agg(
        TotalRevenue="sum",
        AvgTransactionAmount="mean",
        TransactionCount="count"
    )
    .reset_index()
)

print("\nTotal revenue:")
print(
    revenue_data["TotalRevenue"].sum()
)


# ============================================================
# 8. TEACHER FEATURES
# ============================================================

print("\n" + "-" * 70)
print("TEACHER FEATURE CALCULATION")
print("-" * 70)

teacher_features = (
    course_transactions
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
        )
    )
    .reset_index()
)


# ============================================================
# 9. EXPERTISE MATCH
# ============================================================

print("\n" + "-" * 70)
print("EXPERTISE MATCH ANALYSIS")
print("-" * 70)

course_transactions["ExpertiseMatch"] = (
    course_transactions["CourseCategory"]
    ==
    course_transactions["Expertise"]
)

expertise_match_data = (
    course_transactions
    .groupby("CourseID")["ExpertiseMatch"]
    .mean()
    .reset_index(
        name="ExpertiseMatchRate"
    )
)

print("\nOverall expertise match rate:")
print(
    course_transactions["ExpertiseMatch"].mean()
)


# ============================================================
# 10. MERGE ALL COURSE-LEVEL INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("CREATING COURSE-LEVEL AUDIT DATASET")
print("-" * 70)

course_audit = courses.copy()

course_audit = course_audit.merge(
    enrollment_data,
    on="CourseID",
    how="left",
    validate="one_to_one"
)

course_audit = course_audit.merge(
    revenue_data,
    on="CourseID",
    how="left",
    validate="one_to_one"
)

course_audit = course_audit.merge(
    teacher_features,
    on="CourseID",
    how="left",
    validate="one_to_one"
)

course_audit = course_audit.merge(
    expertise_match_data,
    on="CourseID",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 11. FILL COURSE-LEVEL DEFAULTS
# ============================================================

course_audit["EnrollmentCount"] = (
    course_audit["EnrollmentCount"]
    .fillna(0)
)

course_audit["TotalRevenue"] = (
    course_audit["TotalRevenue"]
    .fillna(0)
)

course_audit["AvgTransactionAmount"] = (
    course_audit["AvgTransactionAmount"]
    .fillna(0)
)

course_audit["TransactionCount"] = (
    course_audit["TransactionCount"]
    .fillna(0)
)

course_audit["TeacherCount"] = (
    course_audit["TeacherCount"]
    .fillna(0)
)

course_audit["ExpertiseMatchRate"] = (
    course_audit["ExpertiseMatchRate"]
    .fillna(0)
)


# ============================================================
# 12. REVENUE PER ENROLLMENT
# ============================================================

course_audit["RevenuePerEnrollment"] = (
    course_audit["TotalRevenue"]
    /
    course_audit["EnrollmentCount"].replace(0, pd.NA)
)

course_audit["RevenuePerEnrollment"] = (
    course_audit["RevenuePerEnrollment"]
    .fillna(0)
)


# ============================================================
# 13. REVENUE CONSISTENCY CHECK
# ============================================================

print("\n" + "-" * 70)
print("REVENUE CONSISTENCY CHECK")
print("-" * 70)

course_audit["ExpectedRevenue"] = (
    course_audit["EnrollmentCount"]
    *
    course_audit["CoursePrice"]
)

course_audit["RevenueDifference"] = (
    course_audit["TotalRevenue"]
    -
    course_audit["ExpectedRevenue"]
)

print("\nMaximum absolute revenue difference:")
print(
    course_audit["RevenueDifference"]
    .abs()
    .max()
)


# ============================================================
# 14. COURSE SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("COURSE-LEVEL AUDIT SUMMARY")
print("-" * 70)

print("\nNumber of courses:")
print(len(course_audit))

print("\nTotal enrollments:")
print(
    course_audit["EnrollmentCount"].sum()
)

print("\nTotal revenue:")
print(
    course_audit["TotalRevenue"].sum()
)

print("\nAverage enrollment per course:")
print(
    course_audit["EnrollmentCount"].mean()
)

print("\nAverage revenue per course:")
print(
    course_audit["TotalRevenue"].mean()
)


# ============================================================
# 15. TOP COURSES BY ENROLLMENT
# ============================================================

print("\n" + "-" * 70)
print("TOP 10 COURSES BY ENROLLMENT")
print("-" * 70)

top_enrollment = (
    course_audit[
        [
            "CourseID",
            "CourseName",
            "CourseCategory",
            "CourseType",
            "EnrollmentCount"
        ]
    ]
    .sort_values(
        "EnrollmentCount",
        ascending=False
    )
    .head(10)
)

print(top_enrollment.to_string(index=False))


# ============================================================
# 16. TOP COURSES BY REVENUE
# ============================================================

print("\n" + "-" * 70)
print("TOP 10 COURSES BY REVENUE")
print("-" * 70)

top_revenue = (
    course_audit[
        [
            "CourseID",
            "CourseName",
            "CourseCategory",
            "CourseType",
            "TotalRevenue"
        ]
    ]
    .sort_values(
        "TotalRevenue",
        ascending=False
    )
    .head(10)
)

print(top_revenue.to_string(index=False))


# ============================================================
# 17. TOP COURSES BY REVENUE PER ENROLLMENT
# ============================================================

print("\n" + "-" * 70)
print("TOP 10 COURSES BY REVENUE PER ENROLLMENT")
print("-" * 70)

top_revenue_efficiency = (
    course_audit[
        [
            "CourseID",
            "CourseName",
            "CourseType",
            "EnrollmentCount",
            "TotalRevenue",
            "RevenuePerEnrollment"
        ]
    ]
    .sort_values(
        "RevenuePerEnrollment",
        ascending=False
    )
    .head(10)
)

print(
    top_revenue_efficiency.to_string(
        index=False
    )
)


# ============================================================
# 18. CATEGORY PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("CATEGORY PERFORMANCE")
print("-" * 70)

category_summary = (
    course_audit
    .groupby("CourseCategory")
    .agg(
        CourseCount=(
            "CourseID",
            "nunique"
        ),

        TotalEnrollment=(
            "EnrollmentCount",
            "sum"
        ),

        TotalRevenue=(
            "TotalRevenue",
            "sum"
        ),

        AverageEnrollment=(
            "EnrollmentCount",
            "mean"
        ),

        AverageRevenue=(
            "TotalRevenue",
            "mean"
        )
    )
    .sort_values(
        "TotalRevenue",
        ascending=False
    )
)

print(
    category_summary.to_string()
)


# ============================================================
# 19. FREE VS PAID PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("FREE VS PAID COURSE PERFORMANCE")
print("-" * 70)

type_summary = (
    course_audit
    .groupby("CourseType")
    .agg(
        CourseCount=(
            "CourseID",
            "nunique"
        ),

        TotalEnrollment=(
            "EnrollmentCount",
            "sum"
        ),

        TotalRevenue=(
            "TotalRevenue",
            "sum"
        ),

        AverageEnrollment=(
            "EnrollmentCount",
            "mean"
        ),

        AverageRevenue=(
            "TotalRevenue",
            "mean"
        )
    )
)

print(
    type_summary.to_string()
)


# ============================================================
# 20. COURSE LEVEL PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("COURSE LEVEL PERFORMANCE")
print("-" * 70)

level_summary = (
    course_audit
    .groupby("CourseLevel")
    .agg(
        CourseCount=(
            "CourseID",
            "nunique"
        ),

        TotalEnrollment=(
            "EnrollmentCount",
            "sum"
        ),

        TotalRevenue=(
            "TotalRevenue",
            "sum"
        ),

        AverageEnrollment=(
            "EnrollmentCount",
            "mean"
        ),

        AverageRevenue=(
            "TotalRevenue",
            "mean"
        )
    )
)

print(
    level_summary.to_string()
)


# ============================================================
# 21. SAVE COURSE AUDIT
# ============================================================

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "course_level_audit.csv"
)

course_audit.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 22. DISPLAY FINAL COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("FINAL COURSE AUDIT COLUMNS")
print("-" * 70)

print(
    course_audit.columns.tolist()
)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("COURSE-LEVEL BUSINESS AUDIT COMPLETED")
print("=" * 70)

print("\nOutput file created:")
print(OUTPUT_FILE)