import os
import pandas as pd


# ============================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 5A: COURSE-LEVEL MASTER DATASET
# ============================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 5A: COURSE-LEVEL MASTER DATASET")
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
# 2. CHECK REQUIRED FILES
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
# 4. BASIC DATA VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("BASIC DATA VALIDATION")
print("-" * 70)


required_course_columns = [
    "CourseID",
    "CourseName",
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
    "UserID",
    "CourseID",
    "TransactionDate",
    "Amount",
    "PaymentMethod",
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
# 5. CHECK UNIQUE COURSE IDs
# ============================================================

print("\n" + "-" * 70)
print("COURSE ID VALIDATION")
print("-" * 70)


course_id_duplicates = (
    courses["CourseID"].duplicated().sum()
)

print("\nDuplicate CourseIDs in Courses:")
print(course_id_duplicates)


if course_id_duplicates > 0:

    raise ValueError(
        "Courses dataset contains duplicate CourseIDs."
    )


# ============================================================
# 6. CHECK UNIQUE TEACHER IDs
# ============================================================

print("\n" + "-" * 70)
print("TEACHER ID VALIDATION")
print("-" * 70)


teacher_id_duplicates = (
    teachers["TeacherID"].duplicated().sum()
)

print("\nDuplicate TeacherIDs in Teachers:")
print(teacher_id_duplicates)


if teacher_id_duplicates > 0:

    raise ValueError(
        "Teachers dataset contains duplicate TeacherIDs."
    )


# ============================================================
# 7. COURSE PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING COURSE PERFORMANCE")
print("-" * 70)


course_performance = (
    transactions
    .groupby("CourseID")
    .agg(
        EnrollmentCount=(
            "UserID",
            "nunique"
        ),

        TotalRevenue=(
            "Amount",
            "sum"
        ),

        AvgTransactionAmount=(
            "Amount",
            "mean"
        ),

        TransactionCount=(
            "TransactionID",
            "count"
        )
    )
    .reset_index()
)


print("\nCourse performance calculated.")

print("\nNumber of courses with transactions:")
print(
    course_performance["CourseID"].nunique()
)


# ============================================================
# 8. TRANSACTION + COURSE + TEACHER DATA
# ============================================================

print("\n" + "-" * 70)
print("CREATING COURSE-TEACHER ANALYTICAL DATA")
print("-" * 70)


# ------------------------------------------------------------
# FIRST MERGE:
# TRANSACTIONS + COURSES
# ------------------------------------------------------------

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


print("\nAfter Transactions + Courses merge:")
print(course_teacher_data.shape)


# ------------------------------------------------------------
# SECOND MERGE:
# RESULT + TEACHERS
# ------------------------------------------------------------

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


print("\nAfter adding Teacher information:")
print(course_teacher_data.shape)


# ============================================================
# 9. MERGE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("MERGE VALIDATION")
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
        "Merge created missing course or teacher information."
    )


print("\nAll course and teacher information matched successfully.")


# ============================================================
# 10. EXPERTISE ALIGNMENT
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING EXPERTISE ALIGNMENT")
print("-" * 70)


course_teacher_data["ExpertiseMatch"] = (
    course_teacher_data["CourseCategory"]
    ==
    course_teacher_data["Expertise"]
)


overall_expertise_match_rate = (
    course_teacher_data["ExpertiseMatch"]
    .mean()
)


print("\nOverall expertise match rate:")
print(
    round(
        overall_expertise_match_rate * 100,
        2
    ),
    "%"
)


# ============================================================
# 11. COURSE-LEVEL EXPERTISE MATCH RATE
# ============================================================

expertise_features = (
    course_teacher_data
    .groupby("CourseID")["ExpertiseMatch"]
    .mean()
    .reset_index(
        name="ExpertiseMatchRate"
    )
)


print("\nCourse-level expertise match rates calculated.")


# ============================================================
# 12. TEACHER FEATURES
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING TEACHER FEATURES")
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
        )
    )
    .reset_index()
)


print("\nTeacher features calculated.")


# ============================================================
# 13. CREATE COURSE MASTER DATASET
# ============================================================

print("\n" + "-" * 70)
print("CREATING COURSE MASTER DATASET")
print("-" * 70)


course_master = courses.merge(
    course_performance,
    on="CourseID",
    how="left",
    validate="one_to_one"
)


course_master = course_master.merge(
    teacher_features,
    on="CourseID",
    how="left",
    validate="one_to_one"
)


course_master = course_master.merge(
    expertise_features,
    on="CourseID",
    how="left",
    validate="one_to_one"
)


print("\nCourse master dataset created.")


# ============================================================
# 14. FILL MISSING AGGREGATED VALUES
# ============================================================

print("\n" + "-" * 70)
print("HANDLING MISSING AGGREGATED VALUES")
print("-" * 70)


performance_columns = [
    "EnrollmentCount",
    "TotalRevenue",
    "AvgTransactionAmount",
    "TransactionCount",
    "TeacherCount",
    "ExpertiseMatchRate"
]


for column in performance_columns:

    course_master[column] = (
        course_master[column]
        .fillna(0)
    )


print("\nAggregated missing values handled.")


# ============================================================
# 15. REVENUE PER ENROLLMENT
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING REVENUE PER ENROLLMENT")
print("-" * 70)


course_master["RevenuePerEnrollment"] = (
    course_master["TotalRevenue"]
    /
    course_master["EnrollmentCount"]
    .replace(0, pd.NA)
)


course_master["RevenuePerEnrollment"] = (
    course_master["RevenuePerEnrollment"]
    .fillna(0)
)


print("\nRevenue per enrollment calculated.")


# ============================================================
# 16. REVENUE CONSISTENCY CHECK
# ============================================================

print("\n" + "-" * 70)
print("REVENUE CONSISTENCY CHECK")
print("-" * 70)


course_master["ExpectedRevenue"] = (
    course_master["EnrollmentCount"]
    *
    course_master["CoursePrice"]
)


course_master["RevenueDifference"] = (
    course_master["TotalRevenue"]
    -
    course_master["ExpectedRevenue"]
)


maximum_revenue_difference = (
    course_master["RevenueDifference"]
    .abs()
    .max()
)


print("\nMaximum absolute revenue difference:")
print(maximum_revenue_difference)


if maximum_revenue_difference != 0:

    print(
        "\nWARNING: Revenue consistency check did not produce zero."
    )

else:

    print(
        "\nRevenue consistency check passed."
    )


# ============================================================
# 17. MASTER DATASET VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("MASTER DATASET VALIDATION")
print("-" * 70)


unique_courses = (
    course_master["CourseID"]
    .nunique()
)


duplicate_course_ids = (
    course_master["CourseID"]
    .duplicated()
    .sum()
)


missing_values = (
    course_master.isna()
    .sum()
    .sum()
)


print("\nNumber of courses:")
print(unique_courses)

print("\nDuplicate CourseIDs:")
print(duplicate_course_ids)

print("\nTotal missing values:")
print(missing_values)


if unique_courses != len(courses):

    raise ValueError(
        "Course count changed unexpectedly during merging."
    )


if duplicate_course_ids > 0:

    raise ValueError(
        "Duplicate CourseIDs found in course master dataset."
    )


# ============================================================
# 18. COURSE MASTER SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("COURSE MASTER SUMMARY")
print("-" * 70)


print("\nTotal enrollment:")
print(
    course_master["EnrollmentCount"].sum()
)


print("\nTotal revenue:")
print(
    course_master["TotalRevenue"].sum()
)


print("\nAverage enrollment per course:")
print(
    course_master["EnrollmentCount"].mean()
)


print("\nAverage revenue per course:")
print(
    course_master["TotalRevenue"].mean()
)


print("\nAverage teacher count:")
print(
    course_master["TeacherCount"].mean()
)


print("\nAverage teacher experience:")
print(
    course_master["AvgTeacherExperience"].mean()
)


print("\nAverage teacher rating:")
print(
    course_master["AvgTeacherRating"].mean()
)


print("\nAverage expertise match rate:")
print(
    course_master["ExpertiseMatchRate"].mean()
)


# ============================================================
# 19. TOP COURSES BY ENROLLMENT
# ============================================================

print("\n" + "-" * 70)
print("TOP 10 COURSES BY ENROLLMENT")
print("-" * 70)


top_enrollment = (
    course_master[
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


print(
    top_enrollment.to_string(
        index=False
    )
)


# ============================================================
# 20. TOP COURSES BY REVENUE
# ============================================================

print("\n" + "-" * 70)
print("TOP 10 COURSES BY REVENUE")
print("-" * 70)


top_revenue = (
    course_master[
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


print(
    top_revenue.to_string(
        index=False
    )
)


# ============================================================
# 21. TOP COURSES BY REVENUE PER ENROLLMENT
# ============================================================

print("\n" + "-" * 70)
print("TOP 10 COURSES BY REVENUE PER ENROLLMENT")
print("-" * 70)


top_revenue_efficiency = (
    course_master[
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
# 22. CATEGORY PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("CATEGORY PERFORMANCE")
print("-" * 70)


category_summary = (
    course_master
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
# 23. FREE VS PAID PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("FREE VS PAID COURSE PERFORMANCE")
print("-" * 70)


type_summary = (
    course_master
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
# 24. COURSE LEVEL PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("COURSE LEVEL PERFORMANCE")
print("-" * 70)


level_summary = (
    course_master
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
# 25. FINAL COLUMN LIST
# ============================================================

print("\n" + "-" * 70)
print("FINAL COURSE MASTER COLUMNS")
print("-" * 70)


for column in course_master.columns:

    print("-", column)


# ============================================================
# 26. SAVE DATASET
# ============================================================

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "course_master_dataset.csv"
)


course_master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 27. FINAL CHECK
# ============================================================

if not os.path.exists(OUTPUT_FILE):

    raise FileNotFoundError(
        "Course master dataset was not created."
    )


print("\n" + "=" * 70)
print("COURSE MASTER DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFinal shape:")
print(course_master.shape)

print("\nFinal dataset contains:")
print(
    course_master["CourseID"].nunique(),
    "unique courses"
)

print("\n" + "=" * 70)
print("STEP 5A COMPLETED")
print("=" * 70)