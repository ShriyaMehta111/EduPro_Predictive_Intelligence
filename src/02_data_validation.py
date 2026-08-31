import os
import pandas as pd


# ============================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 2: DATA VALIDATION
# ============================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 2: DATA VALIDATION")
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


# ============================================================
# 2. LOAD DATA
# ============================================================

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


# ============================================================
# 3. USER ID VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("USER ID VALIDATION")
print("-" * 70)

valid_user_ids = set(users["UserID"])

transaction_user_ids = set(
    transactions["UserID"]
)

invalid_user_ids = (
    transaction_user_ids - valid_user_ids
)

print("\nUnique UserIDs in Users:")
print(users["UserID"].nunique())

print("\nUnique UserIDs in Transactions:")
print(transactions["UserID"].nunique())

print("\nTransaction UserIDs not found in Users:")
print(len(invalid_user_ids))

if invalid_user_ids:
    print(invalid_user_ids)


# ============================================================
# 4. TEACHER ID VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("TEACHER ID VALIDATION")
print("-" * 70)

valid_teacher_ids = set(
    teachers["TeacherID"]
)

transaction_teacher_ids = set(
    transactions["TeacherID"]
)

invalid_teacher_ids = (
    transaction_teacher_ids - valid_teacher_ids
)

print("\nUnique TeacherIDs in Teachers:")
print(teachers["TeacherID"].nunique())

print("\nUnique TeacherIDs in Transactions:")
print(transactions["TeacherID"].nunique())

print("\nTransaction TeacherIDs not found in Teachers:")
print(len(invalid_teacher_ids))

if invalid_teacher_ids:
    print(invalid_teacher_ids)


# ============================================================
# 5. COURSE ID VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("COURSE ID VALIDATION")
print("-" * 70)

valid_course_ids = set(
    courses["CourseID"]
)

transaction_course_ids = set(
    transactions["CourseID"]
)

invalid_course_ids = (
    transaction_course_ids - valid_course_ids
)

print("\nUnique CourseIDs in Courses:")
print(courses["CourseID"].nunique())

print("\nUnique CourseIDs in Transactions:")
print(transactions["CourseID"].nunique())

print("\nTransaction CourseIDs not found in Courses:")
print(len(invalid_course_ids))

if invalid_course_ids:
    print(invalid_course_ids)


# ============================================================
# 6. USER-COURSE TRANSACTION REPETITION
# ============================================================

print("\n" + "-" * 70)
print("USER-COURSE TRANSACTION REPETITION")
print("-" * 70)

user_course_counts = (
    transactions
    .groupby(["UserID", "CourseID"])
    .size()
    .reset_index(name="TransactionCount")
)

repeated_user_course = user_course_counts[
    user_course_counts["TransactionCount"] > 1
]

print("\nTotal unique User-Course combinations:")
print(len(user_course_counts))

print("\nUser-Course combinations with multiple transactions:")
print(len(repeated_user_course))

print("\nMaximum transactions for one User-Course combination:")
print(
    user_course_counts["TransactionCount"].max()
)


# ============================================================
# 7. TRANSACTIONS PER COURSE
# ============================================================

print("\n" + "-" * 70)
print("TRANSACTIONS PER COURSE")
print("-" * 70)

transactions_per_course = (
    transactions
    .groupby("CourseID")
    .size()
    .sort_values(ascending=False)
)

print("\nTop courses by transaction count:")
print(
    transactions_per_course.head(10)
)

print("\nMinimum transactions for a course:")
print(
    transactions_per_course.min()
)

print("\nMaximum transactions for a course:")
print(
    transactions_per_course.max()
)

print("\nAverage transactions per course:")
print(
    transactions_per_course.mean()
)


# ============================================================
# 8. TEACHERS ASSOCIATED WITH COURSES
# ============================================================

print("\n" + "-" * 70)
print("TEACHERS ASSOCIATED WITH COURSES")
print("-" * 70)

teachers_per_course = (
    transactions
    .groupby("CourseID")["TeacherID"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nNumber of teachers associated with each course:")
print(teachers_per_course)

print("\nMinimum teachers associated with a course:")
print(teachers_per_course.min())

print("\nMaximum teachers associated with a course:")
print(teachers_per_course.max())

print("\nAverage teachers per course:")
print(teachers_per_course.mean())


# ============================================================
# 9. TEACHER TRANSACTION DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("TRANSACTIONS PER TEACHER")
print("-" * 70)

transactions_per_teacher = (
    transactions
    .groupby("TeacherID")
    .size()
    .sort_values(ascending=False)
)

print("\nTop teachers by transaction count:")
print(
    transactions_per_teacher.head(10)
)

print("\nMinimum transactions for a teacher:")
print(
    transactions_per_teacher.min()
)

print("\nMaximum transactions for a teacher:")
print(
    transactions_per_teacher.max()
)


# ============================================================
# 10. COURSE TYPE VS TRANSACTION AMOUNT
# ============================================================

print("\n" + "-" * 70)
print("COURSE TYPE VS TRANSACTION AMOUNT")
print("-" * 70)

course_transaction_data = transactions.merge(
    courses[
        [
            "CourseID",
            "CourseType",
            "CoursePrice"
        ]
    ],
    on="CourseID",
    how="left"
)

print("\nAverage transaction amount by course type:")

print(
    course_transaction_data
    .groupby("CourseType")["Amount"]
    .agg(
        [
            "count",
            "mean",
            "min",
            "max",
            "sum"
        ]
    )
)


# ============================================================
# 11. FREE COURSE TRANSACTION VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("FREE COURSE REVENUE VALIDATION")
print("-" * 70)

free_transactions = (
    course_transaction_data[
        course_transaction_data["CourseType"] == "Free"
    ]
)

non_zero_free_transactions = (
    free_transactions["Amount"] != 0
).sum()

print("\nTotal Free-course transactions:")
print(len(free_transactions))

print("\nFree-course transactions with non-zero Amount:")
print(non_zero_free_transactions)


# ============================================================
# 12. PAID COURSE PRICE VS AMOUNT
# ============================================================

print("\n" + "-" * 70)
print("PAID COURSE PRICE VS TRANSACTION AMOUNT")
print("-" * 70)

paid_transactions = (
    course_transaction_data[
        course_transaction_data["CourseType"] == "Paid"
    ]
)

paid_transactions["AmountDifference"] = (
    paid_transactions["Amount"]
    - paid_transactions["CoursePrice"]
)

print("\nNumber of Paid transactions:")
print(len(paid_transactions))

print("\nMaximum absolute difference between Amount and CoursePrice:")

print(
    paid_transactions["AmountDifference"]
    .abs()
    .max()
)

print("\nNumber of Paid transactions where Amount != CoursePrice:")

print(
    (
        paid_transactions["AmountDifference"].abs()
        > 0.01
    ).sum()
)


# ============================================================
# 13. DATE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("DATE VALIDATION")
print("-" * 70)

print("\nInvalid TransactionDate values:")
print(
    transactions["TransactionDate"].isna().sum()
)

print("\nEarliest date:")
print(
    transactions["TransactionDate"].min()
)

print("\nLatest date:")
print(
    transactions["TransactionDate"].max()
)


# ============================================================
# 14. NUMERIC RANGE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("NUMERIC RANGE VALIDATION")
print("-" * 70)

print("\nCourses with negative price:")
print(
    (courses["CoursePrice"] < 0).sum()
)

print("\nCourses with non-positive duration:")
print(
    (courses["CourseDuration"] <= 0).sum()
)

print("\nCourses with rating outside 0-5:")
print(
    (
        (courses["CourseRating"] < 0)
        |
        (courses["CourseRating"] > 5)
    ).sum()
)

print("\nTeachers with negative experience:")
print(
    (teachers["YearsOfExperience"] < 0).sum()
)

print("\nTeachers with rating outside 0-5:")
print(
    (
        (teachers["TeacherRating"] < 0)
        |
        (teachers["TeacherRating"] > 5)
    ).sum()
)

print("\nTransactions with negative Amount:")
print(
    (transactions["Amount"] < 0).sum()
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION COMPLETED")
print("=" * 70)