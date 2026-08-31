import os
import pandas as pd


# ============================================================
# EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 4: DATA CLEANING
# ============================================================

print("=" * 70)
print("EDUPRO PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 4: DATA CLEANING")
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


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

print("\n" + "-" * 70)
print("LOADING RAW DATA")
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

print("\nRaw datasets loaded successfully.")


# ============================================================
# 3. STANDARDIZE COLUMN NAMES
# ============================================================

def clean_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


users = clean_column_names(users)
teachers = clean_column_names(teachers)
courses = clean_column_names(courses)
transactions = clean_column_names(transactions)


# ============================================================
# 4. CLEAN TEXT COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("CLEANING TEXT COLUMNS")
print("-" * 70)


def strip_text_columns(df):
    df = df.copy()

    for column in df.select_dtypes(
        include=["object"]
    ).columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    return df


users = strip_text_columns(users)
teachers = strip_text_columns(teachers)
courses = strip_text_columns(courses)
transactions = strip_text_columns(transactions)


# ============================================================
# 5. CONVERT DATE COLUMN
# ============================================================

print("\n" + "-" * 70)
print("CONVERTING TRANSACTION DATE")
print("-" * 70)

transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"],
    errors="coerce"
)

print("\nInvalid dates after conversion:")
print(
    transactions["TransactionDate"].isna().sum()
)


# ============================================================
# 6. ENSURE NUMERICAL TYPES
# ============================================================

print("\n" + "-" * 70)
print("CONVERTING NUMERICAL COLUMNS")
print("-" * 70)

teacher_numeric_columns = [
    "Age",
    "YearsOfExperience",
    "TeacherRating"
]

course_numeric_columns = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating"
]

transaction_numeric_columns = [
    "Amount"
]


for column in teacher_numeric_columns:
    teachers[column] = pd.to_numeric(
        teachers[column],
        errors="coerce"
    )


for column in course_numeric_columns:
    courses[column] = pd.to_numeric(
        courses[column],
        errors="coerce"
    )


for column in transaction_numeric_columns:
    transactions[column] = pd.to_numeric(
        transactions[column],
        errors="coerce"
    )


# ============================================================
# 7. REMOVE EXACT DUPLICATES
# ============================================================

print("\n" + "-" * 70)
print("DUPLICATE HANDLING")
print("-" * 70)

users_before = len(users)
teachers_before = len(teachers)
courses_before = len(courses)
transactions_before = len(transactions)


users = users.drop_duplicates().copy()
teachers = teachers.drop_duplicates().copy()
courses = courses.drop_duplicates().copy()
transactions = transactions.drop_duplicates().copy()


print("\nUsers duplicates removed:")
print(users_before - len(users))

print("Teachers duplicates removed:")
print(teachers_before - len(teachers))

print("Courses duplicates removed:")
print(courses_before - len(courses))

print("Transactions duplicates removed:")
print(transactions_before - len(transactions))


# ============================================================
# 8. RANGE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("RANGE VALIDATION")
print("-" * 70)

invalid_course_prices = (
    courses["CoursePrice"] < 0
).sum()

invalid_course_duration = (
    courses["CourseDuration"] <= 0
).sum()

invalid_course_ratings = (
    (courses["CourseRating"] < 0)
    |
    (courses["CourseRating"] > 5)
).sum()

invalid_teacher_experience = (
    teachers["YearsOfExperience"] < 0
).sum()

invalid_teacher_ratings = (
    (teachers["TeacherRating"] < 0)
    |
    (teachers["TeacherRating"] > 5)
).sum()

invalid_transaction_amounts = (
    transactions["Amount"] < 0
).sum()


print("\nNegative course prices:")
print(invalid_course_prices)

print("\nNon-positive course durations:")
print(invalid_course_duration)

print("\nInvalid course ratings:")
print(invalid_course_ratings)

print("\nNegative teacher experience:")
print(invalid_teacher_experience)

print("\nInvalid teacher ratings:")
print(invalid_teacher_ratings)

print("\nNegative transaction amounts:")
print(invalid_transaction_amounts)


# ============================================================
# 9. FOREIGN KEY VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("FOREIGN KEY VALIDATION")
print("-" * 70)

invalid_users = (
    ~transactions["UserID"].isin(
        users["UserID"]
    )
).sum()

invalid_courses = (
    ~transactions["CourseID"].isin(
        courses["CourseID"]
    )
).sum()

invalid_teachers = (
    ~transactions["TeacherID"].isin(
        teachers["TeacherID"]
    )
).sum()


print("\nTransactions with invalid UserID:")
print(invalid_users)

print("\nTransactions with invalid CourseID:")
print(invalid_courses)

print("\nTransactions with invalid TeacherID:")
print(invalid_teachers)


# ============================================================
# 10. DERIVE DATE FEATURES
# ============================================================

print("\n" + "-" * 70)
print("CREATING DATE FEATURES")
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
    .dt.month_name()
)

transactions["Quarter"] = (
    transactions["TransactionDate"]
    .dt.quarter
)

transactions["Day"] = (
    transactions["TransactionDate"]
    .dt.day
)

transactions["DayOfWeek"] = (
    transactions["TransactionDate"]
    .dt.dayofweek
)

transactions["DayName"] = (
    transactions["TransactionDate"]
    .dt.day_name()
)


# ============================================================
# 11. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "-" * 70)
print("FINAL MISSING VALUE CHECK")
print("-" * 70)

print("\nUsers:")
print(users.isna().sum().sum())

print("\nTeachers:")
print(teachers.isna().sum().sum())

print("\nCourses:")
print(courses.isna().sum().sum())

print("\nTransactions:")
print(transactions.isna().sum().sum())


# ============================================================
# 12. SAVE CLEANED DATA
# ============================================================

print("\n" + "-" * 70)
print("SAVING CLEANED DATA")
print("-" * 70)

users_output = os.path.join(
    OUTPUT_DIR,
    "cleaned_users.csv"
)

teachers_output = os.path.join(
    OUTPUT_DIR,
    "cleaned_teachers.csv"
)

courses_output = os.path.join(
    OUTPUT_DIR,
    "cleaned_courses.csv"
)

transactions_output = os.path.join(
    OUTPUT_DIR,
    "cleaned_transactions.csv"
)


users.to_csv(
    users_output,
    index=False
)

teachers.to_csv(
    teachers_output,
    index=False
)

courses.to_csv(
    courses_output,
    index=False
)

transactions.to_csv(
    transactions_output,
    index=False
)


# ============================================================
# 13. FINAL DATASET SIZES
# ============================================================

print("\n" + "-" * 70)
print("FINAL CLEANED DATASET SIZES")
print("-" * 70)

print("\nUsers:")
print(users.shape)

print("\nTeachers:")
print(teachers.shape)

print("\nCourses:")
print(courses.shape)

print("\nTransactions:")
print(transactions.shape)


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(users_output)
print(teachers_output)
print(courses_output)
print(transactions_output)