import os
import pandas as pd


# ============================================================
# EDUPro PREDICTIVE MODELING & REVENUE FORECASTING
# STEP 1: RAW DATA INSPECTION
# ============================================================

print("=" * 70)
print("EDUPro PREDICTIVE MODELING & REVENUE FORECASTING")
print("STEP 1: RAW DATA INSPECTION")
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

RAW_DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "raw"
)

DATA_FILE = os.path.join(
    RAW_DATA_DIR,
    "EduPro_Dataset.xlsx"
)


print("\n" + "-" * 70)
print("PROJECT PATH INFORMATION")
print("-" * 70)

print("Current script directory:")
print(CURRENT_DIR)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nRaw data directory:")
print(RAW_DATA_DIR)

print("\nDataset file:")
print(DATA_FILE)


# ============================================================
# 2. CHECK WHETHER DATASET EXISTS
# ============================================================

print("\n" + "-" * 70)
print("CHECKING DATASET")
print("-" * 70)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "\nEduPro dataset was not found.\n"
        "Please make sure the file is located at:\n"
        f"{DATA_FILE}"
    )

print("Dataset found successfully.")


# ============================================================
# 3. READ EXCEL WORKBOOK
# ============================================================

print("\n" + "-" * 70)
print("READING EXCEL WORKBOOK")
print("-" * 70)

excel_file = pd.ExcelFile(DATA_FILE)

print("\nAvailable sheets:")

for sheet in excel_file.sheet_names:
    print("-", sheet)


# ============================================================
# 4. LOAD ALL SHEETS
# ============================================================

print("\n" + "-" * 70)
print("LOADING DATASETS")
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

print("\nAll sheets loaded successfully.")


# ============================================================
# 5. DATASET SHAPES
# ============================================================

print("\n" + "-" * 70)
print("DATASET SHAPES")
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
# 6. COLUMN NAMES
# ============================================================

print("\n" + "-" * 70)
print("COLUMN NAMES")
print("-" * 70)

print("\nUsers columns:")
print(users.columns.tolist())

print("\nTeachers columns:")
print(teachers.columns.tolist())

print("\nCourses columns:")
print(courses.columns.tolist())

print("\nTransactions columns:")
print(transactions.columns.tolist())


# ============================================================
# 7. FIRST FIVE ROWS
# ============================================================

print("\n" + "-" * 70)
print("FIRST FIVE ROWS")
print("-" * 70)

print("\n--- USERS ---")
print(users.head())

print("\n--- TEACHERS ---")
print(teachers.head())

print("\n--- COURSES ---")
print(courses.head())

print("\n--- TRANSACTIONS ---")
print(transactions.head())


# ============================================================
# 8. DATA TYPES
# ============================================================

print("\n" + "-" * 70)
print("DATA TYPES")
print("-" * 70)

print("\n--- USERS ---")
print(users.dtypes)

print("\n--- TEACHERS ---")
print(teachers.dtypes)

print("\n--- COURSES ---")
print(courses.dtypes)

print("\n--- TRANSACTIONS ---")
print(transactions.dtypes)


# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n" + "-" * 70)
print("MISSING VALUE ANALYSIS")
print("-" * 70)

print("\n--- USERS ---")
print(users.isnull().sum())

print("\n--- TEACHERS ---")
print(teachers.isnull().sum())

print("\n--- COURSES ---")
print(courses.isnull().sum())

print("\n--- TRANSACTIONS ---")
print(transactions.isnull().sum())


# ============================================================
# 10. DUPLICATE ROWS
# ============================================================

print("\n" + "-" * 70)
print("DUPLICATE ROW ANALYSIS")
print("-" * 70)

print("\nUsers duplicate rows:")
print(users.duplicated().sum())

print("\nTeachers duplicate rows:")
print(teachers.duplicated().sum())

print("\nCourses duplicate rows:")
print(courses.duplicated().sum())

print("\nTransactions duplicate rows:")
print(transactions.duplicated().sum())


# ============================================================
# 11. UNIQUE ID ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("UNIQUE ID ANALYSIS")
print("-" * 70)

print("\nUnique UserID:")
print(users["UserID"].nunique())

print("\nUnique TeacherID:")
print(teachers["TeacherID"].nunique())

print("\nUnique CourseID in Courses:")
print(courses["CourseID"].nunique())

print("\nUnique CourseID in Transactions:")
print(transactions["CourseID"].nunique())

print("\nUnique TransactionID:")
print(transactions["TransactionID"].nunique())


# ============================================================
# 12. NUMERICAL SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("NUMERICAL SUMMARY")
print("-" * 70)

print("\n--- USERS ---")
print(users.describe())

print("\n--- TEACHERS ---")
print(teachers.describe())

print("\n--- COURSES ---")
print(courses.describe())

print("\n--- TRANSACTIONS ---")
print(transactions.describe())


# ============================================================
# 13. CATEGORICAL VALUE COUNTS
# ============================================================

print("\n" + "-" * 70)
print("CATEGORICAL VALUE DISTRIBUTIONS")
print("-" * 70)

print("\nCourse categories:")
print(courses["CourseCategory"].value_counts())

print("\nCourse types:")
print(courses["CourseType"].value_counts())

print("\nCourse levels:")
print(courses["CourseLevel"].value_counts())

print("\nTeacher expertise:")
print(teachers["Expertise"].value_counts())


# ============================================================
# 14. TRANSACTION DATE ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("TRANSACTION DATE ANALYSIS")
print("-" * 70)

transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"],
    errors="coerce"
)

print("\nEarliest transaction date:")
print(transactions["TransactionDate"].min())

print("\nLatest transaction date:")
print(transactions["TransactionDate"].max())

print("\nNumber of unique transaction dates:")
print(transactions["TransactionDate"].nunique())


# ============================================================
# 15. TRANSACTION AMOUNT ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("TRANSACTION AMOUNT ANALYSIS")
print("-" * 70)

print("\nMinimum transaction amount:")
print(transactions["Amount"].min())

print("\nMaximum transaction amount:")
print(transactions["Amount"].max())

print("\nAverage transaction amount:")
print(transactions["Amount"].mean())

print("\nMedian transaction amount:")
print(transactions["Amount"].median())

print("\nTotal transaction revenue:")
print(transactions["Amount"].sum())

print("\nNumber of zero-value transactions:")
print(
    (transactions["Amount"] == 0).sum()
)

print("\nPercentage of zero-value transactions:")
print(
    (transactions["Amount"] == 0).mean() * 100
)


# ============================================================
# 16. COURSE-TEACHER RELATIONSHIP
# ============================================================

print("\n" + "-" * 70)
print("COURSE-TEACHER RELATIONSHIP")
print("-" * 70)

print("\nTeachers sheet columns:")
print(teachers.columns.tolist())

if "CourseID" in teachers.columns and "TeacherID" in teachers.columns:

    teachers_per_course = (
        teachers
        .groupby("CourseID")["TeacherID"]
        .nunique()
    )

    print("\nNumber of courses represented in Teachers:")
    print(teachers["CourseID"].nunique())

    print("\nMinimum teachers assigned to a course:")
    print(teachers_per_course.min())

    print("\nMaximum teachers assigned to a course:")
    print(teachers_per_course.max())

    print("\nAverage teachers per course:")
    print(teachers_per_course.mean())

else:

    print("\nWARNING:")
    print("Expected CourseID and/or TeacherID columns were not found.")
    print("Actual Teachers columns are:")
    print(teachers.columns.tolist())

# ============================================================
# 17. COURSE-TRANSACTION RELATIONSHIP
# ============================================================

print("\n" + "-" * 70)
print("COURSE-TRANSACTION RELATIONSHIP")
print("-" * 70)

transaction_course_ids = set(
    transactions["CourseID"].dropna().unique()
)

course_ids = set(
    courses["CourseID"].dropna().unique()
)

missing_courses = (
    transaction_course_ids - course_ids
)

print("\nCourseIDs appearing in Transactions:")
print(len(transaction_course_ids))

print("\nCourseIDs appearing in Courses:")
print(len(course_ids))

print("\nTransaction CourseIDs not found in Courses:")
print(len(missing_courses))

if missing_courses:
    print(missing_courses)


# ============================================================
# 18. FINAL INSPECTION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("RAW DATA INSPECTION COMPLETED")
print("=" * 70)