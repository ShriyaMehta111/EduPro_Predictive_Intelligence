import joblib
import os

base = r"D:\Projects\EduPro_Predictive_Forecasting\models"

files = [
    "final_enrollment_model.pkl",
    "final_revenue_model.pkl",
    "final_category_revenue_model.pkl"
]

for file in files:
    path = os.path.join(base, file)

    print("\n" + "=" * 70)
    print(file)
    print("=" * 70)

    model = joblib.load(path)

    print("Model type:", type(model))
    print("Number of features:", getattr(model, "n_features_in_", "NOT AVAILABLE"))

    features = getattr(model, "feature_names_in_", None)

    if features is not None:
        print("Features:")
        for i, feature in enumerate(features, 1):
            print(f"{i:02d}. {feature}")
    else:
        print("Feature names: NOT AVAILABLE")