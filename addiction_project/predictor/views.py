import os
import joblib
import pandas as pd

from django.shortcuts import render
from django.conf import settings

# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "addiction_project",
    "model",
    "random_forest_addiction_model.pkl"
)

LABEL_PATH = os.path.join(
    settings.BASE_DIR,
    "addiction_project",
    "model",
    "label_mapping.pkl"
)

FEATURE_PATH = os.path.join(
    settings.BASE_DIR,
    "addiction_project",
    "model",
    "feature_columns.pkl"
)

model = joblib.load(MODEL_PATH)
label_mapping = joblib.load(LABEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

def index(request):

    if request.method == "POST":

        try:

            user_input = {

                "age": int(request.POST.get("age")),

                "gender": request.POST.get("gender"),

                "occupation": request.POST.get("occupation"),

                "region": request.POST.get("region"),

                "most_used_platform": request.POST.get("most_used_platform"),

                "platforms_used_count": int(request.POST.get("platforms_used_count")),

                "daily_screen_hours": float(request.POST.get("daily_screen_hours")),

                "daily_notifications": int(request.POST.get("daily_notifications")),

                "night_time_use": int(request.POST.get("night_time_use")),

                "minutes_to_first_check_after_waking":
                    int(request.POST.get("minutes_to_first_check_after_waking")),

                "primary_purpose": request.POST.get("primary_purpose"),

                "avg_sleep_hours": float(request.POST.get("avg_sleep_hours")),

                "anxiety_score_0to27":
                    int(request.POST.get("anxiety_score_0to27")),

                "low_mood_score_0to27":
                    int(request.POST.get("low_mood_score_0to27")),

                "life_satisfaction_1to10":
                    int(request.POST.get("life_satisfaction_1to10")),

                "loneliness_1to10":
                    int(request.POST.get("loneliness_1to10")),

                "self_esteem_1to10":
                    int(request.POST.get("self_esteem_1to10")),

                "fomo_1to10":
                    int(request.POST.get("fomo_1to10")),

                "social_comparison_1to10":
                    int(request.POST.get("social_comparison_1to10")),

                "physical_activity_days_per_week":
                    int(request.POST.get("physical_activity_days_per_week")),

                "attempted_digital_detox":
                    int(request.POST.get("attempted_digital_detox")),

                "uses_screen_time_limits":
                    request.POST.get("uses_screen_time_limits"),

                "seeks_mental_health_support":
                    request.POST.get("seeks_mental_health_support")

            }

            # Convert to DataFrame
            user_df = pd.DataFrame([user_input])

            # One-hot encoding
            user_encoded = pd.get_dummies(user_df)

            # Match training columns
            user_encoded = user_encoded.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # Prediction
            prediction = model.predict(user_encoded)[0]

            addiction_level = label_mapping[prediction]

            # Confidence
            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(user_encoded)

                confidence = round(max(probabilities[0]) * 100, 2)

            # Recommendation

            if addiction_level == "Low":

                recommendation = (
                    "Maintain your current digital habits and continue balancing online and offline activities."
                )

            elif addiction_level == "Moderate":

                recommendation = (
                    "Reduce daily screen time, avoid excessive night-time use and take regular breaks."
                )

            else:

                recommendation = (
                    "High addiction detected. Reduce screen time, improve sleep, enable screen time limits and practice digital wellness."
                )

            context = {

                "prediction": addiction_level,

                "confidence": confidence,

                "recommendation": recommendation,

                "user": user_input

            }

            return render(request, "result.html", context)

        except Exception as e:

            return render(request, "index.html", {
                "error": str(e)
            })

    return render(request, "index.html")