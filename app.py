from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import sqlite3
from datetime import datetime


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "random_forest_addiction_model.pkl"
)

MAPPING_PATH = os.path.join(
    BASE_DIR,
    "model",
    "label_mapping.pkl"
)


# ============================================================
# DATABASE PATH
# ============================================================

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "predictions.db"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

if os.path.exists(MAPPING_PATH):

    label_mapping = joblib.load(MAPPING_PATH)

else:

    label_mapping = {
        0: "Low",
        1: "Moderate",
        2: "High"
    }


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():

    conn = sqlite3.connect(DATABASE_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            date_time TEXT NOT NULL,

            addiction_level TEXT NOT NULL,

            daily_screen_hours REAL,

            avg_sleep_hours REAL,

            anxiety_score REAL,

            low_mood_score REAL,

            life_satisfaction REAL,

            loneliness REAL,

            self_esteem REAL,

            fomo_score REAL,

            social_comparison REAL,

            physical_activity_days REAL

        )
    """)

    conn.commit()

    conn.close()


# Create database automatically
init_db()


# ============================================================
# WELLNESS RECOMMENDATIONS
# ============================================================

def get_recommendation(addiction_level, scores):

    anxiety = scores["anxiety_score_0to27"]

    low_mood = scores["low_mood_score_0to27"]

    life_satisfaction = scores["life_satisfaction_1to10"]

    loneliness = scores["loneliness_1to10"]

    self_esteem = scores["self_esteem_1to10"]

    fomo = scores["fomo_1to10"]

    social_comparison = scores["social_comparison_1to10"]

    recommendations = []


    # Addiction level recommendation
    if addiction_level == "High":

        recommendations.append(
            "Reduce excessive social media usage and set daily screen-time limits."
        )

    elif addiction_level == "Moderate":

        recommendations.append(
            "Try to maintain a balanced social media routine and reduce unnecessary usage."
        )

    else:

        recommendations.append(
            "Continue maintaining healthy and balanced social media habits."
        )


    # Anxiety
    if anxiety >= 15:

        recommendations.append(
            "Practice relaxation activities such as deep breathing, meditation, or short breaks."
        )


    # Low mood
    if low_mood >= 15:

        recommendations.append(
            "Spend time doing enjoyable offline activities and maintain a regular daily routine."
        )


    # Life satisfaction
    if life_satisfaction <= 4:

        recommendations.append(
            "Focus on personal goals, hobbies, relationships, and activities that improve wellbeing."
        )


    # Loneliness
    if loneliness >= 7:

        recommendations.append(
            "Try to spend more quality time with friends, family, or supportive people."
        )


    # Self esteem
    if self_esteem <= 4:

        recommendations.append(
            "Focus on your strengths and achievements instead of comparing yourself with others."
        )


    # FOMO
    if fomo >= 7:

        recommendations.append(
            "Avoid constantly checking social media and remember that online posts do not show everything."
        )


    # Social comparison
    if social_comparison >= 7:

        recommendations.append(
            "Reduce comparison with people online and focus on your own progress."
        )


    # General recommendation
    recommendations.append(
        "Maintain healthy sleep, physical activity, and regular offline activities."
    )


    return recommendations


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ====================================================
        # BASIC INFORMATION
        # ====================================================

        age = int(
            request.form.get("age")
        )

        gender = request.form.get(
            "gender"
        )

        occupation = request.form.get(
            "occupation"
        )

        region = request.form.get(
            "region"
        )

        platform = request.form.get(
            "most_used_platform"
        )

        daily_screen_hours = float(
            request.form.get(
                "daily_screen_hours"
            )
        )

        avg_sleep_hours = float(
            request.form.get(
                "avg_sleep_hours"
            )
        )


        # ====================================================
        # QUESTIONNAIRE SCORES
        # ====================================================

        anxiety_score = int(
            request.form.get(
                "anxiety_score_0to27"
            )
        )

        low_mood_score = int(
            request.form.get(
                "low_mood_score_0to27"
            )
        )

        life_satisfaction = int(
            request.form.get(
                "life_satisfaction_1to10"
            )
        )

        loneliness = int(
            request.form.get(
                "loneliness_1to10"
            )
        )

        self_esteem = int(
            request.form.get(
                "self_esteem_1to10"
            )
        )

        fomo = int(
            request.form.get(
                "fomo_1to10"
            )
        )

        social_comparison = int(
            request.form.get(
                "social_comparison_1to10"
            )
        )


        # ====================================================
        # PHYSICAL ACTIVITY
        # ====================================================

        physical_activity = int(
            request.form.get(
                "physical_activity"
            )
        )


        # ====================================================
        # CREATE DATAFRAME
        # ====================================================

        user_data = pd.DataFrame([{

            "age": age,

            "gender": gender,

            "occupation": occupation,

            "region": region,

            "most_used_platform": platform,

            "daily_screen_hours":
                daily_screen_hours,

            "avg_sleep_hours":
                avg_sleep_hours,

            "anxiety_score_0to27":
                anxiety_score,

            "low_mood_score_0to27":
                low_mood_score,

            "life_satisfaction_1to10":
                life_satisfaction,

            "loneliness_1to10":
                loneliness,

            "self_esteem_1to10":
                self_esteem,

            "fomo_1to10":
                fomo,

            "social_comparison_1to10":
                social_comparison,

            "physical_activity_days":
                physical_activity

        }])


        # ====================================================
        # PREPROCESSING
        # ====================================================

        user_data = pd.get_dummies(

            user_data,

            columns=[
                "gender",
                "occupation",
                "region",
                "most_used_platform"
            ]

        )


        # ====================================================
        # MATCH MODEL FEATURES
        # ====================================================

        if hasattr(
            model,
            "feature_names_in_"
        ):

            expected_features = (
                model.feature_names_in_
            )

            user_data = user_data.reindex(

                columns=expected_features,

                fill_value=0

            )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            user_data
        )

        predicted_class = int(
            prediction[0]
        )


        # ====================================================
        # CONVERT CLASS TO LEVEL
        # ====================================================

        if predicted_class in label_mapping:

            addiction_level = (
                label_mapping[predicted_class]
            )

        else:

            addiction_level = {

                0: "Low",

                1: "Moderate",

                2: "High"

            }.get(

                predicted_class,

                "Unknown"

            )


        # ====================================================
        # SCORES DICTIONARY
        # ====================================================

        scores = {

            "anxiety_score_0to27":
                anxiety_score,

            "low_mood_score_0to27":
                low_mood_score,

            "life_satisfaction_1to10":
                life_satisfaction,

            "loneliness_1to10":
                loneliness,

            "self_esteem_1to10":
                self_esteem,

            "fomo_1to10":
                fomo,

            "social_comparison_1to10":
                social_comparison

        }


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        recommendations = get_recommendation(

            addiction_level,

            scores

        )


        # ====================================================
        # SAVE PREDICTION
        # ====================================================

        conn = sqlite3.connect(
            DATABASE_PATH
        )

        cursor = conn.cursor()


        cursor.execute("""

            INSERT INTO predictions (

                date_time,

                addiction_level,

                daily_screen_hours,

                avg_sleep_hours,

                anxiety_score,

                low_mood_score,

                life_satisfaction,

                loneliness,

                self_esteem,

                fomo_score,

                social_comparison,

                physical_activity_days

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            addiction_level,

            daily_screen_hours,

            avg_sleep_hours,

            anxiety_score,

            low_mood_score,

            life_satisfaction,

            loneliness,

            self_esteem,

            fomo,

            social_comparison,

            physical_activity

        ))


        conn.commit()

        conn.close()


        # ====================================================
        # RESULT PAGE
        # ====================================================

        return render_template(

            "result.html",

            addiction_level=addiction_level,

            scores=scores,

            recommendations=recommendations

        )


    except Exception as e:

        return f"""

        <h2>Error occurred</h2>

        <p>{str(e)}</p>

        <br>

        <a href="/">Go back</a>

        """


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()


    cursor.execute("""

        SELECT *

        FROM predictions

        ORDER BY id DESC

    """)


    predictions = cursor.fetchall()

    conn.close()


    return render_template(

        "history.html",

        predictions=predictions

    )


# ============================================================
# PROGRESS PAGE
# ============================================================

@app.route("/progress")
def progress():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()


    cursor.execute("""

        SELECT *

        FROM predictions

        ORDER BY id ASC

    """)


    predictions = cursor.fetchall()

    conn.close()


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    message = ""

    status = "none"

    previous_level = None

    current_level = None

    previous_screen_time = None

    current_screen_time = None

    screen_time_change = None


    # ========================================================
    # NO PREDICTIONS
    # ========================================================

    if len(predictions) == 0:

        message = (
            "No prediction data available yet. "
            "Complete the questionnaire to start tracking your progress."
        )

        status = "none"


    # ========================================================
    # ONLY ONE PREDICTION
    # ========================================================

    elif len(predictions) == 1:

        current = predictions[-1]

        current_level = current[
            "addiction_level"
        ]

        current_screen_time = current[
            "daily_screen_hours"
        ]

        message = (
            "This is your first assessment. "
            "Take another assessment later to track your progress."
        )

        status = "first"


    # ========================================================
    # TWO OR MORE PREDICTIONS
    # ========================================================

    else:

        previous = predictions[-2]

        current = predictions[-1]


        previous_level = previous[
            "addiction_level"
        ]

        current_level = current[
            "addiction_level"
        ]


        previous_screen_time = previous[
            "daily_screen_hours"
        ]

        current_screen_time = current[
            "daily_screen_hours"
        ]


        # ====================================================
        # LEVEL COMPARISON
        # ====================================================

        level_value = {

            "Low": 1,

            "Moderate": 2,

            "High": 3

        }


        previous_value = level_value.get(

            previous_level,

            0

        )

        current_value = level_value.get(

            current_level,

            0

        )


        if current_value < previous_value:

            message = (
                "🎉 Great progress! Your addiction level "
                "has decreased compared with your previous assessment."
            )

            status = "improved"


        elif current_value > previous_value:

            message = (
                "Your addiction level has increased compared "
                "with your previous assessment. Try following "
                "the wellness recommendations."
            )

            status = "increased"


        else:

            message = (
                "Your addiction level is unchanged. "
                "Keep working on your digital wellness habits."
            )

            status = "same"


        # ====================================================
        # SCREEN TIME COMPARISON
        # ====================================================

        screen_time_change = round(

            previous_screen_time -
            current_screen_time,

            2

        )


    # ========================================================
    # RENDER PROGRESS PAGE
    # ========================================================

    return render_template(

        "progress.html",

        predictions=predictions,

        message=message,

        status=status,

        previous_level=previous_level,

        current_level=current_level,

        previous_screen_time=previous_screen_time,

        current_screen_time=current_screen_time,

        screen_time_change=screen_time_change

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )