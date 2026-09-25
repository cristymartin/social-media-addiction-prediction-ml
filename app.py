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

    label_mapping = joblib.load(
        MAPPING_PATH
    )

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

    conn = sqlite3.connect(
        DATABASE_PATH
    )

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
# DIGITAL WELLBEING RECOMMENDATIONS
# ============================================================

def get_recommendations(
    addiction_level,
    screen_hours,
    sleep_hours,
    physical_activity,
    anxiety_score,
    low_mood_score,
    life_satisfaction,
    loneliness,
    self_esteem,
    fomo,
    social_comparison
):

    recommendations = []


    # ========================================================
    # LOW ADDICTION
    # ========================================================

    if addiction_level == "Low":

        recommendations.append(
            "Continue maintaining healthy and balanced social media habits."
        )

        recommendations.append(
            "Take short breaks during long periods of screen use."
        )

        recommendations.append(
            "Keep unnecessary social media notifications turned off."
        )

        # Screen time

        if screen_hours >= 4:

            recommendations.append(
                "Your reported screen time is somewhat high. "
                "Try reducing unnecessary scrolling gradually."
            )


        # Sleep

        if sleep_hours < 7:

            recommendations.append(
                "Try to maintain a regular sleep schedule "
                "and avoid social media close to bedtime."
            )


        # Physical activity

        if physical_activity < 3:

            recommendations.append(
                "Include more physical activity during the week "
                "and replace some screen time with exercise or outdoor activities."
            )


        # FOMO

        if fomo >= 7:

            recommendations.append(
                "If FOMO causes frequent checking, "
                "try checking social media only at planned times."
            )


        # Social comparison

        if social_comparison >= 7:

            recommendations.append(
                "Consider reducing exposure to content "
                "that causes excessive social comparison."
            )


        # Loneliness

        if loneliness >= 7:

            recommendations.append(
                "Try spending more quality time with friends, "
                "family, or supportive people."
            )


        # Self-esteem

        if self_esteem <= 4:

            recommendations.append(
                "Focus on your strengths and personal achievements "
                "rather than comparing yourself with others online."
            )


    # ========================================================
    # MODERATE ADDICTION
    # ========================================================

    elif addiction_level == "Moderate":

        recommendations.append(
            "Set a daily social media time limit "
            "and gradually reduce unnecessary usage."
        )

        recommendations.append(
            "Turn off non-essential social media notifications "
            "to reduce frequent checking."
        )

        recommendations.append(
            "Create at least one 30-minute screen-free period every day."
        )

        recommendations.append(
            "Avoid checking social media immediately after waking up."
        )

        recommendations.append(
            "Keep your phone away while studying, working, "
            "or performing important tasks."
        )


        # Screen time

        if screen_hours >= 5:

            recommendations.append(
                "Your reported screen time is relatively high. "
                "Try reducing it gradually by 15–30 minutes each day."
            )


        # Sleep

        if sleep_hours < 7:

            recommendations.append(
                "Avoid social media before bedtime "
                "and keep your phone away while sleeping."
            )


        # Physical activity

        if physical_activity < 3:

            recommendations.append(
                "Replace some social media time with walking, "
                "exercise, hobbies, or outdoor activities."
            )


        # FOMO

        if fomo >= 7:

            recommendations.append(
                "Your FOMO score is high. "
                "Try scheduled social-media checking instead of repeatedly checking for updates."
            )


        # Social comparison

        if social_comparison >= 7:

            recommendations.append(
                "Your social comparison score is high. "
                "Consider muting or unfollowing content that negatively affects your wellbeing."
            )


        # Anxiety

        if anxiety_score >= 15:

            recommendations.append(
                "Your anxiety-related responses are relatively elevated. "
                "Consider regular screen-free breaks and calming offline activities."
            )


        # Low mood

        if low_mood_score >= 15:

            recommendations.append(
                "Your low-mood responses are relatively elevated. "
                "Try incorporating enjoyable offline activities, "
                "exercise, and supportive social interaction."
            )


        # Loneliness

        if loneliness >= 7:

            recommendations.append(
                "Try increasing meaningful offline interaction "
                "with friends, family, or people you trust."
            )


        # Self-esteem

        if self_esteem <= 4:

            recommendations.append(
                "Focus on personal achievements and strengths "
                "instead of comparing yourself with people online."
            )


    # ========================================================
    # HIGH ADDICTION
    # ========================================================

    elif addiction_level == "High":

        recommendations.append(
            "Set a strict daily limit for social media applications."
        )

        recommendations.append(
            "Turn off unnecessary notifications "
            "and avoid repeated checking of social media."
        )

        recommendations.append(
            "Create regular phone-free periods during the day."
        )

        recommendations.append(
            "Consider removing highly distracting social media applications "
            "from your home screen or temporarily restricting them."
        )

        recommendations.append(
            "Replace some social media time with exercise, "
            "hobbies, study, or offline social activities."
        )


        # Screen time

        if screen_hours >= 6:

            recommendations.append(
                "Your reported screen time is high. "
                "Consider following a structured gradual reduction plan "
                "rather than making a sudden change."
            )


        # Sleep

        if sleep_hours < 7:

            recommendations.append(
                "Avoid social media close to bedtime "
                "and keep your phone away from your sleeping area when possible."
            )


        # Physical activity

        if physical_activity < 3:

            recommendations.append(
                "Increase physical activity during the week "
                "and use it as an alternative to unnecessary screen time."
            )


        # FOMO

        if fomo >= 7:

            recommendations.append(
                "Your FOMO score is high. "
                "Try checking social media only at scheduled times "
                "rather than whenever you feel the urge to check."
            )


        # Social comparison

        if social_comparison >= 7:

            recommendations.append(
                "Your social comparison score is high. "
                "Consider muting or unfollowing accounts "
                "that encourage unhealthy comparison."
            )


        # Anxiety

        if anxiety_score >= 15:

            recommendations.append(
                "Your anxiety-related responses are relatively elevated. "
                "Include regular screen-free periods and calming offline activities."
            )


        # Low mood

        if low_mood_score >= 15:

            recommendations.append(
                "Your low-mood responses are relatively elevated. "
                "Try maintaining offline activities, "
                "physical activity, and supportive social connections."
            )


        # Loneliness

        if loneliness >= 7:

            recommendations.append(
                "Try increasing meaningful offline interaction "
                "with friends, family, or people you trust."
            )


        # Self-esteem

        if self_esteem <= 4:

            recommendations.append(
                "Focus on your strengths and achievements "
                "rather than comparing yourself with others online."
            )


        # Professional support

        if anxiety_score >= 15 or low_mood_score >= 15:

            recommendations.append(
                "If social media use is causing significant distress "
                "or interfering with daily life, consider discussing "
                "your concerns with a qualified mental-health professional."
            )


    return recommendations


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# ASSESSMENT PAGE
# ============================================================

@app.route("/assessment")
def assessment():

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
                "life_satisfaction"
            )
        )

        loneliness = int(
            request.form.get(
                "loneliness"
            )
        )

        self_esteem = int(
            request.form.get(
                "self_esteem"
            )
        )

        fomo = int(
            request.form.get(
                "fomo"
            )
        )

        social_comparison = int(
            request.form.get(
                "social_comparison"
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
        # CREATE USER DATAFRAME
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
        # PREDICTION PROBABILITY
        # ====================================================

        confidence = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                user_data
            )

            confidence = round(
                float(probabilities[0].max()) * 100,
                2
            )


        # ====================================================
        # CONVERT CLASS TO ADDICTION LEVEL
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
        # GET PERSONALIZED RECOMMENDATIONS
        # ====================================================

        recommendations = get_recommendations(

            addiction_level,

            daily_screen_hours,

            avg_sleep_hours,

            physical_activity,

            anxiety_score,

            low_mood_score,

            life_satisfaction,

            loneliness,

            self_esteem,

            fomo,

            social_comparison

        )


        # ====================================================
        # SAVE PREDICTION TO DATABASE
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

            confidence=confidence,

            scores=scores,

            recommendations=recommendations

        )


    except Exception as e:

        return f"""

        <html>

        <head>

            <title>SMADS - Error</title>

        </head>

        <body>

            <h2>Error occurred</h2>

            <p>{str(e)}</p>

            <br>

            <a href="/">Go back</a>

        </body>

        </html>

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
                "Your latest predicted addiction level is lower "
                "than your previous assessment."
            )

            status = "improved"


        elif current_value > previous_value:

            message = (
                "Your latest predicted addiction level is higher "
                "than your previous assessment. "
                "Review the digital wellbeing recommendations."
            )

            status = "increased"


        else:

            message = (
                "Your predicted addiction level is unchanged. "
                "Continue working on your digital wellbeing habits."
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