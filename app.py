from flask import Flask, render_template, request, redirect, url_for, flash

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

import sqlite3
import os
import joblib
import pandas as pd


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

app.secret_key = "smads_secret_key_2026"


# =========================================================
# FLASK LOGIN CONFIGURATION
# =========================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"

login_manager.login_message = "Please login to continue."

login_manager.login_message_category = "error"


# =========================================================
# MODEL PATHS
# =========================================================

MODEL_PATH = os.path.join(
    "model",
    "random_forest_addiction_model.pkl"
)

LABEL_MAPPING_PATH = os.path.join(
    "model",
    "label_mapping.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    "model",
    "feature_columns.pkl"
)


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# LOAD LABEL MAPPING
# =========================================================

try:

    label_mapping = joblib.load(
        LABEL_MAPPING_PATH
    )

except Exception:

    label_mapping = {
        0: "Low",
        1: "Moderate",
        2: "High"
    }


# =========================================================
# LOAD FEATURE COLUMNS
# =========================================================

try:

    feature_columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

except Exception:

    feature_columns = None


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db_connection():

    connection = sqlite3.connect(
        "users.db"
    )

    connection.row_factory = sqlite3.Row

    return connection


# =========================================================
# CREATE / UPDATE DATABASE TABLES
# =========================================================

def create_tables():

    connection = get_db_connection()

    cursor = connection.cursor()


    # =====================================================
    # USERS TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT NOT NULL UNIQUE,

            email TEXT NOT NULL UNIQUE,

            password TEXT NOT NULL

        )
    """)


    # =====================================================
    # PREDICTIONS TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            addiction_level TEXT NOT NULL,

            prediction_confidence REAL,

            daily_screen_hours REAL,

            avg_sleep_hours REAL,

            anxiety REAL,

            low_mood REAL,

            life_satisfaction REAL,

            loneliness REAL,

            self_esteem REAL,

            fomo REAL,

            social_comparison REAL,

            physical_activity REAL,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
                REFERENCES users(id)

        )
    """)


    # =====================================================
    # CHECK EXISTING COLUMNS
    # =====================================================

    cursor.execute(
        "PRAGMA table_info(predictions)"
    )

    existing_columns = [
        column[1]
        for column in cursor.fetchall()
    ]


    # =====================================================
    # ADD MISSING COLUMNS TO OLD DATABASE
    # =====================================================

    columns_to_add = {

        "anxiety": "REAL",

        "low_mood": "REAL",

        "life_satisfaction": "REAL",

        "loneliness": "REAL",

        "self_esteem": "REAL",

        "fomo": "REAL",

        "social_comparison": "REAL",

        "physical_activity": "REAL"
    }


    for column_name, column_type in columns_to_add.items():

        if column_name not in existing_columns:

            cursor.execute(
                f"""
                ALTER TABLE predictions
                ADD COLUMN {column_name} {column_type}
                """
            )


    connection.commit()

    connection.close()


# =========================================================
# CREATE / UPDATE TABLES
# =========================================================

create_tables()


# =========================================================
# USER CLASS
# =========================================================

class User(UserMixin):

    def __init__(
        self,
        id,
        username,
        email
    ):

        self.id = id

        self.username = username

        self.email = email


# =========================================================
# FLASK-LOGIN USER LOADER
# =========================================================

@login_manager.user_loader
def load_user(user_id):

    connection = get_db_connection()

    user = connection.execute(
        """
        SELECT *
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()

    connection.close()


    if user:

        return User(
            user["id"],
            user["username"],
            user["email"]
        )


    return None


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if current_user.is_authenticated:

        return redirect(
            url_for("home")
        )


    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        # =================================================
        # FIND USER
        # =================================================

        connection = get_db_connection()

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        connection.close()


        # =================================================
        # CHECK PASSWORD
        # =================================================

        if user and check_password_hash(
            user["password"],
            password
        ):

            logged_user = User(
                user["id"],
                user["username"],
                user["email"]
            )

            login_user(logged_user)


            flash(
                "Login successful!",
                "success"
            )


            return redirect(
                url_for("home")
            )


        else:

            flash(
                "Invalid email or password.",
                "error"
            )


    return render_template(
        "login.html"
    )


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if current_user.is_authenticated:

        return redirect(
            url_for("home")
        )


    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        # =================================================
        # CHECK EMPTY FIELDS
        # =================================================

        if not username or not email or not password:

            flash(
                "Please fill in all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # =================================================
        # CHECK PASSWORD
        # =================================================

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # =================================================
        # PASSWORD LENGTH
        # =================================================

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # =================================================
        # DATABASE CONNECTION
        # =================================================

        connection = get_db_connection()


        # =================================================
        # CHECK EMAIL
        # =================================================

        existing_email = connection.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()


        if existing_email:

            connection.close()

            flash(
                "Email already registered.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # =================================================
        # CHECK USERNAME
        # =================================================

        existing_username = connection.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()


        if existing_username:

            connection.close()

            flash(
                "Username already exists.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # =================================================
        # HASH PASSWORD
        # =================================================

        hashed_password = generate_password_hash(
            password
        )


        # =================================================
        # INSERT USER
        # =================================================

        connection.execute(
            """
            INSERT INTO users
            (
                username,
                email,
                password
            )
            VALUES (?, ?, ?)
            """,
            (
                username,
                email,
                hashed_password
            )
        )


        connection.commit()

        connection.close()


        flash(
            "Registration successful. Please login.",
            "success"
        )


        return redirect(
            url_for("login")
        )


    return render_template(
        "register.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()


    flash(
        "You have been logged out.",
        "success"
    )


    return redirect(
        url_for("login")
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
@login_required
def home():

    return render_template(
        "home.html"
    )


# =========================================================
# ASSESSMENT PAGE
# =========================================================

@app.route("/assessment")
@login_required
def assessment():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():

    try:

        # =================================================
        # BASIC INFORMATION
        # =================================================

        age = float(
            request.form.get(
                "age",
                0
            )
        )


        gender = request.form.get(
            "gender",
            "Male"
        )


        occupation = request.form.get(
            "occupation",
            "Student"
        )


        region = request.form.get(
            "region",
            "Kerala"
        )


        platform = request.form.get(
            "most_used_platform",
            "Instagram"
        )


        platforms_used_count = int(
            request.form.get(
                "platforms_used_count",
                1
            )
        )


        daily_screen_hours = float(
            request.form.get(
                "daily_screen_hours",
                0
            )
        )


        night_time_use = int(
            request.form.get(
                "night_time_use",
                0
            )
        )


        minutes_to_first_check = float(
            request.form.get(
                "minutes_to_first_check_after_waking",
                30
            )
        )


        primary_purpose = request.form.get(
            "primary_purpose",
            "Entertainment"
        )


        avg_sleep_hours = float(
            request.form.get(
                "avg_sleep_hours",
                0
            )
        )


        # =================================================
        # MENTAL WELLBEING SCORES
        # =================================================

        anxiety = float(
            request.form.get(
                "anxiety_score_0to27",
                0
            )
        )


        low_mood = float(
            request.form.get(
                "low_mood_score_0to27",
                0
            )
        )


        life_satisfaction = float(
            request.form.get(
                "life_satisfaction_1to10",
                5
            )
        )


        loneliness = float(
            request.form.get(
                "loneliness_1to10",
                5
            )
        )


        self_esteem = float(
            request.form.get(
                "self_esteem_1to10",
                5
            )
        )


        fomo = float(
            request.form.get(
                "fomo_1to10",
                5
            )
        )


        social_comparison = float(
            request.form.get(
                "social_comparison_1to10",
                5
            )
        )


        physical_activity = float(
            request.form.get(
                "physical_activity_days_per_week",
                0
            )
        )


        uses_screen_time_limits = int(
            request.form.get(
                "uses_screen_time_limits",
                0
            )
        )


        attempted_digital_detox = int(
            request.form.get(
                "attempted_digital_detox",
                0
            )
        )


        seeks_mental_health_support = int(
            request.form.get(
                "seeks_mental_health_support",
                0
            )
        )


        # =================================================
        # CREATE INPUT DATA
        # =================================================

        input_data = {

            "age":
                age,

            "gender":
                gender,

            "occupation":
                occupation,

            "region":
                region,

            "most_used_platform":
                platform,

            "platforms_used_count":
                platforms_used_count,

            "daily_screen_hours":
                daily_screen_hours,

            "night_time_use":
                night_time_use,

            "minutes_to_first_check_after_waking":
                minutes_to_first_check,

            "primary_purpose":
                primary_purpose,

            "avg_sleep_hours":
                avg_sleep_hours,

            "anxiety_score_0to27":
                anxiety,

            "low_mood_score_0to27":
                low_mood,

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

            "physical_activity_days_per_week":
                physical_activity,

            "uses_screen_time_limits":
                uses_screen_time_limits,

            "attempted_digital_detox":
                attempted_digital_detox,

            "seeks_mental_health_support":
                seeks_mental_health_support
        }


        # =================================================
        # CONVERT TO DATAFRAME
        # =================================================

        input_df = pd.DataFrame(
            [input_data]
        )


        # =================================================
        # ONE-HOT ENCODING
        # =================================================

        input_df = pd.get_dummies(
            input_df
        )


        # =================================================
        # MATCH MODEL FEATURES
        # =================================================

        if feature_columns is not None:

            input_df = input_df.reindex(
                columns=feature_columns,
                fill_value=0
            )


        # =================================================
        # MACHINE LEARNING PREDICTION
        # =================================================

        prediction = model.predict(
            input_df
        )[0]


        # =================================================
        # CONVERT PREDICTION TO INTEGER
        # =================================================

        try:

            prediction_int = int(
                prediction
            )

        except Exception:

            prediction_int = prediction


        # =================================================
        # GET ADDICTION LEVEL
        # =================================================

        if isinstance(
            label_mapping,
            dict
        ):

            addiction_level = label_mapping.get(
                prediction_int,
                str(prediction)
            )

        else:

            addiction_level = str(
                prediction
            )


        # =================================================
        # PREDICTION CONFIDENCE
        # =================================================

        confidence = None


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_df
            )[0]

            confidence = round(
                float(
                    max(probabilities)
                ) * 100,
                2
            )


        # =================================================
        # RECOMMENDATIONS
        # =================================================

        recommendations = []


        # =================================================
        # SCREEN TIME
        # =================================================

        if daily_screen_hours >= 6:

            recommendations.append(
                "Try reducing daily social media screen time gradually."
            )

        elif daily_screen_hours >= 4:

            recommendations.append(
                "Set a daily screen-time limit and follow it consistently."
            )

        else:

            recommendations.append(
                "Continue maintaining a balanced social media usage pattern."
            )


        # =================================================
        # SLEEP
        # =================================================

        if avg_sleep_hours < 7:

            recommendations.append(
                "Try to maintain at least 7 hours of sleep and reduce late-night screen use."
            )


        # =================================================
        # ANXIETY
        # =================================================

        if anxiety >= 15:

            recommendations.append(
                "Consider relaxation activities such as breathing exercises, exercise, or talking to someone you trust."
            )


        # =================================================
        # LOW MOOD
        # =================================================

        if low_mood >= 15:

            recommendations.append(
                "Maintain regular physical activity and social interaction."
            )


        # =================================================
        # LIFE SATISFACTION
        # =================================================

        if life_satisfaction <= 4:

            recommendations.append(
                "Spend time on activities and relationships that improve your overall wellbeing."
            )


        # =================================================
        # LONELINESS
        # =================================================

        if loneliness >= 7:

            recommendations.append(
                "Try to increase meaningful offline social interaction."
            )


        # =================================================
        # SELF ESTEEM
        # =================================================

        if self_esteem <= 4:

            recommendations.append(
                "Avoid comparing yourself with unrealistic social media content."
            )


        # =================================================
        # FOMO
        # =================================================

        if fomo >= 7:

            recommendations.append(
                "Consider turning off unnecessary notifications and taking short digital breaks."
            )


        # =================================================
        # SOCIAL COMPARISON
        # =================================================

        if social_comparison >= 7:

            recommendations.append(
                "Remember that social media often shows only selected parts of people's lives."
            )


        # =================================================
        # SCORES FOR RESULT PAGE
        # =================================================

        scores = {

            "anxiety_score_0to27":
                anxiety,

            "low_mood_score_0to27":
                low_mood,

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


        # =================================================
        # SAVE COMPLETE PREDICTION
        # =================================================

        connection = get_db_connection()


        connection.execute(
            """
            INSERT INTO predictions
            (
                user_id,
                addiction_level,
                prediction_confidence,
                daily_screen_hours,
                avg_sleep_hours,
                anxiety,
                low_mood,
                life_satisfaction,
                loneliness,
                self_esteem,
                fomo,
                social_comparison,
                physical_activity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                current_user.id,

                addiction_level,

                confidence,

                daily_screen_hours,

                avg_sleep_hours,

                anxiety,

                low_mood,

                life_satisfaction,

                loneliness,

                self_esteem,

                fomo,

                social_comparison,

                physical_activity
            )
        )


        connection.commit()

        connection.close()


        # =================================================
        # SHOW RESULT PAGE
        # =================================================

        return render_template(

            "result.html",

            addiction_level=
                addiction_level,

            confidence=
                confidence,

            recommendations=
                recommendations,

            daily_screen_hours=
                daily_screen_hours,

            avg_sleep_hours=
                avg_sleep_hours,

            scores=
                scores
        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        return f"""
        <h2>Prediction Error</h2>

        <p>{str(e)}</p>

        <a href="{url_for('assessment')}">
            Go Back
        </a>
        """


# =========================================================
# HISTORY PAGE
# =========================================================

@app.route("/history")
@login_required
def history():

    connection = get_db_connection()


    predictions = connection.execute(
        """
        SELECT
            id,
            user_id,
            addiction_level,
            prediction_confidence,
            daily_screen_hours,
            avg_sleep_hours,
            anxiety,
            low_mood,
            life_satisfaction,
            loneliness,
            self_esteem,
            fomo,
            social_comparison,
            physical_activity,
            created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (current_user.id,)
    ).fetchall()


    connection.close()


    return render_template(
        "history.html",
        predictions=predictions
    )


# =========================================================
# PROGRESS PAGE
# =========================================================

@app.route("/progress")
@login_required
def progress():

    connection = get_db_connection()


    # =====================================================
    # GET CURRENT USER'S PREDICTIONS
    # =====================================================

    predictions = connection.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at ASC
        """,
        (current_user.id,)
    ).fetchall()


    connection.close()


    # =====================================================
    # DEFAULT VALUES
    # =====================================================

    latest_screen_time = 0

    previous_screen_time = 0

    latest_sleep = 0

    previous_sleep = 0

    screen_time_change = 0

    sleep_change = 0

    latest_level = "No assessment"

    previous_level = "No assessment"

    level_change = "No previous assessment"

    total_assessments = len(predictions)


    # =====================================================
    # GET LATEST ASSESSMENT
    # =====================================================

    if total_assessments >= 1:

        latest_prediction = predictions[-1]


        latest_screen_time = (
            latest_prediction["daily_screen_hours"]
            or 0
        )


        latest_sleep = (
            latest_prediction["avg_sleep_hours"]
            or 0
        )


        latest_level = (
            latest_prediction["addiction_level"]
            or "Unknown"
        )


    # =====================================================
    # GET PREVIOUS ASSESSMENT
    # =====================================================

    if total_assessments >= 2:

        previous_prediction = predictions[-2]


        previous_screen_time = (
            previous_prediction["daily_screen_hours"]
            or 0
        )


        previous_sleep = (
            previous_prediction["avg_sleep_hours"]
            or 0
        )


        previous_level = (
            previous_prediction["addiction_level"]
            or "Unknown"
        )


        # =================================================
        # SCREEN TIME CHANGE
        # =================================================

        screen_time_change = round(
            latest_screen_time -
            previous_screen_time,
            2
        )


        # =================================================
        # SLEEP CHANGE
        # =================================================

        sleep_change = round(
            latest_sleep -
            previous_sleep,
            2
        )


        # =================================================
        # LEVEL CHANGE
        # =================================================

        level_order = {

            "Low": 0,

            "Moderate": 1,

            "High": 2
        }


        if (
            latest_level in level_order
            and previous_level in level_order
        ):

            level_difference = (
                level_order[latest_level]
                -
                level_order[previous_level]
            )


            if level_difference < 0:

                level_change = "Improved"

            elif level_difference > 0:

                level_change = "Increased"

            else:

                level_change = "No change"


    # =====================================================
    # SCREEN TIME STATUS
    # =====================================================

    if screen_time_change < 0:

        screen_time_status = "Reduced"

    elif screen_time_change > 0:

        screen_time_status = "Increased"

    else:

        screen_time_status = "No change"


    # =====================================================
    # SLEEP STATUS
    # =====================================================

    if sleep_change > 0:

        sleep_status = "Improved"

    elif sleep_change < 0:

        sleep_status = "Reduced"

    else:

        sleep_status = "No change"


    # =====================================================
    # SCREEN TIME IMPROVEMENT PERCENTAGE
    # =====================================================

    screen_time_improvement = 0


    if previous_screen_time > 0:

        screen_time_improvement = round(
            (
                (
                    previous_screen_time -
                    latest_screen_time
                )
                /
                previous_screen_time
            )
            * 100,
            2
        )


    # =====================================================
    # SLEEP IMPROVEMENT PERCENTAGE
    # =====================================================

    sleep_improvement = 0


    if previous_sleep > 0:

        sleep_improvement = round(
            (
                (
                    latest_sleep -
                    previous_sleep
                )
                /
                previous_sleep
            )
            * 100,
            2
        )


    # =====================================================
    # PROGRESS MESSAGE
    # =====================================================

    if total_assessments == 0:

        progress_message = (
            "Complete your first assessment to start "
            "tracking your digital wellbeing."
        )

    elif total_assessments == 1:

        progress_message = (
            "Complete another assessment later to "
            "compare your progress."
        )

    else:

        if screen_time_change < 0:

            progress_message = (
                "Your recent assessment shows a reduction "
                "in daily screen time."
            )

        elif screen_time_change > 0:

            progress_message = (
                "Your recent assessment shows an increase "
                "in daily screen time."
            )

        else:

            progress_message = (
                "Your daily screen time has remained the same "
                "between the latest assessments."
            )


    # =====================================================
    # SEND DATA TO PROGRESS PAGE
    # =====================================================

    return render_template(

        "progress.html",

        predictions=
            predictions,

        total_assessments=
            total_assessments,

        latest_screen_time=
            latest_screen_time,

        previous_screen_time=
            previous_screen_time,

        screen_time_change=
            screen_time_change,

        screen_time_status=
            screen_time_status,

        screen_time_improvement=
            screen_time_improvement,

        latest_sleep=
            latest_sleep,

        previous_sleep=
            previous_sleep,

        sleep_change=
            sleep_change,

        sleep_status=
            sleep_status,

        sleep_improvement=
            sleep_improvement,

        latest_level=
            latest_level,

        previous_level=
            previous_level,

        level_change=
            level_change,

        progress_message=
            progress_message
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )