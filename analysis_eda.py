import pandas as pd

# -------------------------
# Load Dataset
# -------------------------

def load_file(file):
    try:

        if file.name.endswith(".csv"):

            try:
                df = pd.read_csv(file, encoding="utf-8", low_memory=False)

            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding="latin1", low_memory=False)

        elif file.name.endswith((".xls", ".xlsx")):

            df = pd.read_excel(file)

        else:
            raise ValueError("Unsupported file format")

        return df

    except Exception as e:
        print("Error loading file:", e)
        return None


# -------------------------
# Dataset Summary
# -------------------------

def dataset_summary(df):

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum(),
        "data_types": df.dtypes
    }

    return summary


# -------------------------
# Statistical Analysis
# -------------------------

def statistical_analysis(df):

    return df.describe()


# -------------------------
# Correlation Analysis
# -------------------------

def correlation_analysis(df):

    return df.corr(numeric_only=True)


# -------------------------
# AutoML Model Training
# -------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def train_model(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    # keep numeric features only
    X = X.select_dtypes(include="number")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Detect Problem Type
    # -------------------------

    if y.dtype == "object" or y.nunique() < 10:

        problem_type = "Classification"

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier()
        }

        best_score = 0
        best_model = None
        best_model_name = ""

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds)

            if score > best_score:

                best_score = score
                best_model = model
                best_model_name = name

    else:

        problem_type = "Regression"

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor()
        }

        best_score = -1
        best_model = None
        best_model_name = ""

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = r2_score(y_test, preds)

            if score > best_score:

                best_score = score
                best_model = model
                best_model_name = name

    return best_model, best_score, best_model_name, problem_type, X.columns