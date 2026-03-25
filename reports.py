from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd


# -------------------------
# Generate AI Insights
# -------------------------

def generate_insights(df):

    insights = []

    insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        mean_val = df[col].mean()
        insights.append(f"Average value of {col} is {round(mean_val,2)}")

    missing = df.isnull().sum().sum()
    insights.append(f"Total missing values in dataset: {missing}")

    return insights


# -------------------------
# Create Correlation Chart
# -------------------------

def create_correlation_plot(df):

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    plt.figure(figsize=(6,4))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")

    file_name = "correlation.png"
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    return file_name


# -------------------------
# Create Feature Importance Plot
# -------------------------

def create_feature_importance_plot(model, features):

    if not hasattr(model, "feature_importances_"):
        return None

    importance = model.feature_importances_

    plt.figure(figsize=(6,4))
    plt.barh(features, importance)
    plt.title("Feature Importance")

    file_name = "feature_importance.png"
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    return file_name


# -------------------------
# Generate PDF Report
# -------------------------

def generate_pdf_report(df, insights, model=None, features=None, model_name=None, score=None):

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Data Analyst Report", ln=True)

    # Dataset summary
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Dataset Summary", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Rows: {df.shape[0]}", ln=True)
    pdf.cell(0, 8, f"Columns: {df.shape[1]}", ln=True)

    # Insights
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "AI Insights", ln=True)

    pdf.set_font("Arial", "", 12)

    for i in insights:
        pdf.multi_cell(0, 8, f"- {i}")

    # Correlation analysis
    corr_plot = create_correlation_plot(df)

    if corr_plot:

        pdf.add_page()

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Correlation Analysis", ln=True)

        pdf.image(corr_plot, w=170)

    # Feature importance
    if model is not None and features is not None and len(features) > 0:

        feat_plot = create_feature_importance_plot(model, features)

        if feat_plot:

            pdf.add_page()

            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Feature Importance", ln=True)

            pdf.image(feat_plot, w=170)

    # ML Results
    if model_name:

        pdf.add_page()

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Machine Learning Results", ln=True)

        pdf.set_font("Arial", "", 12)

        pdf.cell(0, 8, f"Best Model: {model_name}", ln=True)

        if score is not None:
            pdf.cell(0, 8, f"Score: {round(score,4)}", ln=True)

    # Save
    file_name = "AI_Data_Report.pdf"

    pdf.output(file_name)

    return file_name