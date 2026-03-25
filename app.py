import streamlit as st
import base64

from analysis_eda import *
from visualization_plots import *
from reports import *
from query import *

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst")

st.info("""
Step 1: Upload Dataset  
Step 2: Explore Data Analysis  
Step 3: Train Machine Learning Model  
Step 4: Ask questions about your data
""")

# -------------------------
# Sidebar Navigation
# -------------------------

st.sidebar.title("AI Data Analyst")

page = st.sidebar.selectbox(
    "Navigation",
    ["EDA", "Machine Learning", "AI Insights"]
)

# -------------------------
# Session State for Model
# -------------------------

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.features = None
    st.session_state.model_name = None
    st.session_state.score = None

# -------------------------
# File Upload
# -------------------------

file = st.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx"])

if file:

    df = load_file(file)

    # -------------------------
    # EDA Section
    # -------------------------

    if page == "EDA":

        st.header("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Summary")

        summary = dataset_summary(df)

        col1, col2 = st.columns(2)

        col1.metric("Rows", summary["rows"])
        col2.metric("Columns", summary["columns"])

        st.write("Missing Values")
        st.write(summary["missing_values"])

        st.write("Data Types")
        st.write(summary["data_types"])

        st.subheader("Statistical Summary")
        st.dataframe(statistical_analysis(df))

        st.subheader("Correlation Heatmap")
        correlation_heatmap(df)

    # -------------------------
    # Machine Learning Section
    # -------------------------

    elif page == "Machine Learning":

        st.header("Machine Learning Analysis")

        target = st.selectbox("Select Target Column", df.columns)

        if st.button("Train Model"):

            try:

                model, score, model_name, problem_type, features = train_model(df, target)

                # Save in session
                st.session_state.model = model
                st.session_state.features = features
                st.session_state.model_name = model_name
                st.session_state.score = score

                st.success("Model Trained Successfully")

                st.write("Problem Type:", problem_type)
                st.write("Best Model:", model_name)
                st.write("Score:", score)

                st.subheader("Feature Importance")
                feature_importance_plot(model, features)

            except Exception as e:

                st.error(f"Model training failed: {e}")

    # -------------------------
    # AI Insights Section
    # -------------------------

    elif page == "AI Insights":

        st.header("AI Insights")

        insights = generate_insights(df)

        for i in insights:
            st.write("•", i)

    # -------------------------
    # Natural Language Query
    # -------------------------

    st.divider()
    st.header("Ask Questions About Your Data")

    user_query = st.text_input(
        "Type your question (example: average income, total sales, max age)"
    )

    if user_query:

        result = simple_nl_query(df, user_query)

        st.subheader("Result")
        st.write(result)

    # -------------------------
    # Generate Business Report
    # -------------------------

    st.divider()
    st.header("Generate Business Report")

    if st.button("Generate PDF Report"):

        insights = generate_insights(df)

        report_file = generate_pdf_report(
            df,
            insights,
            st.session_state.model,
            st.session_state.features,
            st.session_state.model_name,
            st.session_state.score
        )

        with open(report_file, "rb") as f:
            pdf_data = f.read()

        base64_pdf = base64.b64encode(pdf_data).decode("utf-8")

        pdf_display = f"""
        <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="700"
        type="application/pdf">
        </iframe>
        """

        st.markdown("### Report Preview")
        st.markdown(pdf_display, unsafe_allow_html=True)

        st.download_button(
            "Download Report",
            pdf_data,
            file_name="AI_Data_Report.pdf"
        )

else:

    st.warning("Please upload a dataset to begin analysis.")