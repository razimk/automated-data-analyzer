import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def correlation_heatmap(df):

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, ax=ax)
    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)


from sklearn.ensemble import RandomForestRegressor

def feature_importance_plot(model, features):

    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(x=importance, y=features, ax=ax)

    ax.set_title("Feature Importance")

    st.pyplot(fig)