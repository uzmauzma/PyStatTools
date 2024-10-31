import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Function to perform Chi-Square analysis and visualizations
def perform_chisquare_analysis(data, var1, var2):
    # Create contingency table
    contingency_table = pd.crosstab(data[var1].astype(str), data[var2].astype(str))
    
    # Perform Chi-Square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    # Observed counts are the values in the contingency table
    observed = contingency_table.values

    # Display results
    st.write("## Chi-Square Test Results")
    st.write(f"Chi-Square Statistic: {chi2:.2f}")
    st.write(f"p-value: {p:.4f}")
    st.write(f"Degrees of Freedom: {dof}")

    # Balloon Plot
    st.write("## Balloon Plot")
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar_kws={"label": "Count"})
    plt.title('Balloon Plot of Contingency Table')
    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9, right=0.9)  # Adjust margins
    st.pyplot(plt)

    # Residual Plot
    st.write("## Residual Plot")
    residuals = observed - expected.reshape(contingency_table.shape)
    plt.figure(figsize=(10, 6))
    sns.heatmap(residuals, annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar_kws={"label": "Residuals"},
                xticklabels=contingency_table.columns, yticklabels=contingency_table.index)
    plt.title('Residual Plot')
    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9, right=0.9)  # Adjust margins
    st.pyplot(plt)

    # Chi-Square Contribution Plot
    contribution = (observed - expected.reshape(contingency_table.shape)) ** 2 / expected.reshape(contingency_table.shape)
    plt.figure(figsize=(10, 6))
    sns.heatmap(contribution, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={"label": "Chi-Square Contribution"},
                xticklabels=contingency_table.columns, yticklabels=contingency_table.index)
    plt.title('Chi-Square Contribution Plot')
    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9, right=0.9)  # Adjust margins
    st.pyplot(plt)

    # Chi-Square Ratio Plot
    ratio = observed / expected.reshape(contingency_table.shape)
    plt.figure(figsize=(10, 6))
    sns.heatmap(ratio, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={"label": "Chi-Square Ratio"},
                xticklabels=contingency_table.columns, yticklabels=contingency_table.index)
    plt.title('Chi-Square Ratio Plot')
    plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9, right=0.9)  # Adjust margins
    st.pyplot(plt)

# Streamlit app
st.title("Chi-Square Analysis Tool")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Select categorical variables
    var1 = st.selectbox("Select first categorical variable", data.columns)
    var2 = st.selectbox("Select second categorical variable", data.columns)

    if st.button("Run Chi-Square Analysis"):
        perform_chisquare_analysis(data, var1, var2)
