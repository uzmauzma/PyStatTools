import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import statsmodels.api as sm

# Function to create and download plot as PDF
def create_plot_with_download(plot_func, title):
    buf = BytesIO()
    pdf_pages = PdfPages(buf)

    plt.figure(figsize=(10, 6))
    plot_func()  # Generate the plot
    plt.title(title)
    plt.tight_layout()
    
    pdf_pages.savefig()
    pdf_pages.close()
    buf.seek(0)
    
    st.pyplot(plt)
    st.download_button(label=f"Download {title} as PDF", data=buf, file_name=f"{title}.pdf", mime="application/pdf")
    plt.close()

# Chi-Square Analysis Function
def perform_chisquare_analysis(data, var1, var2):
    contingency_table = pd.crosstab(data[var1].astype(str), data[var2].astype(str))
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    observed = contingency_table.values

    st.write("## Chi-Square Test Results")
    st.write(f"Chi-Square Statistic: {chi2:.2f}")
    st.write(f"p-value: {p:.4f}")
    st.write(f"Degrees of Freedom: {dof}")

    # Balloon Plot
    create_plot_with_download(lambda: sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar_kws={"label": "Count"}), 'Balloon Plot of Contingency Table')
    
    # Residual Plot
    create_plot_with_download(lambda: sns.heatmap(observed - expected.reshape(contingency_table.shape), annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar_kws={"label": "Residuals"},
                                                  xticklabels=contingency_table.columns, yticklabels=contingency_table.index), 'Residual Plot')
    
    # Chi-Square Contribution Plot
    contribution = (observed - expected.reshape(contingency_table.shape)) ** 2 / expected.reshape(contingency_table.shape)
    create_plot_with_download(lambda: sns.heatmap(contribution, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={"label": "Chi-Square Contribution"},
                                                  xticklabels=contingency_table.columns, yticklabels=contingency_table.index), 'Chi-Square Contribution Plot')
    
    # Chi-Square Ratio Plot
    ratio = observed / expected.reshape(contingency_table.shape)
    create_plot_with_download(lambda: sns.heatmap(ratio, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={"label": "Chi-Square Ratio"},
                                                  xticklabels=contingency_table.columns, yticklabels=contingency_table.index), 'Chi-Square Ratio Plot')

# GLM Analysis Function
def perform_glm_analysis(data, var1, var2):
    # Create the contingency table and format it as required for GLM
    contingency_table = pd.crosstab(data[var1].astype(str), data[var2].astype(str))
    df = pd.DataFrame(contingency_table.stack(), columns=['Freq']).reset_index()
    df.columns = [var1, var2, "Freq"]

    # Check for NaN values
    if df.isnull().any().any():
        st.error("Data contains NaN values. Please clean your data and try again.")
        return

    # Convert categorical predictors to dummy variables, ensuring numeric format
    X = pd.get_dummies(df[[var1, var2]], drop_first=True)
    y = df["Freq"].astype(float)  # Ensure `y` is in float format for the GLM model

    # Add constant to predictor variables
    X = sm.add_constant(X)

    # Debugging output
    st.write("Predictor Variables (X):")
    st.write(X)
    st.write("Response Variable (y):")
    st.write(y)

    # Fit the GLM model
    try:
        model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        st.write("## GLM Analysis Results")
        st.write(model.summary())

        # Incidence Rate Ratio Plot
        create_plot_with_download(lambda: plt.barh(X.columns[1:], model.params[1:], color='skyblue'), 'Incidence Rate Ratio Plot')
    except Exception as e:
        st.error(f"An error occurred while fitting the GLM model: {e}")