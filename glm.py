#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:22:39 2024

@author: uzma.k.khan
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from statsmodels.graphics.mosaicplot import mosaic

# Function for Generalized Linear Model (GLM) analysis
def perform_glm_analysis(data, var1, var2, label="default"):
    # Create contingency table
    contingency_table = pd.crosstab(data[var1].astype(str), data[var2].astype(str))

    # Prepare data for GLM
    df = contingency_table.stack().reset_index()
    df.columns = [var1, var2, 'Freq']

    # Fit GLM with Poisson family
    formula = f'Freq ~ C({var1}) + C({var2})'
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()

    # Display GLM summary
    st.write("## GLM Summary")
    st.write(glm_model.summary())

    # Plot Incidence Rate Ratio
    st.write("## Incidence Rate Ratio Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    coef = glm_model.params
    coef.plot(kind='barh', ax=ax)
    plt.title(f'Incidence Rate Ratios for {var1} and {var2}')
    st.pyplot(fig)

    # Save plot as PDF
    ir_pdf = BytesIO()
    fig.savefig(ir_pdf, format="pdf")
    st.download_button("Download Incidence Rate Ratio Plot as PDF", ir_pdf, file_name=f"{label}_glm_incidence_rate_ratio.pdf")

    # Mosaic Plot
    st.write("## Mosaic Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    mosaic(df, ['C({})'.format(var1), 'C({})'.format(var2)], ax=ax, title=f'Mosaic Plot of {var1} and {var2}')
    st.pyplot(fig)

    # Save mosaic plot as PDF
    mosaic_pdf = BytesIO()
    fig.savefig(mosaic_pdf, format="pdf")
    st.download_button("Download Mosaic Plot as PDF", mosaic_pdf, file_name=f"{label}_glm_mosaic.pdf")

# Streamlit App Interface
st.title("GLM Analysis Tool for Categorical Variables")

# Upload CSV file
uploaded_file = st.file_uploader("data_table", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Select categorical variables
    var1 = st.selectbox("Select first categorical variable", data.columns)
    var2 = st.selectbox("Select second categorical variable", data.columns)

    if st.button("Run GLM Analysis"):
        perform_glm_analysis(data, var1, var2)
