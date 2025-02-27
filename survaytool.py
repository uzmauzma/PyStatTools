#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:36:09 2025

@author: uzma.k.khan
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic
import vcd
from sklearn.preprocessing import LabelEncoder
import tempfile
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def Multipreprocess_data(data, dependent_var, independent_vars, reference_categories=None):
    """
    Preprocess the dataset by dynamically detecting the type of independent variables
    and encoding them accordingly, with the option to specify reference categories.

    Parameters:
        data (pd.DataFrame): The input dataset.
        dependent_var (str): The name of the dependent variable.
        independent_vars (list of str): The names of the independent variables.
        reference_categories (dict, optional): A dictionary specifying the reference categories
                                                for categorical independent variables.

    Returns:
        pd.DataFrame: The preprocessed dataset.
        dict: Mapping of categories to numeric values for the dependent variable.
        dict: Mapping of categories to numeric values for each independent variable.
    """
    # Preprocess the dependent variable
    data[dependent_var] = data[dependent_var].astype(str).str.strip()
    unique_categories = data[dependent_var].unique()
    dependent_mapping = {category: idx for idx, category in enumerate(sorted(unique_categories))}
    data[dependent_var] = data[dependent_var].map(dependent_mapping)

    # Initialize a mapping dictionary for independent variables
    independent_mappings = {}

    # Iterate over each independent variable
    for var in independent_vars:
        unique_vals = data[var].dropna().unique()

        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):  # Binary variable
            # Binary variable (already numeric and binary, no further action needed)
            independent_mappings[var] = None

        elif data[var].dtype == object or len(unique_vals) <= 50:  # Categorical variable
            # Define reference category handling
            if reference_categories and var in reference_categories:
                # Set the reference category (drop the reference category in the encoding)
                
                reference_category = reference_categories[var]
                data[var] = pd.Categorical(data[var], categories=sorted(unique_vals), ordered=True)
                data[var] = data[var].cat.set_categories(sorted(unique_vals), ordered=True)
                # Apply encoding
                data[var] = data[var].map({category: idx for idx, category in enumerate(sorted(unique_vals))})
                independent_mappings[var] = {
                    'mapping': {category: idx for idx, category in enumerate(sorted(unique_vals))},
                    'reference_category': reference_category
                }
            else:
                # Map categorical values to numeric codes (no reference category)
                data[var] = data[var].astype(str).str.strip()
                unique_categories = data[var].unique()
                mapping = {category: idx for idx, category in enumerate(sorted(unique_categories))}
                data[var] = data[var].map(mapping)
                independent_mappings[var] = mapping

        else:  # Continuous variable
            # Ensure the column is numeric
            data[var] = pd.to_numeric(data[var], errors='coerce')
            if data[var].isnull().any():
                raise ValueError(f"Independent variable '{var}' contains invalid continuous values.")
            independent_mappings[var] = None

    return data, dependent_mapping, independent_mappings



# Function to perform Generalized Linear Model (GLM) analysis
def perform_multinomial_regression(data, dependent_var, independent_vars,ref_values):
    # Check if the independent variable is binary (yes/no or 0/1)
    
    # Define the dependent and independent variables
    y = data[dependent_var]
    X = data[independent_vars]
    
    # One-hot encoding for X
    #X = pd.get_dummies(X, drop_first=True)
    # Initialize an empty list to hold the transformed independent variables
    X_transformed = []

    # Process each independent variable and its corresponding reference value
    for independent_var in independent_vars:
    
        ref_value = ref_values.get(independent_var, None)  # Get the reference value
        
        
        # Create dummy variables
        dummies = pd.get_dummies(data[independent_var])
        # Drop the reference category column if it exists in the dummies
        if ref_value  in dummies.columns:
           
           dummies.drop(columns=[ref_value], inplace=True)
           
        # Rename the columns of the dummies to include the independent variable name
        dummies.columns = [f"{independent_var}_{col}" for col in dummies.columns]
        
        # Add dummies to the list of transformed variables
        X_transformed.append(dummies)

    # Concatenate all the transformed independent variables into a single DataFrame
    X = pd.concat(X_transformed, axis=1)
    X = sm.add_constant(X)  # Add a constant to the independent variables (intercept term)
    
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Fit logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100)
    model.fit(X_scaled, y)

    # Coefficients and intercepts
    coef = model.coef_  # Shape: (n_classes, n_features)
    intercept = model.intercept_  # Shape: (n_classes,)

    # Risk Ratios (exponentiated coefficients)
    risk_ratios = np.exp(coef)

    # Confidence Intervals (mocked for simplicity, adjust based on actual calculations)
    # For example purposes, assuming ±50% bounds around coefficients
    ci_lower = np.exp(coef - 0.5)
    ci_upper = np.exp(coef + 0.5)

  
    # Organize results
    from scipy.stats import norm

    # Compute the covariance matrix and standard errors
    cov_matrix = np.linalg.pinv(np.dot(X_scaled.T, X_scaled))  # Use pseudo-inverse
    std_err = np.sqrt(np.diag(cov_matrix))  # Standard error for each feature

    # Organize results
    results = []
    classes = model.classes_  # Numeric classes (e.g., 0, 1, 2)
    predictors = X.columns  # Feature names

    for class_idx, numeric_class in enumerate(classes):
        class_label = numeric_class  # Map back to category name
        for predictor_idx, predictor in enumerate(predictors):
            coefficient = coef[class_idx, predictor_idx]
            risk_ratio = risk_ratios[class_idx, predictor_idx]
            
            # Compute standard error and Wald statistic
            std_error = std_err[predictor_idx]
            wald_statistic = coefficient / std_error
            
            # Compute p-value
            p_value = 2 * (1 - norm.cdf(abs(wald_statistic)))
            
            # Significance stars
            significance = (
                "***" if p_value < 0.001 else 
                "**" if p_value < 0.01 else 
                "*" if p_value < 0.05 else 
                ""
            )
            
            # Confidence intervals
            ci_lower_value = np.exp(coefficient - 1.96 * std_error)  # 95% CI lower bound
            ci_upper_value = np.exp(coefficient + 1.96 * std_error)  # 95% CI upper bound
            
            # Append results
            results.append([
                predictor, 
                f"{risk_ratio:.2f} {significance}", 
                f"{ci_lower_value:.2f} - {ci_upper_value:.2f}", 
                f"{coefficient:.2f}", 
                f"{std_error:.2f}", 
                f"{p_value:.4f}", 
                class_label
            ])

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results, 
        columns=["Predictors", "Risk Ratios", "95% CI", "Coefficient", "Std. Error", "p-value", "Response"]
    )

    # Add summary statistics
    summary = {
        "Observations": [len(data)],
        "R2": [f"{model.score(X, y):.3f}"],
    }
    summary_df = pd.DataFrame(summary)

    # Print results <title>Logistic Regression Results</title>
    print(results_df)

    # Save as HTML table
    table = results_df.to_html(index=False, escape=False)
    footer = "* p<0.05   ** p<0.01   *** p<0.001"

    html_content = f"""
    <html>
    <head>
        <title></title>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div style="text-align: center;">
              <h1>  <strong>{y.name} </strong> </h1>
        </div>
        {table}
        <h3>Summary</h3>
        {summary_df.to_html(index=False, escape=False)}
        <p>{footer}</p>
    </body>
    </html>
    """

    # Save HTML content
    # Save the HTML content to a temporary file and return the file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        tmp_file.write(html_content.encode())
        tmp_file_path = tmp_file.name

    return tmp_file_path,results_df


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



def perform_chisquare_analysis(data, var1, var2):
    contingency_table = pd.crosstab(data[var1].astype(str), data[var2].astype(str))
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    observed = contingency_table.values

    st.write("## Chi-Square Test Results")
    st.write(f"Chi-Square Statistic: {chi2:.2f}")
    st.write(f"p-value: {p:.4f}")
    st.write(f"Degrees of Freedom: {dof}")

    # Balloon Plot for the observed values in the contingency table
    st.write("### Balloon Plot of Contingency Table")
    create_plot_with_download(lambda: create_balloon_plot(contingency_table), 'Balloon Plot of Contingency Table')

    # Residuals Plot
    residuals = observed - expected.reshape(contingency_table.shape)
    residuals_df = pd.DataFrame(residuals, columns=contingency_table.columns, index=contingency_table.index)
    st.write("### Residual Plot")
    create_plot_with_download(lambda: create_balloon_plot(residuals_df), 'Residual Plot')

    # Chi-Square Contribution Plot
    contribution = (residuals) ** 2 / expected.reshape(contingency_table.shape)
    contribution_df = pd.DataFrame(contribution, columns=contingency_table.columns, index=contingency_table.index)
    st.write("### Chi-Square Contribution Plot")
    create_plot_with_download(lambda: create_balloon_plot(contribution_df), 'Chi-Square Contribution Plot')

    # Chi-Square Ratio Plot
    ratio = observed / expected.reshape(contingency_table.shape)
    ratio_df = pd.DataFrame(ratio, columns=contingency_table.columns, index=contingency_table.index)
    st.write("### Chi-Square Ratio Plot")
    create_plot_with_download(lambda: create_balloon_plot(ratio_df), 'Chi-Square Ratio Plot')


def create_heatmap(data, title, label, fmt='d', center=None):
    """Helper function to create a heatmap with improved layout."""
    plt.figure(figsize=(10, 8))  # Adjust the figure size
    sns.heatmap(data, annot=True, fmt=fmt, cmap='coolwarm', center=center, cbar_kws={"label": label},
                annot_kws={"size": 8}, xticklabels=True, yticklabels=True)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='center')  # Rotate x-axis labels for clarity
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()  # Adjust layout to prevent clipping



def create_balloon_plot(data):
    """Creates a balloon plot with circles of varying sizes inside each cell."""
    plt.figure(figsize=(6, 6))  # Adjust the figure size
    ax = plt.gca()

    # We want to set up an x and y coordinate grid based on the shape of the contingency table
    rows, cols = data.shape
    x_pos, y_pos = np.meshgrid(range(cols), range(rows))

    # Flatten the meshgrid for scatter plot
    x_pos = x_pos.flatten()  # Keep x positions at cell center
    y_pos = y_pos.flatten()  # Keep y positions at cell center
    values = data.values.flatten()

    # Define sizes for the circles based on the values
    sizes = np.clip(values * 50, 50, 2000)  # Clamp the size to avoid too large circles (e.g., 50 to 2000)

    # Create scatter plot (balloon plot)
    scatter = ax.scatter(x_pos + 0.5, y_pos + 0.5, s=sizes, c=values, cmap='coolwarm', edgecolors="w", alpha=0.6)

    # Annotate each circle with its value
    for i, value in enumerate(values):
        ax.annotate(f'{value:.2f}', (x_pos[i] + 0.5, y_pos[i] + 0.5),
                    ha='center', va='center', fontsize=10, color='black')

    # Set the axis labels and title
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(data.columns, rotation=45,ha="center")
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels(data.index,va="center",rotation=60)
    ax.set_title('Balloon Plot of Contingency Table', fontsize=16)

    # Set the axis limits to ensure the circles are properly contained inside cells
    ax.set_xlim(0, cols)  # Fix the x-axis limit to the number of columns
    ax.set_ylim(0, rows)  # Fix the y-axis limit to the number of rows

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Count')

    # Add gridlines to show the cell lines
    ax.grid(True, which='both', axis='both', linestyle='-', color='black', linewidth=1)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()



# Function to perform Generalized Linear Model (GLM) analysis
def perform_glm_analysis(data, dependent_var,dref_value,ref_category):
    data.columns = data.columns.str.strip()
    # Verify if the 'Bedding_Material' column is in the independent variables list
    print("Columns in X:", data.columns)
    
    # Check if the specific column exists in X
    if 'Bedding_Material' in data.columns:
        print("'Bedding_Material' found in X.")
    else:
        print("'Bedding_Material' not found in X.")
    
    
    # Reorder the categories to set the reference first
    # Filter out rows where the dependent variable equals the reference value
    
    # Ensure the dependent variable is binary and map it accordingly
    if dependent_var in data.columns:
        # Identify unique values in the dependent variable
        unique_vals = data[dependent_var].dropna().unique()

        # If there are two unique values, treat it as binary
        if len(unique_vals) == 2:
            # Automatically map the unique values to 0 and 1
            value_map = {dref_value: 0, [val for val in unique_vals if val != dref_value][0]: 1}
            #value_map = {unique_vals[0]: 1, unique_vals[1]: 1}
            data[dependent_var] = data[dependent_var].map(value_map)
        else:
            raise ValueError(f"Dependent variable '{dependent_var}' is not binary, it has {len(unique_vals)} unique values.")
    
    # Define the dependent and independent variables
    y = data[dependent_var]
    independent_vars=list(ref_category.keys())
    X = data[independent_vars]
 
    
    # Reorder the categories to set the reference first
    for col, ref_val in ref_category.items():
             X[col] = pd.Categorical(X[col], categories=[ref_val] + [x for x in X[col].unique() if x != ref_val])
    
    X = pd.get_dummies(X, drop_first=True)
    

    #X = sm.add_constant(X)  # Add a constant to the independent variables (intercept term)
    # Ensure all columns in X are numeric
    X = X.astype(float)
   
    # Split the data into training and testing sets
    # Scale features to avoid overflow in calculations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train=X
    y_train=y
    #X_train = X_train.astype(float)

    # Fit logistic regression model using statsmodels
    # Fit the logistic regression model using statsmodels Logit
   # Fit the logistic regression model using statsmodels Logit
    # Fit the logistic regression model using statsmodels Logit
    model = sm.Logit(y_train, sm.add_constant(X_train))  # Add constant for intercept
    fit = model.fit()
    

    
    # Fit the logistic regression model using statsmodels Logit
    #model = sm.Logit(y_train, sm.add_constant(X_train))  # Add constant for intercept
    #fit = model.fit()
    
    # Get the coefficients and other statistics
    coef = fit.params  # Coefficients for the model
    std_err = fit.bse  # Standard errors of the coefficients
    p_values = fit.pvalues  # p-values for the coefficients
    
    # Risk Ratios (exponentiated coefficients)
    risk_ratios = np.exp(coef)
    
    # Standardized Coefficients (Std. Beta)
    X_std = (X_train - X_train.mean()) / X_train.std()  # Standardize the predictors
    model_std = sm.Logit(y_train, sm.add_constant(X_std))  # Add constant and fit the model
    fit_std = model_std.fit()
    std_beta = fit_std.params  # Standardized coefficients
    
    # Standardized Std. Error
    std_err_std = fit_std.bse  # Standard errors for standardized coefficients
    
    # Confidence Intervals (using the standard errors and a normal distribution)
    ci_lower = np.exp(coef - 1.96 * std_err)
    ci_upper = np.exp(coef + 1.96 * std_err)
    
    # Standardized Confidence Intervals
    ci_lower_std = np.exp(std_beta - 1.96 * std_err_std)
    ci_upper_std = np.exp(std_beta + 1.96 * std_err_std)
    
    # Wald Statistics
    wald_statistic = coef / std_err  # Wald statistic for each coefficient
    
    # Organize results
    results = []
    predictors = X_train.columns 
    
    # Include 'Intercept' explicitly
    predictors = ['Intercept'] + list(predictors)

    # Loop through each predictor to extract relevant statistics
    for predictor_idx, predictor in enumerate(predictors):
        if predictor == 'Intercept':
            coefficient = coef.iloc[0]  # Intercept is the first element in coef
            risk_ratio = risk_ratios.iloc[0]
            std_error = std_err.iloc[0]
            std_beta_value = std_beta.iloc[0]
            std_error_std_value = std_err_std.iloc[0]
            ci_lower_value = ci_lower.iloc[0]
            ci_upper_value = ci_upper.iloc[0]
            ci_lower_std_value = ci_lower_std.iloc[0]
            ci_upper_std_value = ci_upper_std.iloc[0]
            wald_stat = wald_statistic.iloc[0]
        else:
            # For all other predictors
            coefficient = coef.iloc[predictor_idx]
            risk_ratio = risk_ratios.iloc[predictor_idx]
            std_error = std_err.iloc[predictor_idx]
            std_beta_value = std_beta.iloc[predictor_idx]
            std_error_std_value = std_err_std.iloc[predictor_idx]
            ci_lower_value = ci_lower.iloc[predictor_idx]
            ci_upper_value = ci_upper.iloc[predictor_idx]
            ci_lower_std_value = ci_lower_std.iloc[predictor_idx]
            ci_upper_std_value = ci_upper_std.iloc[predictor_idx]
            wald_stat = wald_statistic.iloc[predictor_idx]
    
        # Significance stars
        p_value = p_values.iloc[predictor_idx]
        significance = (
            "***" if p_value < 0.001 else 
            "**" if p_value < 0.01 else 
            "*" if p_value < 0.05 else 
            ""
        )
        
        
        # Append the result for the predictor
        results.append([
            predictor, 
            f"{risk_ratio:.2f} {significance}", 
            f"{std_error:.2f}", 
            f"{std_beta_value:.2f}",
            f"{std_error_std_value:.2f}",
            f"{ci_lower_value:.2f} - {ci_upper_value:.2f}",
            f"{ci_lower_std_value:.2f} - {ci_upper_std_value:.2f}",
            f"{wald_stat:.2f}"
        ])
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results, 
        columns=["Predictors", "Risk Ratios", "Std. Error", "Std. Beta", "Standardized Std. Error", "CI", "Standardized CI", "Statistic"]
    )
    
    # Add summary statistics
    summary = {
        "Observations": [len(y_train)],
        "R2": [f"{fit.prsquared:.3f}"],  # Pseudo R-squared for Logit model
    }
    summary_df = pd.DataFrame(summary)
    
    # Print results <title>Logistic Regression Results</title>
    print(results_df)
    
    # Save as HTML table
    table = results_df.to_html(index=False, escape=False)
    footer = "* p<0.05   ** p<0.01   *** p<0.001"
    
    # HTML content for display
    html_content = f"""
    <html>
    <head>
        <title>Logistic Regression Results</title>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div style="text-align: center;">
            <h1><strong>{y_train.name}</strong></h1>
        </div>
        {table}
        <h3>Summary</h3>
        {summary_df.to_html(index=False, escape=False)}
        <p>{footer}</p>
    </body>
    </html>
    """
    html_file_path = "output_tableuzmaEducation.html"
    with open(html_file_path, "w") as file:
        file.write(html_content)
    # Saving the HTML content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        tmp_file.write(html_content.encode())  # Write the HTML content to the file
        tmp_file_path = tmp_file.name  # Get the file path
    
    # Returning the file path and table as output
    return tmp_file_path, results_df
def main():
    st.title("Survey Analysis Tool")

    # Upload Data
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display the first few rows of the dataset
        st.write("### Data Preview", data)

        # Select Analysis Type
        analysis_type = st.selectbox("Select Analysis Type", ["Select Analysis Type", "Chi-Square", "GLM","Multinomial Regression"])

        if analysis_type != "Select Analysis Type":
            # Run Analysis Button
            if analysis_type == "Chi-Square":
                # Variables for Chi-Square
                var1 = st.selectbox("Select First Variable for Chi-Square", options=data.columns)
                var2 = st.selectbox("Select Second Variable for Chi-Square", options=data.columns)
                
                if st.button("Run Chi-Square Analysis"):
                    data = data.dropna(subset=[var1, var2])
                    perform_chisquare_analysis(data, var1, var2)

            elif analysis_type == "GLM":
                # Variables for GLM
                st.write("### Select Variables for GLM Analysis")
                dependent_var = st.selectbox("Select Dependent Variable", options=data.columns)
                # Filter binary variables: columns with exactly two unique values
               # binary_vars_options = [col for col in data.columns if data[col].nunique() == 2]
                dref_value = st.selectbox(f"Select Reference Value for {dependent_var}", options=data[dependent_var].unique())
                # Ensure the dependent variable is in the binary variables options
                #if dependent_var not in binary_vars_options:
                   # binary_vars_options.append(dependent_var)
                
                # Binary variable selection with valid default
                #binary_vars = st.multiselect("Select Binary Variables", options=binary_vars_options, default=[dependent_var])

                # Independent variables
                #independent_vars = st.multiselect("Select Independent Variables", options=data.columns, default=data.columns.tolist())
                # Streamlit selectbox to select the variable (e.g., 'Age_Group')
                independent_var = st.selectbox("Select Independent Variable", options=data.columns)
                # Streamlit selectbox to select the reference value (e.g., 'L30')
                # Preprocess data: Drop rows with null values in dependent or independent variables
                data = data.dropna(subset=[dependent_var, independent_var])
                #ref_value = st.selectbox(f"Select Reference Value for {independent_var}", options=data[independent_var].unique())
                ref_value = st.selectbox(
                    f"Select Reference Value for {independent_var}", 
                    options=data[independent_var].unique(),
                    key=f"ref_value_{independent_var}"  # Add a unique key for each selectbox
                    )

                #ref_value=int(ref_value)
                # Store the result in the desired form
                independent_vars = {independent_var: ref_value}
               
                st.write("### Data Preview", data)
                if st.button("Run GLM Analysis"):
                    if dependent_var  and independent_vars:
                          # Check type of selected variable

                        #data = preprocess_data(data, dependent_var, binary_vars)  # Preprocess data before analysis
                        html_file_path,table = perform_glm_analysis(data, dependent_var,dref_value, independent_vars)  # GLM analysis
                        # Provide a download button for the HTML file
                        st.write(table)
                        st.download_button(
                        label="Download GLM Results as HTML",
                        data=open(html_file_path, "rb").read(),
                        file_name="glm_results.html",
                        mime="text/html"
                    )
                    else:
                        st.warning("Please make sure to select both dependent and independent variables for GLM analysis.")
            elif analysis_type == "Multinomial Regression":
                # Variables for Multinomial Regression
                st.write("### Select Variables for Multinomial Regression")
                dependent_var = st.selectbox("Select Dependent Variable", options=data.columns)
                independent_vars = st.multiselect("Select Independent Variables", options=data.columns)
                data = data.dropna(subset=[dependent_var] + independent_vars)
                ref_values = {}
                for independent_var in independent_vars:
                    ref_values[independent_var] = st.selectbox(f"Select Reference Value for {independent_var}", options=data[independent_var].unique())

                print(ref_values)
                # ref_values will be a dictionary with the reference values for each independent variable
                if st.button("Run Multinomial Regression"):
                    if dependent_var and independent_vars :
                       #data,dependent_mapping, independent_mappings = Multipreprocess_data(data, dependent_var, independent_vars)
                       results_file,table = perform_multinomial_regression(data, dependent_var, independent_vars,ref_values)
                       st.write(table)
                       st.download_button(
                            label="Download Multinomial Regression Results",
                            data=open(results_file, "rb").read(),
                            file_name="multinomial_regression_results.html",
                            mime="text/html"
                        )
                    else:
                        st.warning("Please select both dependent and independent variables.")

# Run the app
if __name__ == "__main__":
    main()