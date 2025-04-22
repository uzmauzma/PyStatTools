<h1>PyStatTools: A Python Package for Statistical Analysis of Survey Data</h1>
<h3> Overview </h3>
<p>The Survey Analysis Tool is a Python-based application designed for performing various statistical analyses,
  including Chi-Square Tests, Generalized Linear Models (GLM), and Multinomial Logistic Regression, on survey data.
  It provides a user-friendly web interface built with Streamlit that allows users 
  to upload survey datasets, choose analysis types, select variables, and visualize results.</p>

<p>This tool is developed to cater to researchers and data analysts who may not have extensive programming
knowledge but need an accessible method to perform advanced statistical analyses on survey data. The results 
are presented in an easy-to-understand format, enabling users to interpret statistical significance and other key metrics like Risk Ratios and Confidence 
Intervals (CI). </p>
<h1>Methodology</h1>

<h2>Tool Design and Architecture</h2>

<p>The Survey Analysis Tool was designed using Python as the primary programming language, leveraging its robust ecosystem for data analysis and web application development. The architecture follows a systematic flow that guides users through each step of the analysis.</p>

<h2>Frameworks and Libraries Used </h2>

<p>Streamlit: Used to create an intuitive and interactive web interface for the application, enabling users to upload datasets, select analysis types, and visualize results with user-friendly.</p>

<ul>
  <li><strong>Streamlit:</strong> Creates an interactive web interface for dataset uploads and result visualization.</li>
  <li><strong>Pandas:</strong> Handles data manipulation and preprocessing tasks.</li>
  <li><strong>Statsmodels:</strong> Executes advanced statistical analyses such as GLM and Multinomial Logistic Regression.</li>
  <li><strong>SciPy:</strong> Performs Chi-Square Tests for contingency table analysis.</li>
  <li><strong>Matplotlib/Seaborn:</strong> Generates plots to enhance data interpretability.</li>
</ul>

<h2>Tool Workflow</h2>

<ol>
  <li><strong>File Upload:</strong> Users upload and preview survey data in CSV format.</li>
  <li><strong>Analysis Selection:</strong> Users choose statistical methods: Chi-Square, GLM, or Multinomial Logistic Regression.</li>
  <li><strong>Variable Selection:</strong> Users select dependent and independent variables and define reference categories.</li>
  <li><strong>Analysis Execution:</strong> The tool performs calculations and generates detailed results tables, 
      including Risk Ratios, CIs, and p-values.</li>
  <li><strong>Results Display:</strong> Results are presented in both tabular and graphical formats, 
      with options to download for offline use.</li>
</ol>

<h1>How to Use the Tool</h1>

<ol>
  <li>Clone the repository or download the project files.</li>
  <li>Install required dependencies:
    <pre style="background-color: #f4f4f4; color: #333;">pip install -r requirements.txt</pre>
  </li>
  <li>Launch the Streamlit app:
    <pre style="background-color: #f4f4f4; color: #333;">streamlit run surveytool.py</pre>
  </li>
  <li>Follow the interface to upload datasets, select analysis methods, and view results.</li>
</ol>

<h2>Contributions</h2>
<p>This tool is open-source and available on GitHub for public use and collaboration. Contributions are welcome! If you would like to contribute to the development of the tool, please feel free to submit a pull request or open an issue on the GitHub repository.</p>

<p>4. Follow the interface to upload your dataset, select the desired analysis method, and view the results.</p>

![Figure 1](https://github.com/uzmauzma/Survaytool/blob/main/Fig/App.png)
<P>Figure 1: Graphical user interface (GUI) for survey analysis.  A) The layout where users can upload a CSV file. B) The interface for selecting an analysis method.</P>

![Figure 2](https://github.com/uzmauzma/Survaytool/blob/main/Fig/App2.png)
<P>Figure 2: The variable selection interface where users can choose dependent and independent variables.</P>

##  Results

<h2>Figures</h2>

![Figure 3](https://github.com/uzmauzma/Survaytool/blob/main/Fig/SEX_TIT.png)

<p>Figure 3: Balloon plot of Chi-Square contributions and Chi-Square ratio plot for the Titanic dataset. Survival status (Survived) and sex (gender), such as female, show strong contributions to survival outcomes, as indicated by balloon size and colour. </p>

![Figure 4](https://github.com/uzmauzma/Survaytool/blob/main/Fig/pclass_TIT.png)

<p>Figure 4: Survival rates by passenger class in the Titanic dataset. First-class passengers show the highest survival rates, followed by second-class, with third-class passengers having the lowest survival rates.</p>

![Figure 5](https://github.com/uzmauzma/Survaytool/blob/main/Fig/hert_cp_ca.png)

<p>Figure 5: Relationship between Chest Pain Type and NumColoredVessels in the Cleveland Heart Disease dataset. Larger balloons indicate stronger associations, with "Non-Anginal Pain" and "Asymptomatic" categories contributing significantly to the Chi-Square statistic.</p>

![Figure 6](https://github.com/uzmauzma/Survaytool/blob/main/Fig/hert_cp_tal.png)
<p>Figure 6: Relationship between Chest Pain Type and Thalassemia in the Cleveland Heart Disease dataset. Categories such as "Asymptomatic," "Normal (Thalassemia)," and "Reversible Defect" exhibit strong contributions, highlighting their diagnostic significance.</p>

## Table 2: Logistic Regression Results for Titanic Dataset

| Predictors | Risk Ratios | Std. Error | Std. Beta | Standardized Std. Error | CI          | Standardized CI | Statistic |
|------------|-------------|------------|-----------|--------------------------|-------------|-----------------|-----------|
| Intercept  | 0.32 ***    | 0.11       | -0.52     | 0.07                     | 0.26 - 0.39 | 0.51 - 0.69     | -10.82    |
| Pclass_1   | 5.31 ***    | 0.18       | 0.72      | 0.08                     | 3.76 - 7.50 | 1.77 - 2.37     | 9.5       |
| Pclass_2   | 2.80 ***    | 0.18       | 0.42      | 0.07                     | 1.96 - 4.00 | 1.31 - 1.75     | 5.68      |
|Observations|  8911            
|R²          |  0.087 
|* p<0.05 ** p<0.01 *** p<0.001         |        


## Table 3: Logistic Regression Results for Heart Disease Dataset

| Predictors              | Risk Ratios | Std. Error | Std. Beta | Standardized Std. Error | CI             | Standardized CI  | Statistic |
|--------------------------|-------------|------------|-----------|--------------------------|----------------|------------------|-----------|
| Intercept               | 0.21 **     | 0.55       | -0.78     | 0.13                     | 0.07 - 0.62    | 0.36 - 0.59      | -2.83     |
| cp_asymptomatic         | 1.83        | 0.58       | 0.30      | 0.29                     | 0.59 - 5.70    | 0.77 - 2.39      | 1.04      |
| cp_non-anginal          | 3.26 *      | 0.59       | 0.53      | 0.27                     | 1.02 - 10.41   | 1.01 - 2.88      | 2.00      |
| cp_atypical angina      | 2.59        | 0.62       | 0.36      | 0.23                     | 0.76 - 8.79    | 0.90 - 2.26      | 1.53      |
|Observations|  304           
|R²          |  0.018
|* p<0.05 ** p<0.01 *** p<0.001         |  

## Table 4: Multinomial Regression Results for Passenger Class (PClass) and Survival Status

| Predictors    | Risk Ratios | 95% CI          | Coefficient | Std. Error | p-value | Response |
|---------------|-------------|-----------------|-------------|------------|---------|----------|
| const         | 1           | 1.00 - 1.00     | 0           | 0          | nan     | 1        |
| Survived_1    | 1.56 ***    | 1.44 - 1.68     | 0.44        | 0.04       | 0       | 1        |
| Sex_female    | 0.88 **     | 0.82 - 0.95     | -0.12       | 0.04       | 0.0018  | 1        |
| const         | 1           | 1.00 - 1.00     | 0           | 0          | nan     | 2        |
| Survived_1    | 1.04        | 0.96 - 1.12     | 0.04        | 0.04       | 0.3425  | 2        |
| Sex_female    | 1.05        | 0.97 - 1.13     | 0.05        | 0.04       | 0.2386  | 2        |
| const         | 1           | 1.00 - 1.00     | 0           | 0          | nan     | 3        |
| Survived_1    | 0.62 ***    | 0.57 - 0.67     | -0.48       | 0.04       | 0       | 3        |
| Sex_female    | 1.08        | 1.00 - 1.17     | 0.08        | 0.04       | 0.0527  | 3        |
|Observations|  891            
|R²          |  0.549 
|* p<0.05 ** p<0.01 *** p<0.001         |  

## Table 5: Multinomial Regression Results for Chest Pain Type (CP)

| Predictors                 | Risk Ratios | 95% CI          | Coefficient | Std. Error | p-value   | Response          |
|----------------------------|-------------|-----------------|-------------|------------|-----------|-------------------|
| const                      | 1           | 1.00 - 1.00     | 0           | 0          | nan       | asymptomatic      |
| restecg_normal             | 0.85 **     | 0.76 - 0.95     | -0.16       | 0.06       | 0.0055    | asymptomatic      |
| restecg_st-t abnormality   | 1.23 ***    | 1.10 - 1.38     | 0.21        | 0.06       | 0.0004    | asymptomatic      |
| thal_normal                | 0.65 ***    | 0.51 - 0.83     | -0.43       | 0.12       | 0.0005    | asymptomatic      |
| thal_reversable defect     | 1.09        | 0.85 - 1.39     | 0.09        | 0.12       | 0.4916    | asymptomatic      |
| const                      | 1           | 1.00 - 1.00     | 0           | 0          | nan       | atypical angina   |
| restecg_normal             | 1.26 ***    | 1.13 - 1.42     | 0.23        | 0.06       | 0.0001    | atypical angina   |
| restecg_st-t abnormality   | 0.82 ***    | 0.73 - 0.92     | -0.2        | 0.06       | 0.0006    | atypical angina   |
| thal_normal                | 1.27        | 1.00 - 1.62     | 0.24        | 0.12       | 0.0528    | atypical angina   |
| thal_reversable defect     | 0.84        | 0.66 - 1.08     | -0.17       | 0.12       | 0.1696    | atypical angina   |
| const                      | 1           | 1.00 - 1.00     | 0           | 0          | nan       | non-anginal       |
| restecg_normal             | 1.17 **     | 1.04 - 1.31     | 0.15        | 0.06       | 0.0085    | non-anginal       |
| restecg_st-t abnormality   | 1.19 **     | 1.06 - 1.33     | 0.17        | 0.06       | 0.0035    | non-anginal       |
| thal_normal                | 1.52 ***    | 1.19 - 1.94     | 0.42        | 0.12       | 0.0008    | non-anginal       |
| thal_reversable defect     | 1.25        | 0.98 - 1.59     | 0.22        | 0.12       | 0.0762    | non-anginal       |
| const                      | 1           | 1.00 - 1.00     | 0           | 0          | nan       | typical angina    |
| restecg_normal             | 0.80 ***    | 0.71 - 0.89     | -0.23       | 0.06       | 0.0001    | typical angina    |
| restecg_st-t abnormality   | 0.84 **     | 0.75 - 0.94     | -0.18       | 0.06       | 0.0026    | typical angina    |
| thal_normal                | 0.8         | 0.63 - 1.02     | -0.22       | 0.12       | 0.0714    | typical angina    |
| thal_reversable defect     | 0.87        | 0.69 - 1.11     | -0.13       | 0.12       | 0.2769    | typical angina    |
|Observations|  301            
|R²          |  0.502  
|* p<0.05 ** p<0.01 *** p<0.001     

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE.txt](./LICENSE.txt) file for details.

## Citation

If you use this software in your research, please cite it as follows:

Uzma. (2025). PyStatTools: A Python Package for Statistical Analysis of Survey Data. Available at: [https://github.com/uzmauzma/Tool](https://github.com/uzmauzma/PyStatTools).


