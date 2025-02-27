<h1>Survey Analysis Tool</h1>
<h3> Overview </h3>
<p1>The Survey Analysis Tool is a Python-based application designed for performing various statistical analyses,
  including Chi-Square Tests, Generalized Linear Models (GLM), and Multinomial Logistic Regression, on survey data.
  It provides a user-friendly web interface built with Streamlit that allows users 
  to upload survey datasets, choose analysis types, select variables, and visualize results.</p1>

<p>The tool is developed to cater to researchers and data analysts who may not have extensive programming
knowledge but need an accessible method to perform advanced statistical analyses on survey data. The results 
are presented in an easy-to-understand format, enabling users to interpret statistical significance and other key metrics like Risk Ratios and Confidence 
Intervals (CI). </p1>
<h1>Methodology</h1>

<h2>Tool Design and Architecture</h2>

<p>The Survey Analysis Tool was designed using Python as the primary programming language, leveraging its robust ecosystem for data analysis and web application development. The architecture follows a systematic flow that guides users through each step of the analysis.</p>

<h2>Frameworks and Libraries Used </h2>

<p>Streamlit: Used to create an intuitive and interactive web interface for the application, enabling users to upload datasets, select analysis types, and visualize results with minimal technical effort.</p>

<p>Pandas: Utilized for data preprocessing, manipulation, and analysis, allowing seamless handling of tabular data such as CSV files.</p>

<p>Statsmodels: Applied for conducting advanced statistical analyses, including Generalized Linear Models (GLM) and Multinomial Logistic Regression.</p>

<p>SciPy: Used for implementing Chi-Square Tests to assess statistical significance in contingency tables.</p>

<p>Matplotlib/Seaborn: Incorporated for generating visualizations of the results, enhancing interpretability and clarity.</p>

<h2>Tool Workflow</h2>

The tool follows a step-by-step process to ensure a smooth user experience:

<p>1. File Upload: Users upload a CSV file containing the survey data, which is then previewed for correctness.</p>

<p>2.Analysis Selection: Users select one of three statistical methods: Chi-Square, GLM, or Multinomial Logistic Regression.</p>

<p>3.Variable Selection: Users select dependent and independent variables, along with reference categories where applicable.</p>

<p>4.Analysis Execution: The selected statistical method is applied, and the tool outputs a detailed results table, which includes metrics like Risk Ratios, Confidence Intervals (CI), and statistical significance.</p>

<p>5.Results Display: The tool presents the results in both tabular and summarized forms, and users can download the summarized results in CSV, Excel, or HTML formats for offline use.</p>
<h1>How to Use the Tool</h1>

<p? 1. Clone the repository or download the project.</p>

<p> 2. Install required dependencies:</p>
<pre style="background-color: #000; color: #fff;">pip install requirements</pre>
<p>3. Launch the Streamlit app: </p>
<pre style="background-color: #000; color: #fff;">streamlit run survaytool.py</pre>

<h2>Contributions</h2>
<p>This tool is open-source and available on GitHub for public use and collaboration. Contributions are welcome! If you would like to contribute to the development of the tool, please feel free to submit a pull request or open an issue on the GitHub repository.</p>

<p>4. Follow the interface to upload your dataset, select the desired analysis method, and view the results.</p>
<h2>Figures</h2>


## 6. Results

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




