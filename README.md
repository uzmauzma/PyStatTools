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

<p 1. Clone the repository or download the project.</p>

<p> 2. Install required dependencies:</p>
<pre style="background-color: #000; color: #fff;">pip install requirements</pre>
<p>Launch the Streamlit app: </p>
<pre style="background-color: #000; color: #fff;">streamlit run survaytool.py</pre>


<p>4. Follow the interface to upload your dataset, select the desired analysis method, and view the results.</p>
<h2>Figures</h2>
<ul>
  <li>Analyze the network for global network properties.</li>
