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
Streamlit: Used to create an intuitive and interactive web interface for the application, enabling users to upload datasets, select analysis types, and visualize results with minimal technical effort.
Pandas: Utilized for data preprocessing, manipulation, and analysis, allowing seamless handling of tabular data such as CSV files.
Statsmodels: Applied for conducting advanced statistical analyses, including Generalized Linear Models (GLM) and Multinomial Logistic Regression.
SciPy: Used for implementing Chi-Square Tests to assess statistical significance in contingency tables.
Matplotlib/Seaborn: Incorporated for generating visualizations of the results, enhancing interpretability and clarity.
