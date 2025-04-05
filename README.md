# Real_Estate_Solution
This project implements Random Forest Regressor models for predicting housing prices, with a focus on robust data processing, feature engineering, and model evaluation.

## Features
Comprehensive data preprocessing pipeline with logging and error handling
Feature engineering optimized for regression tasks
Model evaluation metrics and analysis
Robust production-ready code structure

## Data Processing
The project includes robust data loading and preprocessing with:

Error handling for missing files and data corruption
Logging of all data transformations
Data validation checks

## Feature Engineering
Feature engineering includes:

Creation of dummy variables for categorical features
Feature selection and transformation
Input validation and error handling

## Technologies Used
Python: Core programming language
Pandas and NumPy: Data manipulation and numerical operations
Scikit-learn: Machine learning algorithms and evaluation
Matplotlib and Seaborn: Data visualization
Logging: Built-in Python module for application logging

## Model
The predictive model is trained using the Real Estate dataset. It applies preprocessing steps like encoding categorical variables and scaling numerical features. The Regression model used Random Forest.


## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/paulhehehe/regression_application-main
   cd Regression_Models_Solution

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

#### Thank you for using the Real_Estate_Solution Application! Feel free to share your feedback.
