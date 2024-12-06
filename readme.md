# Customer Churn Prediction in the Telecom Industry

## Overview
This repository contains all the resources and deliverables for the project **"Customer Churn Prediction in the Telecom Industry"**, completed as part of **DATASCI 3000A - Introduction to Machine Learning** (Fall 2024). The project aims to predict customer churn in the telecom industry using machine learning techniques and data analytics. The results are summarized in the final report provided in this repository.

## Repository Structure
The repository is organized as follows:

Project/ ├── Dataset/ │ ├── archive (1).zip # Compressed raw dataset │ ├── Customer Churn Dataset.csv # Cleaned dataset used for analysis ├── Python Notebooks/ │ ├── catboost_info/ │ │ ├── learn/ │ │ │ └── events.out.tfevents # CatBoost training logs │ │ ├── tmp/ │ │ ├── catboost_training.json # CatBoost training parameters │ │ ├── learn_error.tsv # Training error metrics │ │ ├── time_left.tsv # Time estimation logs │ ├── ML Team 5 Group Project.ipynb # Team's collaborative Jupyter Notebook │ ├── ML_CVUpdatedCode.ipynb # Final notebook with updated model code ├── Report/ │ └── Group5_IEEE_Draft2_CustomerChurnReport.pdf # Final project report


## Project Summary
**Title:** Customer Churn Prediction in the Telecom Industry  
**Authors:** Dani Alex Parayil, Garima Gambhir, Ritika Pandey, Semal Shastri, Sumedha  

### Abstract
This project addresses the challenge of customer churn in the telecom industry by employing machine learning techniques to predict at-risk customers and identify the underlying factors contributing to churn. Models including Logistic Regression, Support Vector Machines (SVC), Random Forest, Gradient Boosting, AdaBoost, and CatBoost were used for analysis, with CatBoost emerging as the best-performing model.

### Key Highlights:
- **Dataset:** The dataset includes 7043 customer records with 20 features detailing customer demographics, service usage, satisfaction, and contract types.
- **Preprocessing:** Steps included handling missing data, encoding categorical variables, and scaling numerical features.
- **Models Used:** Logistic Regression, SVC, Random Forest, Gradient Boosting, AdaBoost, and CatBoost.
- **Evaluation Metrics:** Precision, Recall, F1-Score, Balanced Accuracy, AUC-ROC, and Geometric Mean.
- **Best Model:** CatBoost, due to its handling of categorical variables, robustness with imbalanced datasets, and strong performance metrics.

## Files and Notebooks
- **Dataset:** 
  - `Customer Churn Dataset.csv` contains the cleaned data used for analysis.
- **Notebooks:** 
  - `ML Team 5 Group Project.ipynb` includes EDA, preprocessing, and model training for all algorithms.
  - `ML_CVUpdatedCode.ipynb` contains the final updated code with hyperparameter tuning and cross-validation.
- **Reports:**
  - `Group5_IEEE_Draft2_CustomerChurnReport.pdf` is the final report detailing the methodology, results, and analysis.

## How to Run the Code
1. Clone the repository.
2. Install required Python libraries: `pip install -r requirements.txt`.
3. Open the Jupyter notebooks in the `Python Notebooks/` directory to run the analyses.
4. Refer to `Group5_IEEE_Draft2_CustomerChurnReport.pdf` for detailed explanations of the methodology and results.

## Results
- **Best Performing Model:** CatBoost
- **Key Insights:** 
  - Month-to-month contracts and internet service usage are major factors contributing to churn.
  - Senior citizens and customers without partners show higher churn rates.
  - Retention strategies should focus on high-risk demographics and contract types.

## Future Work
- Hyperparameter optimization using grid search and Bayesian methods.
- Exploring deep learning models for improved performance.
- Enhancing interpretability with tools like SHAP or LIME.

## Team Contributions
1. **Dani Alex Parayil:** EDA, data visualization, CatBoost implementation, and final report preparation.
2. **Garima Gambhir:** Logistic Regression analysis and data preprocessing.
3. **Ritika Pandey:** SVC implementation and report writing.
4. **Semal Shastri:** Random Forest analysis and preprocessing.
5. **Sumedha:** Gradient Boosting, AdaBoost implementation, and final code integration.

## Contact
For further inquiries, please contact:  
- Dani Alex Parayil: dparayil@uwo.ca
- Garima Gambhir: ggambhi@uwo.ca
- Ritika Pandey: rpande6@uwo.ca
- Semal Shastri: sshastr@uwo.ca
- Sumedha: sgalgali@uwo.ca
