Customer Churn Prediction in the Telecom Industry

Overview

This project aims to mitigate customer churn in the telecom industry by identifying at-risk customers using data analytics and machine learning techniques. The analysis involves various models like Logistic Regression, Support Vector Machines, Random Forest, Gradient Boosting, AdaBoost, and CatBoost to predict churn risk, with a focus on model evaluation and selecting the best performing model based on recall and balanced accuracy.

Repository Structure

The repository is organized as follows:

Dataset

Dataset/

archive (1).zip: Contains the original dataset files.

Customer Churn Dataset.csv: The main dataset used for the analysis.

Python Notebooks

Python Notebooks/

Customer Churn Dataset.csv: Another copy of the dataset, used in some analysis.

ML Team 5 Group Project.ipynb: Jupyter notebook containing the full analysis and modeling of customer churn.

ML_CVUpdatedCode.ipynb: Notebook containing updated cross-validation code and metrics.

catboost_info/

learn/ and tmp/: CatBoost training logs and metadata.

catboost_training.json: JSON file containing details of the CatBoost model training.

learn_error.tsv, time_left.tsv: Files that record training error metrics and time estimates during CatBoost training.

Report

Report/

Group5_IEEE_Draft2_CustomerChurnReport.pdf: Final report detailing the project, findings, and analysis.

Project Details

1. Data Pre-processing

The dataset used has 7043 customer records with 20 features, broadly categorized into customer demographics, service usage, satisfaction level, and contract types.

Missing values were treated appropriately (e.g., replacing with mean values), and irrelevant features were dropped to improve the model's accuracy.

Features were scaled and categorical variables encoded using one-hot and label encoding.

2. Machine Learning Models

The following machine learning models were implemented and evaluated for predicting customer churn:

Logistic Regression: Used as the base model due to its simplicity and interpretability.

Support Vector Classifier (SVC): Provided good recall but required intensive computation for larger datasets.

Random Forest: Good at handling class imbalance, with high accuracy in predicting non-churn cases.

Gradient Boosting: Achieved the highest AUC score, balancing precision and recall.

AdaBoost: Focused on reducing false positives, achieved high AUC and geometric mean.

CatBoost: Selected as the best model due to its capability to handle categorical variables without pre-processing and its robustness to imbalanced datasets.

3. Evaluation Metrics

Models were evaluated based on several metrics, including accuracy, precision, recall, F1-score, and AUC-ROC.

CatBoost was chosen as the final model because of its high recall, making it suitable for identifying at-risk customers, thus supporting retention efforts.

Team Contribution

Dani Alex Parayil: Exploratory Data Analysis (EDA), visualizations, CatBoost implementation, and report preparation.

Garima Gambhir: EDA, preprocessing for Logistic Regression, and visualizations.

Ritika Pandey: EDA, preprocessing for SVC, report preparation.

Semal Shastri: EDA, preprocessing for Random Forest and decision trees.

Sumedha: Preprocessing for Gradient Boost and AdaBoost, final code compilation.

Report

For more detailed information about the project, the modeling approaches, and the results, please refer to the final report.

How to Run the Project

Extract the dataset from the Dataset/archive (1).zip and place it in the appropriate directory.

Use the Jupyter notebooks (ML Team 5 Group Project.ipynb and ML_CVUpdatedCode.ipynb) to explore the data, pre-process it, and run machine learning models.

Ensure dependencies are installed. Key dependencies include Scikit-Learn, XGBoost, and CatBoost.

Dependencies

Python 3.8+

Jupyter Notebook

Scikit-Learn

XGBoost

CatBoost

Pandas, Numpy, Matplotlib

Future Work

The next steps include experimenting with deep learning techniques, such as neural networks, to improve prediction rates and using tools like SHAP or LIME for better interpretability of churn predictions.