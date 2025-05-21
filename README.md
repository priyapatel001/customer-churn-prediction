# customer-churn-prediction
Project Overview
This project aims to predict customer churn for a telecom company using machine learning. By analyzing customer behavior and service usage patterns, we identify at-risk customers and recommend retention strategies.

Project Steps
1. Data Collection & Understanding
Dataset Source: WA_Fn-UseC_-Telco-Customer-Churn.csv (Kaggle)

7,043 customers with 20 features (e.g., tenure, contract type, monthly charges).

Target Variable: Churn (Yes/No).

2. Exploratory Data Analysis (EDA)
Univariate Analysis:

Distribution of tenure, MonthlyCharges, and TotalCharges (histograms/boxplots).

Bivariate Analysis:

Correlation between Contract type and churn (countplots).

MonthlyCharges vs. Churn (boxplots).

Multivariate Analysis:

Heatmap of numerical feature correlations.

3. Data Preprocessing
Handled Missing Values:

Replaced empty TotalCharges values with 0.0.

Encoded Categorical Variables:

Label encoding for Contract, PaymentMethod, etc.

Balanced Data:

Applied SMOTE to address class imbalance (26.5% churn rate).

4. Model Training & Evaluation
Algorithms Tested:

Decision Tree, Random Forest, XGBoost.

Best Model: Random Forest (85% cross-validation accuracy).

Test Performance:

Accuracy: 77%

Precision/Recall: 80%/72% for churn class.

5. Deployment
Saved Artifacts:

Model (customer_churn_model.pkl).

Encoders (encoders.pkl).

Predictive System:

Takes user input (e.g., contract type, tenure) and outputs churn risk.

Dataset Sources
Primary Dataset:

Telco Customer Churn on Kaggle

Features:

Demographics (gender, SeniorCitizen).

Account details (tenure, Contract).

Services (InternetService, StreamingTV).

Charges (MonthlyCharges, TotalCharges).

Alternative Datasets (for extension):

IBM Telco Dataset

Google Cloud Public Datasets

Tools Used
Python Libraries: Pandas, Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn.

Environment: Google Colab (GPU runtime).

Key Insights
High Churn Drivers:

Month-to-month contracts.

High monthly charges + low tenure.

Retention Strategies:

Target short-tenure customers with loyalty discounts.

Promote long-term contracts.
