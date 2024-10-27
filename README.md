# Loan-Approval-Prediction

## Introduction

The goal for this project is to predict whether an applicant is approved for a loan. 

## Dataset Overview

Target variable - loan_status - binary (0,1)

Numerical features:

- person_age: Applicant’s age in years.
- person_income: Annual income of the applicant in USD.
- person_emp_length: Length of employment in years.
- loan_amnt: Total loan amount requested by the applicant.
- loan_int_rate: Interest rate associated with the loan.
- loan_percent_income: Percentage of the applicant’s income allocated towards loan repayment.
- cb_person_cred_hist_length: Length of the applicant’s credit history in years.

Categorical features:

- person_home_ownership: Status of homeownership (e.g., Rent, Own, Mortgage).
- loan_intent: Purpose of the loan (e.g., Education, Medical, Personal).
- loan_grade: Risk grade assigned to the loan, assessing the applicant’s creditworthiness.
- loan_status: The approval status of the loan (approved or not approved).cb_person_default_on_file: Indicates if the applicant has a history of default ('Y' for yes, 'N' for no).

## Model Development

### Exploratory Data Analysis (EDA)

1. Descriptive Analysis
2. Check for Missingness
3. Check for imbalanced dataset
4. Univariate Analysis (Categorical & Numerical Features)
5. Bivariate Analysis (Features vs Target)
6. Multivariate Analysis (Pair plots and Correlation Heatmap)
7. Outlier detection (Visual inspection with box plot & Tukey's IQR method)

Key insights:
1. The maximum value of person_age is 144, and the maximum value of person_emp_length is 123, which don't make any sense. The error might stem from the synthetic data generation process. Note that the test set doesn't suffer from outliers as much.
2. Missing values are less than 5%, which won't cause significant information loss when dropped.
3. We have an imbalanced target distribution. Far fewer people get the loan compared to those who don't (17% vs 83%).
4. We don't have a high cardinality problem for the categorical features.
5. Most of the numerical features are right-skewed.
6. There are some counterintuitive findings in the Bivariate Analysis part. For example, the percentage of people who own their property and get their loans approved is lower compared to those who "rent" or have a "mortgage." Additionally, almost one-third of people who have a default record still got their loans approved, which is surprisingly high. 
7. Tukey's IQR method revealed that 838 out of 91,226 data points are outliers and should be removed.

### Data Preprocessing

1. Drop unique identifier (id column)
2. Drop rows with missing values
3. Drop rows with outliers using Tukey's IQR method
4. Split features with target
5. Encoding categorical features
6. Balance the dataset using SMOTE

### Feature Engineering

1. Introduced new feature: `interest_percent_income`

Rationale behind introducing this feature: Interest payments, compared to the principal, represent the real cash outflow that must be paid each year. By calculating interest_percent_income, which measures the proportion of a borrower's income dedicated to interest payments, we gain a clearer understanding of the financial burden the loan places on the borrower. 

### Model Training

- Logistic Regression
- Random Forest
- Xgboost
- Catboost

Evaluation matrics: AUC-ROC score. 
Note: to calculate this score, the model must provide probability estimates rather than just class predictions. This is done by using the predict_proba function, which outputs the probability of each class. The ROC curve is then plotted by varying the decision threshold, and the AUC represents the area under this curve. A higher AUC score indicates a better-performing model.

### Hyperparameter Tuning


### Evaluation


### Feature importance and interpretation



## Model Deployment