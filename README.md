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
3. Drop rows with outliers using Tukey's IQR method (3881 out of 87283 rows)
4. Split features with target
5. Encoding categorical features
6. Balance the dataset using SMOTE

Both dropping outliers and Balancing the dataset boosted the model performance. 

### Feature Engineering

In this project, several new features were introduced to provide the model with additional information that has a strong domain-specific rationale, which helped significantly improve model performance. Each of these features was crafted to capture meaningful financial indicators related to a borrower's loan and income profile, enabling the model to better assess creditworthiness and risk.

1. `interest_percent_income`
Rationale: The `interest_percent_income` feature calculates the proportion of a borrower's income that goes toward paying interest each year. This measure gives insight into the actual financial burden of the loan, as interest payments represent cash outflows that must be met regardless of the principal balance. 

2. `repayment_year`
Rationale: The `repayment_year` feature calculates the number of years required for a borrower to repay their loan. For borrowers who pay some principal annually (non-zero loan_percent_income), repayment_year is calculated as the loan amount divided by the annual repayment amount. However, for cases where `loan_percent_income` is zero, indicating no annual principal repayment, we assign a high value based on the 99th percentile of repayment_year to signify the increased risk associated with prolonged or deferred repayment. This feature provides a clear signal of repayment behavior, helping the model assess long-term credit risk more effectively.

3. `zero_repayment_risk`
Rationale: The `zero_repayment_risk` feature flags cases where loan_percent_income is zero, meaning the borrower does not pay any principal on an annual basis. This binary feature (1 for no repayment, 0 otherwise) highlights high-risk scenarios where the loan's repayment is deferred, potentially increasing the risk of default. 

4. `loan_to_income`
Rationale: The `loan_to_income` ratio measures the loan amount relative to the borrower's income. This feature serves as a proxy for debt-to-income ratio, helping the model understand the borrower’s debt burden. Higher ratios indicate that a larger portion of income would be needed to cover loan obligations, increasing potential financial strain and risk.

Adding these features are able to increase the cross-validation F1 score for base model (LogisticaRegression with default parameters) from 0.9305 to 0.9323.

### Feature Selection

No explicit feature selection was performed for two main reasons:

- Manageable Number of Features: After feature engineering and one-hot encoding, the dataset includes only 30 features. This is a relatively manageable number, allowing models to utilize the full set without creating excessive computational overhead.
- Domain-Specific Relevance: Each feature included has a clear, meaningful relationship to the problem of loan approval, representing valuable financial indicators and risk factors. Excluding features would risk losing critical information relevant to assessing creditworthiness and borrower risk.

Additionally, I observed no significant signs of overfitting during training, indicating that the current feature set was appropriate without further dimensional reduction or selection techniques.

### Model Training

For this project, I focused primarily on tree-based models due to their inherent ability to capture complex, nonlinear relationships in the data.

- Logistic Regression (Base model)
- Xgboost
- LightGBM
- Catboost

#### Cross-Validation Strategy

To ensure robust and generalizable results, I employed stratified cross-validation during training. Stratified cross-validation maintains the same proportion of classes in each fold as in the entire dataset, which is particularly advantageous when dealing with imbalanced datasets like loan approval data. This approach ensures that each fold contains a representative sample of both approved and rejected loans, leading to more stable and reliable performance metrics across all folds and reducing the risk of biased model performance.

#### Evaluation matrics: AUC-ROC score. 

A higher AUC score indicates a better-performing model in terms of its ability to correctly rank loan applicants based on risk.

***Reason for choosing AUC-ROC metric***

- Interpretability: AUC-ROC provides a measure of the model's ability to distinguish between approved and rejected loans across different thresholds. It evaluates how well the model can rank positive (approved) cases higher than negative (rejected) cases.
- Probability-Based Scoring: Since loan approval decisions often depend on risk assessment, having probability estimates is valuable. By using predict_proba to output probabilities rather than binary class predictions, we can assess the model’s confidence in each decision.
- Threshold Flexibility: AUC-ROC also allows flexibility in threshold setting, which is critical in financial contexts where we may adjust thresholds based on changing risk tolerance or economic conditions.

***Note:*** to calculate this score, the model must provide probability estimates rather than just class predictions. This is done by using the predict_proba function, which outputs the probability of each class. The ROC curve is then plotted by varying the decision threshold, and the AUC represents the area under this curve. 

### Hyperparameter Tuning

I used Optuna to find the best parameters for each model. Optuna offers several advantages, such as an efficient search with Tree-structured Parzen Estimator (TPE), dynamic sampling, and early stopping for unpromising trials, making it a powerful tool for hyperparameter optimization.

However, I found that hyperparameter tuning provided only marginal performance improvements compared to feature engineering. In essence, tuning was like adding the final layer of cream to the cake, wchih might be helpful but not quite transformative compared to the foundational impact of well-crafted features.

### Final evaluation

Give a summary of the evaluation matrics for logistic regression, XGBoost, Catbbost and stacking model

provide a picture of the AUC-ROC curve

### Feature importance and interpretation

SHAP plots

## Model Deployment