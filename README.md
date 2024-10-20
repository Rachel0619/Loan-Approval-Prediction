# Loan-Approval-Prediction

## Introduction

The goal for this competition is to predict whether an applicant is approved for a loan. Submissions are evaluated using area under the ROC curve using the predicted probabilities and the ground truth targets.

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

