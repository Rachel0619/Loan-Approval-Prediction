from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

def drop_unique_identifier(df):
    df = df.drop(columns=['id'])
    return df

def drop_missing_values(df):
    # Drop rows with missing values in 'person_emp_length' and 'loan_int_rate'
    df = df.dropna(subset=['person_emp_length', 'loan_int_rate']).reset_index(drop=True)
    return df

def detect_and_remove_outliers(df, n, features):
    """
    Detects and removes observations containing more than n outliers according
    to the Tukey method, and returns the resulting DataFrame.
    
    Parameters:
    - df: DataFrame containing the data
    - n: Number of outliers for rejection (observations with more than n outliers will be removed)
    - features: List of features to check for outliers
    
    Returns:
    - DataFrame with outliers removed
    """
    outlier_indices = []

    # Iterate over features (columns)
    for col in features:
        # 1st quartile (25%) and 3rd quartile (75%)
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = 1.5 * IQR

        # Determine indices of outliers for the feature
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        # Append the found outlier indices for the column to the list
        outlier_indices.extend(outlier_list_col)

    # Identify indices of observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_indices.items() if v > n]

    # Remove outliers from the DataFrame
    df_cleaned = df.drop(multiple_outliers)

    return df_cleaned

def split_features_target(df):
    X_train = df.drop(columns=['loan_status'])
    y_train = df['loan_status']
    return X_train, y_train

def one_hot_encode(df, features):
    """
    This function takes a DataFrame, identifies non-numeric (categorical) columns,
    applies one-hot encoding to them, and returns the modified DataFrame.
    """
    for col in features:
        df_dum = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = df.drop(col, axis=1)
        df = pd.concat([df, df_dum], axis=1)
    return df

def feature_generation(df_):
    df = df_.copy()
    df['interest_percent_income'] = round(df['loan_int_rate'] * df['loan_amnt'] / df['person_income'], 2)
    df['zero_repayment_risk'] = np.where(df['loan_percent_income'] == 0, 1, 0)
    # Calculate repayment_year normally for non-zero loan_percent_income
    df['repayment_year'] = np.where(
        df['loan_percent_income'] != 0,
        df['loan_amnt'] / (df['loan_percent_income'] * df['person_income']),
        np.nan  # Temporarily assign NaN for zero loan_percent_income
    )
    
    # Calculate the 99th percentile of repayment_year for non-zero cases
    high_quantile_value = df['repayment_year'].quantile(0.99)
    
    # Assign the high quantile value to repayment_year for cases with zero loan_percent_income
    df['repayment_year'].fillna(high_quantile_value, inplace=True)
    df['loan_to_income'] = df['loan_amnt'] / df['person_income']

    return df

def apply_smote(X, y, sampling_strategy=1, random_state=42, k_neighbors=3):
    """
    Applies SMOTE to resample the dataset to handle class imbalance.
    
    Parameters:
    - X: Features (input data)
    - y: Target labels
    - sampling_strategy: Desired sampling ratio for the minority class. Default is 1 (equalize classes).
    - random_state: Random state for reproducibility. Default is 42.
    - k_neighbors: Number of nearest neighbors to use for SMOTE. Default is 3.
    
    Returns:
    - X_res: Resampled features
    - y_res: Resampled target labels
    """
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X, y)
    
    return X_res, y_res
