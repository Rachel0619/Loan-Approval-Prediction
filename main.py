import pandas as pd
import numpy as np
from src.preprocessing import *
from src.training import model_training
from xgboost import XGBClassifier

def main():

    train_df = pd.read_csv("data/train.csv")
    X_test = pd.read_csv("data/test.csv")
    original_df = pd.read_csv("data/original.csv")
    train_df = pd.concat([train_df, original_df], axis=0)
    X_test_index = X_test['id']
    
    # Define categotical columns
    categorical_columns = train_df.select_dtypes(include=['object']).columns.to_list()
    if 'loan_status' in categorical_columns:  
        categorical_columns.remove('loan_status')

    # Define numerical columns
    numerical_columns = train_df.select_dtypes(include=['int', 'float']).columns.to_list()
    for column in ['id', 'loan_status']:
        if column in numerical_columns:
            numerical_columns.remove(column)
    
    # Drop unique identifier for train and test set
    train_df = drop_unique_identifier(train_df)
    X_test = drop_unique_identifier(X_test)

    # Drop missing values for train set
    train_df = drop_missing_values(train_df)

    # Detect and drop outliers for train set
    train_df = detect_and_remove_outliers(train_df, 1, numerical_columns)

    # Split features with target
    X_train, y_train = split_features_target(train_df)

    # Encoding categorical variables
    X_train = one_hot_encode(X_train, categorical_columns)
    X_test = one_hot_encode(X_test, categorical_columns)

    # Feature generation
    X_train = feature_generation(X_train)
    X_test = feature_generation(X_test)

    # Dealing with imbalanced dataset
    X_train, y_train = apply_smote(X_train, y_train)

    print("Finished preprocessing and feature engineering")
    print(f"Shape for X_train: {X_train.shape}")
    print(f"Shape for X_test: {X_test.shape}")

    # Save preprocessed test file for deployment
    X_test.to_csv("data/X_test_preprocessed.csv", index=False)

    # Model training
    oof_preds = {}
    test_preds = {}
    xgb_base_model = XGBClassifier()
    oof_preds["xgb_base_model"], test_preds["xgb_base_model"] = model_training(xgb_base_model, X_train, y_train, X_test, numerical_columns)

if __name__=="__main__":
    main()