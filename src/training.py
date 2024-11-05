from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def model_training(model, X, y, X_test, features):

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
    ],
    remainder='passthrough'  
)

    # Set the cross-validation parameters
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize arrays to store the OOF predictions and test predictions
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    
    # Initialize a list to store F1 scores of each fold
    f1_scores = []

    # Looping over cross-validation folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Training the Fold {fold+1}/{n_splits}")

        # Separate training and validation data
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        # Train the model
        pipeline.fit(X_tr, y_tr)

        # Predictions on the validation set
        val_preds_proba = pipeline.predict_proba(X_val)[:, 1]
        val_preds = (val_preds_proba >= 0.5).astype(int)  # Convert probabilities to binary predictions
        oof_preds[val_idx] = val_preds_proba  # Store OOF predictions

        # Predictions on the test set
        test_preds += pipeline.predict_proba(X_test)[:, 1] / n_splits  # Average of predictions for each fold

        # AUC and F1 Score evaluation for this fold
        fold_auc = roc_auc_score(y_val, val_preds_proba)
        fold_f1 = f1_score(y_val, val_preds)
        f1_scores.append(fold_f1)

        print(f"AUC of Fold {fold+1}: {fold_auc:.5f}")
        print(f"F1 Score of Fold {fold+1}: {fold_f1:.5f}")

    # AUC OOF Assessment
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"AUC OOF: {oof_auc:.4f}")

    # Average F1 Score across all folds
    avg_f1_score = np.mean(f1_scores)
    print(f"Average F1 Score: {avg_f1_score:.4f}")
    
    return oof_preds, test_preds

