from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    # Define the hyperparameter search space
    params = {
        'objective': 'binary:logistic',
        'device': 'cuda',
        'eval_metric': 'auc',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'gpu_hist',
        'verbosity': 0  # Suppress warnings and messages from XGBoost
    }
    
    # Fitting LGBM model with parameters from the trials
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(**params))])
    
    # Stratified sampling 
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_splits = cv.split(X, y)
    
    # Creating empty scores list to hold AUC scores from each trialed model
    scores = []
    for train_idx, val_idx in cv_splits:
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_pred_proba)
        scores.append(score)
        
    # Printing and returning mean AUC scores
    mean_score = np.mean(scores)
    print(f"Mean ROC AUC Score = {mean_score:.5f}")
    return mean_score

# When set to True, optuna will create a study to find the optimal parameters
train = False

if train:
    
    # Each optuna study uses an independent sampler with a TPE algorithm
    # For each trial, the TPE essentially uses Gaussian Mixture Models to identify the optimal parameter value
    study = optuna.create_study(sampler=TPESampler(n_startup_trials=30, multivariate=True, seed=42), direction="maximize")
    study.optimize(objective, n_trials=100)
    print('Best value:', study.best_value)
    print('Best trial:', study.best_trial.params)