import logging

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data(feature_path, label_path):
    features_df = pd.read_csv("outputs/orf_features_with_kmers.csv")
    labels_df = pd.read_csv("outputs/orf_labeled.csv")

    merge_keys = ["start", "end", "strand", "frame", "chromosome"]
    df = features_df.merge(labels_df, on=merge_keys)

    le = LabelEncoder()
    df["target"] = le.fit_transform(df["label"])

    X = df.select_dtypes(include=np.number).drop(columns=["target"])
    y = df["target"]

    return X, y, df

def optimize_xgboost(X_train, y_train, n_trials=50):
    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "device": "cuda",
            "eval_metric": "auc",
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "n_estimators": 200,
        }
        model = xgb.XGBClassifier(**param)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50)  # Try 50 trials, can increase if wanted

    logging.info(f"Best parameters: {study.best_params}")
    logging.info(f"Best ROC AUC: {study.best_value}")

    best_params = study.best_params
    best_params.update({
        'n_estimators': 200,
        'device': 'cuda',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    })

    return best_params

def train_and_evaluate(X_train, X_test, y_train, y_test, params):
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]

    logging.info(f"Classification Report (XGBoost+CUDA, Optuna):\n {classification_report(y_test, y_pred)}")
    logging.info(f"ROC AUC Score (XGBoost+CUDA, Optuna): {roc_auc_score(y_test, y_proba):.4f}")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", normalize=None)
    plt.title("Confusion Matrix - Yeast ORF XGB Classifier")
    plt.savefig("outputs/confusion_matrix.png", bbox_inches='tight', dpi=300)
    plt.close()

    xgb.plot_importance(xgb_model, max_num_features=20, importance_type='gain')
    plt.title('Feature Importance (Gain)')
    plt.savefig("outputs/feature_importance.png", bbox_inches='tight', dpi=300)
    plt.close()

    features_to_plot = ['length_y', 'stop_percent', 'length_x']
    PartialDependenceDisplay.from_estimator(xgb_model, X_train, features_to_plot, kind='average')
    plt.tight_layout()
    plt.savefig("outputs/pdp_plot.png", bbox_inches='tight', dpi=300)
    plt.close()

    return xgb_model

def cross_validate_model(xgb_model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    logging.info(f"5-fold CV ROC AUC scores: {scores}")
    logging.info(f"Mean ROC AUC: {scores.mean():.4f} ± {scores.std():.4f}")

def main():
    X, y, df = load_and_prepare_data("outputs/orf_features_with_kmers.csv", "outputs/orf_labeled.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    best_params = optimize_xgboost(X_train, y_train, n_trials=50)
    xgb_model = train_and_evaluate(X_train, X_test, y_train, y_test, best_params)
    cross_validate_model(xgb_model, X, y)

if __name__=="__main__":
    main()