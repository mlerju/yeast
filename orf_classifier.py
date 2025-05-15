import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from xgboost.callback import EarlyStopping

# Load features
features_df = pd.read_csv("orf_features_with_kmers.csv")
labels_df = pd.read_csv("labeled_orfs.csv")

merge_keys = ["start", "end", "strand", "frame", "chromosome"]
df = features_df.merge(labels_df, on=merge_keys)

# Encode labels
df["target"] = df["label"].map({"known_gene": 1, "novel_orf": 0})

# Drop non-numeric or unneeded columns
X = df.drop(columns=["label", "target", "chromosome", "start", "end", "sequence"])
y = df["target"]

print(y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

########------------ RANDOM FOREST ------------########
# Train RF model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred))
print("ROC AUC Score (Random Forest):", roc_auc_score(y_test, y_prob))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["novel", "known"], yticklabels=["novel", "known"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 20 Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

########------------ XGBoost with GPU accel. ------------########

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Bayesian Optimization (Optuna)
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
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    return auc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Try 50 trials, can increase if you want

print("Best parameters:", study.best_params)
print("Best ROC AUC:", study.best_value)

best_params = study.best_params
best_params['n_estimators'] = 200

xgb_model = xgb.XGBClassifier(
    **best_params,
    device='cuda',
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              )

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

# Report
print("Classification Report (XGBoost+CUDA, Optuna):\n", classification_report(y_test, y_pred))
print("ROC AUC Score (XGBoost+CUDA, Optuna):", roc_auc_score(y_test, y_proba))

# Feature Importance
xgb.plot_importance(xgb_model, max_num_features=20, importance_type='gain', title='Feature Importance (Gain)')
plt.show()

# Partial dependence plots (PDPs)
features_to_plot = ['length_y', 'stop_percent', 'length_x']
PartialDependenceDisplay.from_estimator(xgb_model, X_train, features_to_plot, kind='average')
plt.tight_layout()
plt.show()

# Assuming y_test and y_pred are already defined
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", normalize=None)
plt.title("Confusion Matrix - Yeast ORF XGB Classifier")
plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=300)
plt.close()

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"5-fold CV ROC AUC scores: {scores}")
print(f"Mean ROC AUC: {scores.mean():.4f} ± {scores.std():.4f}")