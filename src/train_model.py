# src/train_model.py

import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import base64
import requests

# ---------------- DagsHub / MLflow config ----------------
DAGSHUB_USERNAME = "ranesh88"
DAGSHUB_TOKEN = "264c26ba2a37b8440ddff7f3d3458e34f277c333"
DAGSHUB_REPO = "telecom-churn-prediction"

# Construct MLflow tracking URI
mlflow_tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"

# Encode token for HTTP Basic Auth
token_auth = base64.b64encode(f"{DAGSHUB_USERNAME}:{DAGSHUB_TOKEN}".encode()).decode()
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Telecom Churn Prediction")

# ---------------- Data ----------------
DATA_PATH = r"C:\Users\User\Desktop\Telecom_Customer_Churn_Prediction\data\processed\Churn_Prediction_Final.csv"
print(f"Loading data from: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)

# ---------------- Preprocessing ----------------
y = data["Churn"]
X = data.drop("Churn", axis=1)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Training ----------------
with mlflow.start_run():
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    class_report = classification_report(y_test, y_pred)

    # Console output
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print(f"ROC AUC    : {roc_auc:.4f}")
    print("\n=== Classification Report ===")
    print(class_report)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    print("\n=== Confusion Matrix ===")
    print(cm_df)

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = os.path.join("models", "confusion_matrix.png")
    os.makedirs("models", exist_ok=True)
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join("models", "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path)

    # ---------------- Save model ----------------
    model_path = os.path.join("models", "rf_classifier.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    mlflow.log_artifact(model_path, artifact_path="models")

    # ---------------- Log params & metrics ----------------
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    # Save classification report
    report_path = os.path.join("models", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(class_report)
    mlflow.log_artifact(report_path)

print("\nMLflow run completed. View your run on DagsHub!")
