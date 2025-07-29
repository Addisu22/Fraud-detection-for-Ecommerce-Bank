import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, classification_report

def train_model(X_train, y_train, model_type="logistic"):
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model_type")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, dataset_name, model_name, report_path):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    report_text = (
        f"Dataset: {dataset_name}\n"
        f"Model: {model_name}\n"
        f"F1 Score: {f1:.4f}\n"
        f"AUC-PR: {auc_pr:.4f}\n"
        f"Confusion Matrix:\n{cm}\n"
        f"Classification Report:\n{report}\n"
    )

    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
