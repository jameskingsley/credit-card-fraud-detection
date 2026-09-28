import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from clearml import Task, OutputModel

from src.utils import evaluate_model_performance


def train_fraud_model():
    task = Task.init(
        project_name="Credit Card Fraud Detection",
        task_name="Train Model - Logistic Regression Clean",
        task_type=Task.TaskTypes.training,
        output_uri=True,
    )

    
    task.set_script(entry_point="src/train.py")

    hyperparams = {
        "C": 0.1,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "use_smote": True,
        "test_size": 0.2,
        "random_state": 42,
    }
    task.connect(hyperparams)

    # Read data
    df = pd.read_csv("data/processed/creditcard_scaled.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=hyperparams["test_size"],
        random_state=hyperparams["random_state"],
        stratify=y,
    )

    if hyperparams["use_smote"]:
        smote = SMOTE(random_state=hyperparams["random_state"])
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    model = LogisticRegression(
        C=hyperparams["C"],
        max_iter=hyperparams["max_iter"],
        solver=hyperparams["solver"],
        class_weight=hyperparams["class_weight"],
        random_state=hyperparams["random_state"],
    )
    model.fit(X_train_res, y_train_res)

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc_score, cm_fig = evaluate_model_performance(y_test, y_proba)

    logger = task.get_logger()
    logger.report_single_value(name="PR-AUC", value=float(pr_auc_score))
    logger.report_matplotlib_figure(
        title="Evaluation", series="Confusion Matrix", figure=cm_fig
    )

    # Save and register model artifact
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/logistic_regression_fraud.joblib"
    joblib.dump(model, model_path)

    output_model = OutputModel(task=task, name="Fraud_Logistic_Regression")
    output_model.update_weights(
        weights_filename=model_path, auto_delete_file=False
    )

    print(
        f"Model trained with PR-AUC: {pr_auc_score:.4f} and registered to ClearML."
    )


if __name__ == "__main__":
    train_fraud_model()