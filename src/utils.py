import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import requests

def evaluate_model_performance(y_true, y_proba, threshold=0.5):
    """
    Calculates PR-AUC and Confusion Matrix.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_score = auc(recall, precision)

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (PR-AUC: {pr_auc_score:.4f})')

    return pr_auc_score, fig

def trigger_clearml_alert(message, webhook_url=None):
    """
    Triggers an alert notification via Slack Webhook or logs directly to ClearML console.
    """
    print(f"[ALERT TRIGGERED]: {message}")
    if webhook_url:
        try:
            requests.post(webhook_url, json={"text": message})
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")