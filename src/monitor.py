import numpy as np
import pandas as pd
from clearml import Task, Logger
from src.utils import trigger_clearml_alert

def monitor_data_and_model(reference_csv="data/processed/creditcard_scaled.csv", 
                            production_batch_csv="data/processed/creditcard_scaled.csv",
                            pr_auc_threshold=0.75,
                            drift_std_threshold=3.0,
                            webhook_url=None):
    
    task = Task.init(
        project_name="Credit Card Fraud Detection",
        task_name="Production Data & Model Monitor",
        task_type=Task.TaskTypes.monitor
    )
    logger = task.get_logger()

    ref_df = pd.read_csv(reference_csv)
    prod_df = pd.read_csv(production_batch_csv)

    # Feature Drift Detection 
    print("Checking for Feature Data Drift...")
    drift_detected = False
    
    for col in ref_df.columns:
        if col == 'Class':
            continue
        ref_mean, ref_std = ref_df[col].mean(), ref_df[col].std()
        prod_mean = prod_df[col].mean()
        
        # Calculating Z-score drift shift
        z_score = abs(prod_mean - ref_mean) / (ref_std + 1e-8)
        logger.report_scalar(title="Feature Shift Z-Score", series=col, value=float(z_score), iteration=1)

        if z_score > drift_std_threshold:
            message = f" DATA DRIFT ALERT: Feature '{col}' shifted by {z_score:.2f} standard deviations!"
            trigger_clearml_alert(message, webhook_url=webhook_url)
            drift_detected = True

    if not drift_detected:
        print("No statistical data drift detected.")

if __name__ == "__main__":
    monitor_data_and_model()