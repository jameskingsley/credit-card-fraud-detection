import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
from clearml import Task, Dataset

def load_and_preprocess_data(raw_csv_path="data/raw/creditcard.csv", output_csv_path="data/processed/creditcard_scaled.csv"):
    """
    Reads raw credit card data, scales Amount and Time, and saves the output.
    Registers the dataset in ClearML Data.
    """
    print(f"Loading raw data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)

    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Reorder columns
    scaled_amount = df.pop('scaled_amount')
    scaled_time = df.pop('scaled_time')
    target_class = df.pop('Class')

    df['scaled_amount'] = scaled_amount
    df['scaled_time'] = scaled_time
    df['Class'] = target_class

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Preprocessed dataset saved to {output_csv_path}")

    return df

if __name__ == "__main__":
    task = Task.init(
        project_name="Credit Card Fraud Detection",
        task_name="Data Preprocessing Task",
        task_type=Task.TaskTypes.data_processing
    )
    df = load_and_preprocess_data()
    
    # Upload Data Artifact to ClearML
    task.upload_artifact(name="processed_csv", artifact_object="data/processed/creditcard_scaled.csv")
    print("Preprocessing completed and artifact uploaded.")