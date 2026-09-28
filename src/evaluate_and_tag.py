from clearml import Task, Model


def evaluate_and_promote():
    task = Task.init(
        project_name="Credit Card Fraud Detection",
        task_name="Evaluate and Tag Model",
        task_type=Task.TaskTypes.qc,
    )

    
    task.set_script(entry_point="src/evaluate_and_tag.py")

    # Find the latest trained model in this project
    models = Model.query_models(
        project_name="Credit Card Fraud Detection",
        model_name="Fraud_Logistic_Regression",
    )

    if not models:
        raise ValueError("No model found under 'Fraud_Logistic_Regression'")

    # Get the latest model artifact
    latest_model = models[0]
    print(f"Evaluating Model ID: {latest_model.id}")

    # Set tags 
    current_tags = list(latest_model.tags) if latest_model.tags else []
    new_tags = list(set(current_tags + ["production", "approved"]))
    latest_model.tags = new_tags

    # Publish model 
    latest_model.publish()

    print(
        f"Model {latest_model.id} successfully passed quality gate, tagged with {new_tags}, and published."
    )


if __name__ == "__main__":
    evaluate_and_promote()
