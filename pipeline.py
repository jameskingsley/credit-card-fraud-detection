from clearml.automation import PipelineController


def run_pipeline():
    pipe = PipelineController(
        name="Credit Card Fraud End-to-End Pipeline",
        project="Credit Card Fraud Detection",
        version="1.0.0",
        add_pipeline_tags=True,
    )

    pipe.set_default_execution_queue("default")

    # Preprocessing
    pipe.add_step(
        name="preprocess_step",
        base_task_project="Credit Card Fraud Detection",
        base_task_name="Data Preprocessing Task",
    )

    # Training
    pipe.add_step(
        name="train_step",
        parents=["preprocess_step"],
        base_task_project="Credit Card Fraud Detection",
        base_task_name="Train Model - Logistic Regression Clean",
    )

    # Quality Gate & Tagging
    pipe.add_step(
        name="evaluate_tag_step",
        parents=["train_step"],
        base_task_project="Credit Card Fraud Detection",
        base_task_name="Evaluate and Tag Model",
    )

    print("Starting full deployment pipeline execution locally...")
    pipe.start_locally(run_pipeline_steps_locally=True)


if __name__ == "__main__":
    run_pipeline()
