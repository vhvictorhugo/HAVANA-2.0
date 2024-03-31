import mlflow
import pandas as pd


class MLFlow:
    def __init__(self, state, embeddings_dimension, metadata):
        self.state = state
        self.embeddings_dimension = embeddings_dimension  # 0 if baseline
        self.metadata = metadata

    def run(self):
        metrics_path = self.metadata["processed"]["metrics"]
        f1_score_df = pd.read_csv(metrics_path + "fscore.csv")
        precision_df = pd.read_csv(metrics_path + "precision.csv")
        recall_df = pd.read_csv(metrics_path + "recall.csv")
        category_columns = ["Shopping", "Community", "Food", "Entertainment", "Travel", "Outdoors", "Nightlife"]

        mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
        mlflow.set_experiment(f"{self.state} State")
        with mlflow.start_run(nested=True):
            params = {"embeddings_dimension": self.embeddings_dimension}
            mlflow.log_params(params)
            for category in category_columns:
                # category log
                mlflow.log_metric(f"recall_{category}", recall_df[category].mean())
                mlflow.log_metric(f"precision_{category}", precision_df[category].mean())
                mlflow.log_metric(f"f1_score_{category}", f1_score_df[category].mean())
            # general log
            mlflow.log_metric("recall", recall_df[category_columns].mean().mean())
            mlflow.log_metric("precision", precision_df[category_columns].mean().mean())
            mlflow.log_metric("f1_score", f1_score_df[category_columns].mean().mean())
            mlflow.log_metric("macro_avg", f1_score_df["macro avg"].mean().mean())
            mlflow.log_metric("weighted_avg", f1_score_df["weighted avg"].mean().mean())
            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.end_run()
