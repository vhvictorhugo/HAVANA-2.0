import mlflow
import pandas as pd


class MLFlow:
    def __init__(self, state, embedder, h3_resolution, embeddings_dimension, metadata):
        self.state = state
        self.embedder = embedder
        self.h3_resolution = h3_resolution
        self.embeddings_dimension = embeddings_dimension  # 0 if baseline
        self.metadata = metadata

    def run(self):
        metrics_path = self.metadata["processed"]["metrics"].format(embedder=self.embedder, state=self.state)

        if "baseline" not in metrics_path:
            precision_name = (
                metrics_path + f"precision_{self.embeddings_dimension}_dimension_{self.h3_resolution}_resolution.csv"
            )
            recall_name = (
                metrics_path + f"recall_{self.embeddings_dimension}_dimension_{self.h3_resolution}_resolution.csv"
            )
            fscore_name = (
                metrics_path + f"fscore_{self.embeddings_dimension}_dimension_{self.h3_resolution}_resolution.csv"
            )
        else:
            precision_name = metrics_path + "precision_baseline.csv"
            recall_name = metrics_path + "recall_baseline.csv"
            fscore_name = metrics_path + "fscore_baseline.csv"

        f1_score_df = pd.read_csv(fscore_name)
        precision_df = pd.read_csv(precision_name)
        recall_df = pd.read_csv(recall_name)
        category_columns = ["Shopping", "Community", "Food", "Entertainment", "Travel", "Outdoors", "Nightlife"]

        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment(f"{self.state} State - Region Layer - Balanced Data")
        with mlflow.start_run(nested=True):
            params = {
                "embedder": self.embedder,
                "h3_resolution": self.h3_resolution,
                "embeddings_dimension": self.embeddings_dimension,
            }
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
