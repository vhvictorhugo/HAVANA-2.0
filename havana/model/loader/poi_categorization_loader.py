from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class PoiCategorizationLoader:
    def __init__(self):
        pass

    def heatmap(self, directory, df, filename, title, size, annot):
        return
        plt.figure(figsize=size)
        fig = sns.heatmap(df, annot=annot, cmap="YlGnBu").set_title(title).get_figure()

        self.save_fig(directory, filename + ".png", fig)

    def save_fig(self, directory, filename, fig):
        Path(directory).mkdir(parents=True, exist_ok=True)
        fig.savefig(directory + filename + ".png", bbox_inches="tight", dpi=400)

    def plot_history_metrics(self, folds_histories, folds_reports, output_dir, show=False):
        n_folds = len(folds_histories)
        n_replications = len(folds_histories[0])
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i in range(len(folds_histories)):
            fold_histories = folds_histories[i]
            for j in range(len(fold_histories)):
                h = fold_histories[j]
                file_index = "fold_" + str(i) + "_replication_" + str(j)
                plt.figure(figsize=(12, 12))
                plt.plot(h["accuracy"])
                plt.plot(h["val_accuracy"])
                plt.title("model accuracy")
                plt.ylabel("accuracy")
                plt.xlabel("epoch")
                plt.legend(["train", "test"], loc="upper left")
                if show:
                    plt.show()
                plt.savefig(output_dir + file_index + "_history_accuracy.png")
                # summarize history for loss
                plt.figure(figsize=(12, 12))
                plt.plot(h["loss"])
                plt.plot(h["val_loss"])
                plt.title("model loss")
                plt.ylabel("loss")
                plt.xlabel("epoch")
                plt.legend(["train", "test"], loc="upper left")
                plt.savefig(output_dir + file_index + "_history_loss.png")
                if show:
                    plt.show()

    def save_df(self, df, name):
        df = df.round(5)
        out = f"output/{name}.png"
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.axis("tight")
        ax.axis("off")
        plt.savefig(out)

    def save_model_and_weights(self, model, output_dir, n_folds, n_replications):
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save("model.h5")

    def save_report_to_csv(self, output_dir, report, n_folds, n_replications, usuarios):
        precision_dict, recall_dict, fscore_dict = self.process_report(report, n_folds, n_replications)

        self.save_metrics_to_csv(precision_dict, output_dir, "precision", n_folds, n_replications)
        self.save_metrics_to_csv(recall_dict, output_dir, "recall", n_folds, n_replications)
        self.save_metrics_to_csv(fscore_dict, output_dir, "fscore", n_folds, n_replications)

    def process_report(self, report, n_folds, n_replications):
        precision_dict = {}
        recall_dict = {}
        fscore_dict = {}

        for key, value in report.items():
            if key in ["accuracy", "recall", "f1-score", "support"]:
                continue
            elif key in ["macro avg", "weighted avg"]:
                fscore_dict[key] = value["f1-score"]
                continue

            fscore_data = value["f1-score"]
            precision_data = value["precision"]
            recall_data = value["recall"]

            fscore_dict[key] = self.fill_missing_data(fscore_data, n_folds, n_replications)
            precision_dict[key] = self.fill_missing_data(precision_data, n_folds, n_replications)
            recall_dict[key] = self.fill_missing_data(recall_data, n_folds, n_replications)

        return precision_dict, recall_dict, fscore_dict

    def fill_missing_data(self, data, n_folds, n_replications):
        column_size = 2
        if len(data) < column_size:
            data.extend([np.nan] * (column_size - len(data)))
        return data

    def save_metrics_to_csv(self, metrics_dict, output_dir, filename, n_folds, n_replications):
        df = pd.DataFrame(metrics_dict)
        output_dir = Path(output_dir) / f"{n_folds}_folds" / f"{n_replications}_replications"
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"{filename}.csv", index=False)
        self.save_df(df, f"{filename.capitalize()} Metrics")

    # def save_report_to_csv(self, output_dir, report, n_folds, n_replications, usuarios):
    #     precision_dict = {}
    #     recall_dict = {}
    #     fscore_dict = {}
    #     column_size = n_folds * n_replications
    #     column_size = 2
    #     for key in report:
    #         if key == "accuracy":
    #             column = "accuracy"
    #             fscore_dict[column] = report[key]
    #             continue
    #         elif key == "recall" or key == "f1-score" or key == "support":
    #             continue
    #         if key == "macro avg" or key == "weighted avg":
    #             column = key
    #             fscore_dict[column] = report[key]["f1-score"]
    #             continue
    #         fscore_column = key
    #         fscore_column_data = report[key]["f1-score"]
    #         if len(fscore_column_data) < column_size:
    #             while len(fscore_column_data) < column_size:
    #                 fscore_column_data.append(np.nan)
    #         fscore_dict[fscore_column] = fscore_column_data

    #         precision_column = key
    #         precision_column_data = report[key]["precision"]
    #         if len(precision_column_data) < column_size:
    #             while len(precision_column_data) < column_size:
    #                 precision_column_data.append(np.nan)
    #         precision_dict[precision_column] = precision_column_data

    #         recall_column = key
    #         recall_column_data = report[key]["recall"]
    #         if len(recall_column_data) < column_size:
    #             while len(recall_column_data) < column_size:
    #                 recall_column_data.append(np.nan)
    #         recall_dict[recall_column] = recall_column_data

    #     precision = pd.DataFrame(precision_dict)
    #     print("Métricas precision: \n", precision)
    #     output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
    #     precision.to_csv(output_dir + "precision.csv", index_label=False, index=False)
    #     self.save_df(precision, "Precision Metrics")

    #     recall = pd.DataFrame(recall_dict)
    #     print("Métricas recall: \n", recall)
    #     recall.to_csv(output_dir + "recall.csv", index_label=False, index=False)
    #     self.save_df(recall, "Recall Metrics")

    #     fscore = pd.DataFrame(fscore_dict)
    #     print("Métricas fscore: \n", fscore)
    #     fscore.to_csv(output_dir + "fscore.csv", index_label=False, index=False)
    #     self.save_df(fscore, "F1-Score Metrics")
