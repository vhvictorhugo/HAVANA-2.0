import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class PoiCategorizationLoader:
    def __init__(self):
        pass

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

    def save_model_and_weights(self, model, output_dir, n_folds, n_replications):
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save("model.h5")

    def save_report_to_csv(self, output_dir, report, n_folds, n_replications):
        precision_dict, recall_dict, fscore_dict = self.process_report(report, n_folds, n_replications)

        self.save_metrics_to_csv(precision_dict, output_dir, "precision")
        self.save_metrics_to_csv(recall_dict, output_dir, "recall")
        self.save_metrics_to_csv(fscore_dict, output_dir, "fscore")

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

    def save_metrics_to_csv(self, metrics_dict, output_dir, filename):
        df = pd.DataFrame(metrics_dict)
        logging.info(f"Saving {filename} metrics to csv")
        df.to_csv(output_dir + f"{filename}.csv", index=False)
        logging.info(f"Path: {output_dir + filename}.csv")
