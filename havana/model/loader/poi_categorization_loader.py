import logging

import numpy as np
import pandas as pd


class PoiCategorizationLoader:
    def __init__(self):
        pass

    def save_report_to_csv(self, output_dir, report, embeddings_dimension, h3_resolution):
        precision_dict, recall_dict, fscore_dict = self.process_report(report)

        if "baseline" not in output_dir:
            precision_name = f"precision_{embeddings_dimension}_dimension_{h3_resolution}_resolution"
            recall_name = f"recall_{embeddings_dimension}_dimension_{h3_resolution}_resolution"
            fscore_name = f"fscore_{embeddings_dimension}_dimension_{h3_resolution}_resolution"
        else:
            precision_name = "precision_baseline"
            recall_name = "recall_baseline"
            fscore_name = "fscore_baseline"

        self.save_metrics_to_csv(precision_dict, output_dir, precision_name)
        self.save_metrics_to_csv(recall_dict, output_dir, recall_name)
        self.save_metrics_to_csv(fscore_dict, output_dir, fscore_name)

    def process_report(self, report):
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

            fscore_dict[key] = self.fill_missing_data(fscore_data)
            precision_dict[key] = self.fill_missing_data(precision_data)
            recall_dict[key] = self.fill_missing_data(recall_data)

        return precision_dict, recall_dict, fscore_dict

    def fill_missing_data(self, data):
        column_size = 2
        if len(data) < column_size:
            data.extend([np.nan] * (column_size - len(data)))
        return data

    def save_metrics_to_csv(self, metrics_dict, output_dir, filename):
        df = pd.DataFrame(metrics_dict)
        logging.info(f"Saving {filename} metrics to csv")
        df.to_csv(output_dir + f"{filename}.csv", index=False)
        logging.info(f"Path: {output_dir + filename}.csv")
