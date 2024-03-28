class BasePoiCategorizationConfiguration:
    def __init__(self):
        self.N_SPLITS = ("n_splits", 5, False, "number of splits (minimum 2) for k fold")

        self.N_REPLICATIONS = (
            "n_replications",
            1,
            False,
            "number of replications/executions (minimum 1) of training and evaluation process",
        )

        self.MINIMUM_RECORDS = ("minimum_records", 15)

        self.GOWALLA_7_CATEGORIES = [
            "Shopping",
            "Community",
            "Food",
            "Entertainment",
            "Travel",
            "Outdoors",
            "Nightlife",
        ]

        self.GOWALLA_7_CATEGORIES_TO_INT = {
            self.GOWALLA_7_CATEGORIES[i]: i for i in range(len(self.GOWALLA_7_CATEGORIES))
        }

        self.GOWALLA_7_CATEGORIES = {
            "Shopping": 0,
            "Community": 1,
            "Food": 2,
            "Entertainment": 3,
            "Travel": 4,
            "Outdoors": 5,
            "Nightlife": 6,
        }

        self.INT_TO_CATEGORIES = (
            "int_to_categories",
            {
                "gowalla": {
                    "7_categories": {
                        str(i): list(self.GOWALLA_7_CATEGORIES.keys())[i]
                        for i in range(len(list(self.GOWALLA_7_CATEGORIES)))
                    }
                }
            },
        )

        self.MAX_SIZE_MATRICES = (
            "max_size_matrices",
            3,
            False,
            "max size of the adjacency matrices and features (row size) ones",
        )

        self.REPORT_7_INT_CATEGORIES = (
            "report_7_int_categories",
            {
                "0": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "1": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "2": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "3": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "4": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "5": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "6": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "accuracy": [],
                "macro avg": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "weighted avg": {"precision": [], "recall": [], "f1-score": [], "support": []},
            },
            "report",
        )

        self.REPORT_MODEL = (
            "report_model",
            {
                "7_categories": self.REPORT_7_INT_CATEGORIES[1],
            },
        )
