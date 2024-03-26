class BasePoiCategorizationConfiguration:
    def __init__(self):
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

        self.CATEGORIES_TO_INT = {
            "gowalla": {"7_categories": self.GOWALLA_7_CATEGORIES},
            "user_tracking": {"7_categories": self.GOWALLA_7_CATEGORIES},
        }
