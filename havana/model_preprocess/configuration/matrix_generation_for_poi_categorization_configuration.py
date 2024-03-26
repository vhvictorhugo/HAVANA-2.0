from enum import Enum


class MatrixGenerationForPoiCategorizationConfiguration(Enum):
    DATASET_COLUMNS = (
        "dataset_columns",
        {
            "gowalla": {
                "datetime_column": "local_datetime",
                "userid_column": "userid",
                "locationid_column": "placeid",
                "country_column": "country_name",
                "state_column": "state_name",
                "category_column": "category",
                "category_name_column": "category",
                "latitude_column": "latitude",
                "longitude_column": "longitude",
            }
        },
    )

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_key(self):
        return self.value[0]

    def get_value(self):
        return self.value[1]
