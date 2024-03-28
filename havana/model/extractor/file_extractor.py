import pandas as pd
from scipy import sparse


class FileExtractor:
    def __init__(self):
        pass

    def read_csv(self, filename, dtypes_columns=None):
        if dtypes_columns is None:
            df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename, dtype=dtypes_columns, encoding="utf-8")

        return df.sample(frac=1, random_state=3)

    def read_npz(self, filename):
        return sparse.load_npz(filename)
