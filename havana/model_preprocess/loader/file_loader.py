import os
import time

from scipy import sparse


class FileLoader:
    def save_df_to_csv(self, df, filename, mode="w"):
        header = not os.path.exists(filename)
        try:
            df.to_csv(filename, index=False, mode=mode, header=header)
        except (OSError, ValueError):
            time.sleep(8)
            df.to_csv(filename, index=False, mode=mode, header=header)

    def save_sparse_matrix_to_npz(self, matrix, filename):
        try:
            sparse.save_npz(filename, matrix)

        except (OSError, ValueError):
            time.sleep(8)
            sparse.save_npz(filename, matrix)
