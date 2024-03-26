import time

from model_preprocess.loader.file_loader import FileLoader


class MatrixGenerationForPoiCategorizationLoader(FileLoader):
    def adjacency_features_matrices_to_csv(self, files, files_names):
        for i in range(len(files)):
            try:
                file = files[i]
                file_name = files_names[i]
                self.save_df_to_csv(file, file_name, "a")
            except OSError as e:
                print(f"Erro ao salvar {file_name}: {e}")
                time.sleep(8)
                file = files[i]
                file_name = files_names[i]
                self.save_df_to_csv(file, file_name, "a")
