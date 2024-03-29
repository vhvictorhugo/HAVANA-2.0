import json
import logging

import h3
import pandas as pd


class EmbeddingsPreProcess:
    """
    Preprocess users embeddings data

    Args:
        state (str): State
        embeddings_dimension (int): Embeddings dimension
        embedder (str): Embedder
        metadata (dict): Metadata

    Functions:
        _read_embeddings: Read embeddings from intermediate data
        _read_checkins: Read checkins data
        _generate_h3_cell: Generate H3 cell from latitude and longitude
        _generate_user_embeddings: Generate user embeddings from checkins embeddings
        _write_user_embeddings: Write user embeddings to intermediate data
        run: Preprocess users embeddings data
    """

    def __init__(self, state: str, embeddings_dimension: int, embedder: str, metadata: dict):
        self.state = state
        self.embeddings_dimension = embeddings_dimension
        self.embedder = embedder
        self.metadata = metadata

    def _read_embeddings(self) -> pd.DataFrame:
        """
        Read embeddings from intermediate data

        Returns:
            pd.DataFrame: Embeddings data
        """
        return pd.read_parquet(
            self.metadata["intermediate"]["embeddings"].format(
                embedder=self.embedder, state=self.state, embeddings_dimension=self.embeddings_dimension
            )
        )

    def _read_checkins(self) -> pd.DataFrame:
        """
        Read checkins data

        Returns:
            pd.DataFrame: Checkins data
        """
        return pd.read_csv(self.metadata["intermediate"]["checkins"].format(state=self.state)).rename(
            columns={"userid": "user_id"}
        )

    def _generate_h3_cell(self, row: pd.Series) -> str:
        """
        Generate H3 cell from latitude and longitude

        Args:
            row (pd.Series): Row of a DataFrame

        Returns:
            str: H3 cell
        """
        lat = row["latitude"]
        lon = row["longitude"]
        resolution = 9

        h3_cell = h3.latlng_to_cell(lat, lon, resolution)
        return h3_cell

    def _criar_dicionario(self, n):
        """
        Cria um dicionário com n chaves e valores.

        Argumentos:
            n: Número inteiro que define o número de chaves e valores.

        Retorno:
            Dicionário com chaves de '0' a 'n-1' e valores de 'feature_1' a 'feature_n-1'.
        """
        dicionario = {}
        for i in range(n):
            chave = i
            valor = "feature_" + str(i + 1)
            dicionario[chave] = valor
        return dicionario

    def _generate_user_embeddings(self, checkins_embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate user embeddings from checkins embeddings

        Args:
            checkins_embeddings_df (pd.DataFrame): Checkins embeddings data

        Returns:
            pd.DataFrame: User embeddings data
        """
        users = checkins_embeddings_df["user_id"].unique()
        user_embeddings = pd.DataFrame(columns=["user_id", "embeddings"])

        for user in users:
            user_checkins = checkins_embeddings_df[checkins_embeddings_df["user_id"] == user]
            feature_cols = checkins_embeddings_df.filter(like="feature").columns.tolist()
            user_embedding = user_checkins[feature_cols].values
            user_embedding = json.dumps(user_embedding.tolist())

            user_embeddings = pd.concat(
                [user_embeddings, pd.DataFrame({"user_id": user, "embeddings": [user_embedding]})], ignore_index=True
            )

        return user_embeddings

    def _write_user_embeddings(self, user_embeddings_df: pd.DataFrame) -> None:
        """
        Write user embeddings to intermediate data

        Args:
            user_embeddings_df (pd.DataFrame): User embeddings data
        """
        path = self.metadata["processed"]["user_embeddings"].format(
            embedder=self.embedder, state=self.state, embeddings_dimension=self.embeddings_dimension
        )
        logging.info("Writing user embeddings to processed data")
        user_embeddings_df.to_csv(path, index=False, sep=",")
        logging.info(f"Path: {path}")

    def run(self):
        """
        Preprocess users embeddings data
        """
        embeddings_df = self._read_embeddings().reset_index()
        checkins_df = self._read_checkins()

        checkins_df["region_id"] = checkins_df.apply(self._generate_h3_cell, axis=1)

        rename_columns = self._criar_dicionario(self.embeddings_dimension)

        checkins_embeddings_df = (
            checkins_df.merge(embeddings_df, on="region_id").rename(columns=rename_columns).sort_values(by=["user_id"])
        )

        user_embeddings_df = self._generate_user_embeddings(checkins_embeddings_df)

        self._write_user_embeddings(user_embeddings_df)
