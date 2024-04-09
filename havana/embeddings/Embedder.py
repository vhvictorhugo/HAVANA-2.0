import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class Embedder(ABC):
    """
    Abstract class for Embeddings Generation

    Args:
        state (str): State to generate embeddings for
        embedder_name (str): Embedder name
        embeddings_dimension (int): Embeddings dimensions
        h3_resolution (int): H3 resolution
        metadata (dict): Metadata for intermediate data

    Functions:
        run: Generate embeddings for a given state
        _write_embeddings: Write embeddings to intermediate data
    """

    def __init__(self, state: str, embedder_name: str, embeddings_dimension: int, h3_resolution: int, metadata: dict):
        self.state = state
        self.embedder_name = embedder_name
        self.embeddings_dimension = embeddings_dimension
        self.h3_resolution = h3_resolution
        self.metadata = metadata

    @abstractmethod
    def run(self) -> None:
        """
        Generate embeddings for a given state
        """
        pass

    def _write_embeddings(self, embeddings: pd.DataFrame) -> None:
        """
        Write embeddings to intermediate data

        Args:
            embeddings (pd.DataFrame): Embeddings data
        """
        path = self.metadata["intermediate"]["embeddings"].format(embedder=self.embedder_name, state=self.state)
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + f"{self.embeddings_dimension}_dimension_{self.h3_resolution}_resolution.parquet"
        logging.info(f"Writing {self.embedder_name.upper()} embeddings")
        embeddings.to_parquet(path)
        logging.info(f"Path: {path}")
