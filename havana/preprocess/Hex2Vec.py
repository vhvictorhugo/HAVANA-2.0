import warnings

import pandas as pd
from pytorch_lightning import seed_everything
from srai.embedders import Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMOnlineLoader
from srai.neighbourhoods import H3Neighbourhood
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf


class Hex2Vec:
    """
    Hex2Vec embeddings generation class

    Args:
        state (str): State to generate embeddings
        embeddings_dimension (int): Embeddings dimensions
        h3_resolution (int): H3 resolution
        metadata (dict): Metadata dictionary

    Functions:
        _write_embeddings: Write embeddings to intermediate data
        run: Run embeddings generation
    """

    def __init__(self, state: str, embeddings_dimension: int, h3_resolution: int, metadata: dict):
        self.state = state
        self.embeddings_dimension = embeddings_dimension
        self.h3_resolution = h3_resolution
        self.metadata = metadata

    def _write_embeddings(self, embeddings: pd.DataFrame) -> None:
        """
        Write embeddings to intermediate data

        Args:
            embeddings (pd.DataFrame): Embeddings data
        """
        embeddings.to_parquet(
            self.metadata["intermediate"]["embeddings"].format(
                embedder="hex2vec", state=self.state, embeddings_dimension=self.embeddings_dimension
            )
        )

    def run(self) -> None:
        """
        Generate Hex2Vec embeddings for a given state
        """
        SEED = 71
        seed_everything(SEED)

        area_gdf = geocode_to_region_gdf(f"{self.state}, United States")

        tags = {
            "leisure": "park",
            "landuse": "forest",
            "amenity": ["bar", "restaurant", "cafe"],
            "water": "river",
            "sport": "soccer",
        }

        loader = OSMOnlineLoader()
        features_gdf = loader.load(area_gdf, tags)

        regionalizer = H3Regionalizer(resolution=self.h3_resolution)
        regions_gdf = regionalizer.transform(area_gdf)

        joiner = IntersectionJoiner()
        joint_gdf = joiner.transform(regions_gdf, features_gdf)

        neighbourhood = H3Neighbourhood(regions_gdf)
        embedder = Hex2VecEmbedder([15, self.embeddings_dimension])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = embedder.fit_transform(
                regions_gdf,
                features_gdf,
                joint_gdf,
                neighbourhood,
                trainer_kwargs={"max_epochs": 5, "accelerator": "cpu"},
                batch_size=100,
            )

        self._write_embeddings(embeddings)
