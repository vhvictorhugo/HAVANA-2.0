import logging
import warnings

import torch
from pytorch_lightning import seed_everything
from srai.embedders import GeoVexEmbedder
from srai.h3 import ring_buffer_h3_regions_gdf
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import GEOFABRIK_LAYERS
from srai.neighbourhoods import H3Neighbourhood
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

from havana.embeddings.Embedder import Embedder


class GeoVex(Embedder):
    """
    GeoVex embeddings generation class

    Args:
        state (str): State to generate embeddings
        embeddings_dimension (int): Embeddings dimensions
        h3_resolution (int): H3 resolution
        metadata (dict): Metadata dictionary

    Functions:
        run: Run embeddings generation
    """

    def __init__(self, state: str, embeddings_dimension: int, h3_resolution: int, metadata: dict):
        super().__init__(
            state=state,
            embedder_name="geovex",
            embeddings_dimension=embeddings_dimension,
            h3_resolution=h3_resolution,
            metadata=metadata,
        )

    def run(self) -> None:
        """
        Generate GeoVex embeddings for a given state
        """
        SEED = 71
        seed_everything(SEED)

        # states names that are also cities names
        cities_states = ["New York"]

        logging.info("Generating area GDF")
        area_gdf = (
            geocode_to_region_gdf(f"{self.state} State, United States")
            if self.state in cities_states
            else geocode_to_region_gdf(f"{self.state}, United States")
        )

        k_ring_buffer_radius = 4
        logging.info("Generating regionalizer and base H3 regions")
        regionalizer = H3Regionalizer(resolution=self.h3_resolution)
        base_h3_regions = regionalizer.transform(area_gdf)
        logging.info("Generating buffered H3 regions")
        buffered_h3_regions = ring_buffer_h3_regions_gdf(base_h3_regions, distance=k_ring_buffer_radius)
        buffered_h3_geometry = buffered_h3_regions.unary_union

        tags = GEOFABRIK_LAYERS
        loader = OSMPbfLoader()
        logging.info("Loading features")
        features_gdf = loader.load(buffered_h3_geometry, tags)

        logging.info("Joining features")
        joiner = IntersectionJoiner()
        joint_gdf = joiner.transform(buffered_h3_regions, features_gdf)

        logging.info("Generating neighbourhood")
        neighbourhood = H3Neighbourhood(buffered_h3_regions)

        embedder = GeoVexEmbedder(
            target_features=GEOFABRIK_LAYERS,
            batch_size=10,
            neighbourhood_radius=k_ring_buffer_radius,
            convolutional_layers=2,
            embedding_size=self.embeddings_dimension,
        )

        logging.info("Fitting embeddings")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            embeddings = embedder.fit_transform(
                regions_gdf=buffered_h3_regions,
                features_gdf=features_gdf,
                joint_gdf=joint_gdf,
                neighbourhood=neighbourhood,
                trainer_kwargs={
                    # "max_epochs": 20, # uncomment for a longer training
                    "max_epochs": 5,
                    "accelerator": ("cpu" if torch.backends.mps.is_available() else "auto"),
                },
                learning_rate=0.001,
            )

        self._write_embeddings(embeddings)
