import logging
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
        path = self.metadata["intermediate"]["embeddings"].format(
            embedder="hex2vec", state=self.state, embeddings_dimension=self.embeddings_dimension
        )
        logging.info("Writing hex2vec embeddings")
        embeddings.to_parquet(path)
        logging.info(f"Path: {path}")

    def run(self) -> None:
        """
        Generate Hex2Vec embeddings for a given state
        """
        SEED = 71
        seed_everything(SEED)

        cities_states = ["New York"]

        area_gdf = (
            geocode_to_region_gdf(f"{self.state} State, United States")
            if self.state in cities_states
            else geocode_to_region_gdf(f"{self.state}, United States")
        )

        tags = {
            "shop": True,
            "amenity": [
                "community_centre",
                # "youth_centre",
                "school",
                "library",
                "park",
                "place_of_worship",
                "social_club",
                "community_garden",
                "airport",
                "hotel",
                "station",
                "bus_station",
                "ferry_terminal",
                "bureau_de_change",
                "car_rental",
                "cinema",
                "theatre",
                "concert_hall",
                "museum",
                "bar",
                "restaurant",
                "cafe",
                "fast_food",
                "food_court",
                "ice_cream_parlour",
                "vending_machine",
                "food_truck",
                "biergarten",
                "bbq",  # or bbr_area
                "nightclub",
                "pub",
                "lounge",
                "jazz_club",
                "karaoke_bar",
                "live_music_venue",  # or music_venue
                "comedy_club",
                "casino",
                "adult_entertainment",
                "picnic_area",
            ],
            "place": "square",
            "landuse": "recreation_ground",
            "office": "travel_agent",
            "sport": "climbing",
            "leisure": [
                "amusement_park",
                "bowling_alley",
                "ice_rink",
                "playground",
                "sports_centre",
                "dance_club",
                "fishing_area",
                "park",
                "swimming_pool",
                "campsite",
                "track",
            ],
            "tourism": [
                "information",
                "viewpoint",
                "motel",
            ],
            "natural": [
                "water",
                "waterfall",
                "beach",
                "forest",
                "wood",
                "park",
            ],
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
