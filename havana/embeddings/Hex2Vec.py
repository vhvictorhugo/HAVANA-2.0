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
            "sport": [
                "shooting",
                "cycling",
                "boxing",
                "horse_racing",
                "table_tennis",
                "athletics",
                "yoga",
                "gymnastics",
                "soccer",
                "boules",
                "crossfit",
                "swimming",
                "basketball",
                "climbing",
                "rowing",
                "canoe",
                "running",
                "volleyball",
                "multi",
                "badminton",
                "equestrian",
                "fitness",
                "tennis",
                "skateboard",
            ],
            "amenity": [
                "cinema",
                "studio",
                "pub",
                "college",
                "bbq",
                "nightclub",
                "arts_centre",
                "music_school",
                "social_facility",
                "social_centre",
                "theatre",
                "cafe",
                "library",
                "casino",
                "kindergarten",
                "community_centre",
                "fountain",
                "restaurant",
                "university",
                "fast_food",
                "bar",
                "dancing_school",
                "school",
                "ice_cream",
            ],
            "office": [
                "financial",
                "water_utility",
                "company",
                "foundation",
                "lawyer",
                "government",
                "estate_agent",
                "architect",
                "coworking",
                "accountant",
                "notary",
                "diplomatic",
                "telecommunication",
                "newspaper",
                "research",
                "ngo",
                "engineer",
                "advertising_agency",
                "logistics",
                "it",
                "insurance",
                "yes",
                "association",
            ],
            "shop": [
                "kiosk",
                "carpet",
                "lottery",
                "craft",
                "butcher",
                "military_surplus",
                "hifi",
                "tailor",
                "shoes",
                "religion",
                "locksmith",
                "hardware",
                "farm",
                "convenience",
                "ticket",
                "vacant",
                "appliance",
                "frame",
                "chocolate",
                "lighting",
                "money_lender",
                "copyshop",
                "confectionery",
                "jewelry",
                "garden_centre",
                "car",
                "electrical",
                "anime",
                "nutrition_supplements",
                "houseware",
                "bag",
                "musical_instrument",
                "books",
                "seafood",
                "chemist",
                "doityourself",
                "motorcycle",
                "perfumery",
                "fabric",
                "baby_goods",
                "pawnbroker",
                "stationery",
                "furniture",
                "radiotechnics",
                "pastry",
                "bicycle",
                "hearing_aids",
                "tiles",
                "interior_decoration",
                "supermarket",
                "mall",
                "health_food",
                "pet",
                "second_hand",
                "music",
                "art",
                "tyres",
                "alcohol",
                "optician",
                "gas",
                "erotic",
                "cosmetics",
                "tobacco",
                "fishing",
                "medical_supply",
                "beverages",
                "kitchen",
                "bed",
                "mobile_phone",
                "coffee",
                "newsagent",
                "general",
                "photo",
                "florist",
                "sewing",
                "laundry",
                "outdoor",
                "department_store",
                "wholesale",
                "bakery",
                "variety_store",
                "glaziery",
                "toys",
                "gift",
                "sports",
                "beauty",
                "video_games",
                "herbalist",
                "party",
                "clothes",
                "hairdresser",
                "tea",
                "computer",
                "storage_rental",
            ],
            "leisure": [
                "sports_centre",
                "fishing",
                "track",
                "water_park",
                "playground",
                "marina",
                "horse_riding",
                "adult_gaming_centre",
                "stadium",
                "dog_park",
                "pitch",
                "fitness_centre",
                "fitness_station",
                "golf_course",
                "park",
                "garden",
                "dance",
            ],
            "aeroway": ["aerodrome", "helipad"],
            "tourism": [
                "artwork",
                "picnic_site",
                "motel",
                "hotel",
                "museum",
                "zoo",
                "attraction",
                "information",
                "viewpoint",
                "guest_house",
                "camp_site",
                "hostel",
                "apartment",
                "gallery",
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
