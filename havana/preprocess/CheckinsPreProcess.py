import logging
from pathlib import Path

import pandas as pd


class CheckinsPreProcess:
    """
    Checkins preprocessing class

    Args:
        state (str): State to execute the pipeline
        metadata (dict): Metadata dictionary

    Functions:
        _read_checkins: Read checkins data from raw data
        _format_checkins: Format checkins data
        _filter_checkins: Filter checkins data, removing users with less than 2 locals visited
        _write_checkins: Write checkins data to intermediate data
        run: Run checkins preprocessing
    """

    def __init__(self, state: str, metadata: dict):
        self.state = state
        self.metadata = metadata

    def _read_checkins(self) -> pd.DataFrame:
        """
        Read checkins data from raw data

        Returns:
            pd.DataFrame: Checkins data
        """
        checkins_df = pd.read_csv(
            self.metadata["raw"]["checkins"].format(state=self.state),
            index_col=False,
            usecols=["userid", "datetime", "lat", "lng", "placeid", "categoryid"],
        )

        return checkins_df

    def _format_checkins(self, checkins_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format checkins data

        Args:
            checkins_df (pd.DataFrame): Checkins data

        Returns:
            pd.DataFrame: Formatted checkins data
        """
        checkins_df["country"] = "United States"
        checkins_df["state"] = self.state
        checkins_df["datetime"] = pd.to_datetime(checkins_df["datetime"])

        order_cols = ["userid", "categoryid", "placeid", "datetime", "lat", "lng", "country", "state"]
        checkins_df = checkins_df[order_cols]

        cols = [
            "userid",
            "category",
            "placeid",
            "local_datetime",
            "latitude",
            "longitude",
            "country_name",
            "state_name",
        ]
        maps = dict(zip(checkins_df.columns, cols))
        checkins_df = checkins_df.rename(columns=maps)

        categories_from_to_df = pd.read_csv(self.metadata["raw"]["from_to_category_names"])

        checkins_df = pd.merge(checkins_df, categories_from_to_df, left_on="category", right_on="categoryid")

        checkins_df["category"] = checkins_df["name"]
        checkins_df = checkins_df.drop(["name", "categoryid"], axis=1)

        return checkins_df

    def _filter_checkins(self, checkins_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter checkins data, removing users with less than 2 locals visited

        Args:
            checkins_df (pd.DataFrame): Checkins data

        Returns:
            pd.DataFrame: Filtered checkins data
        """
        places_quantity_per_user = (
            checkins_df.groupby("userid")
            .agg({"placeid": "nunique"})
            .sort_values(by=["placeid"], ascending=False)
            .reset_index()
        )

        places_quantity_per_user = places_quantity_per_user[places_quantity_per_user["placeid"] >= 2]

        valid_users = places_quantity_per_user["userid"].unique()

        checkins_df = checkins_df[checkins_df["userid"].isin(valid_users)]

        return checkins_df

    def _write_checkins(self, checkins_df: pd.DataFrame) -> None:
        """
        Write checkins data to intermediate data

        Args:
            checkins_df (pd.DataFrame): Checkins data
        """
        path = self.metadata["intermediate"]["checkins"]
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + f"{self.state}.csv"
        logging.info("Writing checkins to intermediate data")
        checkins_df.to_csv(path, index=False)
        logging.info(f"Path: {path}")

    def run(self) -> None:
        """
        Run checkins preprocessing
        """
        checkins_df = self._read_checkins()
        checkins_df = self._format_checkins(checkins_df)
        checkins_df = self._filter_checkins(checkins_df)
        self._write_checkins(checkins_df)
