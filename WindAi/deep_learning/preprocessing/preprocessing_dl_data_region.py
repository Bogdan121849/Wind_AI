import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessing_DL_region:
    def __init__(self, path, region_name, features_to_drop, scale=True):
        self.path = path
        self.region_name = region_name
        self.features_to_drop = features_to_drop or ["power_MW", "bidding_area", "time"]
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self.target = None
        self.features_scaled = None

    def _add_cyclic_time_features(self, df: pd.DataFrame, time_col: str = "time", drop_original: bool = True) -> pd.DataFrame:
        """
        Adds sinusoidal time encodings (day and year) to the DataFrame based on a datetime column.
        """
        time_df = pd.to_datetime(df[time_col], format='%d.%m.%Y %H:%M:%S', errors='coerce')

        timestamp_s = time_df.map(pd.Timestamp.timestamp)

        day = 24 * 60 * 60
        year = 365.2425 * day

        df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
        df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
        df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
        df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

        if drop_original:
            df = df.drop(columns=[time_col], errors="ignore")

        return df

    def fit_transform(self, save_path="/home2/s5549329/windAI_rug/WindAi/deep_learning/created_datasets"):
        df = pd.read_parquet(self.path)
        region_df = df[df["bidding_area"] == self.region_name].copy()

        region_df = region_df.dropna(subset=["power_MW"])

        region_df = self._add_cyclic_time_features(region_df)

        self.target = region_df["power_MW"].values

        features = region_df.drop(columns=self.features_to_drop, errors="ignore")
        features = features.select_dtypes(include=["number"])

        if self.scale:
            features_scaled = self.scaler.fit_transform(features)
            self.features_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
        else:
            self.features_scaled = features.copy()

        df_combined = self.features_scaled.copy()
        df_combined["power_MW"] = self.target

        if save_path:
            file_name = f"{save_path}/scaled_features_power_MW_{self.region_name}.parquet"
            df_combined.to_parquet(file_name, index=False)
            print(f"Saved preprocessed data for {self.region_name} to {file_name}")

        return self.features_scaled, self.target


if __name__ == "__main__":
    preprocessor = Preprocessing_DL_region(
        path="/home2/s5549329/windAI_rug/WindAi/deep_learning/created_datasets/region.parquet",
        region_name="ELSPOT NO4",
        features_to_drop=["power_MW", "bidding_area", "time"]
    )

    X, y = preprocessor.fit_transform()