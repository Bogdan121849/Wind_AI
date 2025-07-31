import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def add_lag_features(df, target_col='power_MW', max_lag=7):
    for lag in range(1, max_lag + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


class Preprocessing_DL_region:
    def __init__(self, path, region_name, scale=True):
        self.path = path
        self.region_name = region_name
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
        region_df = df.copy()

        region_df = self._add_cyclic_time_features(region_df, drop_original=False)
        region_df = add_lag_features(region_df, target_col="power_MW", max_lag=7)

        region_df = region_df.dropna().reset_index(drop=True)
        features = region_df.select_dtypes(include=["number"]).copy()

        cyclic_cols = ["Day sin", "Day cos", "Year sin", "Year cos"]
        cyclic_features = features[cyclic_cols]
        features_to_scale = features.drop(columns=cyclic_cols)

        if self.scale:
            scaled_array = self.scaler.fit_transform(features_to_scale)
            scaled_features = pd.DataFrame(scaled_array, columns=features_to_scale.columns, index=features_to_scale.index)
        else:
            scaled_features = features_to_scale.copy()

        self.features_scaled = pd.concat([scaled_features, cyclic_features], axis=1)

        self.target = self.features_scaled["power_MW"].values
        self.features_scaled = self.features_scaled.drop(columns=["power_MW"])

        df_combined = self.features_scaled.copy()
        df_combined["power_MW"] = self.target

        non_numeric = region_df.select_dtypes(exclude=["number"]).copy()
        non_numeric = non_numeric.loc[df_combined.index]

        df_combined = pd.concat([non_numeric, df_combined], axis=1)

        if save_path:
            file_name = f"{save_path}/scaled_features_power_MW_{self.region_name}.parquet"
            df_combined.to_parquet(file_name, index=False)
            print(f"Saved preprocessed data for {self.region_name} to {file_name}")

        return self.features_scaled, self.target


if __name__ == "__main__":
    input_dir = "/home2/s5549329/windAI_rug/WindAi/deep_learning/created_datasets"
    output_dir = input_dir 

    for filename in os.listdir(input_dir):
        if filename.endswith(".parquet"):
            filepath = os.path.join(input_dir, filename)
            region_name = os.path.splitext(filename)[0]
            preproc_dl = Preprocessing_DL_region(filepath, region_name)
            preproc_dl.fit_transform()



  