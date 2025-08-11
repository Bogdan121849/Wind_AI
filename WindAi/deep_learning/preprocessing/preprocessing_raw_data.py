import pandas as pd
import os
import numpy as np


def reshape_power_data(power_df):
    power_long = power_df.reset_index().melt(
        id_vars="index",
        var_name="bidding_area",
        value_name="power_MW"
    ).rename(columns={"index": "time"})

    return power_long

class Preprocessing_raw:
    def __init__(self, **datasets):
        self.datasets_path = datasets   # {name: path}
        self.datasets = {}  # {name: loaded DataFrame}
        
    def read_datasets(self):
        for name, df_path in self.datasets_path.items():
            print(f"Loaded dataset {name} with path {df_path}")
            if df_path.endswith(".parquet"):
                df = pd.read_parquet(df_path)
            elif df_path.endswith(".csv"):
                df = pd.read_csv(df_path)
            else:
                raise ValueError(f"Unsupported file format for {name}: {df_path}")
            
            self.datasets[name] = df
            print(f"Loaded '{name}' with shape {df.shape}")

    def time_cropping(self):
        time_datasets_min = []
        time_datasets_max = []
        for df in self.datasets.values():
            if "time" in df.columns:
                time_datasets_min.append(df["time"].min())
                time_datasets_max.append(df["time"].max())
            
        start = max(time_datasets_min)
        end = min(time_datasets_max)

        for name, df in self.datasets.items():
            if "time" in df.columns:
                df_cropped = df[(df["time"] > start) & (df["time"] <= end)]
                self.datasets[name] = df_cropped

    def filter_common_windparks(self):
        meta_set = set(self.datasets["meta"]["substation_name"])
        nowcast_set = set(self.datasets["met_nowcast"]["windpark"])
        forecast_set = set(self.datasets["met_forecast"]["sid"])

        common = meta_set & nowcast_set & forecast_set
        print(f"Common windparks in all datasets: {len(common)}")
        print(sorted(common))

        self.datasets["meta"] = self.datasets["meta"][self.datasets["meta"]["substation_name"].isin(common)]
        self.datasets["met_nowcast"] = self.datasets["met_nowcast"][self.datasets["met_nowcast"]["windpark"].isin(common)]
        self.datasets["met_forecast"] = self.datasets["met_forecast"][self.datasets["met_forecast"]["sid"].isin(common)]

    def create_region_dataset(self):
        df_forecast = self.datasets["met_forecast"]
        df_nowcasting = self.datasets["met_nowcast"]
        df_metadata = self.datasets["meta"]
        df_power = self.datasets["power"]

        df_power_long = reshape_power_data(df_power)
        df_power_long["bidding_area"] = df_power_long["bidding_area"].str.replace("ELSPOT ", "")

        # Step 1: Compute forecast summary statistics
        forecast_cols_w_speed     = [col for col in df_forecast.columns if "ws10m_" in col]
        forecast_cols_w_direction = [col for col in df_forecast.columns if "wd10m_" in col]
        forecast_cols_t           = [col for col in df_forecast.columns if "t2m_" in col]
        forecast_cols_rh          = [col for col in df_forecast.columns if "rh2m_" in col]
        forecast_cols_mslp        = [col for col in df_forecast.columns if "mslp_" in col]
        forecast_cols_g           = [col for col in df_forecast.columns if "g10m_" in col]

        df_forecast["ws10m_mean"]   = df_forecast[forecast_cols_w_speed].mean(axis=1)
        df_forecast["ws10m_std"]    = df_forecast[forecast_cols_w_speed].std(axis=1)
        df_forecast["ws10m_min"]    = df_forecast[forecast_cols_w_speed].min(axis=1)
        df_forecast["ws10m_max"]    = df_forecast[forecast_cols_w_speed].max(axis=1)
        df_forecast["ws10m_median"] = df_forecast[forecast_cols_w_speed].median(axis=1)

        angles_rad = np.radians(df_forecast[forecast_cols_w_direction])

        # Compute circular mean
        mean_angle_rad = np.arctan2(
            np.mean(np.sin(angles_rad), axis=1),
            np.mean(np.cos(angles_rad), axis=1)
        )
        df_forecast["wd10m_mean"] = (np.degrees(mean_angle_rad) + 360) % 360

        # Linear std deviation
        df_forecast["wd10m_std"] = df_forecast[forecast_cols_w_direction].std(axis=1)


        df_forecast["t2m_mean"]    = df_forecast[forecast_cols_t].mean(axis=1)
        df_forecast["t2m_std"]     = df_forecast[forecast_cols_t].std(axis=1)
        df_forecast["t2m_min"]     = df_forecast[forecast_cols_t].min(axis=1)
        df_forecast["t2m_max"]     = df_forecast[forecast_cols_t].max(axis=1)
        df_forecast["t2m_median"]  = df_forecast[forecast_cols_t].median(axis=1)


        df_forecast["rh2m_mean"]   = df_forecast[forecast_cols_rh].mean(axis=1)
        df_forecast["rh2m_std"]    = df_forecast[forecast_cols_rh].std(axis=1)
        df_forecast["rh2m_min"]    = df_forecast[forecast_cols_rh].min(axis=1)
        df_forecast["rh2m_max"]    = df_forecast[forecast_cols_rh].max(axis=1)
        df_forecast["rh2m_median"] = df_forecast[forecast_cols_rh].median(axis=1)

        df_forecast["mslp_mean"]   = df_forecast[forecast_cols_mslp].mean(axis=1)
        df_forecast["mslp_std"]    = df_forecast[forecast_cols_mslp].std(axis=1)
        df_forecast["mslp_min"]    = df_forecast[forecast_cols_mslp].min(axis=1)
        df_forecast["mslp_max"]    = df_forecast[forecast_cols_mslp].max(axis=1)
        df_forecast["mslp_median"] = df_forecast[forecast_cols_mslp].median(axis=1)

        df_forecast["g10m_mean"]   = df_forecast[forecast_cols_g].mean(axis=1)
        df_forecast["g10m_std"]    = df_forecast[forecast_cols_g].std(axis=1)
        df_forecast["g10m_min"]    = df_forecast[forecast_cols_g].min(axis=1)
        df_forecast["g10m_max"]    = df_forecast[forecast_cols_g].max(axis=1)
        df_forecast["g10m_median"] = df_forecast[forecast_cols_g].median(axis=1)

        summary_cols = [
            "ws10m_mean", "ws10m_std", "ws10m_min", "ws10m_max", "ws10m_median",
            "wd10m_mean", "wd10m_std",
            "t2m_mean", "t2m_std", "t2m_min", "t2m_max", "t2m_median",
            "rh2m_mean", "rh2m_std", "rh2m_min", "rh2m_max", "rh2m_median",
            "mslp_mean", "mslp_std", "mslp_min", "mslp_max", "mslp_median",
            "g10m_mean", "g10m_std", "g10m_min", "g10m_max", "g10m_median"
        ]

        df_forecast_avg = df_forecast.groupby(["sid", "time"])[summary_cols].mean().reset_index()
        df_forecast_avg["time"] = pd.to_datetime(df_forecast_avg["time"])
        print(df_forecast_avg.head(5))

        # Step 2: Rename nowcasting columns
        df_nowcasting = df_nowcasting.rename(columns={"windpark": "sid"})
        df_nowcasting = df_nowcasting.reset_index()
        df_nowcasting["time"] = pd.to_datetime(df_nowcasting["time"])

        df_nowcasting = df_nowcasting.rename(columns={
            "air_temperature_2m": "t2m_now",
            "air_pressure_at_sea_level": "mslp_now",
            "relative_humidity_2m": "rh2m_now",
            "wind_speed_10m": "ws10m_now",
            "wind_direction_10m": "wd10m_now",
            "precipitation_amount": "precip_now",
        })
        print(df_nowcasting.head(5))

        # Step 3: Merge forecast and nowcasting
        df_merged_weather = pd.merge(df_forecast_avg, df_nowcasting, on=["sid", "time"], how="inner")
        print(df_merged_weather.head(5))

        # Step 4: Metadata cleanup
        df_metadata = df_metadata.rename(columns={"substation_name": "sid"})
        df_metadata["bidding_area"] = df_metadata["bidding_area"].str.replace("ELSPOT ", "")
        print(df_metadata.head(5))

        # Step 5: Add bidding_area to weather
        df_power_long_sid = pd.merge(
            df_power_long,
            df_metadata[["bidding_area", "sid"]],
            on="bidding_area",
            how="left"
        )
        print(df_power_long_sid.head(5))

        df_merged_weather["time"] = pd.to_datetime(df_merged_weather["time"])
        df_power_long_sid["time"] = pd.to_datetime(df_power_long_sid["time"])
        
        df_final_sid = pd.merge(
            df_merged_weather,
            df_power_long_sid[["time", "sid", "power_MW", "bidding_area"]],
            on=["time", "sid"],
            how="inner"
        )
        print(df_final_sid.head(5))

        # Step 7: Aggregate per region
        df_final = df_final_sid.drop(columns=["sid"])  # we aggregate across all sids per region
        print(df_final.head(5))

        regions = df_final["bidding_area"].unique()
        df_aggregated_regions = {}

        for region in regions:
            df_region = df_final[df_final["bidding_area"] == region].copy()
            df_region_agg = df_region.groupby("time").mean(numeric_only=True).reset_index()
            df_aggregated_regions[region] = df_region_agg

        print(f"Aggregated region datasets: {list(df_aggregated_regions.keys())}")
        return df_aggregated_regions

    
    def save_datasets(self, save_dir="WindAi/deep_learning/created_datasets", **dataframes):
        os.makedirs(save_dir, exist_ok=True)

        for name, df in dataframes.items():
            path = os.path.join(save_dir, f"{name}.parquet")
            df.to_parquet(path, index=False)
            print(f"Saved '{name}' to: {path}")
