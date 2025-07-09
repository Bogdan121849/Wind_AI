import pandas as pd
import os


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

    def create_windpark_dataset(self):
        forecast = self.datasets["met_forecast"].rename(columns={"sid": "windpark"})
        nowcast = self.datasets["met_nowcast"]
        meta = self.datasets["meta"]
        power = self.datasets["power"]

        if nowcast.index.name == "time":
            nowcast = nowcast.reset_index()
        if forecast.index.name == "time":
            forecast = forecast.reset_index()

        # Merge nowcast and forecast
        weather_combined = pd.merge(
            nowcast, forecast,
            on=["windpark", "time"],
            how="inner",
            suffixes=("_nowcast", "_forecast")
        )

        # Merge with meta
        full_data = pd.merge(
            weather_combined,
            meta,
            left_on="windpark",
            right_on="substation_name",
            how="left"
        )

        # Drop unnecessary columns
        full_data.drop(columns=[
            'eic_code', 'substation_name', 'prod_start_new', 'time_ref', 'lt', 'operating_power_max'
        ], inplace=True, errors='ignore') 

        # Merge with power data
        df_windpark = pd.merge(
            full_data,
            power,
            on=["time", "bidding_area"],
            how="inner"
        )

        # Reorder and sort
        first_cols = ['time', 'bidding_area', 'windpark']
        other_cols = [col for col in df_windpark.columns if col not in first_cols]
        df_windpark = df_windpark[first_cols + other_cols]
        df_windpark = df_windpark.sort_values(by='time').reset_index(drop=True)

        print(f" Final windpark dataset shape: {df_windpark.shape}")
        return df_windpark
    
    def create_region_dataset(self, df_windpark):
        # Count unique windparks per region
        region_park_counts = (
            df_windpark[['bidding_area', 'windpark']]
            .drop_duplicates()
            .groupby('bidding_area')
            .size()
            .reset_index(name='num_windparks')
        )

        # Merge into windpark-level data
        df_region = pd.merge(df_windpark, region_park_counts, on='bidding_area', how='left')

        # Drop windpark to generalize to region-level
        df_region.drop(columns=['windpark'], inplace=True)

        # Sort and reorder columns
        df_region = df_region.sort_values(by='time').reset_index(drop=True)
        first_cols = ['time', 'bidding_area', 'num_windparks']
        other_cols = [col for col in df_region.columns if col not in first_cols]
        df_region = df_region[first_cols + other_cols]

        print(f"Final region dataset shape: {df_region.shape}")
        return df_region
    
    def save_datasets(self, save_dir="WindAi/deep_learning/created_datasets", **dataframes):
        os.makedirs(save_dir, exist_ok=True)

        for name, df in dataframes.items():
            path = os.path.join(save_dir, f"{name}.parquet")
            df.to_parquet(path, index=False)
            print(f"Saved '{name}' to: {path}")

if __name__ == "__main__":
    
    
    preproc = Preprocessing_raw(
        met_forecast="/home2/s5549329/windAI_rug/WindAi/given_datasets/met_forecast.parquet",
        met_nowcast="/home2/s5549329/windAI_rug/WindAi/given_datasets/met_nowcast.parquet",
        power="/home2/s5549329/windAI_rug/WindAi/given_datasets/wind_power_per_bidzone.parquet",
        meta="/home2/s5549329/windAI_rug/WindAi/given_datasets/windparks_bidzone.csv"
    )

    preproc.read_datasets()
    preproc.datasets["power"] = reshape_power_data(preproc.datasets["power"])
    preproc.time_cropping()
    preproc.filter_common_windparks()
    df_windpark = preproc.create_windpark_dataset()
    df_region = preproc.create_region_dataset(df_windpark)
    preproc.save_datasets(
        windpark=df_windpark,
        region=df_region
        )       