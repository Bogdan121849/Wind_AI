import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_parquet("model_dataset.parquet")


df['valid_time'] = pd.to_datetime(df['valid_time'])
df = df.sort_values(['zone', 'valid_time'])


df['power_pred'] = df.groupby('zone')['power'].shift(1)


df_eval = df.dropna(subset=['power_pred']).copy()

# 5. Evaluate RMSE per zone
rmse_per_zone = (
    df_eval
      .groupby('zone')
      .apply(lambda g: np.sqrt(mean_squared_error(g['power'], g['power_pred'])))
      .rename("rmse")
)
print("RMSE by zone:\n", rmse_per_zone)

# 6. If you want to “forecast” the next hour for each zone:
#    simply take the last observed power in each zone
next_hour_forecast = (
    df
      .groupby('zone')
      .apply(lambda g: g.iloc[-1]['power'])
      .rename("forecast_next_hour")
)
print("\nNext-hour persistence forecast:\n", next_hour_forecast)

plt.figure(figsize=(15, 5))
plt.plot(df_eval['valid_time'], df_eval['power'], label='Actual', alpha=0.6)
plt.plot(df_eval['valid_time'], df_eval['power_pred'], label='Persistence Prediction', alpha=0.6)
plt.xlabel('Time')
plt.ylabel('Power')
plt.title('Actual vs. Persistence Forecast (All Zones)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/actual_vs_persistence_all_zones.png')
plt.show()

# Scatter‐plot of predicted vs. actual power for the whole dataset
plt.figure(figsize=(6, 6))
plt.scatter(df_eval['power'], df_eval['power_pred'], s=2, alpha=0.3)
min_val = min(df_eval['power'].min(), df_eval['power_pred'].min())
max_val = max(df_eval['power'].max(), df_eval['power_pred'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
plt.xlabel('Actual Power')
plt.ylabel('Predicted Power')
plt.title('Actual vs. Predicted Scatter (All Zones)')
plt.tight_layout()
plt.savefig('plots/actual_vs_pred_scatter.png')
plt.show()
