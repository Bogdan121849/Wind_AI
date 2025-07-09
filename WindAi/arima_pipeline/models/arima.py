from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

number = 1
print(f"NO_{number}")
df = pd.read_parquet(f"WindAi/created_datasets/arima_power_no{number}.parquet")
df = df.drop(columns=["bidding_area", "time"])


power_series = df["power_MW"].values

forecast_horizon = 61
train = train = power_series[:-forecast_horizon]
test = power_series[-forecast_horizon:]


forecast_horizon = 61  
errors = []

model = ARIMA(train, order=(24, 1, 2))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))
rmse = mean_squared_error(test, forecast)
print(f"ARIMA Test RMSE (train/test only): {rmse:.2f}")

print(model_fit.summary())
model_fit.save(f"WindAi/arima_regions_weights/arima_no{number}_model.pkl")


plt.figure(figsize=(10, 5))
plt.plot(test[:61], label="True", marker="o")
plt.plot(forecast[:61], label="Forecast", marker="x")
plt.legend()
plt.title("First 61-hour Forecast vs Actual")
plt.xlabel("Hour")
plt.ylabel("Power (MW)")
plt.grid(True)

plt.savefig(f"WindAi/forecast_vs_actual_61h_{number}.png", dpi=300)  
plt.show()
