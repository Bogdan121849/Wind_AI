import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from WindAi.deep_learning.preprocessing.preprocess_windowing_region import WindowGenerator
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class RNN:
    def __init__(self, input_width, label_width, num_features, region_number, name="GRU_weak"):
        self.input_width = input_width
        self.label_width = label_width
        self.num_features = num_features
        self.region_number = region_number
        self.name = name

        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_width, self.num_features)))
        model.add(tf.keras.layers.GaussianNoise(0.1))
        model.add(tf.keras.layers.GRU(256, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.GRU(128, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.GRU(64, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

        def combined_loss(y_true, y_pred):
            mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
            mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
            huber = tf.keras.losses.huber(y_true, y_pred, delta=1.0)

            pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            smoothness = tf.reduce_mean(tf.square(pred_diff))

            return (0.4 * tf.reduce_mean(mse) + 
                   0.3 * tf.reduce_mean(mae) + 
                   0.25 * tf.reduce_mean(huber) + 
                   0.05 * smoothness)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4,
                        beta_1=0.9, 
                        beta_2=0.999,
                        epsilon=1e-7,  
                        clipnorm=1.0
                        ),
            loss=combined_loss,
            metrics=["mae", "mse"]
        )

        return model
    
    def fit(self, window, weights_dir, epochs=100):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            min_delta=1e-6
        )

        callbacks = [early_stop]

        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss",
        #     factor=0.5,  
        #     patience=5,   
        #     min_lr=1e-8,  
        #     verbose=1,
        #     cooldown=3    
        # )

            # Reduce on plateau - more responsive
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.7,  # Less aggressive reduction
            patience=8,   # Faster response
            min_lr=1e-7,
            verbose=1,
            cooldown=2
        )
        callbacks.append(reduce_lr)

        history = self.model.fit(
                window.train,
                validation_data=window.val,
                epochs=epochs,
                callbacks=callbacks
            )
        return history
    
    def summary(self):
        self.model.summary()

    def predict_last_window(self, window):
        for x, y in window.test.take(1):
            prediction = self.model.predict(x)
        return prediction, y
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        print(f" Model weights saved to: {filepath}")
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        print(f" Model weights loaded from: {filepath}")
    
    def plot_prediction(self, prediction, y_true, save_path):
        plt.figure(figsize=(12, 4))
        plt.plot(y_true[0, :, 0], label="Actual")
        plt.plot(prediction[0, :, 0], label="Predicted")
        plt.title("Power_MW Forecast: Next 61 Hours")
        plt.xlabel("Hour")
        plt.ylabel("Power_MW")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def plot_learning_curves(self, history, save_path):

        history_df = pd.DataFrame(history.history)

        plt.figure(figsize=(10, 5))
        plt.plot(history_df["loss"], label="Train Loss (MSE)")
        plt.plot(history_df["val_loss"], label="Val Loss (MSE)")
        if "mae" in history_df and "val_mae" in history_df:
            plt.plot(history_df["mae"], label="Train MAE", linestyle="--")
            plt.plot(history_df["val_mae"], label="Val MAE", linestyle="--")

        plt.title(f"Learning Curves - Region {self.region_number}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / MAE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Learning curves saved to: {save_path}")

    def evaluate_model(self, dataset, dataset_name):
        y_trues = []
        y_preds = []

        for x, y_true in dataset:
            y_pred = self.model.predict(x, verbose=0)
            y_trues.append(y_true.numpy().reshape(-1))
            y_preds.append(y_pred.reshape(-1))

        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)

        mse = mean_squared_error(y_trues, y_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_trues, y_preds)

        print(f"\n {dataset_name} Set Evaluation (Region {self.region_number}):")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - MSE:  {mse:.2f}")
        print(f"   - MAE:  {mae:.2f}")


if __name__ == "__main__":
    region_number = 1  # You can loop over multiple later
    input_width = 61
    label_width = 61
    shift = 0

    data_dir = "/home2/s5549329/windAI_rug/WindAi/deep_learning/created_datasets"
    weight_dir = "/home2/s5549329/windAI_rug/WindAi/deep_learning/weights"
    plot_dir = "/home2/s5549329/windAI_rug/WindAi/deep_learning/results"

    for region_number in range(1, 5):  # Loop over NO1 to NO4
        print(f"\n========== Training Region NO{region_number} ==========")

        path = f"{data_dir}/scaled_features_power_MW_NO{region_number}.parquet"
        df = pd.read_parquet(path).drop(columns=["time"], errors="ignore")

        test_df = df[-(input_width + label_width):]
        usable_df = df[:-(input_width + label_width)]
        n_usable = len(usable_df)
        train_df = usable_df[:int(n_usable * 0.7)]
        val_df = usable_df[int(n_usable * 0.7):]

        window = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["power_MW"]
        )

        for x_batch, _ in window.train.take(1):
            input_shape = x_batch.shape[1:]

        rnn = RNN(input_width, label_width, input_shape[-1], region_number)
        rnn.summary()

        history = rnn.fit(window, weight_dir, epochs=100)

        pred, y_true = rnn.predict_last_window(window)
        forecast_plot_path = os.path.join(plot_dir, f"forecast_plot_trial_NO{region_number}_{rnn.name}.png")
        rnn.plot_prediction(pred, y_true, save_path=forecast_plot_path)

        learning_plot_path = os.path.join(plot_dir, f"learning_curve_trial_NO{region_number}_{rnn.name}.png")
        rnn.plot_learning_curves(history, save_path=learning_plot_path)

        # Optional: Save weights
        rnn.save_weights(os.path.join(weight_dir, f"rnn_weights_NO{region_number}_{rnn.name}.h5"))

        # Optional: Evaluate performance on test set
        rnn.evaluate_model(window.test, dataset_name="Test")