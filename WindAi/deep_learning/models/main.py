from GRU_deep import GruDeep
from GRU_weak import GruWeak
from LSTM_main import LSTM
from transformer import TransformerForecast
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from WindAi.deep_learning.preprocessing.preprocess_windowing_region import WindowGenerator

MODEL_REGISTRY =  {
    #"Gru Deep" : GruDeep,
    #"Gru Weak" : GruWeak,
    #"LSTM" : LSTM,
    "Transformer": TransformerForecast
}


def run(model_class, model_name,input_width=336, label_width=61, epochs=100):
    shift = 0
    data_dir   = "/home2/s5549329/windAI_rug/WindAi/deep_learning/created_datasets"
    weight_dir = "/home2/s5549329/windAI_rug/WindAi/deep_learning/weights"
    plot_dir   = "/home2/s5549329/windAI_rug/WindAi/deep_learning/results"

    for region_number in range(1, 5):
        print(f"\n========== Training Region NO{region_number} ==========")

        # Load data
        path = f"{data_dir}/scaled_features_power_MW_NO{region_number}.parquet"
        df = pd.read_parquet(path).drop(columns=["time"], errors="ignore")

        #  397
        test_df   = df[-(input_width + label_width):]
        # 44505 - 397 = 44108
        usable_df = df[:-(input_width + label_width)]
        n = len(usable_df)
        # 30875
        train_df = usable_df[:int(n * 0.7)]
        # 13232
        val_df   = usable_df[int(n * 0.7):]

        # Create data window
        window = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["power_MW"]
        )

        # Initialize and run model
        model = model_class(
            input_width=input_width,
            label_width=label_width,
            num_features=window.train.element_spec[0].shape[-1],
            region_number=region_number,
            name=model_name
        )

        model.summary()
        history = model.fit(window, weight_dir, epochs=epochs)

        # Save forecast plot
        pred, y_true = model.predict_last_window(window)
        model.plot_prediction(
            pred, y_true,
            os.path.join(plot_dir, f"forecast_NO{region_number}_{model.name}.png")
        )

        # Save learning curve plot
        model.plot_learning_curves(
            history,
            os.path.join(plot_dir, f"learning_NO{region_number}_{model.name}.png")
        )

        # Evaluate performance
        model.evaluate_model(window.test, dataset_name="Test")


if __name__ == "__main__":
    for model_name, model_class in MODEL_REGISTRY.items():
        run(model_class, model_name)


