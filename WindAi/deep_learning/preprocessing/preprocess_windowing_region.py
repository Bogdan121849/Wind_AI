import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                train_df, val_df, test_df,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
            self.column_indices = {name: i for i, name in
                                enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift + label_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
            
            label_indices = [self.column_indices[name] for name in self.label_columns]
            input_indices = [i for i in range(features.shape[2]) if i not in label_indices]
            inputs = tf.gather(inputs, input_indices, axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
  
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=64)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

if __name__ == "__main__":
    df = pd.read_parquet("/home2/s5549329/windAI_rug/WindAi/deep_learning/created_datasets/scaled_features_power_MW_ELSPOT NO1.parquet")

    df = df.drop(columns=["time", "bidding_area"], errors="ignore")
    
    print(f"Loaded full dataset with shape: {df.shape}")
    input_width = 48
    label_width = 61
    test_size = input_width + label_width

    test_df = df[-test_size:]
    usable_df = df[:-test_size]

    n_usable = len(usable_df)
    train_df = usable_df[:int(n_usable * 0.7)]
    val_df   = usable_df[int(n_usable * 0.7):]

    print(f"\nSplitting:")
    print(f"Usable data: {usable_df.shape}")
    print(f"Train: {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Test (last 85 rows): {test_df.shape}")


    window = WindowGenerator(
        input_width=48,
        label_width=61,
        shift=0,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=["power_MW"]
    )

    for batch_inputs, batch_labels in window.train.take(1):
        print("Inputs shape:", batch_inputs.shape)
        print("Labels shape:", batch_labels.shape)







  
    