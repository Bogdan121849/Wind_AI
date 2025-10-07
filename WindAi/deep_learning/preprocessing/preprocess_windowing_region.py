import numpy as np
import tensorflow as tf
import pandas as pd

class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None,
                 future_len=48,
                 pad_values=None,
                 past_only_col=None,
                 future_only_col=None,
                 # legacy booleans still accepted:
                 pad_past_only_in_future: bool = True,
                 pad_future_only_in_past: bool = False,
                 # new unified switch: "none" | "past-in-future" | "future-in-past" | "both"
                 padding_mode: str | None = None):

        # store dfs
        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df

        # column -> index (ALWAYS build)
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # labels
        self.label_columns = label_columns or []
        if self.label_columns:
            self.label_columns_indices = {n: i for i, n in enumerate(self.label_columns)}

        # window params
        self.input_width  = int(input_width)
        self.label_width  = int(label_width)
        self.shift        = int(shift)
        self.future_len   = int(future_len)
        assert 0 < self.future_len <= self.input_width, "future_len must be in (0, input_width]"

        self.total_window_size = self.input_width + self.shift + self.label_width
        self.input_slice   = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start   = self.total_window_size - self.label_width
        self.labels_slice  = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # padding config
        self.pad_values = pad_values or {}
        self.past_only_cols   = set(past_only_col or [])
        self.future_only_cols = set(future_only_col or [])

        # --- Unified padding mode (overrides legacy bools if provided) ---
        if padding_mode is not None:
            mode = str(padding_mode).strip().lower()
            if mode not in {"none", "past-in-future", "future-in-past", "both"}:
                raise ValueError(f"Invalid padding_mode={padding_mode!r}")
            self.pad_past_only_in_future = (mode in {"past-in-future", "both"})
            self.pad_future_only_in_past = (mode in {"future-in-past", "both"})
        else:
            # fallback to the legacy flags
            self.pad_past_only_in_future = bool(pad_past_only_in_future)
            self.pad_future_only_in_past = bool(pad_future_only_in_past)

        self.enable_padding = self.pad_past_only_in_future or self.pad_future_only_in_past

        # fixed channel counts (for static shapes)
        self.num_label_features = len(self.label_columns)
        self.num_input_features = len(self.column_indices) - self.num_label_features

        # indices for masks
        self.past_columns_indices = tf.constant(
            [self.column_indices[c] for c in self.past_only_cols if c in self.column_indices],
            dtype=tf.int32
        )
        self.future_columns_indices = tf.constant(
            [self.column_indices[c] for c in self.future_only_cols if c in self.column_indices],
            dtype=tf.int32
        )

        # per-feature pad vector (standardized-zero per feature)
        D_all = len(self.column_indices)
        pad_vec = np.zeros(D_all, dtype=np.float32)
        for name, j in self.column_indices.items():
            pad_vec[j] = float(self.pad_values.get(name, 0.0))
        self.pad_vec = tf.constant(pad_vec, dtype=tf.float32)

    def __repr__(self):
        mode = ("none" if not self.enable_padding else
                "both" if (self.pad_past_only_in_future and self.pad_future_only_in_past) else
                "past-in-future" if self.pad_past_only_in_future else
                "future-in-past")
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Labels: {self.label_columns}',
            f'future_len: {self.future_len}',
            f'num_input_features: {self.num_input_features}',
            f'padding_mode: {mode}',
        ])

    # inputs: (batch, input_width, D)
    def apply_segment_padding(self, inputs):
        # quick exit if disabled
        if not self.enable_padding:
            return inputs

        T = self.input_width
        past_T = self.input_width - self.future_len
        D = tf.shape(inputs)[-1]  # dynamic

        pad_vec = tf.reshape(self.pad_vec, (1, 1, -1))
        x = inputs

        # time masks
        fut_time_mask = tf.concat(
            [tf.zeros((past_T, 1), tf.float32),
             tf.ones((self.future_len, 1), tf.float32)], axis=0
        )  # (T,1)
        fut_time_mask = tf.reshape(fut_time_mask, (1, T, 1))

        past_time_mask = tf.concat(
            [tf.ones((past_T, 1), tf.float32),
             tf.zeros((self.future_len, 1), tf.float32)], axis=0
        )
        past_time_mask = tf.reshape(past_time_mask, (1, T, 1))

        # feature masks via dynamic eye -> gather -> sum
        if self.pad_past_only_in_future and tf.size(self.past_columns_indices) > 0:
            feat_mask = tf.reduce_sum(tf.gather(tf.eye(D), self.past_columns_indices), axis=0)  # (D,)
            feat_mask = tf.reshape(feat_mask, [1, 1, -1])
            mask_fut = fut_time_mask * feat_mask
            x = tf.where(mask_fut > 0, pad_vec, x)

        if self.pad_future_only_in_past and tf.size(self.future_columns_indices) > 0:
            feat_mask2 = tf.reduce_sum(tf.gather(tf.eye(D), self.future_columns_indices), axis=0)
            feat_mask2 = tf.reshape(feat_mask2, [1, 1, -1])
            mask_past = past_time_mask * feat_mask2
            x = tf.where(mask_past > 0, pad_vec, x)

        return x

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # padding inside the input window (if enabled)
        inputs = self.apply_segment_padding(inputs)

        if self.label_columns:
            # labels: keep only label cols
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
            # inputs: drop label cols
            label_indices = [self.column_indices[name] for name in self.label_columns]
            all_idx = tf.range(tf.shape(features)[2])
            keep_mask = ~tf.reduce_any(tf.equal(all_idx[:, None], tf.constant(label_indices)[None, :]), axis=1)
            keep_idx = tf.cast(tf.where(keep_mask)[:, 0], tf.int32)
            inputs = tf.gather(inputs, keep_idx, axis=-1)

        # STATIC shapes so Dense can build
        inputs.set_shape([None, self.input_width, self.num_input_features])
        labels.set_shape([None, self.label_width, self.num_label_features])
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=64
        )
        ds = ds.map(self.split_window, num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    @property
    def train(self): return self.make_dataset(self.train_df)
    @property
    def val(self):   return self.make_dataset(self.val_df)
    @property
    def test(self):  return self.make_dataset(self.test_df)
