# train_main.py
from GRU_deep import GruDeep
from GRU_weak import GruWeak
from LSTM_main import LSTM
from transformer import TransformerForecast
import pandas as pd
import os, re, sys, pickle
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from WindAi.deep_learning.preprocessing.preprocess_windowing_region import WindowGenerator
import numpy as np
from sklearn.preprocessing import StandardScaler

MODEL_REGISTRY = {
    #"Gru Deep": GruDeep,
    #"Gru Weak": GruWeak,
    #"LSTM": LSTM,
    "Transformer_Masked": TransformerForecast,
}

HERE     = Path(__file__).resolve()
DL_ROOT  = HERE.parents[1]
DATA_DIR = Path(os.environ.get("WIND_CREATED_DIR", DL_ROOT / "created_datasets")).resolve()
WEIGHT_DIR = (DL_ROOT / "weights").resolve()
PLOT_DIR   = (DL_ROOT / "results").resolve()

def _region_dirs(base_dir=DATA_DIR):
    out = []
    if not os.path.isdir(base_dir):
        return out
    for name in sorted(os.listdir(base_dir)):
        rdir = os.path.join(base_dir, name)
        proc = os.path.join(rdir, "processed")
        if os.path.isdir(proc):
            need = [os.path.join(proc, f"{s}_scaled.parquet") for s in ("train", "val", "test")]
            if all(os.path.exists(p) for p in need):
                out.append((name, rdir))
    return out

def _ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def _load_meta(proc_dir: str):
    scaler = None
    cols_to_scale = None
    feature_order = None

    sp = os.path.join(proc_dir, "scaler.pkl")
    if os.path.exists(sp):
        with open(sp, "rb") as f:
            s = pickle.load(f)
        # saved exactly like in your preprocessing class
        scaler = s["scaler"]
        cols_to_scale = s["cols_to_scale"]

    mp = os.path.join(proc_dir, "preproc_meta.pkl")
    if os.path.exists(mp):
        with open(mp, "rb") as f:
            m = pickle.load(f)
        feature_order = m.get("feature_order", feature_order)
        # fallbacks in case user didn’t save both places
        cols_to_scale = m.get("cols_to_scale", cols_to_scale)

    return scaler, cols_to_scale, feature_order

def _compute_pad_values(scaler: StandardScaler|None, cols_to_scale, feature_order):
    """Standardized-zero pad per feature; 0 for unscaled."""
    if feature_order is None:
        return {}
    if scaler is None or not cols_to_scale:
        return {c: 0.0 for c in feature_order}
    means  = dict(zip(cols_to_scale, scaler.mean_))
    scales = dict(zip(cols_to_scale, scaler.scale_))
    pads = {}
    for c in feature_order:
        if c in means:
            s = scales.get(c, 1.0) or 1.0
            pads[c] = -(means[c]) / s
        else:
            pads[c] = 0.0
    return pads

def _align_and_pad(df: pd.DataFrame, feature_order, pad_values):
    """Ensure df has exactly feature_order columns (order fixed). Keep power_MW."""
    tgt = df.get("power_MW", None)
    X = df.drop(columns=["time", "power_MW"], errors="ignore").copy()
    # add missing
    for c in feature_order:
        if c not in X.columns:
            X[c] = pad_values.get(c, 0.0)
    # reorder + drop extras
    X = X[feature_order]
    if tgt is not None:
        X["power_MW"] = tgt.values
    return X

def _load_processed(region_dir: str, split: str, feature_order, pad_values) -> pd.DataFrame:
    path = os.path.join(region_dir, "processed", f"{split}_scaled.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_parquet(path)
    return _align_and_pad(df, feature_order, pad_values)

def run(model_class, model_name, input_width=72, label_width=48, epochs=100, shift=1, 
        padding_mode="past-in-future"): # {"none", "past-in-future", "future-in-past", "both"}
    """168 past + 48 future -> predict next 61."""
    regions = _region_dirs(DATA_DIR)
    if not regions:
        raise RuntimeError(f"No processed regions found in {DATA_DIR}")

    _ensure(WEIGHT_DIR, PLOT_DIR)

    for region_name, region_dir in regions:
        print(f"\n========== Training Region {region_name} ==========")
        proc_dir = os.path.join(region_dir, "processed")

        # --- meta & pads ---
        scaler, cols_to_scale, feature_order = _load_meta(proc_dir)
        if feature_order is None:
            # fallback: infer from train file
            tmp = pd.read_parquet(os.path.join(proc_dir, "train_scaled.parquet"))
            feature_order = [c for c in tmp.columns if c not in ("time", "power_MW")]
        pad_values = _compute_pad_values(scaler, cols_to_scale, feature_order)

        # --- load + align ---
        train_df = _load_processed(region_dir, "train", feature_order, pad_values)
        val_df   = _load_processed(region_dir, "val",   feature_order, pad_values)
        test_df  = _load_processed(region_dir, "test",  feature_order, pad_values)

        # --- define past/future feature sets from names you showed ---
        cols = feature_order
        past_only = [c for c in cols
                     if c.endswith("_now") or c.startswith("power_MW_lag") or c == "precip_now"]
        future_only = [c for c in cols
                       if c.endswith(("_mean","_std","_min","_max","_median")) and not c.endswith("_now")]

        # --- windowed dataset with in-window padding ---
        window = WindowGenerator(
            input_width=input_width,       # 216 = 168 past + 48 future
            label_width=label_width,       # 61
            shift=shift,                   # labels start at t+1
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["power_MW"],
            future_len=48,
            pad_values=pad_values,
            past_only_col=past_only,
            future_only_col=future_only,
            padding_mode=padding_mode,
        )

        # region number if name like "NO1"
        m = re.search(r"(\d+)$", region_name)
        region_number = int(m.group(1)) if m else None

        # --- init & train ---
        num_features = window.num_input_features
        model = model_class(
            input_width=input_width,
            label_width=label_width,
            num_features=num_features,
            region_number=region_number,
        )

        out_w = os.path.join(WEIGHT_DIR, model.name, region_name)
        out_p = os.path.join(PLOT_DIR, model.name, region_name)
        _ensure(out_w, out_p)

        model.summary()
        history = model.fit(window, out_w, epochs=epochs)

        # plots & eval
        pred, y_true = model.predict_test(window, first_batch_only=False)
        model.plot_prediction(pred, y_true, os.path.join(out_p, f"forecast_{region_name}_{model.name}.png"))
        model.plot_learning_curves(history, os.path.join(out_p, f"learning_{region_name}_{model.name}.png"))
        model.evaluate_model(window.test, dataset_name="Test")


if __name__ == "__main__":
    for model_name, model_class in MODEL_REGISTRY.items():
        run(model_class, model_name)
