"""Load, clean, scale and split the dataset (Section 5.3 of the report)."""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATASET_PATH, FEATURES, LEVELS, MODEL_DIR, RANDOM_STATE, TARGET

SCALER_PATH = MODEL_DIR / "scaler.joblib"
BOUNDS_PATH = MODEL_DIR / "feature_bounds.joblib"


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        from generate_dataset import generate
        path.parent.mkdir(exist_ok=True)
        generate().to_csv(path, index=False)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, fix missing values (median imputation), drop bad labels."""
    df = df.drop_duplicates().copy()
    df = df[df[TARGET].isin(LEVELS)]
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """70% train / 15% validation / 15% test, stratified so all classes appear."""
    X = df[FEATURES].values.astype("float32")
    y = df[TARGET].map({lvl: i for i, lvl in enumerate(LEVELS)}).values
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit the scaler on the training data only (no leakage) and save it."""
    MODEL_DIR.mkdir(exist_ok=True)
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({"min": X_train.min(axis=0), "max": X_train.max(axis=0)}, BOUNDS_PATH)
    return scaler


def load_scaler():
    return joblib.load(SCALER_PATH), joblib.load(BOUNDS_PATH)


def transform(X: np.ndarray, scaler=None, bounds=None) -> np.ndarray:
    """Clip to the range seen in training, then standardise."""
    if scaler is None:
        scaler, bounds = load_scaler()
    X = np.asarray(X, dtype="float32")
    if bounds is not None:
        X = np.clip(X, bounds["min"], bounds["max"])
    return scaler.transform(X).astype("float32")


def one_hot(y: np.ndarray, n_classes: int = len(LEVELS)) -> np.ndarray:
    return np.eye(n_classes, dtype="float32")[y]
