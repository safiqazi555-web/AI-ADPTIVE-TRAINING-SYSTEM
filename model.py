"""Build, train, save and load the ANN (Section 5.4 / Table 3 of the report).

Architecture : 7 inputs -> Dense(64, ReLU) + Dropout(0.3)
                        -> Dense(32, ReLU) + Dropout(0.2)
                        -> Dense(3, Softmax)
Loss / optim : categorical cross-entropy, Adam (lr 0.001)
Training     : up to 50 epochs, batch size 32, Early Stopping on val loss

TensorFlow/Keras is used when it is installed.  If it is not (for example on a
Python version without a TensorFlow wheel) an equivalent scikit-learn MLP with
the same layer sizes is used automatically, so the project still runs.
"""
import copy
import json

import joblib
import numpy as np

from config import FEATURES, LEVELS, MODEL_DIR, RANDOM_STATE
from data_preprocessing import clean, fit_scaler, load_dataset, one_hot, split_data, transform

KERAS_PATH = MODEL_DIR / "ann_model.keras"
SKLEARN_PATH = MODEL_DIR / "ann_model.joblib"
INFO_PATH = MODEL_DIR / "model_info.json"
HISTORY_PATH = MODEL_DIR / "history.json"

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 8

try:  # TensorFlow is optional at run time
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except Exception:  # pragma: no cover
    HAS_TF = False


# --------------------------------------------------------------------------- #
# Keras version
# --------------------------------------------------------------------------- #
def build_keras_model(n_features: int = len(FEATURES), n_classes: int = len(LEVELS)):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _train_keras(X_tr, y_tr, X_val, y_val):
    keras.utils.set_random_seed(RANDOM_STATE)
    model = build_keras_model()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                         restore_best_weights=True)
    hist = model.fit(X_tr, one_hot(y_tr), validation_data=(X_val, one_hot(y_val)),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stop], verbose=0)
    model.save(KERAS_PATH)
    return model, {k: [float(v) for v in vals] for k, vals in hist.history.items()}


# --------------------------------------------------------------------------- #
# scikit-learn fallback (same layer sizes)
# --------------------------------------------------------------------------- #
def _train_sklearn(X_tr, y_tr, X_val, y_val):
    from sklearn.metrics import log_loss
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                        learning_rate_init=LEARNING_RATE, batch_size=BATCH_SIZE,
                        alpha=1e-3, random_state=RANDOM_STATE)
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best, best_loss, wait = None, np.inf, 0
    classes = np.arange(len(LEVELS))
    for _ in range(EPOCHS):
        clf.partial_fit(X_tr, y_tr, classes=classes)
        p_tr, p_val = clf.predict_proba(X_tr), clf.predict_proba(X_val)
        val_loss = log_loss(y_val, p_val, labels=classes)
        hist["loss"].append(float(log_loss(y_tr, p_tr, labels=classes)))
        hist["val_loss"].append(float(val_loss))
        hist["accuracy"].append(float((p_tr.argmax(1) == y_tr).mean()))
        hist["val_accuracy"].append(float((p_val.argmax(1) == y_val).mean()))
        if val_loss < best_loss - 1e-4:
            best, best_loss, wait = copy.deepcopy(clf), val_loss, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    joblib.dump(best, SKLEARN_PATH)
    return best, hist


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
class ANN:
    """Small wrapper so the rest of the project does not care about the backend."""

    def __init__(self, model, backend: str):
        self.model, self.backend = model, backend

    def predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.backend == "keras":
            return np.asarray(self.model(X_scaled, training=False))
        return self.model.predict_proba(X_scaled)

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_scaled).argmax(axis=1)


def train(verbose: bool = True) -> dict:
    """Full training pipeline. Saves scaler, model, history and split info."""
    MODEL_DIR.mkdir(exist_ok=True)
    df = clean(load_dataset())
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df)
    scaler = fit_scaler(X_tr)
    X_tr_s, X_val_s, X_te_s = (transform(a, scaler) for a in (X_tr, X_val, X_te))

    if HAS_TF:
        model, history = _train_keras(X_tr_s, y_tr, X_val_s, y_val)
        backend = "keras"
    else:
        model, history = _train_sklearn(X_tr_s, y_tr, X_val_s, y_val)
        backend = "sklearn"

    ann = ANN(model, backend)
    test_acc = float((ann.predict(X_te_s) == y_te).mean())
    info = {"backend": backend, "epochs_run": len(history["loss"]),
            "n_train": int(len(y_tr)), "n_val": int(len(y_val)), "n_test": int(len(y_te)),
            "test_accuracy": test_acc, "features": FEATURES, "levels": LEVELS}
    INFO_PATH.write_text(json.dumps(info, indent=2))
    HISTORY_PATH.write_text(json.dumps(history))
    if verbose:
        print(f"[model] backend={backend}  epochs={info['epochs_run']}  test accuracy={test_acc:.3f}")
    return info


def is_trained() -> bool:
    return INFO_PATH.exists() and (KERAS_PATH.exists() or SKLEARN_PATH.exists())


def load_model() -> ANN:
    if not is_trained():
        raise FileNotFoundError("Model not trained yet. Run:  python train_all.py")
    backend = json.loads(INFO_PATH.read_text())["backend"]
    if backend == "keras":
        if not HAS_TF:
            raise RuntimeError("The model was trained with TensorFlow, but TensorFlow is not "
                               "installed. Install it or delete the models folder and retrain.")
        return ANN(keras.models.load_model(KERAS_PATH), "keras")
    return ANN(joblib.load(SKLEARN_PATH), "sklearn")


if __name__ == "__main__":
    train()
