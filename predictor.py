"""Glue between the database, the ANN and the adaptive engine."""
import numpy as np

import database as db
from adaptive_engine import build_suggestions, recommend, risk_level
from config import FEATURE_LABELS, FEATURES, LEVELS
from data_preprocessing import load_scaler, transform

_cache = {}


def _load():
    if not _cache:
        from model import load_model  # imported late: TensorFlow takes a few seconds to load
        _cache["ann"] = load_model()
        _cache["scaler"], _cache["bounds"] = load_scaler()
    return _cache["ann"], _cache["scaler"], _cache["bounds"]


def predict(features: dict) -> dict:
    """features: dict with the 7 keys of config.FEATURES -> level, probabilities, risk, factors."""
    ann, scaler, bounds = _load()
    x = np.array([[float(features[k]) for k in FEATURES]], dtype="float32")
    probs = ann.predict_proba(transform(x, scaler, bounds))[0]
    idx = int(np.argmax(probs))

    # "normalised input factors": how far each value is from the typical trainee (z-score,
    # sign flipped for features where a high value is bad)
    z = (np.clip(x[0], bounds["min"], bounds["max"]) - scaler.mean_) / scaler.scale_
    bad = {"attempts_per_quiz", "days_since_last_login"}
    factors = {FEATURE_LABELS[k]: float(-z[i] if k in bad else z[i]) for i, k in enumerate(FEATURES)}

    risk = risk_level(float(probs[0]))
    return {"level": LEVELS[idx], "confidence": float(probs[idx]),
            "probs": {lvl: float(p) for lvl, p in zip(LEVELS, probs)},
            "risk": risk, "factors": factors,
            "suggestions": build_suggestions(features, LEVELS[idx], risk)}


def analyse_user(user_id: int, save: bool = True) -> dict:
    """Full pipeline for one trainee: features -> ANN -> progress table -> next lesson."""
    results = db.get_results(user_id)
    features = db.compute_features(user_id)
    if features is None:
        return {"features": None, "prediction": None, "recommendation": recommend(None, results),
                "results": results}
    pred = predict(features)
    if save:
        db.save_progress(user_id, features["lessons_completed"], pred["level"], pred["confidence"])
    return {"features": features, "prediction": pred,
            "recommendation": recommend(pred["level"], results), "results": results}
