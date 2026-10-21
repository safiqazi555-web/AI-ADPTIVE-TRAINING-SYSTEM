"""Create a synthetic trainee dataset (used when no public dataset is available).

Run:  python generate_dataset.py
Each row is one trainee described by the 7 features of Table 2 and a label
(Weak / Average / Strong).  The labels overlap on purpose, so the task is not
trivial and the ANN has something to learn beyond simple thresholds.
"""
import numpy as np
import pandas as pd

from config import DATASET_PATH, DATA_DIR, FEATURES, LEVELS, RANDOM_STATE, TARGET


def generate(n_samples: int = 4000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # hidden "ability" of every trainee; the observed features are noisy views of it
    ability = rng.normal(0, 1, n_samples)

    avg_score = np.clip(66 + 16 * ability + rng.normal(0, 5, n_samples), 5, 100)
    last_score = np.clip(avg_score + 4 * ability + rng.normal(0, 8, n_samples), 0, 100)
    time_spent = np.clip(rng.gamma(4.0, 32, n_samples) * (1 + 0.12 * ability), 10, 600)
    attempts = np.clip(2.1 - 0.6 * ability + rng.normal(0, 0.4, n_samples), 1, 5)
    completed = np.clip(55 + 22 * ability + rng.normal(0, 9, n_samples), 0, 100)
    logins = np.clip(np.round(4 + 1.5 * ability + rng.normal(0, 1.1, n_samples)), 0, 14)
    days_idle = np.clip(np.round(np.exp(0.9 - 0.5 * ability + rng.normal(0, 0.4, n_samples))), 0, 30)

    # Label rule: a non-linear "performance index" (so a neural network has an edge over
    # simple linear baselines). Idle trainees are penalised, consistent high scorers get a bonus.
    perf = (0.50 * avg_score / 100 + 0.15 * last_score / 100 + 0.15 * completed / 100
            + 0.10 / attempts + 0.06 * np.tanh((logins - 4) / 3)
            - 0.10 / (1 + np.exp(-(days_idle - 8) / 3))
            + 0.04 * ((avg_score >= 75) & (last_score >= 75))
            + rng.normal(0, 0.03, n_samples))
    lo, hi = np.quantile(perf, [0.28, 0.70])           # about 28% Weak, 42% Average, 30% Strong
    labels = np.where(perf < lo, 0, np.where(perf < hi, 1, 2))

    df = pd.DataFrame({
        "avg_quiz_score": avg_score.round(1),
        "last_quiz_score": last_score.round(1),
        "time_spent": time_spent.round(1),
        "attempts_per_quiz": attempts.round(2),
        "lessons_completed": completed.round(1),
        "login_frequency": logins.astype(int),
        "days_since_last_login": days_idle.astype(int),
        TARGET: [LEVELS[i] for i in labels],
    })

    # make the data a little messy so preprocessing has real work to do
    for col in FEATURES:
        mask = rng.random(n_samples) < 0.015
        df.loc[mask, col] = np.nan
    df = pd.concat([df, df.sample(25, random_state=seed)], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    data = generate()
    data.to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(data)} rows to {DATASET_PATH}")
    print(data[TARGET].value_counts())
