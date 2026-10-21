"""One command to prepare everything:  python train_all.py
1) creates the dataset  2) trains the ANN  3) evaluates against the baselines."""
from config import DATASET_PATH, DATA_DIR

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    if not DATASET_PATH.exists():
        from generate_dataset import generate
        generate().to_csv(DATASET_PATH, index=False)
        print(f"[data] created {DATASET_PATH.name}")
    import model, evaluate
    model.train()
    evaluate.evaluate()
