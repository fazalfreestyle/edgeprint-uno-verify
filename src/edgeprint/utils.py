import os, json, random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
