import os, random
import numpy as np

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # torch & other libs would go here if/when you add them

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
