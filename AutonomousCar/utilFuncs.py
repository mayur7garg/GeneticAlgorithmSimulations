import numpy as np
from collections.abc import Iterable

def WeightInitializer(shape: Iterable[int], scale: int = 10):
    return np.random.random(size = shape) / scale

def LeakyRelu(x: np.ndarray, alpha = 0.5):
    return np.maximum(alpha * x, x)

def Sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))