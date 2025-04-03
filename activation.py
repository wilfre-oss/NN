import numpy as np
from numpy.typing import NDArray

def sigmoid(input: NDArray) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(input))