from layer import Layer
import numpy as np
from typing import List
from numpy.typing import NDArray

class NeuralNetwork:

    def __init__(self, layer_sizes: List[int]):
        self.layers: List[Layer] = [Layer() for _ in layer_sizes]

    def calculate_output(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def classify(self, inputs: NDArray[np.float64]) -> int:
        outputs = self.calculate_output(inputs)
        return np.argmax(outputs)