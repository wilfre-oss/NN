from layer import Layer
import numpy as np
from typing import List
from numpy.typing import NDArray

class NeuralNetwork:

    def __init__(self, layer_sizes: List[int]):
        self.layers: List[Layer] = [Layer() for _ in layer_sizes]
        self.learn_rate = 0.01

    def calculate_outputs(self, inputs: NDArray) -> NDArray:
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def classify(self, inputs: NDArray) -> int:
        self.outputs = self.calculate_output(inputs)
        return np.argmax(self.outputs)
    
    def cost(self, label: NDArray) -> float:
        return 1 / len(self.outputs) * np.sum((self.outputs - label) ** 2, axis=0)
    
    def cost(self, inputs: NDArray, label: NDArray) -> float:
        self.outputs = self.calculate_outputs(inputs)
        return 1 / len(self.outputs) * np.sum((self.outputs - label) ** 2, axis=0)