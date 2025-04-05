from layer import Layer
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

class NeuralNetwork:

    def __init__(self, layer_sizes: List[Tuple[int, int]] = []):
        self.layers: List[Layer] = [Layer(x, y) for x,y in layer_sizes]
        self.learn_rate = 0.01

    def create_layers(self, input_size: int, output_size: int, hidden_layers: List[int]):
        for layer in hidden_layers:
            self.layers.append(Layer(input_size, layer))
            input_size = layer
        self.layers.append(Layer(input_size, output_size))

    def calculate_outputs(self, inputs: NDArray) -> NDArray:
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def classify(self, inputs: NDArray) -> int:
        self.outputs = self.calculate_outputs(inputs)
        return np.argmax(self.outputs)
    
    def cost(self, label: NDArray) -> float:
        return 1 / len(self.outputs) * np.sum((self.outputs - label) ** 2, axis=0)
    
    def calc_cost(self, inputs: NDArray, label: NDArray) -> float:
        self.outputs = self.calculate_outputs(inputs)
        return 1 / len(self.outputs) * np.sum((self.outputs - label) ** 2, axis=0)
    
    def back_propagate(self, img: NDArray, label: NDArray) -> None:
        delta = self.outputs - label
        self.layers[-1].weights += -self.learn_rate * delta @ np.transpose(self.layers[-2].outputs)
        self.layers[-1].biases += -self.learn_rate * delta
        
        for i in range(len(self.layers) - 2, 0, -1):
            prev_layer = self.layers[i + 1]
            curr_layer = self.layers[i]
            next_layer = self.layers[i - 1]

            delta = np.transpose(prev_layer.weights) @ delta * (curr_layer.outputs * (1 - curr_layer.outputs))
            curr_layer.weights += -self.learn_rate * delta @ np.transpose(next_layer.outputs)
            curr_layer.biases += -self.learn_rate * delta

        delta = self.layers[1].weights.T @ delta * (self.layers[0].outputs * (1 - self.layers[0].outputs))
        self.layers[0].weights += -self.learn_rate * delta @ img.T
        self.layers[0].biases += -self.learn_rate * delta