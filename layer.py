import numpy as np
from numpy.typing import NDArray

class Layer:

    def __init__(self, nodes_in: int, nodes_out: int):
        self.num_nodes_in = nodes_in
        self.num_nodes_out = nodes_out
        self.weights = np.random.uniform(-0.5, 0.5, (nodes_out, nodes_in))
        self.biases = np.zeros((nodes_out, 1))
        
    
    def calculate_outputs(self, inputs: NDArray) -> NDArray:
        output = self.biases + self.weights @ inputs
        self.outputs = 1 / (1 + np.exp(output)) 
        return self.outputs
    

    