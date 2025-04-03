import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data import get_mnist

epochs = 3
correct = 0

def main() -> None:
    inputs, labels = get_mnist()
    
    print('input shape: ', inputs.shape[1])
    print('label shape: ', labels.shape[1])
    
    first_layer = (inputs.shape[1], 20)
    output_layer = (20, labels.shape[1])
    nn = NeuralNetwork([first_layer, output_layer])

    for epoch in range(epochs):
        for input, label in zip(inputs, labels):
            input.shape += (1,)
            label.shape += (1,)
            
            #forward propagation
            classification = nn.classify(input)
            error = nn.cost(label)
            correct += int(classification == np.argmax(label))

            #back propagation
            nn.back_propagate(input, label)
        print(f"Acc: {round((correct / inputs.shape[0]) * 100, 2)}%")
        correct = 0



if __name__ == "__main__":
    main()