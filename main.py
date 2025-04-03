import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data import get_mnist



def main() -> None:
    epochs = 5
    correct = 0
    inputs, labels = get_mnist()
        
    first_layer = (inputs.shape[1], 20)
    output_layer = (20, labels.shape[1])
    nn = NeuralNetwork([first_layer, output_layer])

    for epoch in range(epochs):
        
        for img, label in zip(inputs, labels):
            img.shape += (1,)
            label.shape += (1,)
            
            #forward propagation
            classification = nn.classify(img)
            error = nn.cost(label)
            correct += int(classification == np.argmax(label))

            #back propagation
            nn.back_propagate(img, label)
        print(f"Acc: {round((correct / inputs.shape[0]) * 100, 2)}%")
        correct = 0

    def display() -> None:
        index = int(input("Enter a number (0 - 59999): "))
        img = inputs[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        img.shape += (1,)
        
        o = nn.classify

        plt.title(f"Subscribe if its a {np.argmax(o)} :)")
        plt.show()

if __name__ == "__main__":
    main()