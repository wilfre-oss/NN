import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data import get_mnist
import pickle
from pathlib import Path
from typing import Optional, Tuple, Callable

def train_model(epochs: int, progress_callback: Optional[Callable] = None) -> Tuple[NeuralNetwork, float]:
    """
    Train the neural network model.
    
    Args:
        epochs: Number of epochs to train for
        progress_callback: Optional callback function to report progress (epoch, total_epochs, accuracy)
    
    Returns:
        Trained NeuralNetwork model
    """
    inputs, labels = get_mnist()
    
    first_layer = (inputs.shape[1], 20)
    output_layer = (20, labels.shape[1])
    nn = NeuralNetwork([first_layer, output_layer])
    nn.learn_rate = 0.01
    accuracy = None

    for epoch in range(epochs):
        correct = 0
        if progress_callback:
            progress_callback(epoch, epochs, accuracy)
        for img, label in zip(inputs, labels):
            img = img.reshape(-1, 1)
            label = label.reshape(-1, 1)
            
            classification = nn.classify(img)
            correct += int(classification == np.argmax(label))
            nn.back_propagate(img, label)
        
        accuracy = (correct / inputs.shape[0]) * 100
        
    
    return nn, accuracy

def save_model(model: NeuralNetwork, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    Args:
        model: The trained NeuralNetwork model to save
        file_path: Path where to save the model
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

def display_prediction(model: NeuralNetwork) -> None:
    """
    Display a prediction for a user-selected digit.
    
    Args:
        model: The trained neural network model
    """
    inputs, _ = get_mnist()
    index = int(input("Enter a number (0 - 59999): "))
    img = inputs[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img = img.reshape(-1, 1)
    classification = model.classify(img)

    plt.title(f"Predicted digit: {classification}")
    plt.show()

def main() -> None:
    # Import gui module and start the application
    from gui import create_gui
    window = create_gui()
    window.mainloop()

if __name__ == "__main__":
    main()