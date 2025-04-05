import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data import get_mnist
import pickle
from pathlib import Path
from typing import Optional, Tuple, Callable
from tkinter import filedialog
from typing import List

def train_model(
    epochs: int, 
    hidden_layer_sizes: List[int] = [20], 
    learn_rate: float = 0.01, 
    progress_callback: Optional[Callable] = None,
    network: Optional[NeuralNetwork] = None
) -> Tuple[NeuralNetwork, float]:
    """
    Train the neural network model.
    
    Args:
        epochs: Number of epochs to train for
        hidden_layer_sizes: Size of the hidden layers
        learn_rate: Learning rate for the neural network
        progress_callback: Optional callback function to report progress (epoch, total_epochs, accuracy)
    
    Returns:
        Tuple of (trained_model, final_accuracy)
    """
    inputs, labels = get_mnist()
    
    if network is None:
        nn = NeuralNetwork()
        nn.create_layers(inputs.shape[1], labels.shape[1], hidden_layer_sizes)
    else:
        nn = network
    nn.learn_rate = learn_rate
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

def save_model(model: NeuralNetwork) -> str | None:
    """
    Save the trained model to a file.
    
    Args:
        model: The trained NeuralNetwork model to save
    
    Returns:
        Path to the saved model or None if saving was cancelled
    """
    file_path = filedialog.asksaveasfilename(
        initialdir=Path.cwd(),
        defaultextension=".pkl",
        filetypes=[("Pickle Files", "*.pkl")],
        title="Save trained model"
    )
    if file_path:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        return file_path
    return None

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