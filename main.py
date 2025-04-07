import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Callable, List
import pickle
from tkinter import filedialog

from gui import create_gui
from data import get_mnist
from neural_network import NeuralNetwork

class NNController:
    """
    Controller class for the Neural Network application.
    Handles the business logic and coordination between model and view.
    """
    
    def __init__(self):
        self.model = None
        self.epochs = 5
        self.hidden_layer_sizes = [20]
        self.learn_rate = 0.01
        self.data = self.load_data()
        
    def train_model(self, progress_callback: Optional[Callable] = None) -> float:
        """
        Train the neural network model.
        
        Args:
            progress_callback: Optional callback function to report progress (epoch, total_epochs, accuracy)
        
        Returns:
            Final accuracy of the model
        """
        inputs, labels = self.data
        
        if self.model is None:
            nn = NeuralNetwork()
            nn.create_layers(inputs.shape[1], labels.shape[1], self.hidden_layer_sizes)
        else:
            nn = self.model
        nn.learn_rate = self.learn_rate
        accuracy = None

        for epoch in range(self.epochs):
            correct = 0
            if progress_callback:
                progress_callback(epoch, self.epochs, accuracy)
            for img, label in zip(inputs, labels):
                img = img.reshape(-1, 1)
                label = label.reshape(-1, 1)
                
                classification = nn.classify(img)
                correct += int(classification == np.argmax(label))
                nn.back_propagate(img, label)
            
            accuracy = (correct / inputs.shape[0]) * 100
        
        self.model = nn
        return accuracy
    
    def save_model(self) -> str | None:
        """
        Save the trained model to a file.
        
        Returns:
            Path to the saved model or None if saving was cancelled or failed
        """
        if self.model is None:
            return None
            
        file_path = filedialog.asksaveasfilename(
            initialdir=Path.cwd(),
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl")],
            title="Save trained model"
        )
        if file_path:
            with open(file_path, "wb") as f:
                pickle.dump(self.model, f)
            return file_path
        return None
    
    def load_model(self) -> Tuple[bool, str]:
        """
        Load a trained model from a file.
        
        Returns:
            Tuple of (success, message)
        """
        file_path = filedialog.askopenfilename(
            initialdir=Path.cwd(),
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl")],
            title="Load trained model"
        )
        if file_path:
            try:
                self.model = pickle.load(open(file_path, "rb"))
                return True, f"Model loaded from {file_path}"
            except Exception as e:
                return False, f"Failed to load model: {str(e)}"
        return False, "No file selected"
    
    def test_model(self, img_index: int) -> int:
        """
        Test the model with a specific image from the dataset.
        
        Args:
            img_index: Index of the image in the MNIST dataset
            
        Returns:
            Classification result (predicted digit)
        """
        if self.model is None:
            return -1
            
        inputs, _ = get_mnist()
        if 0 <= img_index < len(inputs):
            img = inputs[img_index].reshape(-1, 1)
            return self.model.classify(img)
        return -1
    
    def has_model(self) -> bool:
        """Check if a model is loaded or trained."""
        return self.model is not None
    
    def set_parameters(self, epochs: int, hidden_layer_sizes: List[int], learn_rate: float) -> None:
        """
        Set the training parameters.
        
        Args:
            epochs: Number of epochs to train for
            hidden_layer_sizes: Size of the hidden layers
            learn_rate: Learning rate for the neural network
        """
        self.epochs = epochs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learn_rate = learn_rate 
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the MNIST dataset. Will eventually load chosen dataset.
        
        Returns:
            Tuple of (inputs, labels)
        """ 
        return get_mnist()

def display_prediction(controller: NNController, index: int) -> None:
    """
    Display a prediction for a selected digit.
    
    Args:
        controller: The neural network controller
        index: Index of the image in the MNIST dataset
    """
    
    
    if not controller.has_model():
        print("No model available. Train or load a model first.")
        return
        
    inputs, _ = controller.data
    if 0 <= index < len(inputs):
        img = inputs[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        classification = controller.test_model(index)
        plt.title(f"Predicted digit: {classification}")
        plt.show()
    else:
        print(f"Invalid index. Must be between 0 and {len(inputs)-1}")

def main() -> None:
    # Create controller
    controller = NNController()
    
    # Create and start GUI
    window = create_gui(controller)
    window.mainloop()

if __name__ == "__main__":
    main()