import numpy as np
import pathlib
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class DatasetLoader(ABC):
    """Base class for loading datasets."""
    
    def __init__(self):
        self.label_to_index: Dict[Any, int] = {}
        self.index_to_label: Dict[int, Any] = {}
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess the dataset.
        
        Returns:
            Tuple of (inputs, one-hot encoded labels)
        """
        pass
    
    def _create_label_mapping(self, unique_labels: list) -> None:
        """
        Create mapping between labels and indices.
        
        Args:
            unique_labels: List of unique labels in the dataset
        """
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
    
    def encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert labels to one-hot encoding.
        
        Args:
            labels: Array of labels
            
        Returns:
            One-hot encoded labels
        """
        num_classes = len(self.label_to_index)
        one_hot = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            idx = self.label_to_index[label]
            one_hot[i, idx] = 1
        return one_hot
    
    def decode_labels(self, one_hot: np.ndarray) -> np.ndarray:
        """
        Convert one-hot encoded labels back to original labels.
        
        Args:
            one_hot: One-hot encoded labels
            
        Returns:
            Original labels
        """
        indices = np.argmax(one_hot, axis=1)
        return np.array([self.index_to_label[idx] for idx in indices])

class MNISTLoader(DatasetLoader):
    """Loader for the MNIST dataset."""
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the MNIST dataset."""
        with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
            images, labels = f["x_train"], f["y_train"]
        
        images = images.astype("float32") / 255
        images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
        
        self._create_label_mapping(list(range(10)))
        
        one_hot_labels = self.encode_labels(labels)
        
        return images, one_hot_labels

def get_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the MNIST dataset.
    
    Returns:
        Tuple of (inputs, one-hot encoded labels)
    """
    loader = MNISTLoader()
    return loader.load_data()