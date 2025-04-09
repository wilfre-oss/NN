# 🧠 Neural Network Trainer

A simple feedforward neural network written in Python and NumPy, designed for experimenting with handwritten digit recognition (MNIST dataset). Comes with a modern GUI for training and saving models.

## 🚀 Features
- Custom feedforward architecture
- Manual backpropagation implementation
- MNIST support with flexible dataset loading system
- MVC architecture for clean code organization
- Train & save models from a modern-looking GUI
- Cross-platform (Tkinter-based)
- Lightweight, no deep learning frameworks required

## 🖼️ GUI Preview

### Main Window
> Modern UI with draggable title bar, styled buttons, and taskbar visibility support (Windows only)

![GUI Preview](https://github.com/user-attachments/assets/c9b6224c-1c44-40c3-ad01-875090b3afd6)

### Settings Window
> Configure training parameters including epochs, hidden layer sizes, and learning rate

![Settings Window](path/to/settings_window.png)

### Test Window
> Visualize model predictions with real-time testing on random MNIST digits, showing both predicted and actual labels

![Test Window](path/to/test_window.png)

## 📦 Requirements

- Python 3.8+
- NumPy
- Matplotlib (optional, for visualizing digits)
- Tkinter (comes bundled with most Python installations)

## 🏗️ Usage

```bash
pip install -r requirements.txt
python main.py
```

## 💾 Saving Models

After training in the GUI, click **"Save Model"** to store your `.pkl` file for future use.

## 🧪 Example Architecture

```python
# Create a neural network with one hidden layer
first_layer = (784, 20)  # Input layer (28x28 pixels) -> Hidden layer (20 neurons)
output_layer = (20, 10)  # Hidden layer -> Output layer (10 digits)
nn = NeuralNetwork([first_layer, output_layer])
```

## 🏗️ Project Structure

```
nn/
├── main.py          # Application entry point and controller
├── gui.py           # View layer (GUI implementation)
├── data.py          # Data loading and preprocessing
├── neural_network.py # Model layer (NN implementation)
└── data/            # Dataset storage
    └── mnist.npz    # MNIST dataset
```

## 📊 Data Format Requirements

### NPZ Dataset Format
The application currently supports NumPy's `.npz` format with the following requirements:

```
dataset.npz
├── x_train: numpy.ndarray    # Image data
│   └── shape: (n_samples, height, width)
│   └── type: uint8 or float32 (0-255)
└── y_train: numpy.ndarray    # Labels
    └── shape: (n_samples,)
    └── type: any            # Can be int, str, or any other type
```

**Requirements:**
- Images must be grayscale (2D arrays)
- Images will be normalized to 0-1 range automatically
- Images will be flattened to 1D arrays automatically
- Labels can be of any type (integers, strings, etc.)
- All images must have the same dimensions

**Examples:**

```python
# Example 1: Numeric labels (0-9)
images = np.random.randint(0, 256, (1000, 28, 28), dtype=np.uint8)  # 1000 28x28 images
labels = np.random.randint(0, 10, (1000,), dtype=np.int)            # 1000 labels (0-9)

# Save in required format
np.savez('digits.npz', x_train=images, y_train=labels)

# Example 2: String labels
images = np.random.randint(0, 256, (100, 32, 32), dtype=np.uint8)   # 100 32x32 images
labels = np.array(['cat', 'dog'] * 50)                              # 100 labels ('cat'/'dog')

# Save in required format
np.savez('animals.npz', x_train=images, y_train=labels)
```

The loader will automatically:
1. Normalize image values to range [0, 1]
2. Flatten images to 1D arrays
3. Create a mapping between your labels and indices
4. Convert labels to one-hot encoding for training

## 🔮 Planned Features

### Enhanced Training Settings
- Activation function selection (ReLU, Sigmoid, Tanh)
- Batch training support
  - Configurable batch size
  - Number of batches per epoch
- Learning rate scheduling
- Early stopping options

### Dataset Support
- Custom dataset loading
  - Support for various image formats
  - CSV/JSON data import
  - Custom label mapping
  - Flexible label type support (numeric, string, etc.)
- Dataset preprocessing options
  - Image resizing
  - Normalization methods
  - Data augmentation

### Model Improvements
- Dropout layers
- Weight initialization options
- Layer visualization
- Training history graphs

### UI Enhancements
- Dark/Light theme toggle
- Training progress visualization
- Model architecture visualization
- Dataset preview window
- Custom dataset selection dialog

## 📚 Learning Outcome

This project was made to deepen understanding of:
- Neural network internals
- Gradient descent and backpropagation
- Custom activation and loss functions
- GUI development with Tkinter
- MVC architecture in Python
- Dataset handling and preprocessing
- Object-oriented design patterns

---

### 🔗 Connect
- 🏠 **Project Repository:** [GitHub](https://github.com/wilfre-oss/NN)
