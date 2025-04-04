# 🧠 Neural Network Trainer

A simple feedforward neural network written in Python and NumPy, designed for experimenting with handwritten digit recognition (MNIST dataset). Comes with a modern GUI for training and saving models.

## 🚀 Features
- Custom feedforward architecture
- Manual backpropagation implementation
- MNIST support
- Train & save models from a modern-looking GUI
- Cross-platform (Tkinter-based)
- Lightweight, no deep learning frameworks required

## 🖼️ GUI Preview

> Modern UI with draggable title bar, styled buttons, and taskbar visibility support (Windows only)

![GUI Preview](preview.png) <!-- Replace with an actual screenshot if available -->

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

Or use the GUI directly by running:

```bash
python gui.py
```

## 💾 Saving Models

After training in the GUI, click **"Save Model"** to store your `.pkl` file for future use.

## 🧪 Example Architecture

```python
first_layer = (784, 20)
output_layer = (20, 10)
nn = NeuralNetwork([first_layer, output_layer])
```

## 📚 Learning Outcome

This project was made to deepen understanding of:
- Neural network internals
- Gradient descent and backpropagation
- Custom activation and loss functions
- GUI development with Tkinter

---

### 🔗 Connect
- 🏠 **Project Repository:** [GitHub](https://github.com/wilfre-oss/NN)
