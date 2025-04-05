import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
import sys
import pickle
from main import train_model, save_model

if sys.platform.startswith("win"):
    import ctypes

    def show_in_taskbar(window):
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        # WS_EX_TOOLWINDOW = 0x00000080, WS_EX_APPWINDOW = 0x00040000
        ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        ex_style = ex_style & ~0x00000080  # Remove tool window flag.
        ex_style = ex_style | 0x00040000   # Add app window flag.
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, ex_style)
        # Sometimes withdrawing and deiconifying forces the update.
        window.wm_withdraw()
        window.after(10, lambda: window.wm_deiconify())
else:
    def show_in_taskbar(window):
        pass  # No-op on non-Windows platforms

def create_gui():
    window = tk.Tk()
    window.overrideredirect(True)
    window.geometry("400x320")
    window.resizable(False, False)
    
    trained_model = None
    epochs = tk.StringVar(value="5")
    hidden_layers = tk.StringVar(value="20")
    learning_rate = tk.StringVar(value="0.01")

    def train_network():
        nonlocal trained_model
        try:
            epoch_value = int(epochs.get())
            hidden_layer_sizes = [int(layer) for layer in hidden_layers.get().split(",")]
            learn_rate_value = float(learning_rate.get())
        except ValueError:
            status_label.config(text="Please enter valid values in settings.")
            return

        def update_progress(epoch: int, total_epochs: int, accuracy: float | None):
            if accuracy is None:
                status_label.config(text=f"Epoch {epoch}/{total_epochs} - Accuracy: N/A")
            else:
                status_label.config(text=f"Epoch {epoch}/{total_epochs} - Accuracy: {accuracy:.2f}%")
            window.update_idletasks()

        trained_model, accuracy = train_model(
            epochs=epoch_value, 
            hidden_layer_sizes=hidden_layer_sizes,
            learn_rate=learn_rate_value,
            progress_callback=update_progress,
            network=trained_model
        )
        status_label.config(text=f"Training complete! Epoch {epoch_value}/{epoch_value} - Accuracy: {accuracy:.2f}%")

    def save_trained_model():
        if trained_model is None:
            status_label.config(text="Train the model first!")
            return

        file_path = save_model(trained_model)
        if file_path:
            status_label.config(text=f"Model saved to {file_path}")
        else:
            status_label.config(text="Failed to save model")

    def load_trained_model():
        nonlocal trained_model
        file_path = filedialog.askopenfilename(
            initialdir=Path.cwd(),
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl")],
            title="Load trained model"
        )
        if file_path:
            try:
                trained_model = pickle.load(open(file_path, "rb"))
                status_label.config(text=f"Model loaded from {file_path}")
            except Exception as e:
                status_label.config(text=f"Failed to load model: {str(e)}")
        else:
            status_label.config(text="Failed to load model")

    def open_settings():
        settings_window = tk.Toplevel(window)
        settings_window.title("Neural Network Settings")
        settings_window.geometry("400x320")
        settings_window.resizable(False, False)

        # Create a frame for the settings
        settings_frame = ttk.Frame(settings_window, padding="20 20 20 20")
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid for settings frame
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=1)
        for row in range(5):
            settings_frame.rowconfigure(row, weight=1)
            
        # Create settings header
        settings_header = ttk.Label(settings_frame, text="Training Settings", font=('Helvetica', 14, 'bold'))
        settings_header.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 15))
        
        # Create training parameters frame
        training_frame = ttk.Frame(settings_frame)
        training_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        training_frame.columnconfigure(0, weight=1)
        training_frame.columnconfigure(1, weight=1)
        
        # Epochs setting
        epoch_label = ttk.Label(training_frame, text="Epochs:")
        epoch_label.grid(row=0, column=0, sticky="e", padx=5, pady=5)
        
        epoch_entry = ttk.Entry(training_frame, textvariable=epochs, width=10)
        epoch_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Hidden layers setting
        hidden_layer_label = ttk.Label(training_frame, text="Hidden Layer Sizes:")
        hidden_layer_label.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        
        hidden_layer_entry = ttk.Entry(training_frame, textvariable=hidden_layers, width=10)
        hidden_layer_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Learning rate setting
        learning_rate_label = ttk.Label(training_frame, text="Learning Rate:")
        learning_rate_label.grid(row=2, column=0, sticky="e", padx=5, pady=5)
        
        learning_rate_entry = ttk.Entry(training_frame, textvariable=learning_rate, width=10)
        learning_rate_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Add save button to settings window
        save_settings_button = ttk.Button(settings_frame, text="Save Settings", 
                                         command=lambda: settings_window.destroy())
        save_settings_button.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

    def test_model():
        if trained_model is None:
            status_label.config(text="Train the model first!")
            return


    def start_move(event):
        window.x = event.x
        window.y = event.y

    def stop_move(event):
        window.x = None
        window.y = None

    def on_move(event):
        deltax = event.x - window.x
        deltay = event.y - window.y
        x = window.winfo_x() + deltax
        y = window.winfo_y() + deltay
        window.geometry(f"+{x}+{y}")

    def close_window():
        window.destroy()

    # Create a frame for the custom title bar
    title_bar = tk.Frame(window, bg="#2e3f4f", relief="raised", bd=0)
    title_bar.pack(fill=tk.X)

    # Title label in the custom title bar
    title_label = tk.Label(title_bar, text="Neural Network Trainer", bg="#2e3f4f", fg="white", font=("Helvetica", 14))
    title_label.pack(side=tk.LEFT, padx=10, pady=5)

    # Close button in the custom title bar
    close_button = tk.Button(title_bar, text="✕", bg="#2e3f4f", fg="white", bd=0, command=close_window, font=("Helvetica", 14))
    close_button.pack(side=tk.RIGHT)
    close_button.bind("<Enter>", lambda e: close_button.config(bg="#ff0000"))
    close_button.bind("<Leave>", lambda e: close_button.config(bg="#2e3f4f"))

    # Bind the title bar for moving the window
    for widget in (title_bar, title_label):
        widget.bind("<ButtonPress-1>", start_move)
        widget.bind("<ButtonRelease-1>", stop_move)
        widget.bind("<B1-Motion>", on_move)

    # Use the themed ttk style for the rest of the GUI
    style = ttk.Style(window)
    style.theme_use('clam')
    style.configure('TLabel', font=('Helvetica', 12))
    style.configure('TButton', font=('Helvetica', 12), padding=6)
    style.configure('TEntry', font=('Helvetica', 12), padding=4)
    style.configure('TLabelframe', font=('Helvetica', 12))
    style.configure('TLabelframe.Label', font=('Helvetica', 12, 'bold'))

    # Create a main frame with padding for the rest of the widgets
    mainframe = ttk.Frame(window, padding="20 20 20 20")
    mainframe.pack(fill=tk.BOTH, expand=True)
      
    # Configure grid column and row weights
    columns = 2
    rows = 4
    for col in range(columns):
        mainframe.columnconfigure(col, weight=1)
    for row in range(rows):
        mainframe.rowconfigure(row, weight=1)

    # Button widgets in main frame
    settings_button = ttk.Button(mainframe, text="Settings", command=open_settings)
    settings_button.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

    save_button = ttk.Button(mainframe, text="Save Model", command=save_trained_model)
    save_button.grid(row=1, column=0, columnspan=1, sticky="ew", padx=5, pady=10)

    load_button = ttk.Button(mainframe, text="Load Model", command=load_trained_model)
    load_button.grid(row=1, column=1, columnspan=1, sticky="ew", padx=5, pady=10)

    train_button = ttk.Button(mainframe, text="Train", command=train_network)
    train_button.grid(row=2, column=0, columnspan=1, sticky="ew", padx=5, pady=10)

    test_button = ttk.Button(mainframe, text="Test", command=test_model)
    test_button.grid(row=2, column=1, columnspan=1, sticky="ew", padx=5, pady=10)

    status_label = ttk.Label(mainframe, text="Status: Waiting...", anchor="center")
    status_label.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

    # Optional: Create a frame for a custom border if desired
    border_frame = tk.Frame(window, bg="#2e3f4f")
    border_frame.place(x=0, y=0, relwidth=1, height=2)

    window.update_idletasks()
    win_width = window.winfo_width()
    win_height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (win_width // 2)
    y = (screen_height // 2) - (win_height // 2)
    window.geometry(f"{win_width}x{win_height}+{x}+{y}")

    show_in_taskbar(window)
    return window

