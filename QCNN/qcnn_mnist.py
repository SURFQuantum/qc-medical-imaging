import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import matplotlib.pyplot as plt
from tensorflow import keras
import os

# --- Constants and Configuration ---
# Setting up a directory for saving pre-processed data
SAVE_PATH = "quanvolution/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Constants
N_EPOCHS = 30  # Number of optimization epochs
MAX_LAYERS = 10  # Maximum number of random layers to test (will iterate from 0 to this value)
STEP = 2
N_TRAIN = 50  # Size of the train dataset
N_TEST = 30  # Size of the test dataset
PREPROCESS = True  # If False, skip quantum processing and load data from SAVE_PATH

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Simulating 4 qubits.
DEV = qml.device("default.qubit", wires=4)


# --- Data Loading and Preprocessing Functions ---

def load_and_preprocess_data(n_train, n_test):
    """Loads and preprocesses the MNIST dataset."""
    mnist_dataset = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

    # Reduce dataset size
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

    # Normalize pixel values within 0 and 1
    train_images = train_images / 255
    test_images = test_images / 255

    # Add extra dimension for convolution channels (for CNN compatibility)
    train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
    test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)

    return train_images, train_labels, test_images, test_labels


# --- Quantum Circuit and Convolution Functions ---

# NOTE: The quantum circuit is now defined inside the main loop to handle the
# varying number of layers. The functions below now accept a 'circuit' argument.

def quanv(image, circuit):
    """
    Convolves the input image with many applications of the same quantum circuit.
    The image is divided into 2x2 squares, each processed by the quantum circuit,
    halving the resolution and producing 4 output channels.
    """
    out = np.zeros((14, 14, 4))

    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            pixels = [
                image[j, k, 0],
                image[j, k + 1, 0],
                image[j + 1, k, 0],
                image[j + 1, k + 1, 0]
            ]
            q_results = circuit(pixels)
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out


def quantum_preprocess_dataset(images, name, n_samples, n_layers, circuit):
    """Applies the quanvolution layer to a dataset and saves the result."""
    q_images = []
    print(f"\nQuantum pre-processing of {name} images with {n_layers} layer(s):")
    for idx, img in enumerate(images):
        print("{}/{}        ".format(idx + 1, n_samples), end="\r")
        q_images.append(quanv(img, circuit))
    q_images = np.asarray(q_images)
    # Save with a name that includes the number of layers
    np.save(SAVE_PATH + f"q_{name}_images_{n_layers}_layers.npy", q_images)
    return q_images


def get_preprocessed_images(train_images, test_images, n_train, n_test, n_layers, circuit, preprocess_flag):
    """Manages the loading or creation of quantum-preprocessed images for a specific layer count."""
    q_train_path = SAVE_PATH + f"q_train_images_{n_layers}_layers.npy"
    q_test_path = SAVE_PATH + f"q_test_images_{n_layers}_layers.npy"

    if preprocess_flag or not os.path.exists(q_train_path) or not os.path.exists(q_test_path):
        q_train_images = quantum_preprocess_dataset(train_images, "train", n_train, n_layers, circuit)
        q_test_images = quantum_preprocess_dataset(test_images, "test", n_test, n_layers, circuit)
    else:
        print(f"\nLoading pre-processed data for {n_layers} layer(s) from {SAVE_PATH}...")
        q_train_images = np.load(q_train_path)
        q_test_images = np.load(q_test_path)

    return q_train_images, q_test_images


# --- Classical Model and Training Functions ---

def create_classical_model():
    """Initializes and returns a custom Keras model."""
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_data, train_labels, test_data, test_labels, n_epochs, label):
    """Trains and validates the Keras model."""
    print(f"\n--- Training Model {label} ---")
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(test_data, test_labels),
        batch_size=4,
        epochs=n_epochs,
        verbose=2,
    )
    return history


# --- Visualization Functions ---

def plot_quanv_output(train_images, q_train_images):
    """Plots a sample of the input and quantum-convolved images."""
    n_samples = 4
    n_channels = 4
    fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))

    for k in range(n_samples):
        axes[0, 0].set_ylabel("Input")
        if k != 0:
            axes[0, k].yaxis.set_visible(False)
        axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")

        for c in range(n_channels):
            axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
            if k != 0:
                axes[c + 1, k].yaxis.set_visible(False)
            axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")

    plt.tight_layout()
    plt.show()


def plot_all_histories(histories, c_history):
    """Plots the validation accuracy and loss for all models."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    # Define a color cycle for the plots
    colors = plt.cm.viridis(np.linspace(0, 1, len(histories)))

    # Plot Accuracy
    for i, (n_layers, history) in enumerate(histories.items()):
        ax1.plot(history.history["val_accuracy"], "-o", label=f"Quantum ({n_layers} layers)", color=colors[i])
    ax1.plot(c_history.history["val_accuracy"], "-o", label="Classical baseline", color="red")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Plot Loss
    for i, (n_layers, history) in enumerate(histories.items()):
        ax2.plot(history.history["val_loss"], "-o", label=f"Quantum ({n_layers} layers)", color=colors[i])
    ax2.plot(c_history.history["val_loss"], "-o", label="Classical baseline", color="red")
    ax2.set_ylabel("Loss")
    ax2.set_ylim(top=2.5)
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.show()


# --- Main Execution Block ---

def main():
    """Executes the Quanvolutional Neural Network workflow for multiple layer configurations."""
    # 1. Load and prepare classical data (done once)
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data(N_TRAIN, N_TEST)

    # 2. Train classical baseline model (done once)
    c_model = create_classical_model()
    c_history = train_model(
        c_model, train_images, train_labels, test_images, test_labels, N_EPOCHS, "Classical Baseline"
    )

    # Dictionary to store the history of each quantum model
    q_histories = {}

    # 3. Loop over the number of quantum layers
    for n_layers in range(MAX_LAYERS + STEP):
        print(f"\n{'=' * 30}\nProcessing for N_LAYERS = {n_layers}\n{'=' * 30}")

        # Define random parameters and the quantum circuit for the current n_layers
        rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

        @qml.qnode(DEV)
        def circuit(phi):
            # Encoding of 4 classical input values
            for j in range(4):
                qml.RY(np.pi * phi[j], wires=j)

            # Add RandomLayers only if n_layers > 0
            if n_layers > 0:
                RandomLayers(rand_params, wires=list(range(4)))

            # Measurement producing 4 classical output values
            return [qml.expval(qml.PauliZ(j)) for j in range(4)]

        # Quantum Pre-processing
        q_train_images, q_test_images = get_preprocessed_images(
            train_images, test_images, N_TRAIN, N_TEST, n_layers, circuit, PREPROCESS
        )
        print("\nPre-processing complete.")

        # Visualize Quanvolution output only for the 1-layer case to avoid clutter
        if n_layers == 1:
            plot_quanv_output(train_images, q_train_images)

        # Train model with Quantum-processed data
        q_model = create_classical_model()
        q_history = train_model(
            q_model, q_train_images, train_labels, q_test_images, test_labels, N_EPOCHS,
            f"with Quantum Layer(s): {n_layers}"
        )

        # Store the history for final plotting
        q_histories[n_layers] = q_history

    # 4. Plot all results
    plot_all_histories(q_histories, c_history)


if __name__ == "__main__":
    main()