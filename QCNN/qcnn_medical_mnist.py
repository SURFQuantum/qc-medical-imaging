import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import matplotlib.pyplot as plt
from tensorflow import keras
import medmnist
from medmnist import INFO
from medmnist.dataset import PathMNIST
import os

# --- GLOBAL CONSTANTS ---
# N_LAYERS is now the maximum number of layers to test (N)
MAX_N_LAYERS = 5  # Changed from N_LAYERS=1 to allow iteration up to 5 layers
N_EPOCHS = 50
N_TRAIN = 30
N_TEST = 30
IMG_SIZE = 28
N_CHANNELS_INPUT = 3
N_CLASSES = 9
SAVE_PATH = "quanvolution/"
PREPROCESS = True

# Set seeds
np.random.seed(0)
tf.random.set_seed(0)


# ------------------------

# --- PENNYLANE SETUP (Modified to accept layers for dynamic initialization) ---

def initialize_quantum_circuit(n_layers):
    """Initializes the device, random parameters, and the QNode for a given number of layers."""
    dev = qml.device("default.qubit", wires=4)
    # The rand_params size depends on n_layers
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

    @qml.qnode(dev)
    def circuit(phi):
        for j in range(4):
            qml.RY(np.pi * phi[j], wires=j)

        # Only call RandomLayers if n_layers > 0
        if n_layers > 0:
            RandomLayers(rand_params, wires=list(range(4)))

        return [qml.expval(qml.PauliZ(j)) for j in range(4)]

    return circuit


def quanv_process(image, circuit_func):
    """Convolves the input image using the provided quantum circuit function."""
    out = np.zeros((IMG_SIZE // 2, IMG_SIZE // 2, 4))

    for j in range(0, IMG_SIZE, 2):
        for k in range(0, IMG_SIZE, 2):
            q_results = circuit_func(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ]
            )
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out


# --- DATA FUNCTIONS (Unchanged) ---

def load_and_preprocess_data():
    print("Loading PathMNIST data...")
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    train_dataset = DataClass(split='train', download=True)
    test_dataset = DataClass(split='test', download=True)

    train_images = train_dataset.imgs
    train_labels = train_dataset.labels.flatten()
    test_images = test_dataset.imgs
    test_labels = test_dataset.labels.flatten()

    train_images = train_images[:N_TRAIN]
    train_labels = train_labels[:N_TRAIN]
    test_images = test_images[:N_TEST]
    test_labels = test_labels[:N_TEST]

    train_images = train_images / 255
    test_images = test_images / 255

    train_images = np.array(train_images, requires_grad=False)
    test_images = np.array(test_images, requires_grad=False)

    return train_images, train_labels, test_images, test_labels


def pre_process_quantum(train_images, test_images, n_layers, circuit_func):
    """Applies quanvolution for a specific number of layers and handles saving/loading."""

    q_train_path = os.path.join(SAVE_PATH, f"q_train_images_l{n_layers}.npy")
    q_test_path = os.path.join(SAVE_PATH, f"q_test_images_l{n_layers}.npy")

    if not PREPROCESS and os.path.exists(q_train_path) and os.path.exists(q_test_path):
        q_train_images = np.load(q_train_path)
        q_test_images = np.load(q_test_path)
        return q_train_images, q_test_images

    q_train_images = []
    print(f"Quantum pre-processing ({n_layers} layers) of train images:")
    for idx, img in enumerate(train_images):
        print("{}/{}        ".format(idx + 1, N_TRAIN), end="\r")
        q_train_images.append(quanv_process(img, circuit_func))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print(f"\nQuantum pre-processing ({n_layers} layers) of test images:")
    for idx, img in enumerate(test_images):
        print("{}/{}        ".format(idx + 1, N_TEST), end="\r")
        q_test_images.append(quanv_process(img, circuit_func))
    q_test_images = np.asarray(q_test_images)

    # Save pre-processed images
    os.makedirs(SAVE_PATH, exist_ok=True)
    np.save(q_train_path, q_train_images)
    np.save(q_test_path, q_test_images)

    return q_train_images, q_test_images


# --- MODEL AND TRAINING FUNCTIONS (Unchanged) ---

def create_classical_model(output_classes):
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(output_classes, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(model, train_data, train_labels, test_data, test_labels):
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(test_data, test_labels),
        batch_size=4,
        epochs=N_EPOCHS,
        verbose=0,
    )
    return history


# --------------------------------------------------------------------------------------
## Experiment Runner
# --------------------------------------------------------------------------------------

def run_experiment_for_layers(n_layers, train_images, train_labels, test_images, test_labels):
    """Initializes the circuit, pre-processes data, trains the model, and returns final metrics."""

    print(f"\n===== Starting Run: {n_layers} Quantum Layers =====")

    # 1. Initialize Circuit and Pre-process Data
    circuit_func = initialize_quantum_circuit(n_layers)
    q_train_images, q_test_images = pre_process_quantum(train_images, test_images, n_layers, circuit_func)

    # 2. Train Model
    q_model = create_classical_model(N_CLASSES)
    q_history = train_and_evaluate(q_model, q_train_images, train_labels, q_test_images, test_labels)

    # 3. Collect Final Metrics
    final_acc = q_history.history["val_accuracy"][-1]
    final_loss = q_history.history["val_loss"][-1]

    print(f"Final Validation Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")

    return final_acc, final_loss, q_history

# --------------------------------------------------------------------------------------
## Combined Plotting Function (NEW)
# --------------------------------------------------------------------------------------

def plot_combined_epoch_results(all_histories, max_layers):
    """
    Plots validation accuracy and loss over epochs for ALL quantum layer counts
    (0 to max_layers) on a single set of axes.
    """
    plt.style.use("seaborn-v0_8")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Color map for distinct lines
    cmap = plt.cm.get_cmap('hsv', max_layers + 1)

    for n_layers in range(max_layers + 1):
        history = all_histories[n_layers]
        color = cmap(n_layers)
        label = f'Quanv Layer {n_layers}'

        # Plot Validation Accuracy
        ax1.plot(history.history["val_accuracy"], color=color, linestyle='-', label=label)

        # Plot Validation Loss
        ax2.plot(history.history["val_loss"], color=color, linestyle='-', label=label)

    # --- Accuracy Plot Configuration ---
    ax1.set_title("Validation Accuracy vs. Epochs for Various Quantum Layer Counts")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_ylim([0, 1])
    ax1.legend(title="Layers (N)", loc='lower right', fontsize='small')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Loss Plot Configuration ---
    ax2.set_title("Validation Loss vs. Epochs for Various Quantum Layer Counts")
    ax2.set_ylabel("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(top=2.5)
    ax2.legend(title="Layers (N)", loc='upper right', fontsize='small')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------
## Main Execution Block (MODIFIED)
# --------------------------------------------------------------------------------------

def main():
    """Iterates over quantum layers (0 to MAX_N_LAYERS), runs the experiment, and plots results."""

    # Data is loaded only once
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

    # Containers for final metrics across all runs
    layer_counts = list(range(MAX_N_LAYERS + 1))
    all_final_accuracies = {}
    all_final_losses = {}
    all_histories = {}

    # 1. Quantum-Enhanced Experiments (0 to MAX_N_LAYERS)
    for n_layers in layer_counts:
        final_acc, final_loss, history = run_experiment_for_layers(
            n_layers,
            train_images,
            train_labels,
            test_images,
            test_labels
        )
        all_final_accuracies[n_layers] = final_acc
        all_final_losses[n_layers] = final_loss
        all_histories[n_layers] = history

    # 2. Train a fully classical model (for comparison on original 28x28x3 data)
    print("\n===== Starting Fully Classical Baseline (No Quanvolution) =====")
    c_model = create_classical_model(N_CLASSES)
    c_history = train_and_evaluate(c_model, train_images, train_labels, test_images, test_labels)
    c_final_acc = c_history.history["val_accuracy"][-1]
    c_final_loss = c_history.history["val_loss"][-1]
    print(f"Final Classical Baseline Accuracy: {c_final_acc:.4f}, Loss: {c_final_loss:.4f}")

    # 3. Plotting: Combined Epoch-by-Epoch Results for ALL Layers (NEW REQUIREMENT)
    print("\nGenerating combined epoch-by-epoch plot for all layers...")
    plot_combined_epoch_results(all_histories, MAX_N_LAYERS)


    # 4. Plotting: Final Accuracy/Loss vs. Quantum Layers (UNCHANGED)
    print("\nGenerating final accuracy vs. layers plots...")
    layers_plot = list(all_final_accuracies.keys())
    acc_plot = [all_final_accuracies[l] for l in layers_plot]
    loss_plot = [all_final_losses[l] for l in layers_plot]

    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot 4.1: Final Accuracy vs. Quantum Layers
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers_plot, acc_plot, marker='o', color='blue', label='Quanvolution (Final Val. Accuracy)')
    ax.axhline(c_final_acc, color='red', linestyle='--', label='Classical NN (No Quanvolution)')
    ax.set_title(f"Final Validation Accuracy vs. Quantum Layer Count (PathMNIST)")
    ax.set_xlabel("Number of Quantum Layers (N)")
    ax.set_ylabel("Final Validation Accuracy")
    ax.set_xticks(layers_plot)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot 4.2: Final Loss vs. Quantum Layers
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers_plot, loss_plot, marker='o', color='green', label='Quanvolution (Final Val. Loss)')
    ax.axhline(c_final_loss, color='red', linestyle='--', label='Classical NN (No Quanvolution)')
    ax.set_title(f"Final Validation Loss vs. Quantum Layer Count (PathMNIST)")
    ax.set_xlabel("Number of Quantum Layers (N)")
    ax.set_ylabel("Final Validation Loss")
    ax.set_xticks(layers_plot)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()