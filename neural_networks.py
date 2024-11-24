import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        
        # Initialize weights and biases
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.1
        self.b2 = np.zeros((output_dim, 1))

        # Activation function mapping
        self.activation, self.activation_derivative = {
            'tanh': (tanh, tanh_derivative),
            'relu': (relu, relu_derivative),
            'sigmoid': (sigmoid, sigmoid_derivative)
        }[activation]

        # For visualization
        self.hidden_output = None
        self.gradients = None

    def forward(self, X):
        
        self.Z1 = self.W1 @ X.T + self.b1  # Shape: (n_hidden, m)
        self.A1 = self.activation(self.Z1)  # Shape: (n_hidden, m)
        self.Z2 = self.W2 @ self.A1 + self.b2  # Shape: (n_output, m)
        self.A2 = self.activation(self.Z2)  
        self.hidden_output = self.A1  # Store for visualization
        
        return self.A2  # Shape: (1, m)

    def backward(self, X, y):
        m = X.shape[0]  # Number of samples

        # Output layer gradients
        dZ2 = self.A2 - y  # Shape: (1, m)
        dW2 = (dZ2 @ self.A1.T) / m  # Shape: (1, n_hidden)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # Shape: (1, 1)

        # Hidden layer gradients
        dA1 = self.W2.T @ dZ2  # Shape: (n_hidden, m)
        dZ1 = dA1 * self.activation_derivative(self.Z1)  # Shape: (n_hidden, m)
        dW1 = (dZ1 @ X) / m  # Shape: (n_hidden, n_input)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # Shape: (n_hidden, 1)

        # Update weights and biases
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Store gradients for visualization
        self.gradients = {'dW1': dW1, 'dW2': dW2}

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(1, -1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Train the MLP for 10 steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden space features
    hidden_features = mlp.hidden_output.T  # Shape: (n_samples, 3)

    # Plot points in the hidden space
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7
    )

    # Add hyperplane in the hidden space
    w1, w2, w3 = mlp.W2.flatten()  # Weights from hidden to output layer
    b = mlp.b2[0, 0]  # Bias from hidden to output layer

    # Define a grid for the hyperplane
    h = 0.5
    x_range = np.linspace(-1, 1, 30)
    y_range = np.linspace(-1, 1, 30)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = -(w1 * xx + w2 * yy + b) / (w3 + 1e-10)  # Avoid divide by zero

    # Plot the hyperplane
    ax_hidden.plot_surface(xx, yy, zz, color='orange', alpha=0.3)

    # Set titles and limits for the hidden space plot
    ax_hidden.set_title(f"Hidden Space at Step {frame}")
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_ylim(-1, 1)
    ax_hidden.set_zlim(-1, 1)

    # Input space decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], cmap='bwr', alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame}")

    # Gradient visualization
    # Visualize features and gradients as circles and edges
    dW1 = mlp.gradients['dW1']
    dW2 = mlp.gradients['dW2']

    # Plot gradients as edges
    for i in range(dW1.shape[0]):
        for j in range(dW1.shape[1]):
            ax_gradient.plot([0, 0.5], [j / (dW1.shape[1] - 1), i / (dW1.shape[0] - 1)], 'k-', alpha=0.5, lw=np.abs(dW1[i, j]) * 10)

    for i in range(dW2.shape[0]):
        for j in range(dW2.shape[1]):
            ax_gradient.plot([0.5, 1], [j / (dW2.shape[1] - 1), 0], 'k-', alpha=0.5, lw=np.abs(dW2[i, j]) * 10)

    # Plot nodes as circles
    ax_gradient.scatter([0, 0], [0, 1], s=100, c='blue', zorder=5)  # Input layer
    ax_gradient.scatter([0.5] * dW1.shape[0], [i / (dW1.shape[0] - 1) for i in range(dW1.shape[0])], s=100, c='green', zorder=5)  # Hidden layer
    ax_gradient.scatter([1], [0], s=100, c='red', zorder=5)  # Output layer

    ax_gradient.set_title("Gradient Visualization")
    ax_gradient.set_xticks([0, 0.5, 1])
    ax_gradient.set_xticklabels(['Input Layer', 'Hidden Layer', 'Output Layer'])
    ax_gradient.set_yticks([0, 0.5, 1])
    ax_gradient.set_yticklabels(['0', '0.5', '1'])


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, update, frames=step_num, repeat=False, 
                        fargs=(mlp, ax_input, ax_hidden, ax_gradient, X, y))

    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
