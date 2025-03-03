import numpy as np


class FNN:
    """
    A simple Feedforward Neural Network (FNN) implementation with customizable architecture, activation functions,
    and loss functions. Uses Glorot/Xavier initialization for weight initialization.

    Attributes:
        architecture (dict): A dictionary defining the number of neurons in each layer.
        activation_func (function): Activation function for hidden layers.
        out_activation (function): Activation function for the output layer.
        loss_function (str or function): Loss function ('mse', 'binary_crossentropy', etc.) or a callable function.
        weights (dict): Dictionary storing the weight matrices for each layer.
        biases (dict): Dictionary storing the bias vectors for each layer.
    """

    def __init__(self, architecture, activation_func, out_activation, loss_function):
        """
        Initializes the model with the given architecture and activation functions.

        Args:
            architecture (dict): Dictionary defining the number of neurons per layer.
            activation_func (function): Activation function for hidden layers.
            out_activation (function): Activation function for the output layer.
            loss_function (str or function): Loss function ('mse', 'binary_crossentropy', etc.) or a callable function.
        """
        self.architecture = architecture
        self.activation_func = activation_func
        self.out_activation = out_activation
        self.loss_function = loss_function
        self.weights = {}
        self.biases = {}

        if loss_function == 'mse':
            self.loss = self._mean_squared_error
        elif loss_function == 'binary_crossentropy':
            self.loss = self._binary_crossentropy
        elif not isinstance(loss_function, str):
            self.loss = loss_function  # Placeholder; later, handle string-based loss functions separately
        else:
            raise ValueError(f"Invalid loss function: {loss_function}")

        # Initialize weights and biases using Glorot/Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes weights using Glorot/Xavier initialization.
        The weights are drawn from a normal distribution scaled by sqrt(2 / (fan_in + fan_out)).
        Biases are initialized to zero.
        """
        layers = list(self.architecture.keys())  # ['input', 'hidden1', ..., 'output']

        for i in range(len(layers) - 1):
            n_in = self.architecture[layers[i]]
            n_out = self.architecture[layers[i + 1]]

            # Glorot/Xavier initialization for weights
            self.weights[i] = np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))
            self.biases[i] = np.zeros((1, n_out))

    def forward_pass(self, X):
        """
        Performs a forward pass through the network.

        Args:
            X (numpy.ndarray): Input data of shape (num_samples, num_features).

        Returns:
            numpy.ndarray: The network's output after the forward pass.
        """
        self.layer_inputs = {}  # Stores weighted inputs before activation
        self.activations = {0: X}  # Stores activations of each layer

        layers = list(self.architecture.keys())
        for i in range(len(layers) - 1):
            W, b = self.weights[i], self.biases[i]
            self.layer_inputs[i] = np.dot(self.activations[i], W) + b  # Linear transformation

            if i == len(layers) - 2:  # Last layer
                self.activations[i + 1] = self.out_activation(self.layer_inputs[i])
            else:
                self.activations[i + 1] = self.activation_func(self.layer_inputs[i])

        return self.activations[len(layers) - 1]

    def backpropagation(self, X_train, y_train, lr):
        """
        Performs backpropagation and updates weights and biases.

        Args:
            X_train (numpy.ndarray): Training input data.
            y_train (numpy.ndarray): Training target labels.
            lr (float): Learning rate.
        """
        layers = list(self.architecture.keys())
        num_layers = len(layers) - 1  # Number of weight layers (excluding input layer)

        # Compute output layer error (based on loss function)
        if self.loss_function == "mse":
            dA = self.activations[num_layers] - y_train
        elif self.loss_function == "binary_crossentropy":
            # Numerical stability: Clip activations to avoid log(0) issues
            eps = 1e-10
            y_pred = np.clip(self.activations[num_layers], eps, 1 - eps)
            dA = -(y_train / y_pred - (1 - y_train) / (1 - y_pred))
        else:
            raise ValueError("Unknown loss function")

        # Backpropagation loop (from output to input layer)
        for i in reversed(range(num_layers)):
            W, b = self.weights[i], self.biases[i]

            # Compute activation function derivative
            if i == num_layers - 1:  # Output layer
                dZ = dA * self.out_activation(self.layer_inputs[i], derivative=True)
            else:  # Hidden layers
                dZ = dA * self.activation_func(self.layer_inputs[i], derivative=True)

            # Compute gradients
            dW = np.dot(self.activations[i].T, dZ) / X_train.shape[0]
            db = np.sum(dZ, axis=0, keepdims=True) / X_train.shape[0]

            # Update weights and biases using gradient descent
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

            # Propagate error to the previous layer
            dA = np.dot(dZ, W.T)

    def train(self, X_train, y_train, epochs=100, lr=0.01):
        """
        Trains the neural network using forward propagation and (later) backpropagation.

        Args:
            X_train (numpy.ndarray): Training input data.
            y_train (numpy.ndarray): Training target labels.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            lr (float, optional): Learning rate. Defaults to 0.01.
        """
        for epoch in range(epochs):
            y_pred = self.forward_pass(X_train)
            loss = self.loss(y_train, y_pred)
            self.backpropagation(X_train, y_train, lr)  # Currently not implemented
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")
