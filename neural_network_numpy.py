import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    A modernized 'from-scratch' Neural Network implementation using NumPy.
    
    Improvements over the original script:
    - Encapsulated in a class for better organization.
    - Uses vectorized operations for massive performance gains.
    - Implements mini-batch gradient descent instead of stochastic.
    - Better weight initialization (He initialization for ReLU).
    - Clear separation of forward, backward, and update steps.
    - Includes plotting for the cost function.
    """

    def __init__(self, layer_dims, learning_rate=0.01):
        """
        Initializes the neural network.
        
        Args:
            layer_dims (list): A list of integers representing the number of neurons in each layer.
                               e.g., [784, 128, 64, 10] for a 2-hidden-layer network.
            learning_rate (float): The learning rate for gradient descent.
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.parameters = {}
        self.num_layers = len(layer_dims)
        self.costs = []
        
        # He initialization for weights, zeros for biases
        for l in range(1, self.num_layers):
            # Weights: Using He initialization which is good for ReLU activations
            self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            # Biases: Initialized to zeros
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)

    def _softmax(self, Z):
        # Subtract max for numerical stability
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward_propagation(self, X):
        """Performs one forward pass through the network."""
        cache = {'A0': X}
        A = X
        
        # Loop through hidden layers with ReLU activation
        for l in range(1, self.num_layers - 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            A = self._relu(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
            
        # Output layer with Softmax activation
        l = self.num_layers - 1
        W = self.parameters[f'W{l}']
        b = self.parameters[f'b{l}']
        Z = np.dot(W, A) + b
        A = self._softmax(Z)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
        
        return A, cache

    def compute_cost(self, AL, Y):
        """Computes the cross-entropy cost."""
        m = Y.shape[1]
        # Using cross-entropy loss for classification
        cost = -1/m * np.sum(Y * np.log(AL + 1e-8)) # Add small epsilon for numerical stability
        return np.squeeze(cost)

    def backward_propagation(self, AL, Y, cache):
        """Performs one backward pass to calculate gradients."""
        grads = {}
        m = Y.shape[1]
        
        # --- Output Layer ---
        l = self.num_layers - 1
        # Gradient of the cost with respect to Z for the output layer (Softmax derivative)
        dZ = AL - Y
        
        A_prev = cache[f'A{l-1}']
        grads[f'dW{l}'] = 1/m * np.dot(dZ, A_prev.T)
        grads[f'db{l}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        
        # --- Hidden Layers (in reverse) ---
        for l in reversed(range(1, self.num_layers - 1)):
            W_next = self.parameters[f'W{l+1}']
            dZ_next = dZ
            
            dA = np.dot(W_next.T, dZ_next)
            Z = cache[f'Z{l}']
            dZ = dA * self._relu_derivative(Z)
            
            A_prev = cache[f'A{l-1}']
            grads[f'dW{l}'] = 1/m * np.dot(dZ, A_prev.T)
            grads[f'db{l}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)

        return grads

    def update_parameters(self, grads):
        """Updates weights and biases using gradient descent."""
        for l in range(1, self.num_layers):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']

    def train(self, X_train, Y_train, epochs, batch_size=64):
        """Trains the neural network."""
        m = X_train.shape[1]
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_cost = 0.
            
            # Mini-batch gradient descent
            permutation = np.random.permutation(m)
            shuffled_X = X_train[:, permutation]
            shuffled_Y = Y_train[:, permutation]
            
            for i in range(0, m, batch_size):
                X_batch = shuffled_X[:, i:i+batch_size]
                Y_batch = shuffled_Y[:, i:i+batch_size]

                # Forward -> Cost -> Backward -> Update
                AL, cache = self.forward_propagation(X_batch)
                cost = self.compute_cost(AL, Y_batch)
                grads = self.backward_propagation(AL, Y_batch, cache)
                self.update_parameters(grads)
                
                epoch_cost += cost * X_batch.shape[1]

            epoch_cost /= m
            self.costs.append(epoch_cost)
            
            duration = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - Cost: {epoch_cost:.4f} - Time: {duration:.2f}s")
    
    def predict(self, X):
        """Makes predictions for a given input."""
        AL, _ = self.forward_propagation(X)
        return np.argmax(AL, axis=0)

    def evaluate(self, X, Y_labels):
        """Evaluates the model's accuracy."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y_labels) * 100
        return accuracy
        
    def plot_cost(self):
        """Plots the cost over training epochs."""
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.title("Training Cost")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.show()

def load_and_prepare_data(train_path, test_path):
    """Loads and preprocesses the MNIST data from CSV files."""
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate labels and features
    Y_train_labels = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values / 255.0
    
    Y_test_labels = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values / 255.0
    
    # Transpose for our network architecture (features x examples)
    X_train = X_train.T
    X_test = X_test.T
    
    # One-hot encode labels
    Y_train_one_hot = np.zeros((10, Y_train_labels.size))
    Y_train_one_hot[Y_train_labels, np.arange(Y_train_labels.size)] = 1
    
    print("Data loaded and preprocessed.")
    return X_train, Y_train_one_hot, Y_train_labels, X_test, Y_test_labels


def main():
    """Main function to run the NumPy-based neural network."""
    # --- Configuration ---
    # Network architecture: 784 inputs -> 50 hidden neurons -> 10 outputs
    LAYERS = [784, 50, 10]
    LEARNING_RATE = 0.1
    EPOCHS = 10
    BATCH_SIZE = 128
    
    # --- Data Loading ---
    X_train, Y_train, Y_train_labels, X_test, Y_test_labels = load_and_prepare_data('mnist_train.csv', 'mnist_test.csv')

    # --- Model Training ---
    print("\nStarting training with NumPy network...")
    nn = NeuralNetwork(layer_dims=LAYERS, learning_rate=LEARNING_RATE)
    nn.train(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # --- Evaluation ---
    print("\nEvaluating model...")
    train_accuracy = nn.evaluate(X_train, Y_train_labels)
    test_accuracy = nn.evaluate(X_test, Y_test_labels)
    
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # --- Visualization ---
    nn.plot_cost()

if __name__ == "__main__":
    main()
