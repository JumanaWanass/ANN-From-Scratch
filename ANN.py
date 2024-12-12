import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.biases
        self.output = self.activation.forward(self.z)

    def backward(self, dvalues):
        self.dweights = np.dot(dvalues, self.inputs.T)
        self.dbiases = np.sum(dvalues, axis=1, keepdims=True)
        self.dinputs = np.dot(self.weights.T, dvalues)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

class Activation_Functions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_backward(x):
        return x * (1 - x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_backward(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)

    @staticmethod
    def softmax_backward(dvalues, y_true):
        samples = len(dvalues)
        dvalues[range(samples), y_true] -= 1
        return dvalues / samples

class Loss_Functions:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean(0.5 * (y_true - y_pred) ** 2)

    @staticmethod
    def mean_squared_error_backward(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def cross_entropy(y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    @staticmethod
    def cross_entropy_backward(y_true, y_pred):
        return -(y_true / y_pred)


class Optimisation:
    @staticmethod
    def stochastic_gradient_descent(parameters, learning_rate):
        for layer in parameters:
            layer.update(learning_rate)

    @staticmethod
    def batch_gradient_descent(parameters, learning_rate):
        for layer in parameters:
            layer.update(learning_rate)

    @staticmethod
    def pattern_search(parameters, learning_rate):
        pass  # we will choose an algorithm later

class ANN:
    def __init__(self, loss, optimisation, learning_rate):
        self.layers = []
        self.loss = loss
        self.optimisation = optimisation
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, y_true):
        dvalues = self.loss.backward(y_true, self.layers[-1].output)
        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs

    def update(self):
        self.optimisation(self.layers, self.learning_rate)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                # Forward pass
                output = self.forward(X[i].reshape(-1, 1))
                # Compute loss 
                loss = self.loss.forward(y[i].reshape(-1, 1), output)
                # Backward pass
                self.backward(y[i].reshape(-1, 1))
                # Update weights
                self.update()
            print(f"Epoch {epoch + 1}, Loss: {loss}")

    def predict(self, X):
        outputs = [self.forward(x.reshape(-1, 1)) for x in X]
        return [np.argmax(output) for output in outputs]
