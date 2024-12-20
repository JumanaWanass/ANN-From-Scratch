import numpy as np
from tabulate import tabulate

class MSE:
    def forward(self, y_true, y_pred):
        return np.mean(0.5 * (y_true - y_pred) ** 2, dtype=np.float64)
    
    def backward(self, y_true, y_pred):
        return y_pred - y_true

class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x, dtype=np.float64))
        return self.output
    
    def backward(self, dvalues):
        return dvalues * self.output * (1 - self.output)

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.randn(output_size, input_size).astype(np.float64) * 0.01
        self.biases = np.zeros((output_size, 1), dtype=np.float64)

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.biases
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(self, dvalues):
        dvalues = self.activation.backward(dvalues)
        self.dweights = np.dot(dvalues, self.inputs.T).astype(np.float64)
        self.dbiases = dvalues
        self.dinputs = np.dot(self.weights.T, dvalues).astype(np.float64)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= (learning_rate * self.dbiases).astype(np.float64)

class ANN:
    def __init__(self, loss, optimisation, learning_rate):
        self.layers = []
        self.loss = loss
        self.optimisation = optimisation
        self.learning_rate = learning_rate
        self.metrics_history = []

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

    def train_batch(self, X, y, batch_size, epochs, X_val=None, y_val=None):
        headers = ['Epoch', 'Train_Loss', 'Train_RMSE', 'Train_R2', 'Train_Acc_5%',
                  'Val_Loss', 'Val_RMSE', 'Val_R2', 'Val_Acc_5%']
        
        print("\nTraining Progress:")
        print("-" * 100)
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]
            
            total_loss = 0
            all_predictions = []
            all_true_values = []
            num_batches = int(np.ceil(X.shape[0] / batch_size))
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, X.shape[0])
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                batch_predictions = []
                batch_losses = []
                
                for i in range(X_batch.shape[0]):
                    output = self.forward(X_batch[i].reshape(-1, 1))
                    loss = self.loss.forward(y_batch[i].reshape(-1, 1), output)
                    batch_losses.append(loss)
                    batch_predictions.append(output)
                    self.backward(y_batch[i].reshape(-1, 1))
                
                self.update()
                total_loss += np.mean(batch_losses, dtype=np.float64)
                all_predictions.extend(batch_predictions)
                all_true_values.extend(y_batch)
            
            train_predictions = np.array(all_predictions).reshape(-1, 1)
            train_true = np.array(all_true_values).reshape(-1, 1)
            train_metrics = Metrics.calculate_metrics(train_true, train_predictions, total_loss/num_batches)
            
            val_metrics = None
            if X_val is not None and y_val is not None:
                val_predictions = []
                val_losses = []
                for i in range(X_val.shape[0]):
                    val_output = self.forward(X_val[i].reshape(-1, 1))
                    val_loss = self.loss.forward(y_val[i].reshape(-1, 1), val_output)
                    val_losses.append(val_loss)
                    val_predictions.append(val_output)
                
                val_predictions = np.array(val_predictions).reshape(-1, 1)
                val_metrics = Metrics.calculate_metrics(y_val, val_predictions, np.mean(val_losses, dtype=np.float64))
            
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            self.metrics_history.append(epoch_metrics)
            
            metrics_row = [
                epoch + 1,
                f"{train_metrics['MSE']:.6f}",
                f"{train_metrics['RMSE']:.6f}",
                f"{train_metrics['R2']:.6f}",
                f"{train_metrics['Acc_5%']:.2f}%"
            ]
            
            if val_metrics:
                metrics_row.extend([
                    f"{val_metrics['MSE']:.6f}",
                    f"{val_metrics['RMSE']:.6f}",
                    f"{val_metrics['R2']:.6f}",
                    f"{val_metrics['Acc_5%']:.2f}%"
                ])
            
            if epoch == 0:
                print(tabulate([metrics_row], headers=headers, tablefmt='grid'))
            elif (epoch + 1) % 10 == 0:
                print(tabulate([metrics_row], headers=headers, tablefmt='grid'))
                
        print("-" * 100)

    def test(self, X_test, y_test):
        print("\nTest Results:")
        print("-" * 50)
        
        test_predictions = []
        test_losses = []
        
        for i in range(X_test.shape[0]):
            test_output = self.forward(X_test[i].reshape(-1, 1))
            test_loss = self.loss.forward(y_test[i].reshape(-1, 1), test_output)
            test_losses.append(test_loss)
            test_predictions.append(test_output)
        
        test_predictions = np.array(test_predictions).reshape(-1, 1)
        test_metrics = Metrics.calculate_metrics(y_test, test_predictions, np.mean(test_losses, dtype=np.float64))
        
        headers = ['Metric', 'Value']
        test_results = [
            ['Mean Squared Error', f"{test_metrics['MSE']:.6f}"],
            ['Root Mean Squared Error', f"{test_metrics['RMSE']:.6f}"],
            ['Mean Absolute Error', f"{test_metrics['MAE']:.6f}"],
            ['R-squared Score', f"{test_metrics['R2']:.6f}"],
            ['Accuracy (within 1%)', f"{test_metrics['Acc_1%']:.2f}%"],
            ['Accuracy (within 5%)', f"{test_metrics['Acc_5%']:.2f}%"],
            ['Accuracy (within 10%)', f"{test_metrics['Acc_10%']:.2f}%"]
        ]
        
        print(tabulate(test_results, headers=headers, tablefmt='grid'))
        print("-" * 50)
        
        return test_metrics

class Metrics:
    @staticmethod
    def calculate_metrics(y_true, y_pred, loss):
        mse = loss
        rmse = np.sqrt(mse, dtype=np.float64)
        mae = np.mean(np.abs(y_true - y_pred), dtype=np.float64)
        
        ss_tot = np.sum((y_true - np.mean(y_true, dtype=np.float64)) ** 2, dtype=np.float64)
        ss_res = np.sum((y_true - y_pred) ** 2, dtype=np.float64)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        acc_1p = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.01, dtype=np.float64) * 100
        acc_5p = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.05, dtype=np.float64) * 100
        acc_10p = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.10, dtype=np.float64) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Acc_1%': acc_1p,
            'Acc_5%': acc_5p,
            'Acc_10%': acc_10p
        }

def generate_data(num_samples,range):
    # X = np.random.uniform(0, range, (int(num_samples), 3)).astype(np.float64)
    X = np.random.randn(int(num_samples), 3).astype(np.float64) * range
    # X = np.dot(X, 10)
    exponent = -(X[:, 0]**2 + 2*X[:, 1]**2 + 3*X[:, 2]**2)
    # exponent = np.clip(exponent, -0.0001, 0.0001)  # Clip to avoid extremely small values
    y = np.exp(exponent, dtype=np.float64)
    # y = exponent
    y = y.reshape(-1, 1)
    return X, y

def split_data_with_test(X, y, train_ratio=0.7, val_ratio=0.15):
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    
    train_idx = int(train_ratio * num_samples)
    val_idx = train_idx + int(val_ratio * num_samples)
    
    train_indices = indices[:train_idx]
    val_indices = indices[train_idx:val_idx]
    # test_indices = indices[val_idx:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = generate_data(1000*val_ratio,0.2)
    X_test, y_test = generate_data(1000*val_ratio,0.2)
    
    print(f"\nData Split Information:")
    print(f"Total samples: {num_samples}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/num_samples*100:.1f}%)")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/num_samples*100:.1f}%)")
    print(f"Test samples: {len(X_test)} ({len(X_test)/num_samples*100:.1f}%)\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def batch_gradient_descent(parameters, learning_rate):
    for layer in parameters:
        layer.update(learning_rate)

X, y = generate_data(10000,0.2)
X_train, X_val, X_test, y_train, y_val, y_test = split_data_with_test(X, y, train_ratio=0.7, val_ratio=0.15)

network = ANN(
    loss=MSE(),
    optimisation=batch_gradient_descent,
    learning_rate= 0.1
)

network.add_layer(Layer(3, 10, Sigmoid()))
network.add_layer(Layer(10, 1, Sigmoid()))
# network.add_layer(Layer(3, 1, Sigmoid()))

network.train_batch(X_train, y_train, batch_size=32, epochs=100, X_val=X_val, y_val=y_val)

test_metrics = network.test(X_test, y_test)
