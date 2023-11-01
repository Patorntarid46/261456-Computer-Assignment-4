import numpy as np
import matplotlib.pyplot as plt

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

# Define the Multilayer Perceptron (MLP) class
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(self.hidden_size, self.input_size)
        self.bias_hidden = np.random.rand(self.hidden_size, 1)
        self.weights_hidden_output = np.random.rand(self.output_size, self.hidden_size)
        self.bias_output = np.random.rand(self.output_size, 1)
        
    def forward(self, x):
        # Calculate input for hidden layer
        self.hidden_input = np.dot(self.weights_input_hidden, x) + self.bias_hidden
        # Apply activation function to hidden layer
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Calculate input for output layer
        self.output_input = np.dot(self.weights_hidden_output, self.hidden_output) + self.bias_output
        # Apply activation function to output layer
        self.output = sigmoid(self.output_input)
        
        return self.output

# Define the Particle class for Particle Swarm Optimization (PSO)
class Particle:
    def __init__(self, num_params):
        self.position = np.random.rand(num_params)
        self.velocity = np.random.rand(num_params)
        self.best_position = self.position
        self.best_value = float('inf')

# Train the model using PSO
def train_PSO(mlp, X, y, num_particles, max_iterations, c1, c2):
    num_params = mlp.hidden_size * (mlp.input_size + 1) + mlp.output_size * (mlp.hidden_size + 1)
    particles = [Particle(num_params) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')
    
    for _ in range(max_iterations):
        for particle in particles:
            # Update weights and biases of MLP with particle's position
            mlp.weights_input_hidden = particle.position[:mlp.hidden_size*mlp.input_size].reshape(mlp.hidden_size, mlp.input_size)
            mlp.bias_hidden = particle.position[mlp.hidden_size*mlp.input_size:mlp.hidden_size*(mlp.input_size+1)].reshape(mlp.hidden_size, 1)
            mlp.weights_hidden_output = particle.position[mlp.hidden_size*(mlp.input_size+1):mlp.hidden_size*(mlp.input_size+1)+mlp.output_size*mlp.hidden_size].reshape(mlp.output_size, mlp.hidden_size)
            mlp.bias_output = particle.position[mlp.hidden_size*(mlp.input_size+1)+mlp.output_size*mlp.hidden_size:].reshape(mlp.output_size, 1)
            
            # Calculate MSE for the current position
            y_pred = mlp.forward(X)
            mse = np.mean((y - y_pred)**2)
            
            # Update best position and value of the particle
            if mse < particle.best_value:
                particle.best_position = particle.position
                particle.best_value = mse
            
            # Update global best position and value
            if mse < global_best_value:
                global_best_position = particle.position
                global_best_value = mse
        
        for particle in particles:
            # Update velocity and position
            particle.velocity = 0.5 * particle.velocity + c1 * np.random.rand() * (particle.best_position - particle.position) + c2 * np.random.rand() * (global_best_position - particle.position)
            particle.position = particle.position + particle.velocity
            
    # Update MLP with global best position
    mlp.weights_input_hidden = global_best_position[:mlp.hidden_size*mlp.input_size].reshape(mlp.hidden_size, mlp.input_size)
    mlp.bias_hidden = global_best_position[mlp.hidden_size*mlp.input_size:mlp.hidden_size*(mlp.input_size+1)].reshape(mlp.hidden_size, 1)
    mlp.weights_hidden_output = global_best_position[mlp.hidden_size*(mlp.input_size+1):mlp.hidden_size*(mlp.input_size+1)+mlp.output_size*mlp.hidden_size].reshape(mlp.output_size, mlp.hidden_size)
    mlp.bias_output = global_best_position[mlp.hidden_size*(mlp.input_size+1)+mlp.output_size*mlp.hidden_size:].reshape(mlp.output_size, 1)

# Set parameters for PSO
input_size = 8
hidden_size = 8
output_size = 1
num_particles = 20
max_iterations = 100
c1 = 1.5
c2 = 1.5

# Load data from the file "AirQualityUCI.xlsx"
import pandas as pd

data = pd.read_excel("D:/CI/comassign4byme/AirQualityUCI.xlsx")
X = data.iloc[:, [2, 5, 7, 9, 10, 11, 12, 13]].values  # Select attributes 3, 6, 8, 10, 11, 12, 13, 14
y = data.iloc[:, 5].values  # Use attribute 5 as target

# Normalize the data
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Define the cross validation function
def cross_validation(X, y, num_folds, input_size, hidden_size, output_size, num_particles, max_iterations, c1, c2):
    fold_size = len(X) // num_folds
    mae_values = []

    for i in range(num_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size

        X_train = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
        y_train = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)
        X_test = X[start_idx:end_idx]
        y_test = y[start_idx:end_idx]

        mlp = MLP(input_size, hidden_size, output_size)
        train_PSO(mlp, X_train.T, y_train.reshape(1, -1), num_particles, max_iterations, c1, c2)
        y_pred = mlp.forward(X_test.T).flatten()

        mae = np.mean(np.abs(y_test - y_pred))
        mae_values.append(mae)

    return np.mean(mae_values)

# Perform cross validation
num_folds = 10
mae_cv = cross_validation(X, y, num_folds, input_size, hidden_size, output_size, num_particles, max_iterations, c1, c2)
print(f'MAE for 10-fold cross validation: {mae_cv}')

# Test the model performance by comparing results using different configurations of hidden layers and nodes
configs = [(4, 4), (4, 8), (8, 8), (8, 16)]  # Different combinations of hidden layers and nodes

for config in configs:
    hidden_size = config[0]
    num_nodes = config[1]
    mae = cross_validation(X, y, num_folds, input_size, hidden_size, output_size, num_particles, max_iterations, c1, c2)
    print(f'Hidden Layers: {hidden_size}, Nodes per Layer: {num_nodes}, MAE: {mae}')

# Define a function for making predictions
def predict_future(mlp, input_data, num_days):
    predictions = []

    for _ in range(num_days):
        # Predict for the next day
        prediction = mlp.forward(input_data).flatten()
        predictions.append(prediction[0])  # เพิ่ม prediction เข้าไปใน list

        # Shift input data to incorporate the prediction for the next day
        input_data = np.roll(input_data, -1, axis=0)
        input_data[-1] = prediction[-1]

    return predictions


# Select the most recent data as the initial input for prediction
initial_input = X[-1]

# Create an instance of MLP
mlp = MLP(input_size, hidden_size, output_size)

#  Benzene 5days
num_days_5 = 5
prediction_5_days = predict_future(mlp, initial_input, num_days_5)
print(f'Predicted Benzene concentration for the next 5 days: {prediction_5_days}')

#  Benzene 10days
num_days_10 = 10
prediction_10_days = predict_future(mlp, initial_input, num_days_10)
print(f'Predicted Benzene concentration for the next 10 days: {prediction_10_days}')

# Function of plotting
def plot_predictions(actual, predicted, num_days):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(range(len(actual), len(actual) + num_days), predicted, label='Predicted', marker='x')
    plt.xlabel('Days')
    plt.ylabel('Benzene Concentration')
    plt.title(f'Predicted Benzene Concentration ')
    plt.legend()
    plt.show()

# plot of 5 days
plot_predictions(y[-5:], prediction_5_days, num_days_5)

# plot of 10 days
plot_predictions(y[-10:], prediction_10_days, num_days_10)