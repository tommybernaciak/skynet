import numpy as np

# Set a random seed for reproducibility
np.random.seed(0)

# Number of data points
num_samples = 100

# Define the parameters for the linear boundary
slope = 2
intercept = -1

# Generate random data points
X = np.random.rand(num_samples, 2) * 10

# Create labels based on the linear boundary
y = (X[:, 1] > slope * X[:, 0] + intercept).astype(int)

# Split the data into a training set and a test set
split_ratio = 0.8
split_index = int(num_samples * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Print the shapes of the training and test sets
print("Training set shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("\nTest set shapes:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("Training set:")
print("X_train:", X_train)
print("y_train:", y_train)
print("\nTest set:")
print("X_test:", X_test)
print("y_test:", y_test)