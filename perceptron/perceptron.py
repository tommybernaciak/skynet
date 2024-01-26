import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, dataset_filename, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        df = pd.read_csv(dataset_filename)
         # Split the data into features (X) and labels (y)
        self.X = df[['X_0', 'X_1']].values
        self.y = df['y'].values


        self.weights = np.zeros(self.X.shape[1])
        self.bias = 0

    def draw_plot(self):
        # Separate the data form y by class
        # Create a scatter plot for class 0 (where y == 0)
        plt.scatter(self.X[self.y == 0][:, 0], self.X[self.y == 0][:, 1], label='Class 0', marker='o')

        # Create a scatter plot for class 1 (where y == 1)
        plt.scatter(self.X[self.y == 1][:, 0], self.X[self.y == 1][:, 1], label='Class 1', marker='x')

        # Add labels and a legend
        plt.xlabel('X_0')
        plt.ylabel('X_1')
        plt.legend()

        # Show the plot
        plt.title('Data plot')
        plt.show()

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def show_predictions(self):
        for j in range(len(self.X)):
            prediction = self.predict(self.X[j])
            print(f"{self.X[j]} -> {prediction} -> {self.y[j]}")


#     def fit(self, X, y):
#         self.weights = np.zeros(X.shape[1])
#         self.bias = 0

#         for _ in range(self.n_iterations):
#             for i in range(X.shape[0]):
#                 y_pred = self.predict(X[i])
#                 update = self.learning_rate * (y[i] - y_pred)
#                 self.weights += update * X[i]
#                 self.bias += update

# def train(self, X, Y):
#     for j in range(len(X)):
#         # compute prediction error
#         error = Y[j] - self.predict(X[j])

#         # go though all the input features and update the weights
#         for i in range(len(X[j])):
#             self.weights[i] += self.learning_rate * error * X[j][i]
        
#         # update the bias
#         self.bias += self.learning_rate * error
    
# # Evaluate the Perceptron
# y_pred = [perceptron.predict(x) for x in X_test]
# accuracy = np.mean(y_pred == y_test)
# print(f"Accuracy: {accuracy * 100:.2f}%")



# def perceptron(inputs, weights, threshold):
#     # Compute the weighted sum of inputs and weights
#     net_input = sum(x * w for x, w in zip(inputs, weights))

#     # Apply the step activation function
#     if net_input >= threshold:
#         output = 1
#     else:
#         output = 0

#     return output

# Example inputs, weights, and threshold
# inputs = [1, 0, 1]
# weights = [0.5, -0.5, 0.2]
# threshold = 0.0

# # Calculate the output of the perceptron
# output = perceptron(inputs, weights, threshold)

# # Display the result
# print("Output:", output)



perceptron = Perceptron('training-set.csv')
perceptron.draw_plot()
perceptron.show_predictions()