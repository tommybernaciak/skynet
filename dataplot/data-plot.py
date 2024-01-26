import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV file
df = pd.read_csv('training-set.csv') 

# Separate the data points by class for coloring
class_0 = df[df['y'] == 0]
class_1 = df[df['y'] == 1]

# Create a scatter plot
plt.scatter(class_0['X_0'], class_0['X_1'], label='Class 0', c='blue', marker='o')
plt.scatter(class_1['X_0'], class_1['X_1'], label='Class 1', c='red', marker='x')

# Add labels and a legend
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Show the plot
plt.title('Data plot')
plt.show()