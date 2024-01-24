import pandas as pd
import numpy as np
dataset = pd.read_csv('house-prices.csv')
# get all data columns as X and prices as y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Count the missing values (NaN) in each column
missing_counts = dataset.isnull().sum()

# Print the number of missing entries for each column
print("Number of missing entries for each column:")
print(missing_counts)

# take care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer.fit(X[:, 1:5])
X[:, 1:5] = imputer.transform(X[:, 1:5])
print("Fixed missing data:")
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print("After splitting to training and test set:")
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 1:] = sc.fit_transform(X_train[:, 1:])
X_test[:, 1:] = sc.transform(X_test[:, 1:])
print("Feature scaling:")
print("Training set features:")
print(X_train)
print("Training set target:")
print(y_train)

print("Test set features:")
print(X_test)
print("Test set target:")
print(y_test)
