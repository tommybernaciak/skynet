import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('house-prices-reg-data.csv')

# LinearRegression
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, -1].values

X_reshaped = X.reshape(len(X), 1)

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size = 1/4, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('House prices - linear regression (training set)')
plt.xlabel('size')
plt.ylabel('price')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('House prices (test set)')
plt.xlabel('size')
plt.ylabel('price')
plt.show()

print(regressor.predict([[77]]))
