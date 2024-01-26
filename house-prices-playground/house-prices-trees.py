import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('house-prices-reg-data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

test_data = [[77,2,2,1]]

prediction = regressor.predict(test_data)
print('DecisionTreeRegressor prediction:')
print(prediction)

from sklearn.ensemble import RandomForestRegressor
forestRegressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
forestRegressor.fit(X, y)

prediction2 = forestRegressor.predict(test_data)
print('RandomForestRegressor prediction:')
print(prediction2)