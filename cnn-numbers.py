# Importing the libraries
import tensorflow as tf
import keras as ks
from keras.datasets import mnist
# from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

print(tf.__version__)
print(ks.__version__)

# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# plot first few images
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
plt.show()

# Building the CNN

cnn = ks.models.Sequential()
# Convolution layer
cnn.add(ks.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# Pooling
cnn.add(ks.layers.MaxPool2D(pool_size=2, strides=2))
# Convolution layer
cnn.add(ks.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(ks.layers.MaxPool2D(pool_size=2, strides=2))
# Flattening
cnn.add(ks.layers.Flatten())
# Full Connection
cnn.add(ks.layers.Dense(units=128, activation='relu'))
# Output Layer
cnn.add(ks.layers.Dense(units=1, activation='sigmoid'))