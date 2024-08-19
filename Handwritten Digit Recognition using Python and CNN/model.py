import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Loading Data from MNIST dataset and dividing into training and testing data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Normalising Data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Reshaping the Image arrays
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
x_testr = np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)

# Building Deep learning model
model = Sequential()

#Adding layers
model.add(Conv2D(64,(3,3), input_shape = x_trainr.shape[1:]))   #1 convolutional layer
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))                                #1 Maxpooling layer

model.add(Conv2D(64,(3,3)))                                               #2 convolutional layer
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))                                #2 Maxpooling layer

model.add(Conv2D(64,(3,3)))                                               #3 convolutional layer
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))                                #3 Maxpooling layer

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))

# Compiling- Training the model
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(x_trainr,y_train,epochs=5, validation_split=0.3, batch_size = 1)

# Testing the model- Evaluating loss and accuracy
(test_loss, test_acc)= model.evaluate(x_testr,y_test, batch_size=1)
print(test_loss)
print(test_acc)

# serialize model to JSON
json_file = model.to_json()
json_file_path = "trained_model_010921.json"
with open(json_file_path, "w") as file:
   file.write(json_file)                                  #writing in jason file

# serialize weights to HDF5
h5_file = "weights_010921.hdf"
model.save_weights(h5_file)
