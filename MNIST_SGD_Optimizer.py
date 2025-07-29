#MLP
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

#one-dimensional structure transformation
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
#normalization
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

#One-Hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

mlp = Sequential()
#Hidden layer node = 512 , activation func = tanh, input type 784   (model choice)
mlp.add(Dense(units=512, activation='tanh', input_shape=(784, )))
#Output layer node = 10, activation func = softmax                  (model choice)
mlp.add(Dense(units=10, activation='softmax'))

#MSE loss function, SGD optimizer, learning rate = 0,01             (learn)
mlp.compile(loss='MSE', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

#train dataset , batch_size = 128, epochs = 50                      (learn)
mlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

#predictor
res = mlp.evaluate(x_test, y_test, verbose=0)
print(res[1]*100)


