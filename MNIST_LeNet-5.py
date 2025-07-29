#CNN - LeNet-5 model
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout,Dense
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

#two-dimensional structure transformation
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)
#normalization
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

#One-Hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

cnn = Sequential()
#Convoultion Layer two dimensional, cnn(relu func)                 (model choice)
cnn.add(Conv2D(6,(5,5), padding='same',activation='relu', input_shape=(28,28,1 )))
#Pooling Layer(Max Pooling), stride = 2                           (model choice)
cnn.add(MaxPooling2D(pool_size=(2,2),strides=2))
#Convoultion Layer two dimensional, cnn(relu func)                 (model choice)
cnn.add(Conv2D(16,(5,5), padding='valid',activation='relu'))
#Pooling Layer(Max Pooling), stride = 2                           (model choice)
cnn.add(MaxPooling2D(pool_size=(2,2),strides=2))
#Convoultion Layer two dimensional, cnn(relu func)                 (model choice)
cnn.add(Conv2D(120,(5,5), padding='valid',activation='relu'))
#one-dimension transform
cnn.add(Flatten())

cnn.add(Dense(units=84, activation='relu'))
cnn.add(Dense(units=10, activation='softmax'))


#Crossentropy loss function, Adam optimizer, learning rate = 0,01             (learn)
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#train dataset , batch_size = 128, epochs = 30                      (learn)
cnn.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)

#predictor
res = cnn.evaluate(x_test, y_test, verbose=0)
print(res[1]*100)


