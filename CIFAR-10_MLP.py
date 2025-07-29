#MLP
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()

#one-dimensional structure transformation
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)
#normalization
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

#One-Hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

dmlp = Sequential()
#Layer = 4 floor
#Hidden layer node = 512 , activation func = relu, input type 784   (model choice)
dmlp.add(Dense(units=1024, activation='relu', input_shape=(3072, )))
#Output layer node = 512, activation func = relu                    (model choice)
dmlp.add(Dense(units=512, activation='relu'))
#Output layer node = 512, activation func = relu                    (model choice)
dmlp.add(Dense(units=512, activation='relu'))
#Output layer node = 10, activation func = softmax                  (model choice)
dmlp.add(Dense(units=10, activation='softmax'))

#crossentropy loss function, SGD optimizer, learning rate = 0,01             (learn)
dmlp.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


#train dataset , batch_size = 128, epochs = 50                      (learn)
hist = dmlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

#predictor
print('정확률', dmlp.evaluate(x_test, y_test, verbose=0)[1]*100)


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.grid()
plt.show()

#overfitting 정확률 하락