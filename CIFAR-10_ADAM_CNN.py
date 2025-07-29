#CNN - LeNet-5 model
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout,Dense
from tensorflow.keras.optimizers import Adam


(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()

#normalization
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

#One-Hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

cnn = Sequential()
#Convoultion Layer two dimensional, cnn(relu func)                 (model choice)
cnn.add(Conv2D(32,(3,3),activation='relu', input_shape=(32,32,3 )))
cnn.add(Conv2D(32,(3,3),activation='relu'))
#Pooling Layer(Max Pooling), stride = 2                           (model choice)
cnn.add(MaxPooling2D(pool_size=(2,2)))
#regularization
cnn.add(Dropout(0.25))
#Convoultion Layer two dimensional, cnn(relu func)                 (model choice)
cnn.add(Conv2D(64,(3,3), activation='relu'))
cnn.add(Conv2D(64,(3,3), activation='relu'))
#Pooling Layer(Max Pooling), stride = 2                           (model choice)
cnn.add(MaxPooling2D(pool_size=(2,2)))
#regularization
cnn.add(Dropout(0.25))
#one-dimension transform
cnn.add(Flatten())
cnn.add(Dense(units=512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(units=10, activation='softmax'))


#Crossentropy loss function, Adam optimizer, learning rate = 0,01             (learn)
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#train dataset , batch_size = 128, epochs = 30                      (learn)
hist = cnn.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), verbose=2)

#predictor
res = cnn.evaluate(x_test, y_test, verbose=0)
print(res[1]*100)

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


