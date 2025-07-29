import tensorflow as tf
import matplotlib.pyplot as plt
import keras.datasets as ds
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()     #MNIST DATASET
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
plt.figure(figsize=(24, 3))
plt.suptitle('MNIST', fontsize=30)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(str(y_train[i]),fontsize=30)
#plt.show()

(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()     #CIFAR-10 DATASET
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
plt.figure(figsize=(24, 3))
plt.suptitle('CIFAR-10', fontsize=30)
class_name = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(class_name[y_train[i,0]],fontsize=30)

plt.show()