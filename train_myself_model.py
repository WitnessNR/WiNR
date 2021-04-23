import gzip
import numpy as np
import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.contrib.keras.api.keras.layers import Add, Lambda, Conv2D, AveragePooling2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.optimizers import Adam, SGD
from tensorflow.contrib.keras.api.keras.datasets import mnist, cifar10
from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random
import os
import glob
from skimage import io
from skimage import transform
from skimage import exposure, color
from tensorflow.contrib.keras.api.keras.utils import to_categorical
from tensorflow.python.keras.engine.base_layer import Layer
import h5py
    
def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_train /= 255.
    x_test /= 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def get_fashion_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_train /= 255.
    x_test /= 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def get_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
        
    return  x_train, y_train, x_test, y_test
    
def train(file_name, dataset, filters, kernels, num_epochs=5, activation = tf.nn.sigmoid, bn=False):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    elif dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = get_GTSRB_dataset()
        
    batch_size = 128
    nb_classes = 10
    nb_epoch = num_epochs
    img_rows, img_cols, img_channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    input_shape = (img_rows, img_cols, img_channels)
    
    model = Sequential()
    model.add(Convolution2D(filters[0], kernels[0], activation=activation, input_shape=input_shape))
    for f, k in zip(filters[1:], kernels[1:]):
        model.add(Convolution2D(f, k, activation=activation))
    # the output layer, with 10 classes
    model.add(Flatten())
    if dataset == 'gtsrb':
        model.add(Dense(43, activation='softmax'))
    else:
        model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    
    print("Traing a {} layer model, saving to {}".format(len(filters) + 1, file_name))
 
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name+'.h5')
    
    return {'model':model, 'history':history}
    
    
def train_lenet(file_name, dataset, params, num_epochs=10, activation=tf.nn.sigmoid, batch_size=128, train_temp=1, pool = True):
    """
    Standard neural network training procedure. Trains LeNet-5 style model with pooling optional.
    """
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    elif dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = get_GTSRB_dataset()
        
    img_rows, img_cols, img_channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    input_shape = (img_rows, img_cols, img_channels)
    
    model = Sequential()
    
    model.add(Convolution2D(params[0], (5, 5), activation=activation, input_shape=input_shape, padding='same'))
    if pool:
        model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(params[1], (5, 5), activation=activation))
    if pool:
        model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[2], activation=activation))
    model.add(Dense(10, activation='softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)
    

    if file_name != None:
        model.save(file_name+'.h5')

    return model

class ResidualStart(Layer):
    def __init__(self, **kwargs):
        super(ResidualStart, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ResidualStart, self).build(input_shape)

    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class ResidualStart2(Layer):
    def __init__(self, **kwargs):
        super(ResidualStart2, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ResidualStart2, self).build(input_shape)

    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def Residual(f, activation):
    def res(x):
        x = ResidualStart()(x)
        x1 = Conv2D(f, 3, strides=1, padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Lambda(activation)(x1)
        x1 = Conv2D(f, 3, strides=1, padding='same')(x1)
        x1 = BatchNormalization()(x1)
        return Add()([x1, x])
    return res
   
def Residual2(f, activation):
    def res(x):
        x = ResidualStart2()(x)
        x1 = Conv2D(f, 3, strides=2, padding='same')(x)
        #x1 = BatchNormalization()(x1)
        x1 = Lambda(activation)(x1)
        x1 = Conv2D(f, 3, strides=1, padding='same')(x1)
        #x1 = BatchNormalization()(x1)
        x2 = Conv2D(f, 3, strides=2, padding='same')(x)
        #x2 = BatchNormalization()(x2)
        return Add()([x1, x2])
    return res

def train_resnet(file_name, dataset, nlayer, num_epochs=10, activation=tf.nn.sigmoid):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    elif dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = get_GTSRB_dataset()
        
    print('dataset:', dataset)
    print('x_train.shape', x_train.shape)
    print('y_train.shape', y_train.shape)
        
    inputs = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    if nlayer == 2:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(y_train.shape[1], activation='softmax')(x)
    if nlayer == 3:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(y_train.shape[1], activation='softmax')(x)
    if nlayer == 4:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = Residual(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(y_train.shape[1], activation='softmax')(x)
    if nlayer == 5:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = Residual(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(y_train.shape[1], activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    # initiate the Adam optimizer
    sgd = Adam()    

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.summary()
  
    history = model.fit(x_train, y_train,
              batch_size=128,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name+'.h5')
    
    return {'model':model, 'history':history}
    
    
if __name__ == '__main__':
    train(file_name="models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself",dataset='fashion_mnist', filters=[5,5,5], kernels = [3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    train(file_name="models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself",dataset='fashion_mnist', filters=[5,5,5,5,5], kernels = [3,3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    train(file_name="models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself",dataset='fashion_mnist', filters=[5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    train(file_name="models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself",dataset='fashion_mnist', filters=[5,5,5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    train_lenet(file_name="models/cifar10_cnn_lenet_averpool_sigmoid_myself", dataset='cifar10', params=[6, 16, 100], num_epochs=10, activation = tf.nn.sigmoid)
    train_resnet(file_name='models/cifar10_resnet_2_sigmoid_myself', dataset='cifar10', nlayer=2, num_epochs=10)
    train_resnet(file_name='models/cifar10_resnet_3_sigmoid_myself', dataset='cifar10', nlayer=3, num_epochs=10)
    train_resnet(file_name='models/cifar10_resnet_4_sigmoid_myself', dataset='cifar10', nlayer=4, num_epochs=10)
    train_resnet(file_name='models/cifar10_resnet_5_sigmoid_myself', dataset='cifar10', nlayer=5, num_epochs=10)

    
