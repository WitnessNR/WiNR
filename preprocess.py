# -*- coding: utf-8 -*-

import os
import random
import argparse
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
from tensorflow.contrib.keras.api.keras.utils import to_categorical
import numpy as np

from skimage import transform
from skimage import exposure
from skimage import io, color
import pandas as pd

def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.h5']:
        raise argparse.ArgumentTypeError('only .h5 formats supported')
    return fname 

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (48, 48))


    return img

def get_gtsrb_test_dataset():
    test = pd.read_csv('data/GT-final_test.csv',sep=';')

    x_test = []
    y_test = []
    i = 0
    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('data/GTSRB-2/Final_Test/Images/',file_name)
        x_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)
        
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)
    
    #x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
    #print('after x_test.shape:', x_test.shape)
    
    return x_test, y_test

def load_data(dataset):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'gtsrb':
        print("[INFO] loading GTSRB testing data...")
        (x_test, y_test) = get_gtsrb_test_dataset()
        
    return x_test, y_test

def parse_argument(netname, net_type, dataset):

    assert netname, 'a network has to be provided for analysis'

    filename, file_extension = os.path.splitext(netname)

    assert file_extension==".h5", "file extension not supported"

    assert net_type in ['fnn', 'cnn'], "only fnn and cnn network type are supported"

    assert dataset in ['mnist', 'cifar10', 'fashion_mnist', 'gtsrb'], "only mnist, cifar10, fashion and gtsrb datasets are supported"

    x_test, y_test = load_data(dataset)
    print('x_test.shape', x_test.shape)
    print('y_test.shape', y_test.shape)
    
    return netname, net_type, dataset, x_test, y_test

def predict_label(nn, net_type, dataset, x_test, img):
    if net_type == 'fnn':
        if dataset in ['mnist', 'fashion_mnist']:
            shape = x_test.shape[1] * x_test.shape[2]
            a = img.reshape(1,shape)
            label = np.argmax(nn.predict(a))
            img = np.reshape(img, shape)
        elif dataset in ['cifar10', 'gtsrb']:
            shape = x_test.shape[1] * x_test.shape[2] * x_test.shape[3]
            a = img.reshape(1,shape)
            label = np.argmax(nn.predict(a))
            img = np.reshape(img, shape)
    elif net_type == 'cnn':
        if dataset in ['mnist', 'fashion_mnist']:
            a = img[np.newaxis, :, :, np.newaxis]
            label = np.argmax(nn.predict(a))
            img = img[:, :, np.newaxis]
        elif dataset in ['cifar10', 'gtsrb']:
            a = img[np.newaxis, :, :, :]
            label = np.argmax(nn.predict(a))

    return label, img
