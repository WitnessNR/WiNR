import numpy as np
import random
import os
import pandas as pd
from train_myself_model import *
from PIL import Image
import pandas as pd
from tensorflow.contrib.keras.api.keras.datasets import mnist, cifar10
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.python.keras.datasets import fashion_mnist

random.seed(1215)
np.random.seed(1215)

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
        img_path = os.path.join('data/GTSRB/Final_Test/Images/',file_name)
        x_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)
        
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = to_categorical(y_test, 43)
    
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)
    
    return x_test, y_test

def generate_data_myself(dataset, model, samples=10, start=0, ids=None):
    if dataset == 'gtsrb':
        x_test, y_test = get_gtsrb_test_dataset()
        print('get gtsrb test datasets')
        
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.astype('float32')
        x_test = np.expand_dims(x_test, axis=3)
        x_test /= 255.
        y_test = to_categorical(y_test, 10)
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype('float32')
        x_test = np.expand_dims(x_test, axis=3)
        x_test /= 255.
        y_test = to_categorical(y_test, 10)
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32')
        x_test /= 255.
        y_test = to_categorical(y_test, 10)
    
    f = open('generate data.txt', 'w')
    inputs = []
    targets = []
    targets_labels = []
    true_labels = []
    true_ids = []
    
    print('generating labels...', file = f)
    if ids is None:
        ids = range(samples)
    else:
        ids = ids[start:start+samples]
        start = 0
    total = 0
    # traverse images
    for i in ids:
        original_predict = np.squeeze(model.predict(np.array([x_test[start+i]])))
        num_classes = len(original_predict)
        predicted_label = np.argmax(original_predict)
        print('predicted_label:', predicted_label, file = f)
        
        targets_labels = np.argsort(original_predict)[:-1]
        # sort label
        targets_labels = targets_labels[::-1]
        print('targets_labels:', targets_labels, file = f)
        
        true_label = np.argmax(y_test[start+i])
        print('true_label:', true_label, file = f)

        if true_label != predicted_label:
            continue
       
        else:
            total += 1 
            
            # images of test set
            inputs.append(x_test[start+i])
            
            true_labels.append(y_test[start+i])
            seq = []
            for c in targets_labels:
                targets.append(c)
                seq.append(c)
                
            print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}".format(total, start + i, 
                np.argmax(y_test[start+i]), predicted_label, np.argmax(y_test[start+i]) == predicted_label, seq), file=f)
            
            true_ids.append(start+i)
        
    print('targets:', targets, file=f)
    print('true_labels:', true_labels)
    # images of test set
    inputs = np.array(inputs)
    # target label
    targets = np.array(targets)
    # true label
    true_labels = np.array(true_labels)
    # id of images
    true_ids = np.array(true_ids)
    print('labels generated', file=f)
    print('{} images generated in total.'.format(len(inputs)),file=f)
    return inputs, targets, true_labels, true_ids

if __name__ == '__main__':
    get_gtsrb_test_dataset()

