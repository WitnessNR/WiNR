import numpy as np
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split
import os
import glob
import h5py
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Conv2D

from tensorflow.keras.optimizers import SGD, Adam
import tensorflow
tensorflow.keras.backend.set_image_data_format('channels_last')

NUM_CLASSES = 43
IMG_SIZE = 48
seed = 42
np.random.seed(seed)

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
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])

def get_data():
    try:
        with  h5py.File('X.h5') as hf: 
            X, Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X.h5")
        
    except (IOError,OSError, KeyError):  
        print("Error in reading X.h5. Processing all images...")
        root_dir = 'data/GTSRB/Final_Training/Images/'
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            try:
                img = preprocess_img(io.imread(img_path))
                label = get_class(img_path)
                imgs.append(img)
                labels.append(label)

                if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass

        X = np.array(imgs, dtype='float32')
        Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

        with h5py.File('X.h5','w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)
    return X,Y
        
def cfnn_model(nlayer):
    model = Sequential()
    
    model.add(Conv2D(100, (IMG_SIZE, IMG_SIZE), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='sigmoid'))

    model.add(Flatten())

    for i in range(nlayer-2):
        model.add(Dense(500, activation='sigmoid', kernel_initializer='glorot_uniform'))
    #model.add(Dropout(0.5))
    
    # model.add(Dense(500, activation='sigmoid', kernel_initializer='glorot_uniform'))
    
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def train_gtsrb_model(file_name, nlayer):
    model = cfnn_model(nlayer)
    # let's train the model using SGD + momentum (how original).
    sgd = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])


    def lr_schedule(epoch):
        return lr*(0.1**int(epoch/10))

    batch_size = 32
    nb_epoch = 30
    X, Y = get_data()
    print(X.shape)
    print(Y.shape)


    model.fit(X, Y,
            batch_size=batch_size,
            epochs=nb_epoch,
            validation_split=0.2,
            shuffle=True)
    model.save(file_name)

if __name__ == '__main__':  
    train_gtsrb_model('models/gtsrb_cnn_5layer_sigmoid_myself.h5', 5)
    