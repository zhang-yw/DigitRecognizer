from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from scipy import ndimage
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 3000

def data_augment(images, labels):
    augmented_images = []
    augmented_labels = []
    j = 0 
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,numpy.size(images,0)))
        augmented_images.append(x)
        augmented_labels.append(y)
        bg_value = 0
        image = numpy.reshape(x, (-1, 28))
        for i in range(4):
            angle = numpy.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)
            augmented_images.append(numpy.reshape(new_img_, 784))
            augmented_labels.append(y)
    total_data = numpy.concatenate((augmented_images, augmented_labels), axis=1)
    numpy.random.shuffle(total_data)
    return total_data

def get_data_and_labels():
    data = pd.read_csv('data/train.csv')
    labels = numpy.array(data.pop('label'))
    labels = LabelEncoder().fit_transform(labels)[:, None]
    labels = OneHotEncoder().fit_transform(labels).toarray()
    
    data = numpy.float32(data.values)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    num_images = numpy.size(data,0)
    data = numpy.reshape(data, [num_images, -1])
    labels = numpy.reshape(labels, [-1, NUM_LABELS]) 
    return data, labels

def prepare_MNIST_data(use_data_augmentation=True):
    train_data, train_labels = get_data_and_labels()

    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE,:]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:,:]

    if use_data_augmentation:
        train_total_data = data_augment(train_data, train_labels)
    else:
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)
    train_size = train_total_data.shape[0]
    
    return train_total_data, train_size, validation_data, validation_labels


