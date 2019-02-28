import os
import h5py
import json
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, load_model, Model
from keras.layers.merge import concatenate
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras import optimizers

# path to the model weights file
location = 'data'

# will be saved into when models are created
top_model_weights_path=location + '/top_model_weights.h5'
fine_tuned_model_path = location + '/ft_model.h5'

# shifting image size to 224 gives slightly better results
# since original VGG model expects images to be of this size
IMG_WIDTH, IMG_HEIGHT = 224, 224

# directories for data
train_data_dir = location + '/training'
validation_data_dir = location + '/validation'

# 3 classes - front, rear and side
# directory structure is -
# data/
#     training/
#             00-front/images
#             01-rear/images
#             02-side/images

train_samples = [len(list(Path(i).iterdir())) for i in sorted(
    list(Path(train_data_dir).iterdir()))]
nb_train_samples = sum(train_samples)
validation_samples = [len(list(Path(sub_dir).iterdir())) for sub_dir in sorted(
    list(Path(validation_data_dir).iterdir()))]
nb_validation_samples = sum(validation_samples)

# convert image to numpy array
def load_image_into_numpy_array(image):
    return np.array(image.getdata()).reshape(
        (IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.int8)

# load VGG16 model with convolution base
def load_vgg16():
    model = Sequential()
    # include_top = False skips fully connected layers and
    # loads rest of the model
    vgg16_model = VGG16(weights='VGG-16',
        include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    model.add(vgg16_model)
    print('VGG16 Model loaded.')

    return model

# prepare train/validation data
def get_data(data_dir, nb_samples):
    # get required data - train or validation
    data = np.zeros(shape=(nb_samples, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    i = 0
    for dir in Path(data_dir).iterdir():
        for file in Path(dir).iterdir():
            if file.is_file():
                image = Image.open(file).resize((IMG_WIDTH, IMG_HEIGHT))
                image_np = load_image_into_numpy_array(image)
                data[i] = image_np
                i = i + 1

    return data

# prepare test data
def get_test_data(data_dir, nb_samples):
    # get required data - train or validation
    data = np.zeros(shape=(nb_samples, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    i = 0
    for file in sorted(list(Path(data_dir).iterdir())):
        if file.is_file():
            image = Image.open(file).resize((IMG_WIDTH, IMG_HEIGHT))
            image_np = load_image_into_numpy_array(image)
            data[i] = image_np
            i = i + 1

    return data

# save bottleneck features to further train top model
def save_bottleneck_features(batch_size=128, epochs=50):
    # load VGG16 model except for fully-connected layers
    model = load_vgg16()

    # get required data for training
    train_data = get_data(train_data_dir, nb_train_samples)
    print('train data reading done')

    # get and save train bottleneck features
    bottleneck_features_train = model.predict(
        train_data, batch_size=batch_size)
    print('train predictions done')
    np.save(open('bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)

    # repeat for validation data
    validation_data = get_data(validation_data_dir, nb_validation_samples)
    print('validation data reading done')

    # get and save validation bottleneck features
    bottleneck_features_validation = model.predict(
        validation_data, batch_size=batch_size)
    print('validation predictions done')
    np.save(open('bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)

# train top model in order to fintune it further
def train_top_model(batch_size=16, epochs=50):
    # the features were saved in order, so recreating the labels is not hard
    train_data = np.load(open('./bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([0] * train_samples[0] +
        [1] * train_samples[1] +
        [2] * train_samples[2])
    train_labels = to_categorical(train_labels)

    validation_data = np.load(
        open('./bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * validation_samples[0] +
        [1] * validation_samples[1] +
        [2] * validation_samples[2])
    validation_labels = to_categorical(validation_labels)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = optimizers.SGD(lr=0.00001)
    model.compile(optimizer=optimizer,
        loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
        epochs=epochs, batch_size=batch_size,
        validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

# finetune the model
def finetune_model(batch_size=16, epochs=50):
    # load VGG16 model convolution base
    model = load_vgg16()

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation = 'relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='softmax'))

    # load weights into top model
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable - weights will not be updated
    for layer in model.layers[:25]:
        layer.trainable=False

    optimizer = optimizers.SGD(lr=0.00000001)
    model.compile(loss='categorical_crossentropy',
        optimizer = optimizer, metrics=['accuracy'])

    # train data
    train_data = get_data(train_data_dir, nb_train_samples)
    train_labels = np.array([0] * train_samples[0] +
        [1] * train_samples[1] +
        [2] * train_samples[2])
    train_labels = to_categorical(train_labels)

    # validation data
    validation_data = get_data(validation_data_dir, nb_validation_samples)
    validation_labels = np.array([0] * validation_samples[0] +
        [1] * validation_samples[1] +
        [2] * validation_samples[2])
    validation_labels = to_categorical(validation_labels)

    model.fit(train_data, train_labels,
        epochs=epochs, batch_size=batch_size,
        validation_data=(validation_data, validation_labels),
        verbose=1, callbacks=[checkpoint])

    return model

def main():
    # parse arguments to get test directory
    parser = argparse.ArgumentParser()

    # gets filename from arguments
    parser.add_argument("--test_dir", "-f", type=str, required=True)
    # args will contain name of the test directory
    args = parser.parse_args()

    save_bottleneck_features()
    train_top_model()
    ft_model = finetune_model()

if __name__ == "__main__":
    main()
