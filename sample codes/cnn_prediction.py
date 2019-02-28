#!/usr/bin/env python3

import time
import sys
import datetime
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from os import listdir
from os.path import isfile, join
from keras import optimizers
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from sklearn.metrics import mean_squared_error
from PIL import Image

# avoid errors caused by truncted files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# image size
IMAGE_WIDTH, IMAGE_HEIGHT = 256, 256

# paths to image directory
PATH_TO_IMAGES_DIR = '/trainImages'

IMAGE_PATHS = [
    f for f in sorted(list(Path(PATH_TO_IMAGES_DIR).iterdir())) if f.is_file()
]

NUM_DAMAGE_CLASSES = 4

LENGTH_OF_IMAGES_DIR = len(IMAGE_PATHS)

def load_image_into_numpy_array(image):
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.float)

def onehot_encode(column):
    return pd.get_dummies(column)

def get_required_features():
    # Load dataset
    parse_dates = ['ReportedDate']

    # stores numeric features of the data
    data = pd.read_csv(
        './damagePredictionCost.csv',
        low_memory=False,
        dtype=object,
        parse_dates=True)

    nonimage_columns = ['IncidentID',
        'AirbagsDeployed',
        'ImpactPoints',
        'PrimaryImpactPoints',
        'SecondaryImpactPoint',
        'IsHeavyEquipment',
        'DamageCost']

    # Taking required columns and dropiing duplicates
    require_features = data[nonimage_columns]

    return required_features

def load_train():

    # get tabular features
    tabular_features = get_required_features()

    # IncidentID -> DamageCost mapping
    data = tabular_features[['IncidentID', 'DamageCost']]
    data['IncidentID'] = data['IncidentID'].astype(str)
    data['DamageCost'] = data['DamageCost'].astype(float)

    # This csv contains class labels based on viewing angle of vehicles
    # format: image_name -> class_label
    reader = csv.reader(open('./class_labels.csv'))
    damageclass_mappings = {row[0]:row[1:] for row in reader if row and row[0]}

    i = 0
    image_train = np.zeros(shape=(
        LENGTH_OF_IMAGES_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    y_train = np.zeros(shape=(LENGTH_OF_IMAGES_DIR), dtype=np.float64)

    start = time.time()

    for image_path in IMAGE_PATHS:

        # read image and convert to numpy array
        image = Image.open(image_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        image_np = load_image_into_numpy_array(image)
        image_train[i] = image_np

        # Get the incident id from image name
        # to fetch corresponding damage cost
        incident_id = image_path.name.split('_')[1]
        incident_id = incident_id.split('.jpg')[0]
        cost = data.loc[
            data['IncidentID'] == incident_id, 'DamageCost'].iloc[0]
        y_train[i] = cost

        # get rest of the corresponding tabular features for IncidentID
        categorical_features = tabular_features.loc[
            tabular_features['IncidentID'] == incident_id]
        categorical_features = categorical_features.drop(
            columns=['IncidentID', 'DamageCost']).iloc[0]

        # code block to one hot encode and
        # add damage class label to tabular data
        class_label = damageclass_mappings[image_path.name][0]
        damage_one_hot_encode = np.zeros(NUM_DAMAGE_CLASSES, dtype=np.uint8)
        damage_one_hot_encode[int(class_label)] = 1

        # append damage class mapping to rest of the tabular features
        all_tabular_features = np.append(
            categorical_features.values, damage_one_hot_encode)

        if(i == 0):
            tabular_train = np.zeros(shape=(
                LENGTH_OF_IMAGES_DIR, all_tabular_features.shape[0]),
                dtype=np.uint8)

        tabular_train[i] = all_tabular_features
        i += 1

    end = time.time()
    time_taken = str(round((end-start)/60))
    print('loading done in {} minutes'.format(time_taken))

    return image_train, tabular_train, y_train

def init_model(tabular_features_num):

    nb_filters = 32
    nb_conv = 3

    # image input shape
    image_input_shape = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # tabular input shape
    tabular_data_input = Input(shape=(tabular_features_num,))

    image_model = Conv2D(nb_filters, nb_conv, nb_conv,
        border_mode='same')(image_input_shape)
    image_model = MaxPooling2D(pool_size=(2, 2))(image_model)
    image_model = Dropout(0.5)(image_model)

    image_model = Conv2D(nb_filters*2, nb_conv, nb_conv,
        activation='relu')(image_model)
    image_model = MaxPooling2D(pool_size=(2, 2))(image_model)
    image_model = Dropout(0.5)(image_model)

    image_model = Conv2D(nb_filters*4, nb_conv, nb_conv,
        activation='relu')(image_model)
    image_model = MaxPooling2D(pool_size=(2, 2))(image_model)
    image_model = Dropout(0.5)(image_model)

    image_model = Conv2D(nb_filters*8, nb_conv, nb_conv,
        activation='relu')(image_model)
    image_model = MaxPooling2D(pool_size=(2, 2))(image_model)
    image_model = Dropout(0.5)(image_model)

    image_model = Conv2D(nb_filters*16, nb_conv, nb_conv,
        activation='relu')(image_model)
    image_model = MaxPooling2D(pool_size=(2, 2))(image_model)
    image_model = Dropout(0.5)(image_model)

    image_model_output = Flatten()(image_model)
    # merge layers
    merged_model = concatenate(
        [image_model_output, tabular_data_input], axis=1)

    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(4096, activation='relu',
        kernel_regularizer=regularizers.l2(0.0001))(merged_model)
    merged_model = Dropout(0.5)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(1024, activation='relu',
        kernel_regularizer=regularizers.l2(0.0001))(merged_model)
    merged_model = Dropout(0.5)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(512, activation='relu',
        kernel_regularizer=regularizers.l2(0.0001))(merged_model)
    merged_model = Dropout(0.5)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(256, activation='relu',
        kernel_regularizer=regularizers.l2(0.0001))(merged_model)
    merged_model = Dropout(0.5)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dense(128, activation='relu',
        kernel_regularizer=regularizers.l2(0.0001))(merged_model)
    merged_model = Dropout(0.5)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    predictions = Dense(1, activation ='linear')(merged_model)

    # Now create the model
    model = Model(
        inputs=[image_input_shape, tabular_data_input], outputs=predictions)

    optimizer = optimizers.Adadelta(lr=0.5)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

# function to split test and train data
# test proportion of n mean 1/n is the size of test data
def shuffle(matrix, target, test_proportion):
    ratio = matrix.shape[0]/test_proportion
    X_train = matrix[int(ratio):,:,:,:]
    X_test =  matrix[:int(ratio),:,:,:]
    Y_train = target[int(ratio):]
    Y_test =  target[:int(ratio)]
    return X_train, X_test, Y_train, Y_test

def run_network(batch_size = 256, nb_epoch = 20):

    # image reading and preprocessing
    image_train_np, tabular_train_np, train_target_np = load_train()
    # split data into test and train (train:test - 0.80:0.20)
    image_train, image_test,
    tabular_train, tabular_test,
    y_train, y_test = shuffle(
        image_train_np, tabular_train_np, train_target_np, 5)

    # split train into train and validation
    image_train, image_valid,
    tabular_train, tabular_valid,
    y_train, y_valid = shuffle(
        image_train, tabular_train, y_train, 5)

    # Compile model
    model = init_model(tabular_train_np.shape[1])

    start_time = time.time()
    # fit the data
    model.fit(
        [image_train, tabular_train],
        y_train, batch_size=batch_size,
        nb_epoch=nb_epoch, verbose=1,
        validation_data=([image_valid, tabular_valid], y_valid))
    end_time = time.time()
    t = str(round((end_time - start_time) / 60))
    print('fit done in {} minutes'.format(t))

    # predict
    predictions_valid = model.predict(
        [image_test, tabular_test], batch_size=batch_size, verbose=1)

    # Write actual and predicted cost in a csv file for comparison
    compare = pd.DataFrame(data={'original':y_test.flatten(),
             'prediction':predictions_valid.flatten()})
    compare.to_csv('compare.csv')

def main():
    run_network(128, 20)

if __name__ == "__main__":
    main()
