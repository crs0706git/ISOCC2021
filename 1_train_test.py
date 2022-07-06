import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm
# import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from keras.layers import BatchNormalization

import csv
import argparse
import tmr
import time

import random

"""
Time settings:
Initialize timer
Record date
"""
time_start = tmr.start()
run_date = tmr.record_start('time_now')
print(run_date)
time_rec = tmr.record_start('time_record')

"""
Parser/Args settings
"""
parser = argparse.ArgumentParser()

parser.add_argument('--epo', type=int, default=1)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--filter', type=int, default=16)

parser.add_argument('--copy', type=int, default=70)

args = parser.parse_args()

# Master directory
master_dir = 'D:/D-Drive Files/000_data_backup'
# master_dir = 'E:/E Drive Files/000_data_backup'

"""
Initialize, variable settings
"""
epo = args.epo
bs = args.bs
mix_mode = 'filename'

the_dataset_is = '/esc10'

# mfcc_single = master_dir + the_dataset_is + '_single_mfcc/'
# mfcc_2mix = master_dir + the_dataset_is + '_2mix_mfcc_xy23/'
mfcc_single = '.' + the_dataset_is + '_shifting_aug_single_50div_mfcc_type1/'
mfcc_2mix = '.' + the_dataset_is + '_2mix_mfcc_xy23/'
mfcc_blank1 = '.' + the_dataset_is + '_nothing_overall/'

csv_single = '.' + the_dataset_is + '_shifting_aug_single_50div.csv'
csv_2mix = '.' + the_dataset_is + '_2_all_mixes.csv'
csv_blank1 = '.' + the_dataset_is + '_nothing_overall.csv'

"""
Loading
"""
# About single audio, mixed_mode = 'single_file'
sf = pd.read_csv(csv_single)
# About mixed audio
mf = pd.read_csv(csv_2mix)
b1f = pd.read_csv(csv_blank1)

# About classes
the_classes = sf.columns[1:]
num_classes = len(the_classes)

# Image size information
the_address = mfcc_single + sf['filename'][0] + '.png'
img = image.load_img(the_address, target_size=(177, 120, 3))
sz_x, sz_y = img.size
print("Image size is:", sz_x, sz_y)
print()

# Image reader
def img_reader(path):
    img = image.load_img(path, target_size=(177, 120, 3))
    img = image.img_to_array(img)
    img = img / 255.
    return img

"""
x: Image Addresses
y: Answers in binary. 1 for yes, 0 for no in the corresponding classes
"""
# Train-Test ratio, from 1~10
train_ratio = 7

# Extract all the answers
y_single = np.array(sf.drop(['filename'], axis=1))
y_single = y_single.tolist()
y_mix = np.array(mf.drop(['filename'], axis=1))
y_mix = y_mix.tolist()
y_b1 = np.array(b1f.drop(['filename'], axis=1))
y_b1 = y_b1.tolist()

# Single
x_single_train = []
x_single_test = []
y_single_train = []
y_single_test = []
ratio_counter = 0
for i in range(sf.shape[0]):
    the_address = mfcc_single + sf['filename'][i] + '.png'

    # Collect training data addresses
    if ratio_counter < train_ratio:
        x_single_train.append(the_address)
        y_single_train.append(y_single[i])
        ratio_counter += 1
    # Collect testing data addresses
    elif ratio_counter < 9:
        x_single_test.append(the_address)
        y_single_test.append(y_single[i])
        ratio_counter += 1
    else:
        x_single_train.append(the_address)
        y_single_train.append(y_single[i])
        ratio_counter = 0
y_single_train = np.array(y_single_train)
y_single_test = np.array(y_single_test)

# 2-mix
x_mix_train = []
x_mix_test = []
y_mix_train = []
y_mix_test = []
ratio_counter = 0
for i in range(mf.shape[0]):
    the_address = mfcc_2mix + mf['filename'][i] + '.png'

    # Collect training data addresses
    if ratio_counter < train_ratio:
        x_mix_train.append(the_address)
        y_mix_train.append(y_mix[i])
        ratio_counter += 1
    # Collect testing data addresses
    elif ratio_counter < 9:
        x_mix_test.append(the_address)
        y_mix_test.append(y_mix[i])
        ratio_counter += 1
    else:
        x_mix_train.append(the_address)
        y_mix_train.append(y_mix[i])
        ratio_counter = 0
y_mix_train = np.array(y_mix_train)
y_mix_test = np.array(y_mix_test)

"""
Model Setting
"""
model = Sequential()

f_mul = args.filter

model.add(Conv2D(filters=f_mul, kernel_size=(5, 5), activation="relu", input_shape=(sz_y, sz_x, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=f_mul*2, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=f_mul*4, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=f_mul*4, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

# model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""
Training
"""

"""
Standard Threshold
"""
th = 0.8


def predicted_scored(predictions, test_label, detect_th):
    test_length, _ = test_label.shape
    correct_counts = 0
    for i in range(test_length):
        actual_ans = []
        predicted_ans = []
        for check_num in range(len(test_label[i])):
            if test_label[i][check_num] >= detect_th:
                actual_ans.append(check_num)
            if predictions[i][check_num] >= detect_th:
                predicted_ans.append(check_num)
        if actual_ans == predicted_ans:
            correct_counts += 1
        """
        else:
            print(actual_ans, "predicted", predictions[i])
        """
    return correct_counts * 100 / test_length


def partial_loading(start_at, the_rate, total_x, total_y):
    cur_x = []
    cur_y = []
    for cur_num in tqdm(range(start_at, len(total_x), the_rate)):
        cur_x.append(img_reader(total_x[cur_num]))
        cur_y.append(total_y[cur_num])
    cur_x = np.array(cur_x)
    cur_y = np.array(cur_y)

    return cur_x, cur_y

print()

pass_score = 85
single_score = 0
mixed_score = 0
blank_score = 0

cur_reps = 1

div_rate = 4

# num_total_data = len(x_single_train) + len(x_single_test) + len(x_mix_train) + len(x_mix_test)

while single_score < pass_score or mixed_score < pass_score:
    partial_single_score = 0
    partial_mixed_score = 0
    ov_single_score = 0
    ov_mixed_score = 0
    for cur_div in range(div_rate):
        print("Single training data, division rate is %d, start at %d" % (div_rate, cur_div))
        cur_x_single_train, cur_y_single_train = partial_loading(cur_div, div_rate, x_single_train, y_single_train)

        print("Single testing data, division rate is %d, start at %d" % (div_rate, cur_div))
        cur_x_single_test, cur_y_single_test = partial_loading(cur_div, div_rate, x_single_test, y_single_test)

        print("Mixed training data, division rate is %d, start at %d" % (div_rate, cur_div))
        cur_x_mix_train, cur_y_mix_train = partial_loading(cur_div, div_rate, x_mix_train, y_mix_train)

        print("Mixed testing data, division rate is %d, start at %d" % (div_rate, cur_div))
        cur_x_mix_test, cur_y_mix_test = partial_loading(cur_div, div_rate, x_mix_test, y_mix_test)

        total_x_training = np.concatenate((cur_x_single_train, cur_x_mix_train), axis=0)
        total_y_training = np.concatenate((cur_y_single_train, cur_y_mix_train), axis=0)
        total_x_testing = np.concatenate((cur_x_single_test, cur_x_mix_test), axis=0)
        total_y_testing = np.concatenate((cur_y_single_test, cur_y_mix_test), axis=0)

        history = model.fit(total_x_training, total_y_training, epochs=1, validation_data=(total_x_testing, total_y_testing), batch_size=bs, shuffle=True)

        single_predictions = model.predict(cur_x_single_test)
        partial_single_score = predicted_scored(single_predictions, cur_y_single_test, th)
        ov_single_score += partial_single_score/div_rate

        mixed_predictions = model.predict(cur_x_mix_test)
        partial_mixed_score = predicted_scored(mixed_predictions, cur_y_mix_test, 0.3)
        ov_mixed_score += partial_mixed_score / div_rate
    single_score = ov_single_score
    mixed_score = ov_mixed_score

    cur_x_single_train = []
    cur_x_single_test = []

    cur_y_single_train = []
    cur_y_single_test = []

    cur_x_mix_train = []
    cur_x_mix_test = []

    cur_y_mix_train = []
    cur_y_mix_test = []

    total_x_training = []
    total_x_testing = []
    total_y_training = []
    total_y_testing = []

    # if 77 < single_score < 92 and 77 < mixed_score < 92:
    if 82 < single_score and 82 < mixed_score:
        print("Current score:", single_score, mixed_score, blank_score)

        total_x = []
        for i in tqdm(range(len(x_single_test))):
            total_x.append(img_reader(x_single_test[i]))
        total_x = np.array(total_x)
        single_predictions = model.predict(total_x)

        total_x = []
        for i in tqdm(range(len(x_mix_test))):
            total_x.append(img_reader(x_mix_test[i]))
        total_x = np.array(total_x)
        mixed_predictions = model.predict(total_x)

        comp_score = single_score
        for cur_th in range(70, 100, 1):
            real_th = cur_th / 100
            single_score = predicted_scored(single_predictions, y_single_test, real_th)

            if single_score > comp_score:
                comp_score = single_score
                print(real_th, comp_score)
        single_score = comp_score

        comp_score = mixed_score
        for cur_th in range(100, 1000, 1):
            real_th = cur_th / 1000
            mixed_score = predicted_scored(mixed_predictions, y_mix_test, real_th)

            if mixed_score > comp_score:
                comp_score = mixed_score
                print(real_th, comp_score)
        mixed_score = comp_score

    print("Prediction with the test data:", single_score, mixed_score)
    print("Current repetition:", cur_reps)

    """
    Model Checkpoints settings
    """
    variable_settings = str(bs) + '_' + str(epo) + '_' + str(args.copy) + '_rep' + str(cur_reps)
    model_location = './model_checkpoint/' + 'model' + '_' + variable_settings + '.h5'
    model.save(model_location)

    cur_reps += 1
    print()

"""
Timer end
"""
print("Spent:", tmr.end(time_start))
print()
