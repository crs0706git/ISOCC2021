import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

# import tensorflow as tf
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
from tqdm import tqdm
from keras.layers import BatchNormalization

import csv
import tmr
import time
import random
import math
import shutil

"""
Time settings:
Initialize timer
Record date
"""
time_start = tmr.start()

# Master directory
master_dir = 'D:/D-Drive Files/000_data_backup'
# master_dir = '.'
data_target = '/esc10'

# Directory to save the model
save_slide_results = './sliding_results_40div/'
if os.path.exists(save_slide_results):
    shutil.rmtree(save_slide_results)
    os.mkdir(save_slide_results)
if not os.path.exists(save_slide_results):
    os.mkdir(save_slide_results)

sz_x, sz_y = 120, 177

selected_model = './model_checkpoint/model_type1_40div.h5'
model = keras.models.load_model(selected_model)

start_num = 0.5
dec_place = 4
divide_fac = int(math.pow(10, dec_place))
start_from = int(start_num * divide_fac)

num_classes = 5
increment = 1

a_sec = int(120 / 5)

sec_test_loc = master_dir + data_target + '_2mix_mfcc_xy323'
sec_test_files = os.listdir(sec_test_loc)
num_to_select = len(sec_test_files)
sec_test_csv = '.' + data_target + '_2_all_mixes.csv'
tf = pd.read_csv(sec_test_csv)

total_data_length = int(num_to_select / increment)


# partial loading
def data_loading(num_to_select, increment, sec_test_loc, tf, sec_mul):
    sec_x = []
    the_names = []
    print()
    print("Sliding Prediction for %d sec - data loading" % sec_mul)
    for i in tqdm(range(0, num_to_select, increment)):
        the_names.append(tf['filename'][i])
        the_address = sec_test_loc + '/' + tf['filename'][i] + '.png'

        # Image length is 177, static
        # Image width is various depending on the length of the audio
        # 5 secs = 120 pixel, 1 sec = 24 pixel
        img = image.load_img(the_address, target_size=(177, 24 * sec_mul, 3))
        img = image.img_to_array(img)
        img = img / 255.

        sec_x.append(img)
    sec_x = np.array(sec_x)

    total_sec_y = np.array(tf.drop(['filename'], axis=1))
    sec_y = []
    for i in range(0, num_to_select, increment):
        sec_y.append(total_sec_y[i])
    sec_y = np.array(sec_y)

    return sec_x, sec_y, the_names


# Save the predicted classes in percentage for each window
def slide_prediction(total_data_length, sec_x, the_names, sec_mul):
    save_results_to = save_slide_results + 'sliding_' + str(sec_mul)
    if os.path.exists(save_results_to):
        shutil.rmtree(save_results_to)
        os.mkdir(save_results_to)
    if not os.path.exists(save_results_to):
        os.mkdir(save_results_to)

    for this_trial in range(total_data_length):
        print(this_trial, end='\r')
        testing_x = sec_x[this_trial]

        # Window analysis for each input
        current_data = []
        for the_slide in range(0, int(24 * sec_mul - 120 + 1), a_sec):
            call_x_start = the_slide
            call_x_end = 120 + the_slide

            part_x = np.zeros((1, sz_y, sz_x, 3))
            part_x[0] = testing_x[:, call_x_start:call_x_end, :]

            prediction = model.predict(part_x)

            current_data.append(prediction[0])
        current_data = np.array(current_data)

        save_as = save_results_to + '/' + the_names[this_trial]

        np.save(save_as, current_data)


sec_x, sec_y_8, the_names = data_loading(num_to_select, increment, sec_test_loc, tf, 8)
slide_prediction(total_data_length, sec_x, the_names, 8)

sec_test_loc = master_dir + data_target + '_2mix_mfcc_xy43'
sec_test_files = os.listdir(sec_test_loc)
num_to_select = len(sec_test_files)
sec_test_csv = '.' + data_target + '_2_all_mixes.csv'
tf = pd.read_csv(sec_test_csv)

sec_x, sec_y_10, the_names = data_loading(num_to_select, increment, sec_test_loc, tf, 10)
slide_prediction(total_data_length, sec_x, the_names, 10)

sec_test_loc = master_dir + data_target + '_2mix_mfcc_xy483'
sec_test_files = os.listdir(sec_test_loc)
num_to_select = len(sec_test_files)
sec_test_csv = '.' + data_target + '_2_all_mixes.csv'
tf = pd.read_csv(sec_test_csv)

# Seminar: ignore silence (black) part for predictions
sec_x, sec_y_12, the_names = data_loading(num_to_select, increment, sec_test_loc, tf, 12)
slide_prediction(total_data_length, sec_x, the_names, 12)

"""
Timer end
"""
print("Spent:", tmr.end(time_start))
print()
