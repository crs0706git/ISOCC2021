import scipy.io as wv
import scipy.io.wavfile as siw
from scipy.io import wavfile
import soundfile as sf

import librosa
import librosa.display
import matplotlib.pyplot as plt
# For plotting headlessly
# https://stackoverflow.com/questions/52432731/store-the-spectrogram-as-image-in-python/52683474
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
from numpy import save
import os
import time
import csv
import math
import argparse

# Dataset directory
data_source = 'D:/D-Drive Files/000_data_backup'
data_location = data_source + '/esc10'

# Making the dataset mixture saving path
data_output_location = data_source + '/esc10_2_all_mixes_ver_max2'
if not os.path.exists(data_output_location):
    os.mkdir(data_output_location)

the_classes = os.listdir(data_location)
num_class = len(the_classes)

count = 0

total_start = time.time()

"""
CSV file to save each file information
File name, Class of A, File number of A file, Class of B, File number of B file
"""
w = open("./esc10_2_all_mixes_ver_max2.csv", "w", newline='')
wr = csv.writer(w)
write_this = ['filename']
write_this.extend(the_classes)
wr.writerow(write_this)

# Calculate the total combinations of mixing two different classes
num_class_combi = int(math.factorial(num_class) / (math.factorial(num_class-2) * math.factorial(2)))

cls_count = 1
print()
print("Started")
print()

# Select one class and extract all the files
for A_class in range(num_class - 1):
    about_A = data_location + '/' + the_classes[A_class]
    A_class_files = os.listdir(about_A)

    # Select the other class and extract all the files
    for B_class in range(A_class + 1, num_class):
        about_B = data_location + '/' + the_classes[B_class]
        B_class_files = os.listdir(about_B)

        single_class_mixing_start = time.time()

        class_check = []

        # Select files from each different classes
        for A_file in range(len(A_class_files)):

            for B_file in range(len(B_class_files)):
                count = count + 1

                # Read each audio
                A_file_loc = about_A + '/' + A_class_files[A_file]
                sr, A_data = wavfile.read(A_file_loc)
                np_A_data = np.array(A_data, dtype=np.int16)

                B_file_loc = about_B + '/' + B_class_files[B_file]
                _, B_data = wavfile.read(B_file_loc)
                np_B_data = np.array(B_data, dtype=np.int16)

                # Compare which audio is louder, normalize both audio and then mix together
                a_rate = max(A_data)
                b_rate = max(B_data)

                if a_rate > b_rate:
                    np_A_data = np_A_data/a_rate*b_rate
                else:
                    np_B_data = np_B_data/b_rate*a_rate

                # Mixing process: 1D matrix addition
                np_mix = np_A_data + np_B_data
                np_mix_16 = np.array(np_mix, dtype=np.int16)

                # Saving the mixture audio
                filename = the_classes[A_class] + str(A_file) + '_' + the_classes[B_class] + str(B_file) + '.wav'

                destination_for_all = data_output_location + '/' + filename
                wavfile.write(destination_for_all, sr, np_mix_16)

                # Saving the mixture audio information on the csv file
                checker = []
                for this_class_B in the_classes:
                    if this_class_B != the_classes[B_class]:
                        checker.append(0)
                    else:
                        checker.append(1)

                for i in range(len(the_classes)):
                    if the_classes[A_class] == the_classes[i]:
                        checker[i] = 1

                mixture_name = the_classes[A_class] + str(A_file) + '_' + the_classes[B_class] + str(B_file)
                write_this = [mixture_name]
                write_this.extend(checker)
                wr.writerow(write_this)

        # Reporting the time spent for each mixture combination
        print("Time for mixing %s and %s (%d/%d):" % (the_classes[A_class], the_classes[B_class], cls_count, num_class_combi), end=" ")
        cls_count = cls_count + 1
        
        single_class_mixing_end = time.time()

        single_class_mixing_time = single_class_mixing_end - single_class_mixing_start
        single_class_mixing_time_min = int(single_class_mixing_time/60)
        single_class_mixing_time_sec = int(single_class_mixing_time%60)
        print("%dmin %dsec" %(single_class_mixing_time_min, single_class_mixing_time_sec))

# Reporting the time spent for each mixture combination
total_end = time.time()

total_time = total_end - total_start
total_time_min = int(total_time/60)
total_time_sec = int(total_time%60)
print()
print("Total time: %dmin %dsec" %(total_time_min, total_time_sec))
print("%d combinations" % count)
print()
