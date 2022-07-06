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
import tmr

parser = argparse.ArgumentParser()

# Settings for the max frequency, number of mels, image width and length, and dots per inches (dpi)
# Number of mels: Scaling in different rates for representing each frequency potions
# dpi: Pixel = Length of the image x dpi (6.4 * 100dpi = 640 pixels)
parser.add_argument('--f', type=int, default=7000)
parser.add_argument('--m', type=int, default=1024)
parser.add_argument('--x', type=float, default=2)
parser.add_argument('--y', type=int, default=3)
parser.add_argument('--dpi', type=int, default=77)

args = parser.parse_args()

# Dataset to save at
data_source = 'D:/D-Drive Files/000_data_backup'
data_target = data_source + '/esc10'

the_classes = os.listdir(data_target)
num_class = len(the_classes)

# Bring all the audio to convert
data_audio_location = data_target + '_2_all_mixes_ver_norm'
files_to_convert = os.listdir(data_audio_location)

# Saving it into new place
data_mfccs = data_audio_location + '_mfcc_type1_1'
if not os.path.exists(data_mfccs):
    os.mkdir(data_mfccs)

count = 0

total_start = time.time()
cls_count = 1
print()
print("Started for", data_mfccs)
print()
f_set = args.f
mel_set = args.m
for the_file in files_to_convert:
    source_file = data_audio_location + '/' + the_file

    # Load each file
    y, sr = librosa.load(source_file)

    # Convert audio into MFCC
    # Seminar: Check if this is MFCC
    S = librosa.feature.melspectrogram(y=y, sr=int(sr / 2), n_mels=mel_set, fmax=f_set)

    # Set figure size
    fig = plt.figure(num=1, clear=True, figsize=(args.x, args.y), dpi=args.dpi)
    # Set dB expression method
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Plotting
    img = librosa.display.specshow(S_dB, sr=sr, fmax=f_set)

    plt.axis('off')

    # Save image
    image_name = data_mfccs + '/' + the_file.replace('.wav', '.png')
    fig.savefig(image_name, bbox_inches='tight', pad_inches=0)

    print("%d/%d" % (cls_count, len(files_to_convert)), end='\r')
    cls_count += 1

print("Total:", tmr.end(total_start))
print("Done", cls_count-1)
