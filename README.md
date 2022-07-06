The overall flow is described as follows:

![isocc flow](https://user-images.githubusercontent.com/67090206/176493496-96642b57-e0f9-4b5e-a282-347169352aee.png)

(a): A multi-label classifier with the t-sec or less duration audio <br/>
(b): A multi-label classification of audio with a longer duration

The training tactic for the model is to train single-label audio and multi-label, simultaneously. For this research, the multi-label audio set has been artificially made with the single-source audio dataset for verifying precise labeling performance. The single-source audio dataset used for this research was ESC10, extracted from ESC50: <br/>
https://github.com/karolpiczak/ESC-50

# Requirements
- tensorflow-gpu 2.1.0
- keras 2.3.1
- segmentation-models 1.0.1
- scipy 1.4.1
- scikit-learn 0.24.2
- numpy 1.19.5
- soundfile 0.10.3
- librosa 0.8.1
- matplotlib
- tqdm

# 0. Preprocessings
Codes:
- 0_2mix_audio.py
- 0_mfcc_converter.py

## 0_2mix_audio.py
As explained above, the training tactic for this research was simultaneously training the model with single-label and multi-label audio set. For this purpose, the multi-label audio set (2-class mixture) was made with mixing each audio in the single-source audio dataset (ESC10).<br/>
The following code prepares the 2-class label dataset. Since ESC10 dataset only contains audio that each is recorded single-source sound. Two main processes are done for mixing:
1. Volume Normalization (comparison approach) <br/>
Before mixing, the volume of each audio are considered. When mixing, certain mixture audio had in case of one audio being louder than the other audio and so causing the other audio to be harder to distinguish. Such a situation defects the model detection performance, so it is considered an impossible case to deal with. The situation is neglected for this research and applied the alternate method, called Volume Normalization, to resolve the situation.
For this research, the volume normalization was done to each mixing audio file. Among each audio to be mixed, the code first determines the loudest audio among the files. Then, the code normalize each audio to the loudest volume value.

2. 1D Matrix Addition
Calculates the total number of files by mixing two audio of different classes. After Volume Normalization, 1D matrix addition is performed between the audio to mix with. The audio files mixed are associated with different classes to each other file.

## 0_mfcc_converter.py
Since the model used in this research was based on CNN, the image data was considered to train the model. The following code converts the designated audio files to Mel Spectrogram image files.

# 1. Multi-label Classification
Code:
- 1_train_test.py

Before start training and validating, single audio and 2-class mixture audio are concatenated. For each epoch, the code test with the model to verify if the model can answer up to certain percentage for each single and 2-class mixture testing data. If both prediction scores are reached to the certain level, the training is finished. <br/>

Pseudocode: <br/>
```
While single_score < 85% and mixed_score < 85%
	training(1-epoch)
	
	single_score = prediction_score(single_audio, threshold)
	mixed_score = prediction_score(mixed_audio, threshold)

	if single_score ≥ 80% and mixed_score ≥ 80%:
		for low-threshold ~ high-threshold:
			single_score = prediction_score(single_audio, threshold)
			mixed_score = prediction_score(mixed_audio, threshold)
```

The training process is as follows:
![isocc train process](https://user-images.githubusercontent.com/67090206/176684107-2a46c80b-a654-4926-8f94-dd3afc4f609a.png)


The model structure is as follows:
![isocc_model](https://user-images.githubusercontent.com/67090206/176517258-99895ebb-685e-44a4-ba64-8bc73bc5b140.png)

# 2. Windowing
Codes:
- 2_2mix_audio_8sec.py
- 2_2mix_audio_10sec.py
- 2_2mix_audio_12sec.py
- 2_windowing.py

For the sample audio that is more than 5-sec, the code applies windowing, analyze for several times until the code analyze the entire duration.

## 2_2mix_audio_8sec.py / 2_2mix_audio_10sec.py / 2_2mix_audio_12sec.py
The following codes are to prepare the audio longer than 5-sec, the audio duration that the model is trained already trained with.

![81012](https://user-images.githubusercontent.com/67090206/176685679-ea84505e-7c3b-484e-92e2-43cbcba3ede2.png)

- 8sec: Overlapped Audio
The 8 seconds duration of the overlapped audio is mixed on only 2 seconds portions from the two single audio are overlapped. The model should detect multiple classes correctly by detecting either a single class from the unmixed portions or two classes from the mixed portion.

- 10sec: Adjacent Audio
The mixture is a connection of two audio of different classes, the model has options for detecting the classes either at a single class from each audio portion or two classes from the middle portion.

- 12sec: Separated Audio
The separated audio is also a connection of two audio, with silence portion is in the mixture between the two audio.

## 2_windowing.py
The following code labels the audio that is longer than 5-sec. The code analyzes first 5-sec of the audio, and then shifts 1-sec of the audio and analyze the audio 5-sec of the audio again.

![isocc windowing](https://user-images.githubusercontent.com/67090206/176685357-307f0f6e-3f02-40fc-993d-e65cdee33dc0.png)


# 9. Utility
Code:
- tmr.py

The following code contains functions for recording operation time and starting operation time. Simply, just use "datatime" library which is a basic Python library given when installed.
