import csv
import numpy as np
import cv2
from random import shuffle
import sklearn


def generator(samples, batch_size=32):
    '''
    Generator function used to extract training and validation samples als batch_samples.
    Using this function allows the training tu run on lower memory resources.
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        batch_size = int(batch_size / 6)
        if batch_size != 0:
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    for i in range(3):
                        source_path = batch_sample[i].strip()
                        filename = source_path.split('/')[-1]
                        current_path = 'data/IMG/' + filename
                        im_cv = cv2.imread(current_path)
                        im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
                        # im_rgb = im_cv
                        # im_rgb = im_cv
                        # Crop image
                        # image = image[60:140,:,:]
                        # Normalize image
                        # image = image.astype('float32')
                        # image = image / 255.0 - 0.5
                        measurement = float(batch_sample[3])
                        if i == 1:
                            measurement += 0.2
                        elif i == 2:
                            measurement -= 0.2
                        images.append(im_rgb)
                        angles.append(measurement)
                        images.append(cv2.flip(im_rgb, 1))
                        angles.append(measurement*-1.0)
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)


# Create a list of all lines in the driving_log.csv file
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# This could be used to delete first line of the csv file read above to avoid the headers (if any)
# print(lines[0])
lines.pop(0)
# print(lines[0])

# Split the available data into two sets (training 80% and validation 20%)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Set batch size
batch_size=36


# Create a generator for training to generate batches of data
train_generator = generator(train_samples, batch_size=batch_size)
# Create a generator for validation to generate batches
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Normalize pixel values to the range [-0.5, +0.5]
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# Crop the top 50 lines and the bottom 20 lines of each input image (since it contains data not relevant for road recognition)
model.add(Cropping2D(cropping=((50,20),(0,0))))

# In the following convolutional layers, 'relu' function was used to introduce non liniearity
# to the network
# Apply a 5x5 convolution with 24 output filters on a 90x320 image:
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))

# Add a 5x5 convolution on top, with 36 output filters:
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))

# Add a 5x5 convolution on top, with 48 output filters:
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))

# Add a 3x3 convolution on top, with 64 output filters:
model.add(Convolution2D(64,3,3,activation='relu'))

# Add a 3x3 convolution on top, with 64 output filters:
model.add(Convolution2D(64,3,3,activation='relu'))

# Add fully connected layers with no activation functions
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compile the model using ADAM algorithm as an optimizer and the mean square error as a measure of error
model.compile(loss='mse', optimizer='adam')

from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping
# Save the best model only using validation loss as metric
checkpoint = ModelCheckpoint(filepath='model_invidia.h5', monitor='val_loss', save_best_only=True)
# Stop training on further epochs once no more progress in the validation has been seen
stopper = EarlyStopping(monitor='val_accuracy', min_delta=0.0003, patience=5)
# Fit the model to the data batch by batch
from math import ceil
history_object = model.fit_generator(train_generator,
 steps_per_epoch=ceil(len(train_samples)/batch_size),
 validation_data=validation_generator,
 validation_steps=ceil(len(validation_samples)/batch_size),
 callbacks=[checkpoint, stopper],
 epochs=10,
 verbose=1)

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
