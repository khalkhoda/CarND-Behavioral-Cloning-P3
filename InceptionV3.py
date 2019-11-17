import csv
import numpy as np
import cv2
import pickle
import joblib
import os
from random import shuffle
import sklearn


def generator(samples, batch_size=36):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        batch_size = int(batch_size / 6)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = line[i].strip()
                    filename = source_path.split('/')[-1]
                    current_path = 'my_data/IMG/' + filename
                    image = cv2.imread(current_path)
                    # Crop image
                    # image = image[60:140,:,:]
                    # Normalize image
                    # image = image.astype('float32')
                    # image = image / 255.0 - 0.5
                    measurement = float(line[3])
                    if i == 1:
                        measurement += 0.2
                    elif i == 2:
                        measurement -= 0.2
                    images.append(image)
                    angles.append(measurement)
                    images.append(cv2.flip(image, 1))
                    angles.append(measurement*-1.0)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# if os.path.exists("training_data.p"):
#     print("Training data is dumped already")
#     training_data = joblib.load("training_data.p", mmap_mode=None)
#     X_train, y_train = training_data
# else:
lines = []
with open('my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print("before", lines[0])
lines.pop(0)
print("after", lines[0])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# print("current_path", current_path)
# joblib.dump(training_data, "training_data.p")

# Set our batch size
batch_size=36

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


from keras.layers import Input, Lambda, Cropping2D, Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import tensorflow as tf


input_width = 320
input_height = 140 - 60
# Using Inception with ImageNet pre-trained weights
inception = InceptionV3(weights='imagenet', include_top=False,
                        input_shape=(input_height, input_width, 3))
# Freeze what already trained
for layer in inception.layers:
    layer.trainable = False

# Input layer
image_input = Input(shape=(160, 320, 3))

# Cropping layer
cropped_input = Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3))(image_input)

# Lambda layer (normalization)
resized_input = Lambda(lambda image: image / 255.0 - 0.5, input_shape=(80,320,3))(cropped_input)

# Inception V3 layers
inp = inception(resized_input)

# Global average pooling layer
glob_avg_pool = GlobalAveragePooling2D()(inp)

# Fully connected layer
pre_output = Dense(512, activation='relu')(glob_avg_pool)

# Output layer (fully connected layer)
predictions = Dense(1, activation='softmax')(pre_output)

# model = Model(inputs=inception_input, outputs=predictions)
model = Model(inputs=image_input, outputs=predictions)

# Configure the model
model.compile(optimizer='adam', loss='mse')

print(model.summary())

from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping
# checkpoint = ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)
# stopper = EarlyStopping(monitor='val_accuracy', min_delta=0.0003, patience=5)
from math import ceil
model.fit_generator(train_generator,
 steps_per_epoch=ceil(len(train_samples)/batch_size),
 validation_data=validation_generator,
 validation_steps=ceil(len(validation_samples)/batch_size),
 # callbacks=[checkpoint, stopper],
 epochs=1,
 verbose=1)

model.save('model.h5')



# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
# model.save('model.h5')
#
