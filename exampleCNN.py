#The below script used for BnW dataset. Not tested for RGB dataset yet.

import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential ([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)), #64 filters, filter size 3 by 3, activation is ReLU (for positive value only), input shape, 1 is single byte for color depth.
    tf.keras.layers.MaxPooling2D(2, 2),  #MaxPooling because i want max value.

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),                        #2nd layer to reduce the size. much smaller, quartered and quarted again.

    tf.keras.layers.Flatten(input_shape=(28, 28)),             #the 28x28 is the shape of the input data, Flatten layer is for flattening image dimension linear array.
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)    #The unit of 10 is number of classified class, for the last layer. Assuming we wanna classify 10 class of images.
])

# model.summary()
model.summary()
