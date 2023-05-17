# Accident detection-twilio
This Deep Learning project basically first identifies uses CNN whether the given picture is an accident or not. In case it is , it calls the nearest ambulance using the twilio api.

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers

from time import perf_counter 

import os

from keras.callbacks import ModelCheckpoint

from keras.models import 

load_model

from tensorflow.keras.utils import plot_model

batch_size = 100

img_height = 250

img_width = 250

training_data = tf.keras.preprocessing.image_dataset_from_directory(
 'C:\\Users\\HP\\Desktop\\gg\\Accident-Detection-System\\data\\train',
seed=42,
image_size= (img_height, img_width),    
batch_size=batch_size,
color_mode='rgb'
)

## loading validation dataset
validation_data =  tf.keras.preprocessing.image_dataset_from_directory(
    'C:\\Users\\HP\\Desktop\\gg\\Accident-Detection-System\\data\\val',
seed=42,
image_size= (img_height, img_width),
batch_size=batch_size,
color_mode='rgb'
)

training_data


class_names = training_data.class_names
class_names

## loading testing dataset
testing_data = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\\Users\\HP\\Desktop\\gg\\Accident-Detection-System\\data\\train',
seed=42,
image_size= (img_height, img_width),
batch_size=batch_size,
color_mode='rgb'
)

testing_data

AUTOTUNE = tf.data.experimental.AUTOTUNE
training_data = training_data.cache().prefetch(buffer_size=AUTOTUNE)
testing_data = testing_data.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.models.Sequential([
  layers.BatchNormalization(),
  layers.Conv2D(32, 3, activation='relu'), ## activation function 
  layers.MaxPooling2D(), # MaxPooling
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(256, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(len(class_names), activation= 'softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.build((None, 250, 250, 3))
model.summary()

import pydot
import graphviz ## to visualize the neural network model
from keras.utils import plot_model

plot_model(model, show_shapes=True)

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(training_data, validation_data=validation_data, epochs = 20, callbacks=callbacks_list)

###### serialize model structure to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

## stats on training data
plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['accuracy'], label = 'training accuracy')
plt.grid(True)
plt.legend()

## stats on training data
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
plt.grid(True)
plt.legend()

AccuracyVector = []
plt.figure(figsize=(30, 30))
for images, labels in testing_data.take(1):
    predictions = model.predict(images)
    predlabel = []
    prdlbl = []
    
    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))
    
    AccuracyVector = np.array(prdlbl) == labels
    for i in range(40):
        ax = plt.subplot(10, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title('Pred: '+ predlabel[i]+' actl:'+class_names[labels[i]] )
        plt.axis('off')
        plt.grid(True)


import os

from twilio.rest import Client

account_sid="AC7908c08e52c95d9c55d06c5c20234423"

auth_token="232cd6881523fab02584653d2a4ce527"

client=Client(account_sid,auth_token)

call=client.calls.create(
   to="+917386320904",
   from_="+12075696430",
   url="http://demo.twilio.com/docs/voice.xml"    
)

if(class_names[labels[i]]=="Accident"):
    print(call.sid)
