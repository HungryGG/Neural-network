
from PIL import Image
import gc
from google.colab import drive
from sklearn.model_selection import train_test_split
import zipfile
from google.colab import output
import os
import numpy as np
import matplotlib.pyplot as plt

import PIL
import tensorflow as tf

from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import regularizers


drive.mount('/content/drive')
!unzip /content/drive/MyDrive/Datasets.zip -d /content/data 

img_h = 256
img_w = 256
images = []
labels = []

classes = ['Roman', 'Gothic']
ind = 0.0
for cl in classes:
  path = '/content/data/Datasets/' + cl
  files = os.listdir(path)
  for img in files:
    im = Image.open(path+'/'+img).convert('RGB')
    imResize = im.resize((img_h, img_w), Image.ANTIALIAS)
    images.append(np.array(imResize))
    labels.append(ind)
  ind = ind + 1.0
  
images = np.array(images) / 255.0
labels = np.array(labels)

images_train, images_test, labels_train, labels_test = train_test_split(images, labels,
                                                    stratify=labels, 
                                                    test_size=0.2,
                                                    random_state = 42)


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.Resizing(50, 50),
  tf.keras.layers.Rescaling(1./2),
  tf.keras.layers.Resizing(25, 25),
])

model = tf.keras.models.Sequential([
    data_augmentation,                    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(600, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(images_train, labels_train, epochs=50, 
                    validation_data=(images_test, labels_test),
                    batch_size = 31)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

scores = model.evaluate(images_test, labels_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

num_images = 19
test_unit = images_test[[num_images]]
test_pred = model.predict(test_unit).squeeze()

test_lbl = labels_test[num_images]

print(test_pred)
print(test_lbl)

def plot_images(pixels: np.array):
     plt.imshow(pixels.reshape(256, 256, 3))
     plt.show()
plot_images(images_test[num_images])