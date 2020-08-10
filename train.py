#importing libraries
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


h, w = 224, 224

# path1 = 'my_data/'         #for user face data
# path2 = 'blank_data/'      #for blank face data
path1 = 'v_data/'            #for gesture data ('V')
path2 = 'blankh_data/'       #for gesture blank data    also you can use blank face data together with this 

train_images, train_labels = [], []

#preprocessing the data before training
for i in range(len(os.listdir(path1))):
	train_image = cv2.imread(path1 + str(i) + '.png')
	train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
	train_image = cv2.resize(train_image, (h, w))
	train_images.append(train_image)
	train_labels.append(0)

for i in range(len(os.listdir(path2))):
	train_image = cv2.imread(path2 + str(i) + '.png')
	train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
	train_image = cv2.resize(train_image, (h, w))
	train_images.append(train_image)
	train_labels.append(1)

train_images = np.array(train_images)

X_train, X_test, y_train, y_test = train_test_split(train_images, to_categorical(train_labels), test_size=0.2, random_state=42)

#building the model
base_model = VGG16(              #use any model you prefer
    weights='imagenet',
    include_top=False, 
    input_shape=(h, w, 3), 
    pooling='avg'
)

base_model.trainable = False

model = Sequential([
  base_model,
  Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 16
epochs = 10

datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
)

#training
model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), validation_data = (X_test, y_test),
                    steps_per_epoch = len(X_train) / batch_size, epochs = epochs)


test_images = train_images[90:110]
test_labels = train_labels[90:110]
labels = model.predict(test_images)
labels = [np.argmax(i) for i in labels]
print(labels)
print("------------------------------------------------------------")
print(test_labels)

# model.save("model_face.h5")          #use when training for face data
model.save("model_hand.h5")            #use when training for gesture data

print("Training Done!")