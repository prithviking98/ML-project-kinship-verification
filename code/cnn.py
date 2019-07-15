import os
import numpy as np
import cv2
import pickle

from imutils import paths
import skimage.data
import skimage.transform

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.input_layer import Input
from keras.utils.vis_utils import plot_model
from keras import backend as K

# set the matplotlib backend so figures can be saved in the background
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use("Agg")

import preprocess as pp
import sys

data_dir = sys.argv[1]

posPairs, negPairs = pp.getCNNSamples(data_dir,1)

posY = np.ones((posPairs.shape[0],1),int)
negY = np.zeros((negPairs.shape[0],1),int)


data_X = np.concatenate((posPairs,negPairs),axis=0)
data_Y = np.concatenate((posY,negY),axis=0)

(trainX, testX, trainY, testY) = train_test_split(data_X,data_Y, test_size=0.2, random_state=0)

trainY = to_categorical(trainY,num_classes=2,dtype='int32')
testY = to_categorical(testY,num_classes=2,dtype='int32')

#bulding the CNN
#input layer
ip = Input(shape=trainX[0].shape)

#first convolutional layer
c1 = Conv2D(filters=16, kernel_size=(5,5), strides=1, padding="same", activation = 'relu')(ip)

#first pooling layer
p1 = MaxPooling2D(pool_size = (2,2), strides=2)(c1)

#second convolutional layer
c2 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same",activation= 'relu')(p1)

#second pooling layer
p2 = MaxPooling2D(pool_size = (2,2), strides=2)(c2)

#third convolutional layer
c3 = Conv2D(filters=128, kernel_size=(5,5), strides=1, padding="same",activation= 'relu')(p2)

#flattening outputs of c3
fl1 = Flatten()(c3)

#fully connected layer
fc1 = Dense(640)(fl1)

#fully connected (final layer - 2 outputs)
fc6 = Dense(2, activation = 'softmax')(fc1)

#model
model = Model(inputs = ip, outputs = fc6)

#model
model = Model(inputs = ip, outputs = fc6)

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.0000000001
MOMENTUM = 0.000000003
DECAY =  0.005

# compile the model using SGD as our optimizer and binary
# cross-entropy loss (logistic regression loss)
opt = SGD(lr = INIT_LR, momentum = MOMENTUM, decay = DECAY)
# opt = SGD(lr = INIT_LR, momentum = MOMENTUM, )
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print(model.summary())
plot_model(model, show_shapes= True)

model_diagram=skimage.data.imread("model.png")
plt.figure(figsize=(10,10),dpi=512)
plt.imshow(model_diagram,aspect='auto')
plt.axis("off");

#train the model
EPOCHS = 500
BATCH_SIZE = 128

gen = ImageDataGenerator(horizontal_flip=True)
# H = model.fit_generator(gen.flow(trainX,trainY), steps_per_epoch = len(trainX)//BATCH_SIZE, epochs=EPOCHS)
H = model.fit_generator(gen.flow(trainX,trainY), validation_data=(testX,testY) , steps_per_epoch = len(trainX)//BATCH_SIZE, epochs=EPOCHS)

# evaluate the network
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), labels=[0,1]))