
from numpy import argmax
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
from keras import layers
from keras.preprocessing import image
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from tqdm import tqdm

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def label_assignment(img,label):
    return label

def training_data(label,data_dir):
    for img in tqdm(os.listdir(data_dir)):
        label = label_assignment(img,label)
        path = os.path.join(data_dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(imgsize,imgsize))
        X.append(np.array(img))
        Z.append(str(label))

X , Z, imgsize= [], [], 150
directory="../input/images/Images"
chihuahua_dir=directory+'/n02085620-Chihuahua'
japanese_spaniel_dir = directory+'/n02085782-Japanese_spaniel'
maltese_dir = directory+'/n02085936-Maltese_dog'
pekinese_dir = directory+'/n02086079-Pekinese'
shitzu_dir = directory+'/n02086240-Shih-Tzu'
blenheim_spaniel_dir = directory+'/n02086646-Blenheim_spaniel'
papillon_dir = directory+'/n02086910-papillon'
toy_terrier_dir = directory+'/n02087046-toy_terrier'
afghan_hound_dir = directory+'/n02088094-Afghan_hound'
basset_dir = directory+'/n02088238-basset'

training_data("chihuahua",chihuahua_dir)
training_data('japanese_spaniel',japanese_spaniel_dir)
training_data('maltese',maltese_dir)
training_data('pekinese',pekinese_dir)
training_data('shitzu',shitzu_dir)
training_data('blenheim_spaniel',blenheim_spaniel_dir)
training_data('papillon',papillon_dir)
training_data('toy_terrier',toy_terrier_dir)
training_data('afghan_hound',afghan_hound_dir)
training_data('basset',basset_dir)


Z = np.array(Z)
n = np.arange(X.shape[0])
np.random.shuffle(n)
X=X[n]
Z=Z[n]
X = X.astype(np.float32)
X=X/255
le= LabelEncoder()
Y = le.fit_transform(Z)
Y = to_categorical(Y,10)
X = np.array(X)
X=X/255
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=69)
print("Train X shape:{} and Train Y shape: {}".format(x_train.shape,y_train.shape))


base_model = ResNet50(include_top=False,
                  input_shape = (imgsize,imgsize,3),
                  weights = 'imagenet')

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=100, epochs=5, verbose=1)
preds = model.evaluate(x_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))