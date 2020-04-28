import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
#from keras.utils.np_utils import to_categorical

from keras.applications import VGG16
from keras import models 
from keras import layers

from keras.preprocessing.image import ImageDataGenerator

images_dir = '../input/images/Images'

breed_list= os.listdir(images_dir)
breed_list = breed_list[:50]
def load_img_nd_labels(breed_list):
    img_lst = []
    lables= []
    for index,category in enumerate(breed_list):
        for image_name in os.listdir(images_dir+"/"+category):
            img = cv2.imread(images_dir+"/"+category+"/"+image_name, 1) 
            img = cv2.resize(img,(150,150))
            
            img_lst.append(np.array(img))
            
            lables.append(str(category))
    return img_lst, lables

images, lables = load_img_nd_labels(breed_list)
images = np.array(images)
label_encoder = LabelEncoder()
lables = label_encoder.fit_transform(lables)
x_train,x_test,y_train,y_test = train_test_split(images,lables,test_size=0.3,random_state=69)

conv_base = VGG16(weights='imagenet',                
                  include_top=False,              
                  input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base) 
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(10,activation='softmax')) 

datagen = ImageDataGenerator(rescale = 1./255,       
        rotation_range=10,  
        zoom_range = 0.2, 
        width_shift_range=0.2,  
        height_shift_range=0.2, 
        horizontal_flip=True,  
        vertical_flip=False) 

train_aug = datagen.flow(x_train, y_train, batch_size = 25)
test_aug = datagen.flow(x_test, y_test, batch_size = 25)

model.compile(loss="sparse_categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit_generator(
    train_aug,
    validation_data  = test_aug,
    epochs = 50,
    validation_steps = 1000,
    steps_per_epoch  = 1000
)