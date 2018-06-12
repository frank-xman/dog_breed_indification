import numpy as np
import pandas as pd
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.applications import InceptionV3
from sklearn.model_selection import train_test_split
from os.path import exists
import h5py
from keras.layers import GlobalAveragePooling2D,Dense,Dropout
from keras.models import Model,load_model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad,SGD
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras import callbacks

'''--------------------------model-----------------------------------'''

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(1024,activation='relu')(
    Dropout(0.5)(x)
)
predictions = Dense(100,activation='softmax')(x)
def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                 )
    
setup_to_transfer_learning(model,base_model)
model = Model(inputs=base_model.input,outputs=predictions)
batch_size=64

def lr_decay(epoch):
    lrs=[]
    lr=0.001
    for i in range(30):
        if i%10==0:
            lr*=0.1
        lrs.append(lr)
    return lrs[epoch]

'''---------------------------------read data and label ------------------------------------'''
train_datagen=ImageDataGenerator(    
    rescale=1./255,
    shear_range=0.2,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True)
train_generator = train_datagen.flow_from_directory(directory='train/',
                                  target_size=(299,299),#Inception V3 image_size
                                  batch_size=batch_size)
test_datagen=ImageDataGenerator(    
    rescale=1./ 255)
test_generator = test_datagen.flow_from_directory(directory='test/',
                                  target_size=(299,299),
                                  batch_size=batch_size)
'''----------------------------------------------------already-----'''

    
'''----------------------fine_tune function-----------------'''
def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 249 #maybe some other num
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    
'''----------------------freeze_train--------------------------------'''

model.fit_generator(generator=train_generator,
                        steps_per_epoch=100, 
                        epochs=10,
                        validation_data=test_generator,
                        validation_steps=1)
                        #callbacks=[early_stopping, auto_lr])
model.save('dog_inception_v3.h5')
'''-------------------------------fine_turn----------------------'''
model=load_model('dog_inception_v3.h5')
setup_to_fine_tune(model)
my_lr = LearningRateScheduler(lr_decay)
save_model = ModelCheckpoint('inception_v3_{epoch:02d}-{val_acc:.2f}.h5')
model.fit_generator(generator=train_generator,
                        steps_per_epoch=16210/batch_size+1, 
                        epochs=30,
                        validation_data=test_generator,
                        validation_steps=1800/batch_size+1,
                    callbacks=[my_lr, save_model])
model.save('dog_inception_v3.h5')

