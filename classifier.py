# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:02:48 2020

@author: K B PRANAV
"""

import numpy as np # linear algebra
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import time
import cv2
import tkinter as tk

class Classifier:
    def __init__(self):
        self.init_gui()
            
    def Load_Model_Arch(self):
        # Initialising the CNN
        self.classifier = Sequential()
        
        # Step 1 - Convolution
        self.classifier.add(Conv2D(32, (2, 2), input_shape = (160, 80, 3), activation = 'relu'))
        
        # Step 2 - Pooling
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a second convolutional layer
        self.classifier.add(Conv2D(64, (2, 2), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a third convolutional layer
        self.classifier.add(Conv2D(128, (2, 2), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a fourth convolutional layer
        self.classifier.add(Conv2D(128, (2, 2), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Step 3 - Flattening
        self.classifier.add(Flatten())
        
        # Step 4 - Full connection
        self.classifier.add(Dense(units = 64, activation = 'relu'))
        self.classifier.add(Dense(units = 1, activation = 'sigmoid'))
        
        # Compiling the CNN
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        print("\n[INFO]Architecture Loaded...")
        #return self.classifier
        
    
    def train_model(self):
        batch_size = 128
        
        
        train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2,horizontal_flip = True)
                                         
        
        test_datagen = ImageDataGenerator(rescale = 1./255)
        
        training_set = train_datagen.flow_from_directory('database/train',target_size = (160, 80), batch_size = 32,class_mode = 'binary')
       
        test_set = test_datagen.flow_from_directory('database/test',target_size = (160, 80), batch_size = 32,class_mode = 'binary')
        
        filepath = "Trained_Weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        start=time.time()
        
        self.classifier.fit_generator(training_set,
                                 steps_per_epoch = 5000//batch_size,
                                 epochs = 10,
                                 validation_data = test_set,
                                 validation_steps = 1000//batch_size,
                                 callbacks = [checkpoint])
        end=time.time()
        self.classifier.save("Trained_model")
        #print(history.history.keys())
        print("\n[INFO]Trained Successfully...")
        print('Time to train:',(end-start)," seconds.")
        #return self.classifier
    def load_trained_model(self):
        #self.classifier.load_weights('best_model_Evaluation_new.hdf5')
        self.classifier.load_weights('Trained_model')
        print("\n[INFO]Model Loaded Successfully...")
        #return self.classifier
        
        
    def detect(self):
        print("\n[INFO]Camera Starting....Detecting Pedestrians now...")
        print("\n[INFO]Press 'q' to exit")
        cap1 = cv2.VideoCapture(0)
        time.sleep(2)
        while(True):
            #name+=1
            ret1,frame1 = cap1.read()
            
            frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            img=np.array(frame)
            i=0
            l=[]
            
            while(i<8):
                crop_img = img[ :,80*i:160+80*i]
                crop=np.array(crop_img)
                crop=cv2.resize(crop, (80, 160))
                test_image = np.expand_dims(crop, axis = 0)
                result = self.classifier.predict(test_image)
                if(result[0][0]>0.5):
                    l.append(i)
                
                i+=1
            for i in l:
                cv2.rectangle(frame,(80*i,10),(80*i+160,460),(0,210,0),3)
        
         
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Pedestrian Detection",frame)
            
            key = cv2.waitKey(1)&0xFF
            if(key==ord('q')):
                break
            
        cap1.release()
        cv2.destroyAllWindows()
        
        
    
    def init_gui(self):
       
        window= tk.Tk()
        
        self.btn_arch = tk.Button(window, text="Load Architecture", width=50, command=self.Load_Model_Arch())
        self.btn_arch.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_train = tk.Button(window ,text="Train Model", width=50, command=lambda: self.train_model())
        self.btn_train.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_load = tk.Button(window,text="Load Trained Model", width=50, command=lambda: self.load_trained_model())
        self.btn_load.pack(anchor=tk.CENTER, expand=True)
        
        
        self.btn_detect = tk.Button(window,text="Detect Pedestrian", width=50, command=lambda: self.detect())
        self.btn_detect.pack(anchor=tk.CENTER, expand=True)
        
        window.mainloop()
        
        
Classifier()