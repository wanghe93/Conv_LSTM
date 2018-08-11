# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 21:38:29 2018

@author: Administrator
"""

import cv2
import os
import numpy as np
import itertools
import time
import sys
import math
from keras.layers import Conv2D, ConvLSTM2D, Lambda, Reshape
from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



def Data_generator(file_path='.', scope=(1,1), img_size=(501,501), batch_size = 1):
    
    def key(path):
        return int(os.path.splitext(path)[0][-3:])
    
    pre, post = scope
    height,width = img_size
    
    Data = np.zeros((batch_size, 60, height, width, 1))
    Target = np.zeros((batch_size, 60, height, width, 1))
    
    listdirs = sorted(os.listdir(file_path))[pre:post]
    
    cs = itertools.cycle(listdirs) # 注意字符串也是序列的一种
    while 1:
        dirs = []
        for i in range(batch_size):
            dirs.append(next(cs))
        for i,dire in enumerate(dirs):
            #print('-'*9 + dire + '-'*9)
            for j,img_path in enumerate(sorted(os.listdir(os.path.join(file_path, dire)), key = key)):
                #print(img_path)
                Img = cv2.imread(os.path.join(file_path, dire, img_path), 0)
                if(Img.shape[0] != height or Img.shape[1] != width):
                    #print('resized')
                    Img = cv2.resize(Img,(width, height), interpolation = cv2.INTER_CUBIC)
                Img = Img/255
                if j >= 0 and j <= 59:                
                   Data[i, j, :, :, 0] = Img
                if j >= 1 and j <= 60:
                   Target[i, j - 1, :, :, 0] = Img
        yield (dirs, Data, Target)
 
       
def Get_Model(kernel = 21, height = 501, width = 501, batch_size = 1, times = 1):
    model = Sequential()
    model.add(ConvLSTM2D(filters=3, batch_input_shape = (batch_size, times, height, width, 1), kernel_size=(kernel, kernel), padding='same', 
                    return_sequences=True, stateful=True))
    model.add(ConvLSTM2D(filters=6, batch_input_shape = (batch_size, times, height, width, 1), kernel_size=(kernel, kernel), padding='same', 
                    return_sequences=True, stateful=True))
    model.add(ConvLSTM2D(filters=6, batch_input_shape = (batch_size, times, height, width, 1), kernel_size=(kernel, kernel), padding='same', 
                    return_sequences=True, stateful=True))
    model.add(TimeDistributed(Conv2D(filters = 1, kernel_size = (9,9), padding = 'same')))
    optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae', optimizer=optimizer)
    return model


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, "." * (40 - rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()
    
kernel = 11
height = 501
width = 501


if os.path.exists('my_model.h5'):
    print('Loading existed model...')
    model = load_model('my_model.h5')
else:
    print('Creating a new model...')
    model = Get_Model(kernel = kernel, height = height, width = width)

optimizer = RMSprop(lr=0.000001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mae', optimizer=optimizer)  
    
train_gen = Data_generator(file_path=r'F:\wanghe\qixiang\SRAD2018_TRAIN_001', scope=(115,135), img_size=(height,width), batch_size = 1)
test_gen = Data_generator(file_path=r'F:\wanghe\qixiang\SRAD2018_TRAIN_001', scope=(303,305), img_size=(height,width), batch_size = 1)
epochs = 20     #训练样例个数
test_nums = 2   #测试样例个数

#for i in range(epochs):
#    print('Epoch', i + 1, '/', epochs)
#    data, target = next(data_generator)
#    for j in range(60):
#        x_batch = data[:,j,:,:,:]
#        y_batch = target[:,j,:,:,:]
#        x_batch = x_batch[:,np.newaxis,:,:,:]
#        y_batch = y_batch[:,np.newaxis,:,:,:]
#        loss = model.train_on_batch(x_batch, y_batch)
#        
#        view_bar('train_loss:'+str(loss), j + 1, 60)
#    model.reset_states()
#    print()
    
t1 = time.time()   
for epoch in range(epochs):
    print('Epoch', epoch + 1, '/', epochs)
    
    dir_name, data, target = next(train_gen)
    train_loss = []
    for j in range(60):
        x_batch = data[:,j,:,:,:]
        y_batch = target[:,j,:,:,:]
        x_batch = x_batch[:,np.newaxis,:,:,:]
        y_batch = y_batch[:,np.newaxis,:,:,:]
        loss = model.train_on_batch(x_batch, y_batch)
        train_loss.append(loss)
        view_bar(dir_name[0], j + 1, 60)
    model.reset_states()
    print()
    print('Train_loss:%s' % (str(np.mean(train_loss))))
          
          
    if (epoch+1)%3 == 0:
        test_loss = []
        for i in range(test_nums):
            _, test_data, test_target = next(test_gen)
            for j in range(60):
                loss = model.test_on_batch(test_data[0, j, :, :, :][np.newaxis, np.newaxis, :, :, :], test_target[0, j, :, :, :][np.newaxis, np.newaxis, :, :, :])    
                test_loss.append(loss)
            model.reset_states()
        model.reset_states()
        print()
        print('*********************************************')
        print('Test_loss:%s' % (str(np.mean(test_loss))))
        print('*********************************************')
        print()
    
       
model.save('my_model.h5') 
t2 = time.time()
print(t2-t1)