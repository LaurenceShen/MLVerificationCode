
# coding: utf-8

# In[8]:


import numpy as np
import cv2
import os
import random
files = os.listdir("letters")
#print (files)

x_train = []
y_train = []
x_test = []
y_test = []

k = 0
a = 0
l = 0
b = 0

for i in files:
    k += 1
for f in files:
    if a > k:
        break
    if f == '.DS_Store':
        continue
    h = os.listdir('letters/' + f)
    for i in h:
        l += 1
    #print(l) 
    for image in h:
        
        if b > l:
            break
        b += 1
        if image == '.DS_Store':
        #print('letters/'+f+'/'+image)
            continue
        '''
        if 'test' in image:
            r = random.randint(1, 20)
            
            if 2 <= r <= 20:
                continue
            else:
                y_test.append(ord(f))
                tmp = cv2.imread('letters/'+f+'/'+image, cv2.IMREAD_GRAYSCALE)
                tmp = cv2.resize(tmp, (128, 128), interpolation=cv2.INTER_CUBIC)
                x_test.append(tmp.astype('float32')/255)

        else:
            r = random.randint(1, 20)
            
            if 2 <= r <= 20:
                continue
            else:
                y_train.append(ord(f))
                tmp = cv2.imread('letters/'+f+'/'+image, cv2.IMREAD_GRAYSCALE)
                tmp = cv2.resize(tmp, (128, 128), interpolation=cv2.INTER_CUBIC)
                x_train.append(tmp.astype('float32')/255)
        '''
        if b > l:
            break
        b += 1
        if image == '.DS_Store':
        #print('letters/'+f+'/'+image)
            continue
        r = random.randint(1, 20)
        if 3<= r <= 20:
            continue
        elif r == 1:
            y_train.append(ord(f))
            tmp = cv2.imread('letters/'+f+'/'+image, cv2.IMREAD_GRAYSCALE)
            tmp = cv2.resize(tmp, (128, 128), interpolation=cv2.INTER_CUBIC)
            x_train.append(tmp.astype('float32')/255)
        else:
            y_test.append(ord(f))
            tmp = cv2.imread('letters/'+f+'/'+image, cv2.IMREAD_GRAYSCALE)
            tmp = cv2.resize(tmp, (128, 128), interpolation=cv2.INTER_CUBIC)
            x_test.append(tmp.astype('float32')/255)
        
    a += 1


x_train = np.array(x_train)
x_train = x_train.reshape(-1, 128, 128, 1)
#print('x: ', x_train)
#print('y: ',y_train)

x_test = np.array(x_test)
x_test = x_test.reshape(-1, 128, 128, 1)
#print('x: ', x_test)
#print('y: ',y_test)




import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#import matplotlib.pyplot as plt
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')




batch_size = 32
#num_classes = k-1 
num_classes = 100
epochs = 20




y_train = keras.utils.to_categorical((y_train), num_classes=num_classes+1, dtype='int8')
print (y_train)
y_test = keras.utils.to_categorical((y_test), num_classes=num_classes+1, dtype='int8')





model = Sequential()
model.add(Conv2D(32, (4, 4), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
#model.add(Conv2D(32, (4, 4)))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (4, 4), padding='same'))
model.add(Activation('relu'))
#model.add(Conv2D(64, (4, 4)))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes+1))
model.add(Activation('softmax'))

opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)




model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


'''
files = os.listdir("jpg")
x_test = []
y_test = []

for f in files:
    y_test.append(int(f.split('-')[1].split('.')[0]))
    tmp = cv2.imread('jpg1/'+f, cv2.IMREAD_GRAYSCALE)
    tmp = cv2.resize(tmp, (128, 128), interpolation=cv2.INTER_CUBIC)
    x_test.append(tmp.astype('float32')/255)
x_test = np.array(x_test)
x_test = x_test.reshape(-1, 128, 128, 1)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes+1, dtype='int8')
'''


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
model.save('model.h5')

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
ans = model.predict(x_test)

'''
for i in range(len(ans)):
    print(ans[i], y_test[i])
'''

