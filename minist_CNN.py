from __future__ import absolute_import
from __future__ import print_function
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.optimizers import SGD,Adagrad,Adadelta
from keras.utils import np_utils,generic_utils
from keras.preprocessing.image import ImageDataGenerator#图片预处理
from six.moves import range#doubt，这是啥，有些模块没用过
from keras.callbacks import EarlyStopping
from keras_demo import minist_loaddata
#加载Keras自带数据
# (x_train,y_train),(x_test,y_test)=mnist.load_data()
# number=10000
# x_train = x_train[0:number]
# y_train = y_train[0:number]
# x_train=x_train.reshape(number,28,28,1)
# x_test=x_test.reshape(x_test.shape[0],28,28,1)
# x_train=x_train.astype('float32')
# x_test=x_test.astype('float32')
# x_train = x_train / 255
# x_test = x_test / 255
# y_train=np_utils.to_categorical(y_train,10)
# y_test=np_utils.to_categorical(y_test,10)
#加载data,label形式
data,label=minist_loaddata.load_data()
# data=data.reshape(42000,28,28,1)
# label=label.reshape(42000,28,28,1)
# print(data.shape[0],'samples',data.shape[1],data.shape[2],data.shape[3])
# print(label)
#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，
# 直接调用keras提供的这个函数
label=np_utils.to_categorical(label,10)
train_data = data[:40000]
train_labels = label[:40000]
validation_labels = label[40000:]
validation_data = data[40000:]
#加载train,test形式,ImageGenrator

#建立CNN模型
model=Sequential()
# model.add(ZeroPadding2D((1,1)))#input_shape=(28,28,1)
model.add(Conv2D(4,(5,5),input_shape=(28,28,1),activation='relu',name='conv1_1'))
# model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8,(3,3),activation='relu',name='conv2_1'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(16,(3,3),activation='relu',name='conv3_1'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

#开始train
# sgd=SGD(lr=0.05,momentum=0.9,decay=1e-6,nesterov=False)#这个优化器性价比好，但是有时候会出问题
model.compile(
            # optimizer='adam',
            optimizer='rmsprop',
             loss='categorical_crossentropy',
            metrics=['accuracy'],
              )
#调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
#数据经过随机打乱shuffle=True。verbose=1，
# 训练过程中输出的信息，0、1、2三种方式都可以，无关紧要。
# show_accuracy=True，训练时每一个epoch都输出accuracy。
#validation_split=0.2，将20%的数据作为验证集。
# early_stopping = EarlyStopping(monitor='val_loss', patience=1)
#保存达到最好的val-accuracy时的模型
history_fit=model.fit(train_data,
                      train_labels,
                      nb_epoch=20,
                      batch_size=100,
                      validation_data=(validation_data,validation_labels),
                      # callbacks=[early_stopping]
                      )
model.save('E:/keras_data/mnist/minist_CNN_model.h5')
def plot_training(history):
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))
    plt.plot(epochs,acc,'b')
    plt.plot(epochs,val_acc,'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs,loss,'b')
    plt.plot(epochs,val_loss,'r')
    plt.title('Training and validation loss')
    plt.show()
#训练的acc_loss图
plot_training(history_fit)





