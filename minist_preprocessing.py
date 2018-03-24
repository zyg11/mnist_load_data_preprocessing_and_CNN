import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，
#如果是将彩色图作为输入,则将1替换为3，
# 并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
    # num=4200
    data=np.empty((42000,1,28,28),dtype="float32")
    label=np.empty((42000,),dtype='uint8')
    imgs=os.listdir("E:/keras_data/mnist1")
    num=len(imgs)
    for i in range(num):
        img=Image.open("E:/keras_data/mnist1/"+imgs[i])
        arr=np.asarray(img,dtype="float32")
        data[i,:,:,:]=arr
        label[i]=int(imgs[i].split('.')[0])
    #由th格式转成tf格式
    data = data.reshape(42000, 28, 28, 1)
    #归一化
    data/=np.max(data)
    data-=np.mean(data)
    return data,label