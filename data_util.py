#coding: utf-8
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

BATCH_SIZE=16
TRAIN_H='train_data_H.mat'

weights_path=os.path.join('./Multi_Direction_CNN/indian/weights/new_net/')
if not os.path.exists(weights_path):
    os.mkdir(weights_path)
PATH='./Multi_Direction_CNN/indian/file/GRSM2018indian/7/'
TEST_PATH='./Multi_Direction_CNN/indian/file/GRSM2018indian/7/'
NUM_CLASS=16
HEIGHT=145
WIDTH= 145
NUM_CHN= 200

class DataSet(object):
    def __init__(self,hsi,labels):
        self._hsi=hsi
        self._labels=labels
    @property
    def hsi(self):
        return self._hsi
    @property
    def labels(self):
        return self._labels


def read_data(path,filename_H,data_style,key):
    if data_style=='train':
        train_data=sio.loadmat(os.path.join(path,filename_H))
        hsi=np.array(train_data[key])
        train_labl=np.array(train_data['label'])
        return DataSet(hsi,train_labl)
    else:
        test_data=sio.loadmat(os.path.join(path,filename_H))
        hsi=test_data[key]
        test_labl=test_data['label']
        test_labl=np.reshape(test_labl.T,(test_labl.shape[1]))
        idx=test_data['idx']
        idy=test_data['idy']
        idx=np.reshape(idx.T,(idx.shape[1]))
        idy=np.reshape(idy.T,(idy.shape[1]))
        return DataSet(hsi,test_labl),idx,idy


def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)
 

def eval(predication,labels):
    """
    evaluate test score
    """
    num=labels.shape[0]
    count=0
    for i in range(num):
        if(np.argmax(predication[i])==labels[i]):
            count+=1
    return 100.0*count/num

def generate_map(predication,idx,idy,gt):
    maps=gt
    for i in range(len(idx)):
        maps[idx[i],idy[i]]=np.argmax(predication[i])+1
    return maps
def DrawResult(labels,imageID):
    #ID=1:Pavia University
    #ID=2:Indian Pines   
    #ID=6:KSC
    #ID=7:Houston
    global palette
    global row
    global col
    num_class = int(labels.max())
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    elif imageID == 2:
        row = 145
        col = 145
        palette = np.array([[255,0,0],
                            [0,255,0],
                            [0,0,255],
                            [255,255,0],
                            [0,255,255],
                            [255,0,255],
                            [176,48,96],
                            [46,139,87],
                            [160,32,240],
                            [255,127,80],
                            [127,255,212],
                            [218,112,214],
                            [160,82,45],
                            [127,255,0],
                            [216,191,216],
                            [238,0,0]])
        palette = palette*1.0/255  

    elif imageID == 6:
        row = 512
        col = 614
        palette = np.array([[94, 203, 55],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])
        palette = palette*1.0/255

    elif imageID == 7:
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
        palette = palette*1.0/255
    
    
    X_result = np.zeros((labels.shape[0],3))
    for i in range(1,num_class+1):
        X_result[np.where(labels==i),0] = palette[i-1,0]
        X_result[np.where(labels==i),1] = palette[i-1,1]
        X_result[np.where(labels==i),2] = palette[i-1,2]
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result