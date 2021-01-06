import tensorflow as tf
import keras as K
import os
import keras.layers as L
import scipy.io as sio
import argparse
import numpy as np
import time
from data_util import *
from MDN_func import _get_block,basic_block_spc,basic_block,_conv_bn_relu_spc,\
    _residual_block_spc,_bn_relu_spc,_conv_bn_relu,_residual_block,_bn_relu
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

from keras.layers import (
    Activation,
    Flatten)
from keras.layers.convolutional import (
    AveragePooling3D,
    Conv3D
)
from keras.layers.core import Reshape
os.environ['CUDA_VISIBLE_DEVICES']='1'

parser=argparse.ArgumentParser()
parser.add_argument('--NUM_EPOCH',
                    type=int,
                    default=100,
                    help='number of epoch')
parser.add_argument('--ksize',
                    type=int,
                    default=7,
                    help='window size')
parser.add_argument('--mode',
                    type=int,
                    default=1,
                    help='train or test mode')
parser.add_argument('--full_net_',
                    type=bool,
                    default=True,
                    help='train or not')
parser.add_argument('--right_net_',
                    type=bool,
                    default=True,
                    help='train or not')
parser.add_argument('--left_net_',
                    type=bool,
                    default=True,
                    help='train or not')
parser.add_argument('--up_net_',
                    type=bool,
                    default=True,
                    help='train or not')
parser.add_argument('--bottom_net_',
                    type=bool,
                    default=True,
                    help='train or not')

parser.add_argument('--upleft_net_',
                    type=bool,
                    default=True,
                    help='train or not')                    
parser.add_argument('--upright_net_',
                    type=bool,
                    default=True,
                    help='train or not')
parser.add_argument('--bottomleft_net_',
                    type=bool,
                    default=True,
                    help='train or not')
parser.add_argument('--bottomright_net_',
                    type=bool,
                    default=True,
                    help='train or not')
args=parser.parse_args()
window_size=args.ksize

model_name_data=os.path.join(weights_path+'data.h5')
model_name_XR=os.path.join(weights_path+'XR.h5') 
model_name_XL=os.path.join(weights_path+'XL.h5')
model_name_XU=os.path.join(weights_path+'XU.h5') 
model_name_XB=os.path.join(weights_path+'XB.h5') 
model_name_XUL=os.path.join(weights_path+'XUL.h5') 
model_name_XUR=os.path.join(weights_path+'XUR.h5')
model_name_XBL=os.path.join(weights_path+'XBL.h5') 
model_name_XBR=os.path.join(weights_path+'XBR.h5') 

new_model_name=os.path.join(weights_path+'all_mul_cnn_10.h5') 

if not os.path.exists('log/'):
    os.makedirs('log/')

class LossHistory(K.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('./Multi_Direction_CNN/indian_github/result/loss_acc.jpg')

def GW_net(input_spat, block_fn_spc, block_fn, repetitions1, repetitions2):
    
    block_fn_spc = _get_block(block_fn_spc)  #basic_block_spc
    block_fn = _get_block(block_fn)  #basic_block
    conv1_spc = _conv_bn_relu_spc(nb_filter=32, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7,
                                  subsample=(1, 1, 2))(input_spat)#input of the spectral 3DSERes block
    nb_filter = 32
    for i, r in enumerate(repetitions1):
        block_spc = _residual_block_spc(block_fn_spc, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(conv1_spc)
        nb_filter = nb_filter*2
    block_output_spc = _bn_relu_spc(block_spc) # output of the spectral 3DSERes block

    conv_spc_results = _conv_bn_relu_spc(nb_filter=32,kernel_dim1=1,kernel_dim2=1,
                                         kernel_dim3=block_output_spc._keras_shape[3])(block_output_spc)
    block_in = Reshape((conv_spc_results._keras_shape[1],conv_spc_results._keras_shape[2],
                         conv_spc_results._keras_shape[4],1))(conv_spc_results) #input of the spatial 3DSERes block

    conv2_spc = _conv_bn_relu(nb_filter=32, kernel_dim1=3, kernel_dim2=3, kernel_dim3=32,
                          subsample=(1, 1, 1))(block_in)
    nb_filter = 32
    for i, r in enumerate(repetitions2):
        block = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(conv2_spc)
        nb_filter = nb_filter * 2
    block_output = _bn_relu(block)
    pool2 = AveragePooling3D(pool_size=(block._keras_shape[1],
                                        block._keras_shape[2],
                                        block._keras_shape[3],),
                             strides=(1, 1, 1))(block_output) #output of the spatial 3DSERes block

    #Feature aggregation
    flatten1 = Flatten()(pool2)
    dense = L.Dense(units=NUM_CLASS, activation="softmax", kernel_initializer="he_normal")(flatten1)
    inputs = input_spat

    model = K.models.Model(inputs=inputs, outputs=dense)
    RMS=K.optimizers.Adam(lr=1e-4, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])
    
    return model

def concatenate_coff(x,coff):
	coff=tf.split(coff, len(x), axis=1)
	xc=x[0] * coff[0];
	for i in range(1,len(x)):
		xc=L.add([xc, x[i] * (coff[i])])
	return xc

def UNION_net():
    layers=0
    
    input_full=L.Input((window_size,window_size,NUM_CHN,1))
    full_net = GW_net(input_full,basic_block_spc, basic_block, [1],[1])
    full_net.layers.pop() #delete the last layer of full_net
    full_net.trainable=args.full_net_ #full_net_ determines that the parameters of the full new can be trained or not
    full_output=full_net.layers[-1].output
    layers=layers+1
    
    input_right=L.Input((window_size,window_size,NUM_CHN,1))
    right_net = GW_net(input_right,basic_block_spc, basic_block, [1],[1])
    right_net.layers.pop()
    right_net.trainable=args.right_net_
    right_output=right_net.layers[-1].output
    layers=layers+1
    
    input_left=L.Input((window_size,window_size,NUM_CHN,1))
    left_net=GW_net(input_left,basic_block_spc, basic_block, [1],[1])
    left_net.layers.pop()
    left_net.trainable=args.left_net_
    left_output=left_net.layers[-1].output
    layers=layers+1

    input_up=L.Input((window_size,window_size,NUM_CHN,1))
    up_net=GW_net(input_up,basic_block_spc, basic_block, [1],[1])
    up_net.layers.pop()
    up_net.trainable=args.up_net_
    up_output=up_net.layers[-1].output
    layers=layers+1

    input_bottom=L.Input((window_size,window_size,NUM_CHN,1))
    bottom_net=GW_net(input_bottom,basic_block_spc, basic_block, [1],[1])
#    bottom_net.load_weights(model_name_data)
    bottom_net.layers.pop()
    bottom_net.trainable=args.bottom_net_
    bottom_output=bottom_net.layers[-1].output
    layers=layers+1

    input_upleft=L.Input((window_size,window_size,NUM_CHN,1))
    upleft_net = GW_net(input_upleft,basic_block_spc, basic_block, [1],[1])
    upleft_net.layers.pop()
    upleft_net.trainable=args.upleft_net_
    upleft_output=upleft_net.layers[-1].output
    layers=layers+1

    input_upright=L.Input((window_size,window_size,NUM_CHN,1))
    upright_net=GW_net(input_upright,basic_block_spc, basic_block, [1],[1])
    upright_net.layers.pop()
    upright_net.trainable=args.upright_net_
    upright_output=upright_net.layers[-1].output
    layers=layers+1

    input_bottomleft=L.Input((window_size,window_size,NUM_CHN,1))
    bottomleft_net=GW_net(input_bottomleft,basic_block_spc, basic_block, [1],[1])
    bottomleft_net.layers.pop()
    bottomleft_net.trainable=args.bottomleft_net_
    bottomleft_output=bottomleft_net.layers[-1].output
    layers=layers+1

    input_bottomright=L.Input((window_size,window_size,NUM_CHN,1))
    bottomright_net=GW_net(input_bottomright,basic_block_spc, basic_block, [1],[1])
    bottomright_net.layers.pop()
    bottomright_net.trainable=args.bottomright_net_
    bottomright_output=bottomright_net.layers[-1].output
    layers=layers+1

    input_coff=L.Input([layers],name='coff')
    merge0=L.Lambda(lambda x:concatenate_coff(x[0:layers],x[layers]),name="concatenate")\
        ([full_output,right_output,left_output,up_output,bottom_output,upleft_output,
          upright_output,bottomleft_output,bottomright_output,input_coff])

    logits=L.Dense(NUM_CLASS,activation='softmax')(merge0)
    new_model=K.models.Model([input_full,input_right,input_left,input_up,input_bottom,input_upleft,input_upright,input_bottomleft,input_bottomright,input_coff],logits)

    RMS=K.optimizers.Adam(lr=1e-4, decay=1e-6)
    new_model.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return new_model

def train(model,model_name):
    # model_ckt = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
    tensorbd=TensorBoard(log_dir='./log',histogram_freq=0, write_graph=True, write_images=True)
    train_data_full=read_data(PATH,TRAIN_H,'train','data')
    train_data_XR=read_data(PATH,TRAIN_H,'train','XR')
    train_data_XL=read_data(PATH,TRAIN_H,'train','XL')
    train_data_XU=read_data(PATH,TRAIN_H,'train','XU')
    train_data_XB=read_data(PATH,TRAIN_H,'train','XB')

    train_data_XUL=read_data(PATH,TRAIN_H,'train','XUL')
    train_data_XUR=read_data(PATH,TRAIN_H,'train','XUR')
    train_data_XBL=read_data(PATH,TRAIN_H,'train','XBL')
    train_data_XBR=read_data(PATH,TRAIN_H,'train','XBR')

    train_labels=K.utils.np_utils.to_categorical(train_data_XR.labels,NUM_CLASS)
    train_labels = np.squeeze(train_labels)
    
    CTRAIN=os.path.join('train_data_H.mat')
    train_data=sio.loadmat(os.path.join(TEST_PATH,CTRAIN))
    trian_data_att=train_data['train_att'].T

    print('train hsi data shape:{}'.format(train_data_full.hsi.shape))
    print('train XR data shape:{}'.format(train_data_XR.hsi.shape))
    print('train XL data shape:{}'.format(train_data_XL.hsi.shape))
    print('train XU data shape:{}'.format(train_data_XU.hsi.shape))
    print('train XB data shape:{}'.format(train_data_XB.hsi.shape))
    print('train XUL data shape:{}'.format(train_data_XUL.hsi.shape))
    print('train XUR data shape:{}'.format(train_data_XUR.hsi.shape))
    print('train XBL data shape:{}'.format(train_data_XBL.hsi.shape))
    print('train XBR data shape:{}'.format(train_data_XBR.hsi.shape))
    
    print('{} train sample'.format(train_data_XR.hsi.shape[0]))
    #estabish an instance of history
    history = LossHistory()
    model.fit([np.expand_dims(train_data_full.hsi,axis=4),np.expand_dims(train_data_XR.hsi,axis=4),
               np.expand_dims(train_data_XL.hsi,axis=4),np.expand_dims(train_data_XB.hsi,axis=4),
               np.expand_dims(train_data_XU.hsi,axis=4),np.expand_dims(train_data_XUL.hsi,axis=4),
               np.expand_dims(train_data_XUR.hsi,axis=4),np.expand_dims(train_data_XBL.hsi,axis=4),
               np.expand_dims(train_data_XBR.hsi,axis=4),trian_data_att],train_labels, 
             batch_size=BATCH_SIZE,
             epochs=args.NUM_EPOCH,
             verbose=1,
             shuffle=True,
             # validation_split=0.1,
             # callbacks=[model_ckt,tensorbd,history])
             callbacks=[tensorbd,history])
    history.loss_plot('epoch')
    model.save(os.path.join(model_name+'_')) #save the model of the last epoch

def test(model_name,hsi_data,XR_data,XL_data,XU_data,XB_data,XUL_data,XUR_data,XBL_data,XBR_data,test_att):
    model = UNION_net()
    model.load_weights(model_name)
    pred=model.predict([hsi_data,XR_data,XL_data,XU_data,XB_data,XUL_data,XUR_data,XBL_data,XBR_data,test_att],batch_size=BATCH_SIZE)
    return pred


def main(mode=1,show=False):
#    if args.mode==0:
        start_time=time.time()
        model=UNION_net()
        train(model,new_model_name)
        train_duration=time.time()-start_time
        print('TrainingT='+str(train_duration)+'\n')
#    else:
        test_samples_full=[]
        test_samples_XR=[]
        test_samples_XL=[]
        test_samples_XU=[]
        test_samples_XB=[]
        test_samples_XUL=[]
        test_samples_XUR=[]
        test_samples_XBL=[]
        test_samples_XBR=[]
        test_samples_att=[]
        
        idxx=np.zeros(shape=(1,),dtype=np.int64)
        idyy=np.zeros(shape=(1,),dtype=np.int64)
        labels=np.zeros(shape=(1,),dtype=np.int64)
        for iclass in range(1,NUM_CLASS+1):
            CTEST=os.path.join('test_data_H'+str(iclass)+'.mat')
            test_data=sio.loadmat(os.path.join(TEST_PATH,CTEST))
            test_data_full=test_data['hsi']
            test_data_XR=test_data['XR']
            test_data_XL=test_data['XL']
            test_data_XU=test_data['XU']
            test_data_XB=test_data['XB']
            test_data_XUL=test_data['XUL']
            test_data_XUR=test_data['XUR']
            test_data_XBL=test_data['XBL']
            test_data_XBR=test_data['XBR']
            test_data_att=test_data['test_att'].T
            
            test_samples_full.extend(test_data_full)
            test_samples_XR.extend(test_data_XR)
            test_samples_XL.extend(test_data_XL)
            test_samples_XU.extend(test_data_XU)
            test_samples_XB.extend(test_data_XB)
            test_samples_XUL.extend(test_data_XUL)
            test_samples_XUR.extend(test_data_XUR)
            test_samples_XBL.extend(test_data_XBL)
            test_samples_XBR.extend(test_data_XBR)
            test_samples_att.extend(test_data_att)

            test_labl=test_data['label']
            test_labl=np.reshape(test_labl.T,(test_labl.shape[1]))
            idx=test_data['idx']
            idy=test_data['idy']
            idx=np.reshape(idx.T,(idx.shape[1]))
            idy=np.reshape(idy.T,(idy.shape[1])) 

            idxx=np.concatenate((idxx,idx),axis=0)
            idyy=np.concatenate((idyy,idy),axis=0)
            labels=np.concatenate((labels,test_labl),axis=0)
        test_samples_full=np.array(test_samples_full,dtype=np.float32)
        test_samples_XR=np.array(test_samples_XR,dtype=np.float32)
        test_samples_XL=np.array(test_samples_XL,dtype=np.float32)
        test_samples_XU=np.array(test_samples_XU,dtype=np.float32)
        test_samples_XB=np.array(test_samples_XB,dtype=np.float32)
        test_samples_XUL=np.array(test_samples_XUL,dtype=np.float32)
        test_samples_XUR=np.array(test_samples_XUR,dtype=np.float32)
        test_samples_XBL=np.array(test_samples_XBL,dtype=np.float32)
        test_samples_XBR=np.array(test_samples_XBR,dtype=np.float32)
        test_samples_att=np.array(test_samples_att,dtype=np.float32)

        #test process
        start_time=time.time()
        # prediction=np.array(test(new_model_name,np.expand_dims(test_samples_full,axis=4),
        #            np.expand_dims(test_samples_XR,axis=4),np.expand_dims(test_samples_XL,axis=4),
        #            np.expand_dims(test_samples_XU,axis=4),np.expand_dims(test_samples_XB,axis=4),
        #            np.expand_dims(test_samples_XUL,axis=4),np.expand_dims(test_samples_XUR,axis=4),
        #            np.expand_dims(test_samples_XBL,axis=4),np.expand_dims(test_samples_XBR,axis=4),test_samples_att),dtype=np.float32)
        prediction=model.predict([np.expand_dims(test_samples_full,axis=4),
                   np.expand_dims(test_samples_XR,axis=4),np.expand_dims(test_samples_XL,axis=4),
                   np.expand_dims(test_samples_XU,axis=4),np.expand_dims(test_samples_XB,axis=4),
                   np.expand_dims(test_samples_XUL,axis=4),np.expand_dims(test_samples_XUR,axis=4),
                   np.expand_dims(test_samples_XBL,axis=4),np.expand_dims(test_samples_XBR,axis=4),
                   test_samples_att],batch_size=BATCH_SIZE)
        prediction=np.array(prediction,dtype=np.float32)
        test_duration=time.time()-start_time
        print('TestT='+str(test_duration)+'\n')
        idxx=np.delete(idxx,0,axis=0)
        idyy=np.delete(idyy,0,axis=0)
        labels=np.delete(labels,0,axis=0)

        print(prediction.shape,labels.shape)
        print('OA: {}%'.format(eval(prediction,labels)))
        
        # generate confusion_matrix
        prediction=np.asarray(prediction)
        pred=np.argmax(prediction,axis=1)
        pred=np.asarray(pred,dtype=np.int8)
        print(confusion_matrix(labels,pred))

        # generate accuracy
        f = open(os.path.join('Wunionprediction.txt'), 'w')
        n = prediction.shape[0]
        
        for i in range(n):
            pre_label = np.argmax(prediction[i],0)
            f.write(str(pre_label)+'\n')
        f.close()

        print(classification_report(labels, pred))
        
        matrix = confusion_matrix(labels, pred)
        OA=np.sum(np.trace(matrix)) / float(labels.shape[0])
#    print('OA = '+str(OA)+'\n')
#    average accuracy
#    print('ua =')
        ua = np.diag(matrix)/np.sum(matrix, axis=0)
#    precision
#    print('precision =')
        precision = np.diag(matrix)/np.sum(matrix, axis=1)  
#    Kappa
        matrix = np.mat(matrix);
        Po = OA;
        xsum = np.sum(matrix, axis=1);
        ysum = np.sum(matrix, axis=0);
        Pe = float(ysum*xsum)/(np.sum(matrix)**2);
        Kappa = float((Po-Pe)/(1-Pe));
        
        for i in ua:
             print(i)
        print(str(np.sum(ua)/matrix.shape[0]))
        print(str(OA))
        print(str(Kappa));
        print()
        for i in precision:
             print(i)  
        print(str(np.sum(precision)/matrix.shape[0]))
    # generate classification map        
        gt_uPavia = sio.loadmat('./Multi_Direction_CNN/indian_github/data/GRSM2018indian/Indian_gt.mat')
        gt= gt_uPavia['image_gt']
        resultpath = './Multi_Direction_CNN/indian_github/result/'
        pred_map=generate_map(prediction,idxx,idyy,gt)
        sio.savemat(resultpath+'pred_map.mat', {'pred_map': pred_map})
        pred_map = pred_map.reshape((pred_map.shape[0]*pred_map.shape[1],1))
        img = DrawResult(pred_map,2)
        plt.imsave(resultpath+'Wunion'+'_'+repr(int(OA*10000))+'.png',img)
        plt.show()

if __name__ == '__main__':

    main()
