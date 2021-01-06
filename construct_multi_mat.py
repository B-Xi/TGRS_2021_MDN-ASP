# coding:utf-8
import argparse
import numpy as np
import os
import scipy.io as sio
import math

parser=argparse.ArgumentParser()
parser.add_argument('--train_label_name',
                    type=str,
                    default='mask_train_200_10',
                    help='random sample sets')
parser.add_argument('--ksize',
                    type=int,
                    default=7,
                    help='window size')
args=parser.parse_args()
r=args.ksize//2

NUM_train='GRSM2018indian'

NUM_CLASS=16
window_size=str(args.ksize)
mdata_name='indian.mat'
data_type='mat'
dataset='indian'
alpha=1

PATH = os.path.join('./Multi_Direction_CNN/'+ dataset+'/data/'+NUM_train+'/')
save_path = os.path.join('./Multi_Direction_CNN/'+ dataset+'/file/'+NUM_train+'/'+window_size+'/')
if os.path.exists(save_path):
    print("Saving Path is no problem.")
else:
    print("Saving Path has problem, ready for constructing a new one")
    os.makedirs(os.path.join('./Multi_Direction_CNN/'+ dataset+'/file/'+NUM_train+'/'+window_size+'/'))

def image_pad(data,r):
    if len(data.shape)==3:
        data_new=np.lib.pad(data,((r,r),(r,r),(0,0)),'symmetric')
        return data_new
    if len(data.shape)==2:
        data_new=np.lib.pad(data,r,'constant',constant_values=0)
        return data_new

def featureNormalize(X,type):
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
    elif type==3:
        maxX = np.max(X,0)
        X_norm = 2*X-maxX
        X_norm = X_norm/maxX
    return X_norm    

def SADis(patch,central_spectral,alpha):
    patch_aver = patch.mean(1).mean(0)
    normab = np.sqrt(np.dot(patch_aver,patch_aver.T))*np.sqrt(np.dot(central_spectral,central_spectral.T))
    errRadians = math.acos((np.dot(patch_aver,central_spectral)/normab))
    errRadians = -alpha*errRadians
    errRadians=np.exp(errRadians)
    return errRadians

def Eucdis(patch,central_spectral,alpha):
    patch_aver = patch.mean(1).mean(0)
    errRadians = np.sum(np.square(patch_aver-central_spectral))
    errRadians = -alpha*errRadians
    errRadians=np.exp(errRadians)
    return errRadians

def softmax(x):
    return x/x.sum(axis=0)

def construct_spatial_patch(mdata,mlabel,r,patch_type,alpha,seed=0):
    patch=[]
    patch_right=[]
    patch_left=[]
    patch_up=[]
    patch_bottom=[]
    
    patch_upleft=[]
    patch_upright=[]
    patch_bottomleft=[]
    patch_bottomright=[]
    label=[]

    if patch_type=='train':
        count_idx = 0
        num_class=np.max(mlabel)
        train_att = np.zeros((9,(np.where(mlabel!=0)[0].shape[0])),dtype=np.float32)
        for c in range(1,num_class+1):
            idx,idy=np.where(mlabel==c)
            
            for i in range(len(idx)):
                patch.append(mdata[idx[i]-r:idx[i]+r+1,idy[i]-r:idy[i]+r+1,...])
                patch_right.append(mdata[idx[i]:idx[i]+2*r+1,idy[i]-r:idy[i]+r+1,...])
                patch_left.append(mdata[idx[i]-2*r:idx[i]+1,idy[i]-r:idy[i]+r+1,...])
                patch_up.append(mdata[idx[i]-r:idx[i]+r+1,idy[i]-2*r:idy[i]+1,...])
                patch_bottom.append(mdata[idx[i]-r:idx[i]+r+1,idy[i]:idy[i]+2*r+1,...])
                patch_upleft.append(mdata[idx[i]-2*r:idx[i]+1,idy[i]-2*r:idy[i]+1,...])
                patch_upright.append(mdata[idx[i]:idx[i]+2*r+1,idy[i]-2*r:idy[i]+1,...])
                patch_bottomleft.append(mdata[idx[i]-2*r:idx[i]+1,idy[i]:idy[i]+2*r+1,...])
                patch_bottomright.append(mdata[idx[i]:idx[i]+2*r+1,idy[i]:idy[i]+2*r+1,...])
            
                train_att[0,count_idx+i]=Eucdis(patch[i],mdata[idx[i],idy[i]],alpha)
                train_att[1,count_idx+i]=Eucdis(patch_right[i],mdata[idx[i],idy[i]],alpha)
                train_att[2,count_idx+i]=Eucdis(patch_left[i],mdata[idx[i],idy[i]],alpha)
                train_att[3,count_idx+i]=Eucdis(patch_up[i],mdata[idx[i],idy[i]],alpha)
                train_att[4,count_idx+i]=Eucdis(patch_bottom[i],mdata[idx[i],idy[i]],alpha)
                train_att[5,count_idx+i]=Eucdis(patch_upleft[i],mdata[idx[i],idy[i]],alpha)
                train_att[6,count_idx+i]=Eucdis(patch_upright[i],mdata[idx[i],idy[i]],alpha)
                train_att[7,count_idx+i]=Eucdis(patch_bottomleft[i],mdata[idx[i],idy[i]],alpha)
                train_att[8,count_idx+i]=Eucdis(patch_bottomright[i],mdata[idx[i],idy[i]],alpha)
                label.append(mlabel[idx[i],idy[i]]-1)
            count_idx = count_idx + len(idx)
        train_att=softmax(train_att)
        result_patchs=np.asarray(patch,dtype=np.float32)
        result_labels=np.asarray(label,dtype=np.int8)
        
        XR=np.asarray(patch_right,dtype=np.float32)
        XL=np.asarray(patch_left,dtype=np.float32)
        XU=np.asarray(patch_up,dtype=np.float32)
        XB=np.asarray(patch_bottom,dtype=np.float32)
        XUL=np.asarray(patch_upleft,dtype=np.float32)
        XUR=np.asarray(patch_upright,dtype=np.float32)
        XBL=np.asarray(patch_bottomleft,dtype=np.float32)
        XBR=np.asarray(patch_bottomright,dtype=np.float32)
        np.random.seed(seed)
        index=np.random.permutation(result_patchs.shape[0])       
        return result_patchs[index],XR[index],XL[index],XU[index],XB[index],XUL[index],XUR[index],XBL[index],XBR[index],result_labels[index],(train_att.T[index]).T
    if patch_type=='test':
        idx,idy=np.nonzero(mlabel)
        test_att = np.zeros((9,(np.where(mlabel!=0)[0].shape[0])),dtype=np.float32)
        for i in range(len(idx)):
            patch.append(mdata[idx[i]-r:idx[i]+r+1,idy[i]-r:idy[i]+r+1,...])
            patch_right.append(mdata[idx[i]:idx[i]+2*r+1,idy[i]-r:idy[i]+r+1,...])
            patch_left.append(mdata[idx[i]-2*r:idx[i]+1,idy[i]-r:idy[i]+r+1,...])
            patch_up.append(mdata[idx[i]-r:idx[i]+r+1,idy[i]-2*r:idy[i]+1,...])
            patch_bottom.append(mdata[idx[i]-r:idx[i]+r+1,idy[i]:idy[i]+2*r+1,...])
            patch_upleft.append(mdata[idx[i]-2*r:idx[i]+1,idy[i]-2*r:idy[i]+1,...])
            patch_upright.append(mdata[idx[i]:idx[i]+2*r+1,idy[i]-2*r:idy[i]+1,...])
            patch_bottomleft.append(mdata[idx[i]-2*r:idx[i]+1,idy[i]:idy[i]+2*r+1,...])
            patch_bottomright.append(mdata[idx[i]:idx[i]+2*r+1,idy[i]:idy[i]+2*r+1,...])

            test_att[0,i]=Eucdis(patch[i],mdata[idx[i],idy[i]],alpha)
            test_att[1,i]=Eucdis(patch_right[i],mdata[idx[i],idy[i]],alpha)
            test_att[2,i]=Eucdis(patch_left[i],mdata[idx[i],idy[i]],alpha)
            test_att[3,i]=Eucdis(patch_up[i],mdata[idx[i],idy[i]],alpha)
            test_att[4,i]=Eucdis(patch_bottom[i],mdata[idx[i],idy[i]],alpha)
            test_att[5,i]=Eucdis(patch_upleft[i],mdata[idx[i],idy[i]],alpha)
            test_att[6,i]=Eucdis(patch_upright[i],mdata[idx[i],idy[i]],alpha)
            test_att[7,i]=Eucdis(patch_bottomleft[i],mdata[idx[i],idy[i]],alpha)
            test_att[8,i]=Eucdis(patch_bottomright[i],mdata[idx[i],idy[i]],alpha)
            label.append(mlabel[idx[i],idy[i]]-1)
        test_att=softmax(test_att)
        result_patchs=np.asarray(patch,dtype=np.float32)
        result_labels=np.asarray(label,dtype=np.int8)
        XR=np.asarray(patch_right,dtype=np.float32)
        XL=np.asarray(patch_left,dtype=np.float32)
        XU=np.asarray(patch_up,dtype=np.float32)
        XB=np.asarray(patch_bottom,dtype=np.float32)
        XUL=np.asarray(patch_upleft,dtype=np.float32)
        XUR=np.asarray(patch_upright,dtype=np.float32)
        XBL=np.asarray(patch_bottomleft,dtype=np.float32)
        XBR=np.asarray(patch_bottomright,dtype=np.float32)        
        idx=idx-2*r
        idy=idy-2*r
        return result_patchs,XR,XL,XU,XB,XUL,XUR,XBL,XBR,result_labels,idx,idy,test_att

def random_flip(data,xr,xl,xu,xb,xul,xur,xbl,xbr,label,train_att,seed=0):
    num=data.shape[0]
    datas=[]
    xrs=[]
    xls=[]
    xus=[]
    xbs=[]
    xuls=[]
    xurs=[]
    xbls=[]
    xbrs=[]
    labels=[]
    train_atts=[]
    for i in range(num):
        datas.append(data[i])
        xrs.append(xr[i])
        xls.append(xl[i])
        xus.append(xu[i])
        xbs.append(xb[i])
        xuls.append(xul[i])
        xurs.append(xur[i])
        xbls.append(xbl[i])
        xbrs.append(xbr[i])
        
        if len(data[i].shape)==3:
            noise=np.random.normal(0.0,0.05,size=(data[i].shape))
            datas.append(np.fliplr(data[i])+noise)
            noise=np.random.normal(0.0,0.05,size=(xr[i].shape))
            xrs.append(np.fliplr(xr[i])+noise)
            noise=np.random.normal(0.0,0.05,size=(xl[i].shape))
            xls.append(np.fliplr(xl[i])+noise)
            noise=np.random.normal(0.0,0.05,size=(xu[i].shape))
            xus.append(np.fliplr(xu[i])+noise)
            noise=np.random.normal(0.0,0.05,size=(xb[i].shape))
            xbs.append(np.fliplr(xb[i])+noise)
            
            noise=np.random.normal(0.0,0.05,size=(xul[i].shape))
            xuls.append(np.fliplr(xul[i])+noise)
            noise=np.random.normal(0.0,0.05,size=(xur[i].shape))
            xurs.append(np.fliplr(xur[i])+noise)
            noise=np.random.normal(0.0,0.05,size=(xbl[i].shape))
            xbls.append(np.fliplr(xbl[i])+noise)
            noise=np.random.normal(0.0,0.05,size=(xbr[i].shape))
            xbrs.append(np.fliplr(xbr[i])+noise)
                        
        labels.append(label[i])
        labels.append(label[i])
        train_atts.append(train_att[:,i])
        train_atts.append(train_att[:,i])
        
    datas=np.asarray(datas,dtype=np.float32)
    xrs=np.asarray(xrs,dtype=np.float32)
    xls=np.asarray(xls,dtype=np.float32)
    xus=np.asarray(xus,dtype=np.float32)
    xbs=np.asarray(xbs,dtype=np.float32)
    xuls=np.asarray(xuls,dtype=np.float32)
    xurs=np.asarray(xurs,dtype=np.float32)
    xbls=np.asarray(xbls,dtype=np.float32)
    xbrs=np.asarray(xbrs,dtype=np.float32)
    train_atts=np.asarray(train_atts,dtype=np.float32)
    labels=np.asarray(labels,dtype=np.float32)
    np.random.seed(seed)
    index=np.random.permutation(datas.shape[0])
    return datas[index],xrs[index],xls[index],xus[index],xbs[index],xuls[index],xurs[index],xbls[index],xbrs[index],labels[index],train_atts[index].T

def read_data(path,file_name,data_name,data_type):
    # if data_type=='tif':
    #     mdata=tiff.imread(os.path.join(path,file_name))
    #     return mdata
    if data_type=='mat':
        mdata=sio.loadmat(os.path.join(path,file_name))
        mdata=np.array(mdata[data_name])
        return mdata

def main():
    mdata=read_data(PATH,mdata_name,'data',data_type)
    mdata=np.asarray(mdata,dtype=np.float32)
    [row,col,n_band]=mdata.shape
    mdata = np.reshape(featureNormalize(np.reshape(mdata,-1),2),(row,col,n_band))
    mdata=image_pad(mdata,r+r)

    mlabel_train=read_data(PATH,args.train_label_name,'mask_train',data_type)
    mlabel_train=image_pad(mlabel_train,r+r)
    train_data_H,XR,XL,XU,XB,XUL,XUR,XBL,XBR,train_label_H,train_att=construct_spatial_patch(mdata,mlabel_train,r,'train',alpha)
#    train_data_H,XR,XL,XU,XB,XUL,XUR,XBL,XBR,train_label_H,train_att=random_flip(train_data_H,XR,XL,XU,XB,XUL,XUR,XBL,XBR,train_label_H,train_att)
    print('train data shape:{}'.format(train_data_H.shape))
    print('right data shape:{}'.format(XR.shape))
    print('left  data shape:{}'.format(XL.shape))
    print('up    data shape:{}'.format(XU.shape))
    print('botom data shape:{}'.format(XB.shape))
    print('Saving train data...')
    data={
        'data':train_data_H,
        'XR':XR,
        'XL':XL,
        'XU':XU,
        'XB':XB,
        'XUL':XUL,
        'XUR':XUR,
        'XBL':XBL,
        'XBR':XBR,
        'train_att':train_att,
        'label':train_label_H
    }
    path_train =os.path.join(save_path+'train_data_H.mat')
    sio.savemat(path_train,data)

    # SAVE TEST_DATA TO MAT FILE
    for iclass in range(1,NUM_CLASS+1):
        test_label_name=os.path.join('mask_test_patch'+str(iclass)+'.mat')
        mlabel_test=read_data(PATH,test_label_name,'mask_test',data_type)
        mlabel_test=image_pad(mlabel_test,r+r)
        test_data_H,XR,XL,XU,XB,XUL,XUR,XBL,XBR,test_label_H,idx,idy,test_att=construct_spatial_patch(mdata,mlabel_test,r,'test',alpha)
        print('test data shape:{}'.format(test_data_H.shape))
        print('right data shape:{}'.format(XR.shape))
        print('left  data shape:{}'.format(XL.shape))
        print('up    data shape:{}'.format(XU.shape))
        print('botom data shape:{}'.format(XB.shape))
        
        print('Saving test data...')
        data={
            'hsi':test_data_H,
            'XR':XR,
            'XL':XL,
            'XU':XU,
            'XB':XB,
            'XUL':XUL,
            'XUR':XUR,
            'XBL':XBL,
            'XBR':XBR,

            'label':test_label_H,
            'idx':idx,
            'idy':idy,
            'test_att':test_att 
        }
        path_test=os.path.join(save_path,'test_data_H'+str(iclass)+'.mat')
        sio.savemat(path_test,data,format='5')
    print('Done')

if __name__=='__main__':
    main()






