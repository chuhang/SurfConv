import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

def get_db_sizes(data_path,number_of_splits):
    d=os.listdir(data_path+'/train')
    file_name_base=d[0][:-5]
    db_sizes=[]
    for i in range(number_of_splits):
        file_name=data_path+'/train/'+file_name_base+str(i+1)+'.png'
        img=m.imread(file_name)
        img=np.array(img,dtype=np.uint8)
        db_sizes.append([img.shape[0],img.shape[1]])
    return db_sizes

def kitti_loader_nway_with_d_with_hha(data_path_rgb,data_path_lbl,depth_path,data_path_hha,trainvaltest,number_of_splits,db_sizes,batch_idx_list):
    mean_rgb=np.array([90.96, 93.08, 92.71])
    mean_hha=np.array([43.12, 86.08, 96.97])
    batch_size=len(batch_idx_list)
    img_tensor_list=[]
    lbl_tensor_list=[]
    d_tensor_list=[]
    hha_tensor_list=[]
    for s in range(number_of_splits):
        img_tensor=np.zeros((batch_size,3,db_sizes[s][0],db_sizes[s][1]),dtype=np.float64)
        lbl_tensor=np.zeros((batch_size,db_sizes[s][0],db_sizes[s][1]),dtype=np.int32)
        d_tensor=np.zeros((batch_size,db_sizes[s][0],db_sizes[s][1]),dtype=np.uint8)
        hha_tensor=np.zeros((batch_size,3,db_sizes[s][0],db_sizes[s][1]),dtype=np.float64)
        for i,idx in enumerate(batch_idx_list):
            file_name=data_path_rgb+'/'+trainvaltest+'/img_'+str(idx+1)+'p'+str(s+1)+'.png'
            img=m.imread(file_name,mode='RGB')
            img=np.array(img,dtype=np.uint8)
            img=img[:, :, ::-1]
            img=img.astype(np.float64)
            img-=mean_rgb
            img=img.astype(float)/255.0
            #NHWC->NCHW
            img=img.transpose(2,0,1)
            img_tensor[i]=img
            file_name=data_path_lbl+'/'+trainvaltest+'/img_'+str(idx+1)+'p'+str(s+1)+'.png'
            lbl=m.imread(file_name)
            lbl=np.array(lbl,dtype=np.int32)
            lbl_tensor[i]=lbl
            file_name=depth_path+'/'+trainvaltest+'/img_'+str(idx+1)+'.png'
            d=m.imread(file_name)
            d=m.imresize(d,(db_sizes[s][0],db_sizes[s][1]),interp='nearest')
            d=np.array(d,dtype=np.uint8)
            d_tensor[i]=d
            file_name=data_path_hha+'/'+trainvaltest+'/img_'+str(idx+1)+'p'+str(s+1)+'.png'
            hha=m.imread(file_name,mode='RGB')
            hha=np.array(hha,dtype=np.uint8)
            hha=hha.astype(np.float64)
            hha-=mean_hha
            hha=hha.astype(float)/255.0
            #NHWC->NCHW
            hha=hha.transpose(2,0,1)
            hha_tensor[i]=hha
        img_tensor=torch.from_numpy(img_tensor).float()
        lbl_tensor=torch.from_numpy(lbl_tensor).long()
        d_tensor=torch.from_numpy(d_tensor).long()
        hha_tensor=torch.from_numpy(hha_tensor).float()
        img_tensor_list.append(img_tensor)
        lbl_tensor_list.append(lbl_tensor)
        d_tensor_list.append(d_tensor)
        hha_tensor_list.append(hha_tensor)
    return img_tensor_list,lbl_tensor_list,d_tensor_list,hha_tensor_list
