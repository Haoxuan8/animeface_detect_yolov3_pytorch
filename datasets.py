# -*- coding: utf-8 -*-
"""
"""

from __future__ import division

from torch.utils.data import Dataset
import os
import util 
import cv2
import numpy as np
import torch

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 1] = 1 - targets[:, 1]
    return images, targets


class LoadTrainDataset(Dataset):
    def __init__(self, imgs_path, label_path, net_h, net_w,augment=True, max_objects=50):
        imgname_list=os.listdir(imgs_path)
        self.img_list=[]
        self.label_list=[]
        self.net_h=net_h
        self.net_w=net_w
        self.augment=augment
        self.max_objects=max_objects
        for name in imgname_list:
            if name.find('.jpg')>0 or name.find('.png')>0:
                labelfile=name.replace('.jpg','.txt').replace('.png','.txt')
                label_np=np.loadtxt(label_path+'/'+labelfile)
                if len(label_np)==0:
                    continue
                self.label_list.append(label_path+'/'+labelfile)
                self.img_list.append(imgs_path+'/'+name)
            
    
    def __len__(self):
        return len(self.img_list)
    
    
    def __getitem__(self,index):
        img_path=self.img_list[index%len(self.img_list)]
        img=cv2.imread(img_path)
        
        #transform to torchtensor
        _,img_tensor=util.resize_img(img,self.net_h,self.net_w)        
        img_tensor=img_tensor[0]
        #label load
        label_path=self.label_list[index%len(self.label_list)]
        label_np=np.loadtxt(label_path).reshape(-1,5)
        label=torch.from_numpy(label_np)
        height,width=img.shape[0],img.shape[1]
        
        if len(label)==0:
            return None,None,None
        
        #transform to (x,y,w,h) in resized image
        scaling=min(self.net_h/height,self.net_w/width)
        label[:,1]*=width
        label[:,1]*=scaling
        label[:,1]+=(self.net_w-width*scaling)/2
        label[:,1]/=self.net_w
        
        label[:,2]*=height
        label[:,2]*=scaling
        label[:,2]+=(self.net_h-height*scaling)/2
        label[:,2]/=self.net_h
        
        label[:,3]*=width*scaling
        label[:,3]/=self.net_w
        
        label[:,4]*=height*scaling
        label[:,4]/=self.net_h
                        
        
        if self.augment:
            if np.random.random() <0.5:
                img,label=horisontal_flip(img_tensor, label)
                
        return img_path, img_tensor, label
    
    def collate_fn(self,batch):
         paths, imgs, targets = list(zip(*batch))
         labels_list=[]
         # transform targets to bsxmax_objectsx5
         for i in range(len(targets)):
             filled_labels = np.zeros((self.max_objects, 5), np.float32)
             for j in range(len(targets[i])):
                 filled_labels[j]=targets[i][j]
             
             labels_list.append(torch.from_numpy(filled_labels))
             
         
         return paths, torch.stack(imgs,0), torch.stack(labels_list,0)
             
         
    