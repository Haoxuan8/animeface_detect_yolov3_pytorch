# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:26:49 2019

@author: hasee
"""

import  yolov3
import util
import torch
import cv2

num_classes=1 
confidence=0.3
nms_theshold=0.4

classes=util.load_classes('data/animeface.names')
model=yolov3.yolov3_darknet('cfg/animeface.cfg')
#model.load_weights('weights/yolov3.weights')
model.load_state_dict(torch.load('model_state_dict.pt'))
img=cv2.imread('samples/test2.png')

net_h,net_w=int(model.net_info['height']),int(model.net_info['width'])

new_img,img_tensor=util.resize_img(img,net_h,net_w)
_,prediction=model(img_tensor,torch.cuda.is_available())
prediction=util.write_results(prediction,confidence,num_classes)

write_img=util.writebox(img,model,prediction,classes)
cv2.imwrite('test.png',write_img)
