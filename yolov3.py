# -*- coding: utf-8 -*-
from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import os
import util


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.stride=0
        self.grid_size=0


class yolov3_darknet(nn.Module):
    def __init__(self,cfg='cfg/yolov3.cfg'):
        super(yolov3_darknet, self).__init__()
        self.blocks=self.parse_cfg(cfg)
        self.net_info,self.module_list=self.create_modules(self.blocks,torch.cuda.is_available())  #convert to pytorch models
        
    def forward(self, x, CUDA):
        if CUDA:
            x=x.cuda()
        
        
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        detections=[]
        detections_prev=[]
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                detections_prev.append(x)
                x = x.data
                self.module_list[i][0].stride =  inp_dim // x.size(2)
                self.module_list[i][0].grid_size = inp_dim // self.module_list[i][0].stride
                x = util.predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                detections.append(x)
                '''
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.stack((detections, x), 0)
                '''
        
            outputs[i] = x
        
        return detections_prev,detections
    
    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
    
    
    
        
        
    def parse_cfg(self,cfg):
        if not os.path.exists(cfg):
            raise RuntimeError('cfg does not exist.')
        
        cfgfile=open('cfg/yolov3.cfg','r')
        lines=cfgfile.read().split('\n')
        blocks=[]
        block={}
        for line in lines:
            if len(line)==0 or line[0]=='#':
                continue
            if line[0]=='[':
                if len(block)!=0:
                    blocks.append(block)
                    block={}    #clear block
                block['type']=line[1:-1]
            else:
                key,value=line.split('=')
                block[key.rstrip()]=value.lstrip()
                
        blocks.append(block)
        return blocks
    
    def create_modules(self,blocks,CUDA):
        net_info=blocks[0]
        prev_filters=3
        output_filters = []
        modules_list=nn.ModuleList()
        
        for index,block in enumerate(blocks[1:]):
            module=nn.Sequential()
            
            # convolutional layer
            if block['type']=='convolutional':
                activation=block['activation']
                try:
                    batch_normalize=int(block['batch_normalize'])
                    bias=False
                except:
                    batch_normalize=0
                    bias=True
                filters=int(block['filters'])
                size=int(block['size'])
                stride=int(block['stride'])
                pad=int(block['pad'])
                if pad:
                    padding=(size-1)//2
                else:
                    padding=0
                
                #add conv to module
                conv=nn.Conv2d(prev_filters,filters,size,stride,padding,bias=bias)
                module.add_module('conv_%d'%index,conv)
                
                if batch_normalize:
                    bn=nn.BatchNorm2d(filters)
                    module.add_module('batch_norm_%d'%index,bn)
                
                if activation=='leaky':
                    activn=nn.LeakyReLU(0.1,inplace=True)
                    module.add_module('leaky_%d'%index,activn)   
                    
            #umsample layer     
            elif block['type']=='upsample':
                stride = int(block['stride'])
                upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
                module.add_module('upsample_%d'%index, upsample)
                
            #route layer
            elif block['type']=='route':
                block['layers'] = block['layers'].split(',')
                #Start  of a route
                start = int(block['layers'][0])
                #end, if there exists one.
                try:
                    end = int(block['layers'][1])
                except:
                    end = 0
                #Positive anotation
                if start > 0: 
                    start = start - index
                if end > 0:
                    end = end - index
                route = EmptyLayer()
                module.add_module('route_%d'%index, route)
                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                else:
                    filters= output_filters[index + start]                   
            #shortcut layer
            elif block['type']=='shortcut':
                shortcut = EmptyLayer()
                module.add_module('shortcut_%d'%index, shortcut)
            #yolo
            elif block['type']=='yolo':
                mask=block['mask'].split(',')
                mask=[int(x) for x in mask]
                
                anchors=block['anchors'].split(',')
                anchors=[int(x) for x in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
                anchors = [anchors[i] for i in mask]
                detection = DetectionLayer(anchors)
                module.add_module("Detection_{}".format(index), detection)
            
            if CUDA:
                module=module.cuda()
            modules_list.append(module)
            prev_filters=filters
            output_filters.append(filters)
        
        return (net_info,modules_list)
                

