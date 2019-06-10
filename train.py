# -*- coding: utf-8 -*-
from __future__ import division

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn

import torch 
import datasets
import yolov3
import util
import yololoss
import time


params = {'xy': 0.2,  # xy loss gain
       'wh': 0.1,  # wh loss gain
       'cls': 0.04,  # cls loss gain
       'conf': 4.5,  # conf loss gain
       'iou_t': 0.5,  # iou target-anchor training threshold
       'lr0': 0.001,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay

savepath='animeface.pt'


def build_yolo_loss(net):
    modules = net.blocks[1:]
    yolo_loss=[]
    for i, module in enumerate(modules): 
        if module['type']=='yolo':
            anchors = net.module_list[i][0].anchors
            classes=int(module['classes'])
            size=(int(net.net_info['height']),int(net.net_info['width']))
            yolo_loss.append(yololoss.YOLOLoss(anchors,classes,size))
    
    return yolo_loss
            

if __name__=='__main__':
    cuda=torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    batch_size=1
    epochs=100
    
    net=yolov3.yolov3_darknet('cfg/animeface.cfg').to(device)
    net_h,net_w=int(net.net_info['height']),int(net.net_info['width'])
    
    yolo_loss=build_yolo_loss(net)
    
    # Optimizer
    optimizer = optim.SGD(nn.ParameterList(net.parameters()), lr=params['lr0'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in (0.8, 0.9)], gamma=0.1)
    
    
    dataset=datasets.LoadTrainDataset('images/data/img','images/data/img', net_h,net_w)
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )

    

    nb = len(dataloader)
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    losses_epoch=[]
    
    for epoch in range(epochs):
        net.train()
        scheduler.step()
        
        for bn,(paths,imgs,targets) in enumerate(dataloader):
            optimizer.zero_grad()
            imgs=imgs.to(device)
            targets=targets.to(device)
            # SGD burn-in
            if epoch == 0 and bn <= n_burnin:
                lr = params['lr0'] * (bn / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr
            
            #run net
            pred=net(imgs,torch.cuda.is_available())
            
            #computer loss
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
                
            for i in range(3):
                _loss_item = yolo_loss[i](net.outputs[i], targets)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
            
            #gradient
            loss.backward()
            
            optimizer.step()
            

            losses_epoch.append(loss)
            s = 'epoch: %d   batch_num: %d   loss: %.3f'%(epoch,bn,loss)
            print(s)
            
            with open('result.txt','a') as f:
                f.write(s+'\n')
        
        if epoch % 1==0: #each epoch save net
            torch.save(net, savepath)
    
    
    torch.save(net, savepath)
    print('finish!')
    print(losses_epoch)
    
    
    
    
        
        