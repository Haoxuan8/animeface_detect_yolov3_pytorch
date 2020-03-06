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
import argparse


params = {'xy': 0.2,  # xy loss gain
       'wh': 0.1,  # wh loss gain
       'cls': 0.04,  # cls loss gain
       'conf': 4.5,  # conf loss gain
       'iou_t': 0.5,  # iou target-anchor training threshold
       'lr0': 0.001,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay

def test(model, path, batch_size, yolo_loss):
    net_h,net_w=int(model.net_info['height']),int(model.net_info['width'])
    dataset=datasets.LoadTrainDataset(path,path, net_h,net_w)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )    
    model.eval()
    test_loss=float(0)
    for bn,(paths,imgs,targets) in enumerate(dataloader):
        pred,_=model(imgs,torch.cuda.is_available())
            
        #computer loss
        losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
        losses = []
        for _ in range(len(losses_name)):
            losses.append([])
            
        for i in range(3):
            _loss_item = yolo_loss[i](pred[i], targets)
            for j, l in enumerate(_loss_item):
                losses[j].append(l)
        losses = [sum(l) for l in losses]
        loss = losses[0] 
        test_loss+=float(loss)
    
    test_loss/=len(dataloader)
    return test_loss
    



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
            

def train(cfg, names, epochs, batch_size, pretrained, trainpath, validpath):
    cuda=torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    
    net=yolov3.yolov3_darknet(cfg).to(device)
    net_h,net_w=int(net.net_info['height']),int(net.net_info['width'])
    
    yolo_loss=build_yolo_loss(net)
    
    # Optimizer
    optimizer = optim.SGD(nn.ParameterList(net.parameters()), lr=params['lr0'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in (0.8, 0.9)], gamma=0.1)
    
    
    dataset=datasets.LoadTrainDataset(trainpath,trainpath, net_h,net_w)
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
    
    
    if pretrained:
        print('loading predtrained net')
        net.load_state_dict(torch.load('model_state_dict.pt'))

    

    nb = len(dataloader)
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    losses_epoch=[]
    mlosses_epoch=[]
    best_loss=float('inf')
    for epoch in range(epochs):
        net.train()
        scheduler.step()
        mloss=torch.zeros(1).to(device)
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
            pred,_=net(imgs,torch.cuda.is_available())
            
            #computer loss
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
                
            for i in range(3):
                _loss_item = yolo_loss[i](pred[i], targets)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
            
            mloss=(mloss*bn+loss)/(bn+1)
            #gradient
            loss.backward()
            
            optimizer.step()
            
            losses_epoch.append(loss)
            mlosses_epoch.append(mloss)
            
            s = 'epoch: %d\tbatch_num: %d\tloss: %.3f\tmloss: %.3f'%(epoch,bn,loss,mloss)
            print(s)
            
            with open('result.txt','a') as f:
                f.write('%d'%epoch+' '+'%g'%loss+'\n')
        
        test_loss=test(net,validpath, batch_size,yolo_loss)
        print('test_loss: %.3f'%test_loss)
        
        if test_loss<best_loss:
            best_loss=test_loss
            print('saving model...')
            torch.save(net.state_dict(), 'model_state_dict.pt')
            
        '''
        if epoch % 1==0: #each epoch save net
            torch.save(net.state_dict(), 'model_state_dict.pt')
        '''
    
    #torch.save(net.state_dict(), 'model_state_dict.pt')
    print('finish!')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--bs', type=int, default=1, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/animeface.cfg', help='cfg file path')
    parser.add_argument('--names', type=str, default='data/animeface.names', help='names file path')
    parser.add_argument('--trainpath', type=str, default='imgs/train', help='train file path')
    parser.add_argument('--validpath', type=str, default='imgs/valid', help='valid file path')
    parser.add_argument('--pretrained', type=str, default=0, help='pretrained')
    opt = parser.parse_args()
    print(opt)
    
    train(opt.cfg, opt.names, batch_size=opt.bs, epochs=opt.epochs, trainpath=opt.trainpath, validpath=opt.validpath, pretrained=opt.pretrained)
    

    
    
    
        
        