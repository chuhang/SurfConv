import os
import sys
import torch
import argparse
import numpy as np
import scipy.misc as m
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable

import data_loader.kitti_loader_nway_with_d_with_hha as DL
import models.resnet as NET
import loss_metrics.loss as LOSS
import loss_metrics.metrics as METRICS

N_Classes=12
Data_Path='../dataset/KITTI/'
Root_Path='./'

def train(args):
    # Setup dataloader
    number_of_splits=args.sc
    data_path_rgb=Data_Path+'rgb_SurfConv'+str(number_of_splits)+'_gamma'+args.gamma
    data_path_hha=Data_Path+'hha_SurfConv'+str(number_of_splits)+'_gamma'+args.gamma
    data_path_lbl=Data_Path+'label_SurfConv'+str(number_of_splits)+'_gamma'+args.gamma
    data_path_d=Data_Path+'d_max80m_8bit'
    n_classes=N_Classes
    Batch_Size=args.batch_size
    train_image_number=len(os.listdir(data_path_rgb+'/train'))/number_of_splits
    val_image_number=len(os.listdir(data_path_rgb+'/val'))/number_of_splits
    db_sizes=DL.get_db_sizes(data_path_rgb,number_of_splits)

    # Setup logging/checkpointing
    ckpt_path=Root_Path+'kitti_rgbhha_res'+str(args.capacity)+'_sc'+str(number_of_splits)+'_ga'+args.gamma+'_voxloss_run'+str(args.run_id)+'.model'
    log_path=Root_Path+'kitti_rgbhha_res'+str(args.capacity)+'_sc'+str(number_of_splits)+'_ga'+args.gamma+'_voxloss_run'+str(args.run_id)+'.log'
    logfile=open(log_path,'w')

    # Setup model
    model=NET.resnet18(capacity=args.capacity,n_classes=n_classes,in_channels=6)
    model.cuda(0)

    # Optimizer
    optimizer=torch.optim.SGD(model.parameters(),lr=args.l_rate,momentum=0.9,weight_decay=5e-4)

    # Misc
    iter_ct=0
    now_lr=args.l_rate

    # Training
    meanious=[]
    globalaccs=[]
    spatialious=[]
    spatialaccs=[]
    lr_adjust_times_max=4
    lr_adjust_times=0
    last_early_stop=0
    for epoch in range(args.n_epoch):
        # plateau with patience
        is_early_stop=0
        if epoch>(last_early_stop+args.patience+1):
            recent_metrics=spatialious[-args.patience:]
            if max(recent_metrics)<max(spatialious):
                is_early_stop=1
        if epoch<args.min_epoch:
            is_early_stop=0
        if is_early_stop==1:
            last_early_stop=epoch
            # load previous best checkpoint
            ckpt=torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            printstr='loaded previous best checkpoint'
            print(printstr)
            logfile.write(printstr+'\n')
            # lr adjust
            now_lr*=0.1
            lr_adjust_times+=1
            if lr_adjust_times==lr_adjust_times_max:
                printstr='lr too low, training terminates.'
                print(printstr)
                logfile.write(printstr+'\n')
                break
            for param_group in optimizer.param_groups:
                param_group['lr']=now_lr
                printstr='lr: '+str(param_group['lr'])
                print(printstr)
                logfile.write(printstr+'\n')
        
        # Training
        model.train()
        order_list=np.random.permutation(train_image_number)
        losses=[]
        counters=[]
        for j in range(number_of_splits):
            counters.append([])
        for i in range(0,train_image_number,Batch_Size):
            # load batch
            if i+Batch_Size<=train_image_number:
                batch_idx_list=order_list[i:i+Batch_Size]
            else:
                batch_idx_list=[]
                for j in range(i,train_image_number):
                    batch_idx_list.append(order_list[j])
                extraneeded=Batch_Size-len(batch_idx_list)
                for j in range(extraneeded):
                    batch_idx_list.append(order_list[j])
            img_tensor_list,lbl_tensor_list,d_tensor_list,hha_tensor_list=DL.kitti_loader_nway_with_d_with_hha(data_path_rgb,data_path_lbl,data_path_d,data_path_hha,'train',number_of_splits,db_sizes,batch_idx_list)
            for j in range(number_of_splits):
                img_tensor_list[j]=Variable(img_tensor_list[j].cuda(0))
                lbl_tensor_list[j]=Variable(lbl_tensor_list[j].cuda(0))
                d_tensor_list[j]=Variable(d_tensor_list[j].cuda(0))
                hha_tensor_list[j]=Variable(hha_tensor_list[j].cuda(0))
            # forward model
            optimizer.zero_grad()
            loss_list=[]
            doback_list=[]
            n_pix_list=[]
            pixel_weight_list=[]
            for j in range(number_of_splits):
                outputs=model(torch.cat((hha_tensor_list[j],img_tensor_list[j]),1))
                loss,doback,n_pix=LOSS.cross_entropy2d_mask(outputs,lbl_tensor_list[j],[0],size_average=False)
                # reweight efficient to make loss pixel-based
                pixel_weight=float(db_sizes[j][0])*db_sizes[j][0]/db_sizes[0][0]/db_sizes[0][0]
                loss_list.append(loss)
                doback_list.append(doback)
                n_pix_list.append(float(n_pix))
                pixel_weight_list.append(float(pixel_weight))
            # compute loss
            for j in range(number_of_splits):
                if doback_list[j]==1:
                    loss_list[j]/=float(n_pix_list[j])
                    loss_list[j].backward()
                    counters[j].append(n_pix_list[j])
            optimizer.step()

            iter_ct+=1
            #print('Iter: '+str(iter_ct)+' loss: '+str(loss_overall.data[0]))
            #losses.append(loss_overall.data[0])
        printstr="Epoch [%d/%d]" % (epoch+1,args.n_epoch)
        print(printstr)
        logfile.write(printstr+'\n')
        logfile.flush()

        # Validate every iteration
        model.eval()
        gts=[]
        preds=[]
        drefs=[]
        for j in range(number_of_splits):
            gts.append([])
            preds.append([])
            drefs.append([])
        for i in range(val_image_number):
            img_tensor_list,lbl_tensor_list,d_tensor_list,hha_tensor_list=DL.kitti_loader_nway_with_d_with_hha(data_path_rgb,data_path_lbl,data_path_d,data_path_hha,'val',number_of_splits,db_sizes,[i])
            for j in range(number_of_splits):
                img_tensor_list[j]=Variable(img_tensor_list[j].cuda(0))
                lbl_tensor_list[j]=Variable(lbl_tensor_list[j].cuda(0))
                d_tensor_list[j]=Variable(d_tensor_list[j].cuda(0))
                hha_tensor_list[j]=Variable(hha_tensor_list[j].cuda(0))
                gt=lbl_tensor_list[j].data.cpu().numpy()
                dref=d_tensor_list[j].data.cpu().numpy().astype(float)
                outputs=model(torch.cat((hha_tensor_list[j],img_tensor_list[j]),1))
                pred=outputs.data.max(1)[1].cpu().numpy()
                for k in range(len(gt)):
                    gts[j].append(gt[k])
                    preds[j].append(pred[k])
                    drefs[j].append(dref[k])
        score,class_iou=METRICS.scores_dontcare_nway(gts,preds,db_sizes,n_class=n_classes,dontcare_list=[0])
        score_spatial,class_iou_spatial=METRICS.scores_dontcare_nway_weighted(gts,preds,drefs,db_sizes,n_class=n_classes,dontcare_list=[0])
        printstr="--Validation: Epoch [%d/%d] Mean IOU: %.4f" % (epoch+1,args.n_epoch,score['Mean_IoU'])
        print(printstr)
        logfile.write(printstr+'\n')        
        printstr="--Validation: Epoch [%d/%d] Global Acc: %.4f" % (epoch+1,args.n_epoch,score['Overall_Acc'])
        print(printstr)
        logfile.write(printstr+'\n')
        printstr="--Validation: Epoch [%d/%d] Spatial IOU: %.4f" % (epoch+1,args.n_epoch,score_spatial['Mean_IoU'])
        print(printstr)
        logfile.write(printstr+'\n')        
        printstr="--Validation: Epoch [%d/%d] Spatial Acc: %.4f" % (epoch+1,args.n_epoch,score_spatial['Overall_Acc'])
        print(printstr)
        logfile.write(printstr+'\n')
        logfile.flush()
        # Save model
        if epoch==0 or score_spatial['Mean_IoU']>max(spatialious):
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'val_metrics':[score['Mean_IoU'],score['Overall_Acc'],score_spatial['Mean_IoU'],score_spatial['Overall_Acc']]},ckpt_path)
        meanious.append(score['Mean_IoU'])
        globalaccs.append(score['Overall_Acc'])
        spatialious.append(score_spatial['Mean_IoU'])
        spatialaccs.append(score_spatial['Overall_Acc'])
    printstr='*****'+str(max(meanious))+', '+str(max(globalaccs))
    print(printstr)
    logfile.write(printstr+'\n')
    logfile.write('done'+'\n')
    logfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=300, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--run_id', nargs='?', type=int, default=0, 
                        help='repeat experiment id')
    parser.add_argument('--capacity', nargs='?', type=int, default=1, 
                        help='model capacity')
    parser.add_argument('--patience', nargs='?', type=int, default=10, 
                        help='plateau patience for lr adjustment')
    parser.add_argument('--sc', nargs='?', type=int, default=4, 
                        help='surface convolution levels')
    parser.add_argument('--gamma', nargs='?', type=str, default='1.0', 
                        help='depth gamma value')
    parser.add_argument('--min_epoch', nargs='?', type=int, default=100, 
                        help='minumum epoch before first decay')
    args = parser.parse_args()
    if torch.cuda.is_available():
        train(args)
    else:
        print('No CUDA device found!')
