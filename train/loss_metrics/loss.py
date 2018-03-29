import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def cross_entropy2d_ignore(input, target, ignore_id, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False, ignore_index=ignore_id)
    if size_average:
        loss /= mask.data.sum()
    return loss

def cross_entropy2d_mask(input,target,outside_mask_classes,weight=None,size_average=True):
    n,c,h,w=input.size()
    log_p=F.log_softmax(input)
    mask=target[0]!=outside_mask_classes[0]
    if len(outside_mask_classes)>1:
        for x in range(1,len(outside_mask_classes)):
            tmpmask=target[0]!=outside_mask_classes[x]
            mask=(mask+tmpmask)==2
    if mask.data.sum()>0:
        target_final=target[0][mask]
        log_p_final=torch.unsqueeze(log_p[0][0][mask],1)
        if c>1:
            for cc in range(1,c):
                log_p_tmp=torch.unsqueeze(log_p[0][cc][mask],1)
                log_p_final=torch.cat((log_p_final,log_p_tmp),1)
    else:
        target_final=Variable(torch.LongTensor(0).cuda(0))
        log_p_final=Variable(torch.LongTensor(0,0).cuda(0))
    if n>1:
        for idx in range(1,n):
            mask=target[idx]!=outside_mask_classes[0]
            if len(outside_mask_classes)>1:
                for x in range(1,len(outside_mask_classes)):
                    tmpmask=target[idx]!=outside_mask_classes[x]
                    mask=(mask+tmpmask)==2
            if mask.data.sum()>0:
                target_final_tmp=target[idx][mask]
                log_p_final_tmp=torch.unsqueeze(log_p[idx][0][mask],1)
                if c>1:
                    for cc in range(1,c):
                        log_p_tmp=torch.unsqueeze(log_p[idx][cc][mask],1)
                        log_p_final_tmp=torch.cat((log_p_final_tmp,log_p_tmp),1)
                if not log_p_final.size():
                    target_final=target_final_tmp
                    log_p_final=log_p_final_tmp
                else:
                    target_final=torch.cat((target_final,target_final_tmp),0)
                    log_p_final=torch.cat((log_p_final,log_p_final_tmp),0)
    if not log_p_final.size():
        loss=Variable(torch.from_numpy(np.array([0.0])).float().cuda(0))
        doback=0
        n_pix=0
    else:
        loss=F.nll_loss(log_p_final,target_final,weight=weight,size_average=False,ignore_index=outside_mask_classes[0])
        if size_average:
            loss/=log_p_final.size(0)
        doback=1
        n_pix=log_p_final.size(0)
    return loss,doback,n_pix
