from cProfile import label
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def evaluate(net, dataloader, device,batch_size):
    net.eval()
    net.freeze_bn()
    num_val_batches = len(dataloader)
    loss_mse = torch.nn.MSELoss()
    loss_all = 0
    num_all = 0
    dis_all = []
    ctf_all = []
    loss_l1 = torch.nn.L1Loss()
    ok_num = 0
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, score,dis = batch['image'],batch['score'],batch['dis']
        image = image.to(device=device, dtype=torch.float32)
        score=score.to(device=device,dtype=torch.float)
        dis=dis.to(device=device,dtype=torch.float)
        with torch.no_grad():
            out_ctf,pre_dis= net(image)
            
            
            num_all += image.shape[0]
            
            for i in range(pre_dis.shape[0]):
                if abs(pre_dis[i]-dis[i])<0.5:
                    ok_num+=1
                dis_all.append(loss_l1(pre_dis[i],dis[i]).item())
                ctf_all.append(loss_l1(out_ctf[i],score[i]).item())
            
    
    net.train()
    #net.freeze_bn()
    print('z_acc:',ok_num/num_all)
    return ctf_all,dis_all
    
    



