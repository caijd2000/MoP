import argparse
import logging
import sys
import math
from pathlib import Path
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import cv2 as cv
from utils.data_loading_stainaug import BasicDataset, CarvanaDataset
from evaluate import evaluate
from resnet import  resnet as resnet
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from skimage import io
from torch.utils.tensorboard import SummaryWriter 
import random
import numpy as np

seed=2023
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.	
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



writer = SummaryWriter('./log/tensor')


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))





#normal training
dir_img_train = Path(r'data_txt/train_v4.txt')
dir_img_val = Path(r'data_txt/val_v4.txt')
dir_img_test = Path(r'data_txt/test_v4.txt')
dir_checkpoint = Path(r'checkpoints/')


transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()#,
])


transform2 = transforms.Compose([
    transforms.ToTensor()
])


def train_net(net,
              device,
              epochs: int = 50,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1,
              amp: bool = False):
    img_size=1024
    img_scale=1
    # 1. Create dataset
    try:
        train_set = CarvanaDataset(img_size,dir_img_train, img_scale,transform=transform1,modes='train')
        val_set = CarvanaDataset(img_size,dir_img_val, img_scale,transform=transform2,modes='val')
        test_set = CarvanaDataset(img_size,dir_img_test, img_scale,transform=transform2,modes='val')
    except (AssertionError, RuntimeError):
        train_set = BasicDataset(img_size,dir_img_train, img_scale,transform=transform1,modes='train')
        val_set = BasicDataset(img_size,dir_img_val, img_scale,transform=transform2,modes='val')
        test_set = CarvanaDataset(img_size,dir_img_test, img_scale,transform=transform2,modes='val')
    
    n_val=len(val_set)
    n_train=len(train_set)
    
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,**loader_args)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=True, drop_last=True, **loader_args)

    optimizer = optim.Adam(filter(lambda p:p.requires_grad,net.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_l1 = torch.nn.L1Loss()
    loss_mse = torch.nn.MSELoss()
    loss_ce = torch.nn.CrossEntropyLoss()
    global_step = 0
    
    
    for epoch in range(epochs):
        print("lr:",optimizer.param_groups[0]['lr'])
        net.train()
        #net.eval()
        net.freeze_bn()
        epoch_loss = 0
        epoch_num = 0
        
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch_idx,batch in enumerate(train_loader):
                images = batch['image']
                score = batch['score']
                dis = batch['dis']
                loss = 0
                
                optimizer.zero_grad(set_to_none=True)
                images = images.to(device=device, dtype=torch.float32)
                score=score.to(device=device, dtype=torch.float32)
                dis=dis.to(device=device, dtype=torch.float32)
                with torch.cuda.amp.autocast(enabled=amp):
                    out_ctf,out_dis = net(images)
                    
                    loss1 = loss_l1(out_dis,dis) 
                    loss2 = loss_l1(out_ctf,score)
                    loss =  (loss1+loss2)/2
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()*images.shape[0]
                    epoch_num += images.shape[0]
                pbar.set_postfix(**{'loss(batch)': loss.item(),'loss1(batch)': loss1.item(),'loss2(batch)': loss2.item()})
                #pbar.set_postfix(**{'loss(batch)': loss.item()})
                # Evaluation round
                division_step = (n_train // (batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        print(epoch,":")
                        print('train_score:{}'.format(epoch_loss/epoch_num) )
                        ctf_all,dis_all = evaluate(net, val_loader, device,batch_size)
                        print("eval score:",np.mean(ctf_all),np.std(ctf_all),np.mean(dis_all) , np.std(dis_all))
                        scheduler.step()
                        
                        ctf_all,dis_all = evaluate(net, test_loader, device,batch_size)
                        print("test score:",np.mean(ctf_all),np.std(ctf_all),np.mean(dis_all) , np.std(dis_all))
                        

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    
    net = resnet.resnet18(num_classes=1, pretrained=False)
    resnet_dict = torch.load('pretrained_model/resnet18-5c106cde.pth')
    
    net.load_state_dict(resnet_dict,strict=False)
    net.freeze_bn()
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    if torch.cuda.is_available():
        net.cuda()
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
