import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import re


class BasicDataset(Dataset):
    def __init__(self, img_size: int, images_dir: str, scale: float = 1.0, transform=None, modes='train'):
        self.modes = modes
        self.images_dir = images_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.transform = transform
        self.img_size = img_size
        self.images = []
        self.idx = {}
        i = 0
        with open(self.images_dir) as f:
            for line in f:
                lst = line.split(' ')
                name_now = lst[0][:-7]

                if name_now not in self.images:
                    self.images.append(name_now)
                    self.idx[name_now] = i % 14
                    i += 1
        # print(self.images)
        # print(len(self.images))

        if not self.images:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.images)} examples')

    def __len__(self):
        return len(self.images)

    @classmethod
    def preprocess(cls, img_size, pil_img, scale, transform):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        pil_img = transform(pil_img)

        return pil_img

    def random_crop(self, img, label):
        _, h, w = img.shape

        # print(h,w)
        th, tw = 256, 256
        wrange = int((w - tw) / 32)
        hrange = int((h - th) / 32)
        x1 = random.randint(0, wrange)
        y1 = random.randint(0, hrange)
        img = img[:, y1 * 32:y1 * 32 + th, x1 * 32:x1 * 32 + tw]
        label = label[:, y1 * 32:y1 * 32 + th, x1 * 32:x1 * 32 + tw]
        return img, label, x1, y1

    def center_crop(self, img, label, crop_size=(256,256)):
        h, w = img.shape[1], img.shape[2]
        # print(h, w)
        # print((w-crop_size[0])/2)

        bottom = int((h-crop_size[0])/2)
        top = int((h+crop_size[0])/2)
        left = int((w-crop_size[1])/2)
        right = int((w+crop_size[1])/2)
        # print(left,right,bottom,top)
        img = img[:,  bottom: top, left: right]
        label = label[:,  bottom: top, left: right]
        return img, label

    def crop(self, img, x1, y1):
        th, tw = 8, 8
        img = img[:, y1:y1 + th, x1:x1 + tw]
        return img

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        elif ext in ['.jpg', '.png']:
            return Image.open(filename)

    def __getitem__(self, idx):
        # name,val,dis = self.images[idx][0],self.images[idx][1],self.images[idx][2]
        name = self.images[idx]  # ...xx_xx
        lst = re.split('[/\\\\]', name)
        sample = lst[-2]
        name_last = lst[-1]
        # 随机抽取0-8的一个数,且不等于i
        if self.modes == 'train':
            i = random.randint(0, 14)
            # i = 6
        else:
            # i = 3
            i = self.idx[name]
        if i < 10:
            name_i = name_last + '_0' + str(i) + '.jpg'
        else:
            name_i = name_last + '_' + str(i) + '.jpg'
        # name_j = name[:-5]+str(j)+name[-4:]
        # 改成Z_xx_xx.jpg
        name_j = name_last[:-5] + "Z_" + name_last[-5:] + '.jpg'
        name = name_i[:-4]
        name_i = os.path.join('/public_bme/data/v-caijd/20221207', sample, name_i)
        name_j = os.path.join('/public_bme/data/v-caijd/20221207', sample, name_j)

        # print(name_i)
        # print(name_j)
        try:
            img_i = self.load(name_i)
        except:
            name_i = os.path.join('/public_bme/data/v-caijd/20221207', sample, name_last + '_00.jpg')
            img_i = self.load(name_i)
        img_j = self.load(name_j)

        img_i = self.preprocess(img_size=self.img_size, pil_img=img_i, scale=self.scale, transform=self.transform)

        img_j = self.preprocess(img_size=self.img_size, pil_img=img_j, scale=self.scale, transform=self.transform)

        # get depth
        # 多个分割符split /\

        # lst = re.split('[/\\\\]', name)
        # sample = lst[-2]
        # name_dep = os.path.join('D:/mohu/quality_evalute/all_depthmap',sample,"Z_"+name[-5:]+'.npy')
        # print(name_dep)
        # 读 npy
        # dep = np.load(name_dep)

        # dep = cv.resize(dep-8,(1024,768))
        # dep = Image.fromarray(dep)
        # dep = self.load(name_dep)
        # dep = self.preprocess(img_size=self.img_size,pil_img=dep, scale=4,transform=self.transform)
        # dep = torch.tensor(dep)
        # print(sample+'_'+lst[-1]+'_0'+str(i)+'.jpg')
        # print("1", img_i.shape)
        if self.modes == 'train':
            img_i, img_j, x1, y1 = self.random_crop(img_i, img_j)
        else:
            img_i, img_j = self.center_crop(img_i, img_j)
        # print("2", img_i.shape)
        img_i = (img_i - torch.min(img_i)) / (torch.max(img_i) - torch.min(img_i)+1e-6)
        ori_img = img_i.clone()
        ori_np = (ori_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gray_np = cv.cvtColor(ori_np, cv.COLOR_RGB2GRAY)
        threshold1 = 100
        threshold2 = 200
        edge = cv.Canny(ori_np, threshold1, threshold2)
        edge_tensor = torch.Tensor(edge).unsqueeze(0)
        edge_tensor = (edge_tensor - torch.min(edge_tensor)) / (torch.max(edge_tensor) - torch.min(edge_tensor) + 1e-6)
        # img_j = (img_j - torch.min(img_j))/(torch.max(img_j) - torch.min(img_j))
        return img_i, img_j, name_i.split('/')[-2]+name_i.split('/')[-1]


class CarvanaDataset(BasicDataset):
    def __init__(self, img_size, images_dir, scale=1, transform=None, modes='train'):
        super().__init__(img_size, images_dir, scale, transform=transform, modes=modes)

