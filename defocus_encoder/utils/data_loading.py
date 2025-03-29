import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, img_size:int,images_dir: str,  scale: float = 1.0,transform=None):
        self.images_dir = images_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.transform = transform
        self.img_size=img_size
        self.images = []
        with open(self.images_dir) as f:
            for line in f:
                lst = line.split(' ')
                self.images.append(lst)
        #print(self.images)
        #print(len(self.images))
        
        if not self.images:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.images)} examples')

    def __len__(self):
        return len(self.images)

    @classmethod
    def preprocess(cls, img_size,pil_img, scale,transform):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample = Image.BICUBIC)
        pil_img = transform(pil_img)
        
        return pil_img

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        elif ext in ['.jpg','.png']:
            return Image.open(filename)
        
    def __getitem__(self, idx):
        name,val,dis = self.images[idx][0],self.images[idx][1],self.images[idx][2]
        if val[-2:]=="\n":
            val = eval(val[:-2])
        else:
            val = eval(val)
        if dis[-2:]=="\n":
            dis = eval(dis[:-2])
        else:
            dis = eval(dis)
        
        # dis = abs(dis)
        val = [float(val)]
        distance = [float(dis)]
        
        img = self.load(name)
        
        img1 = self.preprocess(img_size=self.img_size,pil_img=img, scale=self.scale,transform=self.transform)
        
        return {
            'image': img1,
            'score':torch.tensor(val),
            'dis':torch.tensor(distance)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self,img_size,images_dir, scale=1,transform=None):
        super().__init__(img_size,images_dir, scale,transform=transform)

