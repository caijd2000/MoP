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
from utils.randstain import RandStainNA as RandStainNA


transforms_single = [
    RandStainNA(
        yaml_file="./CRC_LAB_randomTrue_n0.yaml",
        std_hyper=-0.3,
        probability=0.9,
        distribution="normal",
        is_train=True,
    )
]
aug_sing = transforms.Compose(transforms_single)


class BasicDataset(Dataset):
    def __init__(self, img_size:int,images_dir: str,  scale: float = 1.0,transform=None,modes='train'):
        self.images_dir = images_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.transform = transform
        self.img_size=img_size
        self.modes = modes
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
    def stain_aug(self,img):
        img = np.array(img)
        img = aug_sing(img)
        img = Image.fromarray(img)
        return img
    
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
        
        
        if self.modes == 'train':
            img = self.stain_aug(img)
        
        img1 = self.preprocess(img_size=self.img_size,pil_img=img, scale=self.scale,transform=self.transform)
        img1 = (img1 - torch.min(img1))/(torch.max(img1) - torch.min(img1))
        return {
            'image': img1,
            'score':torch.tensor(val),
            'dis':torch.tensor(distance)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self,img_size,images_dir, scale=1,transform=None,modes='train'):
        super().__init__(img_size,images_dir, scale,transform=transform,modes=modes)

