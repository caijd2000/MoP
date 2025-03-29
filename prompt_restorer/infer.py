import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from resnet.resnet import resnet18
from encoder import get_model
from datapipe.data import CarvanaDataset
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import numpy as np

dir_img_train = Path(r'train.txt')
dir_img_val = Path(r'val.txt')
dir_img_test = Path(r'test.txt')

# encoder_list = ["plip", "uni", "conch", "gigapath"]
encoder_type = "uni"

if encoder_type=="plip":
    model1, _ = get_model(encoder_type=encoder_type)
else:
    model1 = get_model(encoder_type=encoder_type)
model1.to("cuda")

model2 = resnet18(num_classes=1, pretrained=False).to("cuda")


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        return self.out_linear(attention_output)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim_a, embed_dim_b):
        super(CrossAttention, self).__init__()
        self.q_linear = nn.Linear(embed_dim_a, embed_dim_b)
        self.k_linear = nn.Linear(embed_dim_b, embed_dim_b)
        self.v_linear = nn.Linear(embed_dim_b, embed_dim_b)
        self.out_linear = nn.Linear(embed_dim_b, embed_dim_b)

    def forward(self, x_a, x_b):
        q = self.q_linear(x_a)
        k = self.k_linear(x_b)
        v = self.v_linear(x_b)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (x_a.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        return self.out_linear(attention_output)


class CustomBlock(nn.Module):
    def __init__(self, embed_dim_a, embed_dim_b):
        super(CustomBlock, self).__init__()
        self.self_attention = SelfAttention(embed_dim_b)
        self.cross_attention = CrossAttention(embed_dim_a, embed_dim_b)
        self.norm1 = nn.LayerNorm(embed_dim_b)
        self.norm2 = nn.LayerNorm(embed_dim_b)

    def forward(self, x_a, x_b):
        self_att_output = self.self_attention(x_b)
        x_b = x_b + self_att_output
        cross_att_output = self.cross_attention(x_a, x_b)
        x_b = x_b + cross_att_output

        return x_b


class StackedAttentionNetwork(nn.Module):
    def __init__(self, embed_dim_a=512, embed_dim_b=512, num_blocks=2):
        super(StackedAttentionNetwork, self).__init__()
        self.blocks = nn.ModuleList([CustomBlock(embed_dim_a, embed_dim_b) for _ in range(num_blocks)])
        self.out = nn.Linear(embed_dim_b,embed_dim_b)

    def forward(self, x_a, x_b):
        for block in self.blocks:
            x_b = block(x_a, x_b)
        return x_b


num_epochs = 500
image_size = 1024
image_scale = 0.25
batch_size = 1
transform2 = transforms.Compose([
    transforms.ToTensor()
])
loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
val_set = CarvanaDataset(img_size = image_size, images_dir=dir_img_test, scale = image_scale,transform=transform2,modes = 'val')
val_loader = DataLoader(val_set, shuffle=True, drop_last=True,**loader_args)

cross_attention = StackedAttentionNetwork().to('cuda')
cross_attention.load_state_dict(torch.load("./ckpts/best_model.pth"))

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

result_dir = "./result"
tissue_path = os.path.join(result_dir, "tissue")
focus_path = os.path.join(result_dir, "focus")
gt_path = os.path.join(result_dir, "tissue_gt")

os.makedirs(result_dir, exist_ok=True)
os.makedirs(tissue_path, exist_ok=True)
os.makedirs(focus_path, exist_ok=True)
os.makedirs(gt_path, exist_ok=True)

for lq, gt, name in tqdm(val_loader):
    lq = lq.to('cuda')
    gt = gt.to('cuda')
    with torch.no_grad():
        input1 = transform(lq)
        input1 = input1.to("cuda")
        input_gt = transform(gt)
        input_gt = input_gt.to("cuda")
        vec1_val = model1.get_image_features(input1)
        vec2_val = model2(lq)
        vec_gt_val = model1.get_image_features(input_gt)

        vec1_val = vec1_val.to('cuda')
        vec_gt_val = vec_gt_val.to('cuda')
        vec2_val = vec2_val.to('cuda')

        combined_vec = cross_attention(vec2_val, vec1_val)
        torch.save(vec_gt_val, os.path.join(gt_path, "{}_gt.pth".format(name[0])))
        torch.save(combined_vec, os.path.join(tissue_path, "{}.pth".format(name[0])))
        torch.save(vec2_val, os.path.join(focus_path, "{}.pth".format(name[0])))


