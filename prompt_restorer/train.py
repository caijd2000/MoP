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
from torchvision.transforms import Normalize

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
image_scale = 1
batch_size = 16
transform2 = transforms.Compose([
    transforms.ToTensor()
])
loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
train_set = CarvanaDataset(img_size = image_size, images_dir=dir_img_train, scale = image_scale,transform=transform2,modes = 'train')
val_set = CarvanaDataset(img_size = image_size, images_dir=dir_img_val, scale = image_scale,transform=transform2,modes = 'val')
train_loader = DataLoader(train_set, shuffle=True, drop_last=True,**loader_args)
val_loader = DataLoader(val_set, shuffle=True, drop_last=True,**loader_args)

cross_attention = StackedAttentionNetwork().to('cuda')
# cross_attention.load_state_dict(torch.load("./ckpts/latest_model.pth"))
optimizer = optim.Adam(cross_attention.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=100, gamma=0.95)
criterion = nn.L1Loss()

os.makedirs('./ckpts', exist_ok=True)

best_loss = float('inf')
best_epoch = 0
total_iter = 0

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])


for epoch in range(num_epochs):
    train_loss_list = []
    for lq, gt, _ in tqdm(train_loader):
        lq = lq.to('cuda')
        gt = gt.to('cuda')

        with torch.no_grad():
            input1 = transform(lq)
            input1 = input1.to("cuda")
            input_gt = transform(gt)
            input_gt = input_gt.to("cuda")
            vec1 = model1.get_image_features(input1)
            vec2 = model2(lq)
            vec_gt = model1.get_image_features(input_gt)

        combined_vec = cross_attention(vec2, vec1)
        loss = criterion(combined_vec, vec_gt)
        train_loss_list.append(loss)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_iter += 1

        if total_iter % 100 == 0:
            val_loss_list = []
            for lq, gt, _ in tqdm(val_loader):
                lq = lq.to('cuda')
                gt = gt.to('cuda')
                with torch.no_grad():
                    input1_val = transform(lq)
                    input1_val = input1_val.to("cuda")
                    input_gt_val = transform(gt)
                    input_gt_val = input_gt_val.to("cuda")

                    vec1_val = model1.get_image_features(input1_val)
                    vec2_val = model2(lq)
                    vec_gt_val = model1.get_image_features(input_gt_val)

                    vec1_val = vec1_val.to('cuda')
                    vec_gt_val = vec_gt_val.to('cuda')
                    vec2_val = vec2_val.to('cuda')
                    combined_vec = cross_attention(vec2_val, vec1_val)
                    val_loss = criterion(combined_vec, vec_gt_val)
                    val_loss_list.append(val_loss.item())

            avg_val_loss = np.mean(val_loss_list)
            if avg_val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(cross_attention.state_dict(), './ckpt_ca_plip/best_model.pth')
            print(f'Iteration [{total_iter}], Val_Loss: {avg_val_loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Train_Loss: {loss.item():.4f}')
    torch.save(cross_attention.state_dict(), './ckpt_ca_plip/latest_model.pth')
print("Best loss is ", best_loss, f", on epoch {best_epoch}")
print("训练完成，权重已保存。")
