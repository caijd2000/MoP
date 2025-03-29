import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision import utils as vutils

#from positionembedding import Position_Embedding
#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import torch
import cv2
import numpy as np
 
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class FC(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.fc=torch.nn.Linear(in_channels,out_channels)
    def forward(self, x):
        return self.fc(x)



class GMP(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,H,W):
        super().__init__()
        self.gmp=torch.nn.AdaptiveMaxPool2d((H,W))
    def forward(self, x):
        out=self.gmp(x)
        return out.permute(0,2,3,1)

class GAP(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,H,W):
        super().__init__()
        self.gap=torch.nn.AdaptiveAvgPool2d((H,W))
    def forward(self, x):
        out=self.gap(x)
        return out
        


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,padding_mode='replicate', bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2])
        for m in self.modules():
            m.requires_grad_(False)
        self.layer4 = self._make_layer(block, 512, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.sigmoid=torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(512,1)
        self.fc2= torch.nn.Linear(512,1)
        self.softmax=torch.nn.Softmax(dim=1)
        self.gap = GAP(1,1)
        self.dropout=nn.Dropout(0.5)
        self.flatten = nn.Flatten(start_dim=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        
        #outputs=[]#
        img_batch = inputs

        x = self.conv1(img_batch)#512
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#256
        
        #outputs.append(x)#
        x1 = self.layer1(x)#256
        x2 = self.layer2(x1)#128
        x3 = self.layer3(x2)#64
        x4 = self.layer4(x3)#b*2048*32*32

        # out2 = x4
        out =self.gap(x4)
        out = self.flatten(out)
        # out1=self.fc(out)
        # dis=self.fc2(out)
        
        #dis=self.softmax(dis)
        
        return out
def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    model.load_state_dict(torch.load("./resnet/grader.pth"))
    return model