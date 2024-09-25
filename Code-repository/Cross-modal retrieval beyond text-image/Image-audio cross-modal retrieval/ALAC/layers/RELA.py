import torch
import torch.nn as nn

'''
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
'''
def conv3x3(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Location_RELA(nn.Module):
    def __init__(self):
        super(Location_RELA, self).__init__()

        self.inplanes = 384
        self.feature1 = nn.Sequential(
            conv1x1(self.inplanes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.feature2 = nn.Sequential(
            conv1x1(self.inplanes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.feature3 = nn.Sequential(
            conv1x1(self.inplanes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

    def mask(self, feature, x):
        
        with torch.no_grad():
            cam1 = feature.mean(1)
            attn = torch.softmax(cam1.view(x.shape[0], x.shape[2]), dim=1)  # B,H,W
            std, mean = torch.std_mean(attn)
            attn = (attn - mean) / (std ** 0.3) + 1  # 0.15
            attn = (attn.view((x.shape[0], 1, x.shape[2]))).clamp(0, 2)
            
            
        '''
        cam1 = feature.mean(1)
        attn = torch.softmax(cam1.view(x.shape[0], x.shape[2]), dim=1)  # B,H,W
        std, mean = torch.std_mean(attn)
        attn = (attn - mean) / (std ** 0.3) + 1  # 0.15
        attn = (attn.view((x.shape[0], 1, x.shape[2]))).clamp(0, 2)
        '''
        return attn

    def forward(self, av):
        av1 = av.permute(0, 2, 1)
        
        fea1 = self.feature1(av1)  # bs*512*7*7
        attn1 = 2 - self.mask(fea1, av1)

        av2 = av1.mul(attn1.repeat(1, self.inplanes, 1))
        
        fea2 = self.feature2(av2)
        attn2 = 2 - self.mask(fea2, av2)

        av3 = av2.mul(attn2.repeat(1, self.inplanes, 1))
        
        fea3 = self.feature3(av3)

        av = torch.cat([fea1, fea2, fea3], dim=1)
        return av.permute(0, 2, 1)
        
