import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from CBilinearPooling import CompactBilinearPooling
from torch.nn.utils import spectral_norm
def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

class ModalNorm(nn.Module):
    def __init__(self, dim=128):
        super(ModalNorm, self).__init__()
        self.gama = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        cur_mean = x.mean(dim =[2,3], keepdim=True)
        cur_var = x.var(dim =[2,3], keepdim=True)
        x_hat = (x - cur_mean) / torch.sqrt(cur_var + 1e-6)
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)
        #f = gamma * x_hat + beta
        f = self.gama * x_hat + self.beta
        return f

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes # 1024
        self.ratio = ratio #1024 *(1/4-1/16)
        self.lamb = 0.2
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.CBPooling = CompactBilinearPooling(self.planes,self.planes,4000)
        self.LayerNorm = ModalNorm()
        self.softmax  = nn.Softmax(dim=-1)
        self.MAM = MAM(self.inplanes)
        
        self.LeakyReLU = nn.LeakyReLU(0.2, True)
        self.gap = nn.AdaptiveAvgPool2d((3,3))
        
        self.ReLU = nn.ReLU(inplace=True)
        self.IN = nn.InstanceNorm2d(inplanes, track_running_stats=False)
        self.down_dim = nn.Conv2d(self.inplanes, self.planes, kernel_size=1,bias=False)
        self.snconv1x1_xtheta = snconv2d(in_channels=self.inplanes, out_channels=self.planes, kernel_size=1, stride=1, padding=0,bias=False)
        self.snconv1x1_ytheta = snconv2d(in_channels=self.inplanes, out_channels=self.planes, kernel_size=1, stride=1, padding=0,bias=False)
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.planes+1, self.inplanes, kernel_size=1,bias=False),
                nn.Sigmoid())
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                self.LayerNorm,
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

        self.estimation_Coeff = nn.Sequential(
            nn.Linear(4000, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
            nn.Tanh()
            )

    def reset_parameters(self):
        if self.pooling_type == 'att':
            # kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.apply(weights_init_kaiming)
            self.conv_mask.inited = True

    def forward(self, x, x2):
        # [N, C, 1, 1]
        N, C, h, w = x.size()
        _, _, h1, w1 = x2.size()

        contextA = self.snconv1x1_xtheta(x)
        context_A = contextA.view(N,self.planes,-1)
        context2V = self.snconv1x1_ytheta(x2)
        context_2V = context2V.view(N,self.planes,-1)#
        
        contextA = self.gap(contextA).view(N,self.planes,-1)
        context2V = self.gap(context2V).view(N,self.planes,-1)
        
        context = contextA / torch.norm(contextA, 2, 0, True)
        context2 = context2V / torch.norm(context2V, 2, 0, True)
        
        Att_cross = self.CBPooling(context, context2)#B*8000
        ECoefficient = self.estimation_Coeff(Att_cross)
        ECoef = ECoefficient
        shape_coefA= torch.ones(N,1,h*w).cuda()
        shape_coefV= torch.ones(N,1,h1*w1).cuda()
        ECoef = ECoef.unsqueeze(1)
        ECoefxA =torch.bmm(ECoef,shape_coefA)
        ECoefxV =torch.bmm(ECoef,shape_coefV)
        
        Cross_attention = torch.bmm(context, context2.permute(0,2,1))

        Cross_attention = self.softmax(Cross_attention)
        context_cross = torch.bmm(Cross_attention, context_A)#B*C+1*hw
        Label_context_cross = torch.cat([ECoefxA,context_cross],dim=1).view(N,self.planes+1, h, w)
        context2_cross = torch.bmm(Cross_attention, context_2V)#B*C+1*hw
        Label_context2_cross = torch.cat([ECoefxV,context2_cross],dim=1).view(N,self.planes+1, h1, w1)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context_cross))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(Label_context_cross)
            channel_add_term2 = self.channel_add_conv(Label_context2_cross)
            Aout = self.MAM(x, channel_add_term)
            Vout2 = self.MAM(x2, channel_add_term2)
        return Aout, Vout2, ECoefficient
 
  
class MAM(nn.Module):
    def __init__(self, dim, r=4):
        super(MAM, self).__init__()
        self.channel_attention = nn.Sequential(
                nn.Conv2d(dim, dim // r, kernel_size=1, bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, inp, x):
        Avg_pooled = F.avg_pool2d(x, x.size()[2:])
        Max_pooled = F.max_pool2d(x, x.size()[2:])
         
        Avg_mask = self.channel_attention(Avg_pooled)
        Max_mask = self.channel_attention(Max_pooled)
        
        mask = Avg_mask + Max_mask

        output = inp * mask + self.IN(x) * (1 - mask)
        return output     

