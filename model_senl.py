import torch
import torch.nn as nn
from torch.nn import init
import math
from resnet import resnet50, resnet18
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter

class _BatchAttNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_BatchAttNorm, self).__init__(num_features, eps, momentum, affine)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias_readjust = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust.data.fill_(0)
        self.bias_readjust.data.fill_(-1)
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        attention = self.sigmoid(self.avg(input) * self.weight_readjust + self.bias_readjust)
        bn_w = self.weight * attention

        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training, self.momentum, self.eps)
        out_bn = out_bn * bn_w + self.bias

        return out_bn

class BAN2d(_BatchAttNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

# import torch.functional as F
# from attention import PyramidAttention,PyramidAttention2,PyramidAttention3,PYNL,PYNL2,CPYNL

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class MAM(nn.Module):
    def __init__(self, dim, r=16):
        super(MAM, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, x):
        pooled = F.avg_pool2d(x, x.size()[2:])
        mask = self.channel_attention(pooled)
        x = x * mask + self.IN(x) * (1 - mask)

        return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class GeP(nn.Module):
    def __init__(self,p=3.0, eps=1e-12):
        super(GeP, self).__init__()
        self.p=p
        self.eps=eps
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = (torch.mean(x ** self.p, dim=-1) + self.eps**self.p) ** (1 / self.p)
        return x

class res50navie(nn.Module):
    def __init__(self):
        super(res50navie, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)

        self.conv1 = model_base.conv1
        self.bn1 = model_base.bn1
        self.relu = model_base.relu
        self.maxpool = model_base.maxpool
        self.layer1 = model_base.layer1
        self.layer2 = model_base.layer2
        self.layer3 = model_base.layer3
        self.layer4 = model_base.layer4



    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x) # 3
        x = self.layer2(x)#512,4
        x = self.layer3(x)#1024,6
        x = self.layer4(x)#2048,3


        return x

class single50grayenhance(nn.Module):
    def __init__(self, class_num=0):
        super(single50grayenhance, self).__init__()

        self.smodel = res50navie()
        self.pool = GeP(p=3.0,eps=1e-6)  # eps=1e-6


        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim,affine=True)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


        self.cls = nn.Linear(pool_dim, class_num, bias=False)
        self.cls.apply(weights_init_classifier)



    def forward(self, x1, x2, x3, mid=0):  #x1-rgb, x2-gray, x3-nir

        if self.training:
            x = torch.cat((x1, x2, x3), dim=0)
        else:
            x = x1

        x = self.smodel(x)
        pool = self.pool(x).view(x.size(0), -1)
        feat = self.bottleneck(pool)



        if self.training:
            return pool, feat, self.cls(feat)
        else:
            return pool+feat, F.normalize(pool,p=2,dim=1) + F.normalize(feat,p=2,dim=1)

class single50navie(nn.Module):
    def __init__(self, class_num=0):
        super(single50navie, self).__init__()

        self.smodel = res50navie()
        self.pool = GeP(p=3.0,eps=1e-6)  # eps=1e-6


        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim,affine=True)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


        self.cls = nn.Linear(pool_dim, class_num, bias=False)
        self.cls.apply(weights_init_classifier)



    def forward(self, x1, x2, x3=None, mid=0):  #x1-rgb, x2-gray, x3-nir

        if self.training:
            x = torch.cat((x1, x2), dim=0)
        else:
            x = x1

        x = self.smodel(x)
        pool = self.pool(x).view(x.size(0), -1)
        feat = self.bottleneck(pool)



        if self.training:
            return pool, feat, self.cls(feat)
        else:
            return pool+feat, F.normalize(pool,p=2,dim=1) + F.normalize(feat,p=2,dim=1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SENL(nn.Module):# se on nl
    def __init__(self, in_channels, reduc_ratio=16):
        super(SENL, self).__init__()
        print('...... senl welcome you......: in_channels=',in_channels)
        self.in_channels = in_channels
        print('in_channels')
        print(in_channels)
        self.inter_channels = in_channels//reduc_ratio
        print('in_channels//reduc_ratio')
        print(in_channels // reduc_ratio)
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0),
            # nn.Dropout2d(p=0.05),
        )

        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            # nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(h,w),padding=0,bias=False,stride=1,groups=in_channels),
            nn.Conv2d(in_channels=in_channels,out_channels=self.inter_channels,kernel_size=1,stride=1,padding=0,bias= False),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.inter_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0,bias= False),
            nn.Sigmoid()
        )


    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        # print('x.size(0)')
        # print(x.size(0))
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        # print('g_x.shape')
        # print(g_x.shape)
        g_x = g_x.permute(0, 2, 1)
        # print('g_x.shape2')
        # print(g_x.shape)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        # print('theta_x')
        # print(theta_x.shape)
        theta_x = theta_x.permute(0, 2, 1)
        # print('theta_x2')
        # print(theta_x.shape)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # print('phi_x')
        # print(phi_x.shape)
        f = torch.matmul(theta_x, phi_x)
        # print('f.shape')
        # print(f.shape)
        N = f.size(-1)
        # print('N')
        # print(N)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N
        # print('f_div_C')
        # print(f_div_C.shape)

        y = torch.matmul(f_div_C, g_x)
        # print('y1')
        # print(y.shape)
        y = y.permute(0, 2, 1).contiguous()
        # print('y2')
        # print(y.shape)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])#nxc1xhxw
        # print('y3')
        # print(y.shape)

        W_y = self.W(y)#nxcxhxw
        # print('W_y')
        # print(W_y)
        # print(W_y.shape)
        W_y = self.se(W_y)*W_y
        # print('W_y')
        # print(W_y.shape)
        z = W_y + x
        # print('z')
        # print(z.shape)

        att = W_y.view(W_y.size(0),W_y.size(1),-1)# optimize attention
        # print('att')
        # print(att.shape)
        att = att.sum(1)
        # print('attsum')
        # print(att.shape)

        return z, att

class backbone50_senl(nn.Module):
    def __init__(self,layers=[3, 4, 6, 3], att_layers=[0, 2, 3, 0], atttype='senl'):
        super(backbone50_senl, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)

        self.conv1 = model_base.conv1
        self.bn1 = model_base.bn1
        self.relu = model_base.relu
        self.maxpool = model_base.maxpool
        self.layer1 = model_base.layer1
        self.layer2 = model_base.layer2
        self.layer3 = model_base.layer3
        self.layer4 = model_base.layer4

        self.ATT_1 = nn.ModuleList(
            [SENL(in_channels=256, reduc_ratio=16) for i in range(att_layers[0])])
        self.ATT_1_idx = sorted([layers[0] - (i + 1) for i in range(att_layers[0])])
        self.ATT_2 = nn.ModuleList(
            [SENL(in_channels=512, reduc_ratio=16) for i in range(att_layers[1])])
        self.ATT_2_idx = sorted([layers[1] - (i + 1) for i in range(att_layers[1])])
        self.ATT_3 = nn.ModuleList(
            [SENL(in_channels=1024, reduc_ratio=16) for i in range(att_layers[2])])
        self.ATT_3_idx = sorted([layers[2] - (i + 1) for i in range(att_layers[2])])
        self.ATT_4 = nn.ModuleList(
            [SENL(in_channels=2048, reduc_ratio=16) for i in range(att_layers[3])])
        self.ATT_4_idx = sorted([layers[3] - (i + 1) for i in range(att_layers[3])])

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        atts =[]

        NL1_counter = 0
        if len(self.ATT_1_idx) == 0: self.ATT_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.ATT_1_idx[NL1_counter]:
                x, att = self.ATT_1[NL1_counter](x)
                atts.append(att)
                NL1_counter += 1

        # Layer 2
        NL2_counter = 0
        if len(self.ATT_2_idx) == 0: self.ATT_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.ATT_2_idx[NL2_counter]:
                x, att = self.ATT_2[NL2_counter](x)
                atts.append(att)
                NL2_counter += 1


        # Layer 3
        NL3_counter = 0
        if len(self.ATT_3_idx) == 0: self.ATT_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.ATT_3_idx[NL3_counter]:
                x, att = self.ATT_3[NL3_counter](x)
                atts.append(att)
                NL3_counter += 1


        # Layer 4
        NL4_counter = 0
        if len(self.ATT_4_idx) == 0: self.ATT_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.ATT_4_idx[NL4_counter]:
                x, att = self.ATT_4[NL4_counter](x)
                atts.append(att)
                NL4_counter += 1


        return x, atts

class single50_senl(nn.Module):
    def __init__(self, class_num=150, attloc=[0, 1, 1, 0],atttype='senl'):
        super(single50_senl, self).__init__()



        self.smodel = backbone50_senl(att_layers=attloc,atttype=atttype)
        self.pool = GeP(p=3.0,eps=1e-6)  # eps=1e-6

        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim,affine=True)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


        self.cls = nn.Linear(pool_dim, class_num, bias=False)
        self.cls.apply(weights_init_classifier)



    def forward(self, x1, x2, mid=0):

        if self.training:
            x = torch.cat((x1, x2), dim=0)
            x, atts = self.smodel(x)
        else:
            x = x1
            x, atts = self.smodel(x)


        pool = self.pool(x).view(x.size(0), -1) #+ self.pool2(x).view(x.size(0), -1)

        feat = self.bottleneck(pool)


        if self.training:
            f1 = []
            f2 = []

            for i in range(len(atts)):
                cur_att = atts[i]
                f1.append(cur_att[0:feat.size(0) // 2, :])
                f2.append(cur_att[feat.size(0) // 2:, :])
            # print('f1')
            # print(f1)
            # print('f2')
            # print(f2)
            return f1, f2, pool, feat, self.cls(feat) #self.mcls(mx), self.mcls(feat)
        else:
            return pool+feat, F.normalize(pool,p=2,dim=1) + F.normalize(feat,p=2,dim=1)