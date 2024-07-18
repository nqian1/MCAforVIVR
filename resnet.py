import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# class SwitchNorm1d(nn.Module):
#   def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
#     super(SwitchNorm1d, self).__init__()
#     self.eps = eps
#     self.momentum = momentum
#     self.using_moving_average = using_moving_average
#     self.weight = nn.Parameter(torch.ones(1, num_features))
#     self.bias = nn.Parameter(torch.zeros(1, num_features))
#     self.mean_weight = nn.Parameter(torch.ones(2))
#     self.var_weight = nn.Parameter(torch.ones(2))
#     self.register_buffer('running_mean', torch.zeros(1, num_features))
#     self.register_buffer('running_var', torch.zeros(1, num_features))
#     self.reset_parameters()
#
#   def reset_parameters(self):
#     self.running_mean.zero_()
#     self.running_var.zero_()
#     self.weight.data.fill_(1)
#     self.bias.data.zero_()
#
#   def _check_input_dim(self, input):
#     if input.dim() != 2:
#       raise ValueError('expected 2D input (got {}D input)'
#                        .format(input.dim()))
#
#   def forward(self, x):
#     self._check_input_dim(x)
#     mean_ln = x.mean(1, keepdim=True)
#     var_ln = x.var(1, keepdim=True)
#
#     if self.training:
#       mean_bn = x.mean(0, keepdim=True)
#       var_bn = x.var(0, keepdim=True)
#       if self.using_moving_average:
#         self.running_mean.mul_(self.momentum)
#         self.running_mean.add_((1 - self.momentum) * mean_bn.data)
#         self.running_var.mul_(self.momentum)
#         self.running_var.add_((1 - self.momentum) * var_bn.data)
#       else:
#         self.running_mean.add_(mean_bn.data)
#         self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
#     else:
#       mean_bn = torch.autograd.Variable(self.running_mean)
#       var_bn = torch.autograd.Variable(self.running_var)
#
#     softmax = nn.Softmax(0)
#     mean_weight = softmax(self.mean_weight)
#     var_weight = softmax(self.var_weight)
#
#     mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
#     var = var_weight[0] * var_ln + var_weight[1] * var_bn
#
#     x = (x - mean) / (var + self.eps).sqrt()
#     return x * self.weight + self.bias
#
#
# class SwitchNorm2d(nn.Module):
#   def __init__(self, num_features, eps=1e-6, momentum=0.9, using_moving_average=True, using_bn=True,
#                last_gamma=False):
#     super(SwitchNorm2d, self).__init__()
#     self.eps = eps
#     self.momentum = momentum
#     self.using_moving_average = using_moving_average
#     self.using_bn = using_bn
#     self.last_gamma = last_gamma
#     # print(self.last_gamma)
#     self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
#     self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
#     if self.using_bn:
#       self.mean_weight = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))  # nn.Parameter(torch.ones(3))
#       self.var_weight = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))  # nn.Parameter(torch.ones(3))
#     else:
#       self.mean_weight = nn.Parameter(torch.tensor([0.0, 0.0]))  # nn.Parameter(torch.ones(2))
#       self.var_weight = nn.Parameter(torch.tensor([0.0, 0.0]))  # nn.Parameter(torch.ones(2))
#     if self.using_bn:
#       self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
#       self.register_buffer('running_var', torch.zeros(1, num_features, 1))
#
#     self.softmax = nn.Softmax(0)
#
#     self.reset_parameters()
#
#     # print(self.weight)
#
#   def reset_parameters(self):
#     if self.using_bn:
#       self.running_mean.zero_()
#       self.running_var.zero_()
#     if self.last_gamma:
#       # print('xxxxx')
#       self.weight.data.fill_(0)
#     else:
#       # print('yyyyy')
#       self.weight.data.fill_(1)
#     self.bias.data.zero_()
#
#   def _check_input_dim(self, input):
#     if input.dim() != 4:
#       raise ValueError('expected 4D input (got {}D input)'
#                        .format(input.dim()))
#
#   def forward(self, x):
#     self._check_input_dim(x)
#     N, C, H, W = x.size()
#     x = x.view(N, C, -1)
#     mean_in = x.mean(-1, keepdim=True)
#     var_in = x.var(-1, keepdim=True)
#
#     mean_ln = mean_in.mean(1, keepdim=True)
#     temp = var_in + mean_in ** 2
#     var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
#
#     if self.using_bn:
#       if self.training:
#         mean_bn = mean_in.mean(0, keepdim=True)
#         var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
#         if self.using_moving_average:
#           self.running_mean.mul_(self.momentum)
#           self.running_mean.add_((1 - self.momentum) * mean_bn.data)
#           self.running_var.mul_(self.momentum)
#           self.running_var.add_((1 - self.momentum) * var_bn.data)
#         else:
#           self.running_mean.add_(mean_bn.data)
#           self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
#       else:
#         mean_bn = torch.autograd.Variable(self.running_mean)
#         var_bn = torch.autograd.Variable(self.running_var)
#
#     mean_weight = self.softmax(self.mean_weight)
#     var_weight = self.softmax(self.var_weight)
#
#     # print(mean_weight,var_weight)
#     # mean_weight = self.mean_weight/(torch.sum(self.mean_weight)+1e-6)
#     # var_weight = self.var_weight/(torch.sum(self.var_weight)+1e-6)
#
#     if self.using_bn:
#       mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
#       var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
#     else:
#       mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
#       var = var_weight[0] * var_in + var_weight[1] * var_ln
#
#     x = (x - mean) / (var.sqrt() + self.eps)
#     x = x.view(N, C, H, W)
#     return x * self.weight + self.bias
#
#
# class SwitchNorm3d(nn.Module):
#   def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
#                last_gamma=False):
#     super(SwitchNorm3d, self).__init__()
#     self.eps = eps
#     self.momentum = momentum
#     self.using_moving_average = using_moving_average
#     self.using_bn = using_bn
#     self.last_gamma = last_gamma
#     self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
#     self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
#     if self.using_bn:
#       self.mean_weight = nn.Parameter(torch.ones(3))
#       self.var_weight = nn.Parameter(torch.ones(3))
#     else:
#       self.mean_weight = nn.Parameter(torch.ones(2))
#       self.var_weight = nn.Parameter(torch.ones(2))
#     if self.using_bn:
#       self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
#       self.register_buffer('running_var', torch.zeros(1, num_features, 1))
#
#     self.reset_parameters()
#
#   def reset_parameters(self):
#     if self.using_bn:
#       self.running_mean.zero_()
#       self.running_var.zero_()
#     if self.last_gamma:
#       self.weight.data.fill_(0)
#     else:
#       self.weight.data.fill_(1)
#     self.bias.data.zero_()
#
#   def _check_input_dim(self, input):
#     if input.dim() != 5:
#       raise ValueError('expected 5D input (got {}D input)'
#                        .format(input.dim()))
#
#   def forward(self, x):
#     self._check_input_dim(x)
#     N, C, D, H, W = x.size()
#     x = x.view(N, C, -1)
#     mean_in = x.mean(-1, keepdim=True)
#     var_in = x.var(-1, keepdim=True)
#
#     mean_ln = mean_in.mean(1, keepdim=True)
#     temp = var_in + mean_in ** 2
#     var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
#
#     if self.using_bn:
#       if self.training:
#         mean_bn = mean_in.mean(0, keepdim=True)
#         var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
#         if self.using_moving_average:
#           self.running_mean.mul_(self.momentum)
#           self.running_mean.add_((1 - self.momentum) * mean_bn.data)
#           self.running_var.mul_(self.momentum)
#           self.running_var.add_((1 - self.momentum) * var_bn.data)
#         else:
#           self.running_mean.add_(mean_bn.data)
#           self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
#       else:
#         mean_bn = torch.autograd.Variable(self.running_mean)
#         var_bn = torch.autograd.Variable(self.running_var)
#
#     softmax = nn.Softmax(0)
#     mean_weight = softmax(self.mean_weight)
#     var_weight = softmax(self.var_weight)
#
#     if self.using_bn:
#       mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
#       var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
#     else:
#       mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
#       var = var_weight[0] * var_in + var_weight[1] * var_ln
#
#     x = (x - mean) / (var + self.eps).sqrt()
#     x = x.view(N, C, D, H, W)
#     return x * self.weight + self.bias
#
#
# class SiLU(nn.Module):
#   def __init__(self):
#     super().__init__()
#
#   def forward(self, x):
#     x = x * torch.sigmoid(x)
#     return x
#
#
# class Mish(nn.Module):
#   def __init__(self):
#     super().__init__()
#
#   def forward(self, x):
#     x = x * torch.tanh(nn.functional.softplus(x))
#     return x


# class modal_encode(nn.Module):
#     def __init__(self,mid=1):
#         super(modal_encode, self).__init__()
#         # self.conv1= nn.Sequential(
#         #     nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1, bias=False),
#         #     # nn.InstanceNorm2d(4),
#         #     # nn.ReLU()
#         # )
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(4, 4, kernel_size=5, padding=2, stride=1, bias=False),
#         #     # nn.InstanceNorm2d(4),
#         #     # nn.ReLU()
#         # )
#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(4, 4, kernel_size=7, padding=3, stride=1, bias=False),
#         #     # nn.InstanceNorm2d(4),
#         #     # nn.ReLU()
#         # )
#         model_base = resnet50(pretrained=True,
#                               last_conv_stride=1, last_conv_dilation=1)
#         self.conv1 = model_base.conv1
#         self.bn1 = model_base.bn1
#         self.relu = model_base.relu
#         self.maxpool = model_base.maxpool
#
#         self.fuse = nn.Sequential(
#             # nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1, bias=False),
#             # nn.InstanceNorm2d(8),
#             # nn.ReLU(),
#             # nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1, bias=False),
#             # nn.InstanceNorm2d(8),
#             # nn.ReLU(),
#             # nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1, bias=False),
#             # nn.InstanceNorm2d(8),
#             # nn.ReLU(),
#             nn.Conv2d(4, 3, kernel_size=3, padding=1, stride=1, bias=False),
#             nn.BatchNorm2d(3),
#             nn.Sigmoid()
#         )
#
#
#     def forward(self, x_modal,x):
#         # x1 = self.conv1(x_modal)
#         # x2 = self.conv2(x_modal)
#         # x3 = self.conv3(x_modal)
#         # xf = torch.cat((x1,x2,x3),dim=1)
#         x  = self.fuse(x_modal)  + x
#         return x
#

# class encode(nn.Module):
#     def __init__(self,mid=1):
#         super(encode, self).__init__()
#         self.cc1= nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=2)
#         self.cb1= nn.BatchNorm2d(8)
#         self.cr1= nn.ReLU()
#         # self.cm1= nn.AvgPool2d(kernel_size=3,padding=1,stride=2,return_indices=True)
#         # self.cm1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
#
#         # self.cc2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=False)
#         # self.cb2 = nn.BatchNorm2d(16)
#         # self.cr2 = nn.ReLU()
#         # self.cm2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2, return_indices=True)
#         #
#
#         # self.cc3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False)
#         # self.cb3 = nn.BatchNorm2d(32)
#         # self.cr3 = nn.ReLU()
#         #
#         # # self.cc4 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False)
#         # # self.cb4 = nn.BatchNorm2d(64)
#         # # self.cr4 = nn.ReLU()
#         # #
#         # #
#         # # # self.cc5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False)
#         # # # self.cb5 = nn.BatchNorm2d(128)
#         # # # self.cr5 = nn.ReLU()
#         # # # #
#         # # #
#         # # # self.dc5 = nn.ConvTranspose2d(128, 64, kernel_size=3,padding=1,output_padding=1, stride=2, bias=False)
#         # # # self.db5 = nn.BatchNorm2d(64)
#         # # # self.dr5 = nn.ReLU()
#         # #
#         # #
#         # # self.dc4 = nn.ConvTranspose2d(64, 32, kernel_size=3,padding=1,output_padding=1, stride=2, bias=False)
#         # # self.db4 = nn.BatchNorm2d(32)
#         # # self.dr4 = nn.ReLU()
#         #
#         #
#         # #
#         #
#         #
#         # self.dc3 = nn.ConvTranspose2d(32, 16, kernel_size=3,padding=1,output_padding=1, stride=2, bias=False)
#         # self.db3 = nn.BatchNorm2d(16)
#         # self.dr3 = nn.ReLU()
#         #
#         # self.dc2= nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1,output_padding=1,stride=2, bias=False)
#         # self.db2= nn.BatchNorm2d(8)
#         # self.dr2= nn.ReLU()
#         # self.dm2= nn.MaxUnpool2d(kernel_size=2)
#
#         # self.pad=nn.ZeroPad2d(padding=(1,0))
#         self.dc1= nn.ConvTranspose2d(8, 4, kernel_size=3, padding=1,output_padding=1,stride=2)
#         self.db1= nn.BatchNorm2d(4)
#         self.dr1= nn.ReLU()
#         # self.dm1= nn.UpsamplingBilinear2d([288,144])
#
#
#
#         self.tail = nn.Sequential(
#             # nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1, bias=False),
#             # nn.BatchNorm2d(8),
#             # nn.ReLU(),
#             nn.Conv2d(4, 3, kernel_size=1, padding=0, stride=1),
#             nn.BatchNorm2d(3),
#             nn.Sigmoid()
#             # nn.Tanh()
#         )
#
#
#
#
#
#
#
#     def forward(self, x_modal,x):
#         x_var= torch.var(x,dim=1).unsqueeze(dim=1)
#         xx = torch.cat((x,x_var),dim=1)
#         xm = self.cc1(xx)
#         # xm = self.cc1(x_modal)
#         xm = self.cb1(xm)
#         xm = self.cr1(xm)
#         # xm = self.pad(xm)
#         # print/(xm.size())
#         # xm = self.cm1(xm)
#
#         # xm = self.cc2(xm)
#         # xm = self.cb2(xm)
#         # xm = self.cr2(xm)
#         # # size2= xm.size()
#         # # xm,idx2 = self.cm1(xm)
#         #
#         # # xm = self.cc3(xm)
#         # # xm = self.cb3(xm)
#         # # xm = self.cr3(xm)
#         # # # size2= xm.size()
#         # # # xm,idx2 = self.cm1(xm)
#         # #
#         # # # xm = self.cc4(xm)
#         # # # xm = self.cb4(xm)
#         # # # xm = self.cr4(xm)
#         # # # # size2= xm.size()
#         # # # # xm,idx2 = self.cm1(xm)
#         # # # #
#         # # # # xm = self.cc5(xm)
#         # # # # xm = self.cb5(xm)
#         # # # # xm = self.cr5(xm)
#         # # # # print(xm.size())
#         # # # # # size2= xm.size()
#         # # # # # xm,idx2 = self.cm1(xm)
#         # # # # #
#         # # # # xm = self.dc5(xm)
#         # # # # xm = self.db5(xm)
#         # # # # xm = self.dr5(xm)
#         # # #
#         # # #
#         # # #
#         # # # xm = self.dc4(xm)
#         # # # xm = self.db4(xm)
#         # # # xm = self.dr4(xm)
#         # #
#         # # xm = self.dc3(xm)
#         # # xm = self.db3(xm)
#         # # xm = self.dr3(xm)
#         #
#         # xm = self.dc2(xm)
#         # xm = self.db2(xm)
#         # xm = self.dr2(xm)
#         # # xm = self.dm2(xm,idx2,size2)
#         #
#         #
#
#
#         xm = self.dc1(xm)
#         xm = self.db1(xm)
#         xm = self.dr1(xm)
#
#         # xm = self.dm1(xm)
#
#
#         # print(xm.size())
#
#
#
#
#
#         # print(xm.size())
#         x  = self.tail(xm)
#         return x


def gaussian(window_size, sigma):
  gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
  return gauss / gauss.sum()


def create_window(window_size, channel=1):
  _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
  _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
  window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
  return window


# class encode(nn.Module):
#     def __init__(self,mid=1):
#         super(encode, self).__init__()
#         self.cc1= nn.Conv2d(3+1, 8, kernel_size=3, padding=1, stride=2)
#         self.cb1= nn.BatchNorm2d(8)
#         self.cr1= nn.ReLU()
#         self.dc1= nn.ConvTranspose2d(8, 4, kernel_size=3, padding=1,output_padding=1,stride=2)
#         self.db1= nn.BatchNorm2d(4)
#         self.dr1= nn.ReLU()
#         self.mid = mid
#
#
#         self.tail = nn.Sequential(
#             nn.Conv2d(4, 3, kernel_size=1, padding=0, stride=1),
#             nn.BatchNorm2d(3),
#             nn.Sigmoid()
#             # nn.Tanh()
#         )
#
#
#         #
#
#
#
#
#
#
#
#     def forward(self, x_modal, x):
#
#         xm = self.cc1(x_modal)
#         xm = self.cb1(xm)
#         xm = self.cr1(xm)
#         xm = self.dc1(xm)
#         xm = self.db1(xm)
#         xm = self.dr1(xm)
#         x  = self.tail(xm)+x
#         return x

class GeP(nn.Module):
    def __init__(self, p=3.0, eps=1e-12):
      super(GeP, self).__init__()
      self.p = p
      self.eps = eps

    def forward(self, x):
      b, c, h, w = x.shape
      x = x.view(b, c, -1)
      # p = self.p
      x = (torch.mean(x ** self.p, dim=-1) + self.eps ** self.p) ** (1 / self.p)
      return x

# class lcn(nn.Module):
#   def __init__(self, indim=64, middim=32, ksize=3, eps=1e-6):
#     super(lcn, self).__init__()
#
#     self.downconv = nn.Sequential(
#       nn.Conv2d(in_channels=indim, out_channels=middim, kernel_size=1, padding=0, stride=1),
#       SwitchNorm2d(num_features=middim),
#       nn.LeakyReLU()
#     )
#
#
#     self.group = middim
#     self.g = torch.Tensor(get_gaussian_filter((middim, middim//self.group, ksize, ksize),self.group)).cuda()  # 创建卷积核
#     # print(self.g.size())
#     self.k = ksize
#     self.middim=middim
#     self.indim=indim
#
#
#     # self.deconv = nn.Sequential(
#     # nn.Conv2d(middim, middim, kernel_size=3, padding=1, stride=2),
#     # nn.BatchNorm2d(middim),
#     # nn.ReLU(),
#     # # nn.Conv2d(6, 12, kernel_size=3, padding=1, stride=2),
#     # # nn.BatchNorm2d(12),
#     # # nn.ReLU(),
#     # #
#     # # nn.ConvTranspose2d(12, 6, kernel_size=3, padding=1,output_padding=1,stride=2),
#     # # nn.BatchNorm2d(6),
#     # # nn.ReLU(),
#     # nn.ConvTranspose2d(middim, middim, kernel_size=3, padding=1,output_padding=1,stride=2),
#     # nn.BatchNorm2d(middim),
#     # nn.ReLU(),
#     # )
#     self.upconv = nn.Sequential(
#       nn.Conv2d(in_channels=self.group, out_channels=indim, kernel_size=1, padding=0, stride=1),
#       SwitchNorm2d(num_features=indim),
#       nn.LeakyReLU()
#     )
#     nn.init.constant_(self.upconv[1].bias, 0.0)
#     nn.init.constant_(self.upconv[1].weight, 0.0)
#
#     # self.pool = nn.AdaptiveMaxPool2d((1,1))
#     # self.att = nn.Sequential(
#     #   nn.Linear(in_features=indim,out_features=indim//16),
#     #   SwitchNorm1d(indim//16),
#     #   nn.ReLU(),
#     #   nn.Linear(in_features=indim//16, out_features=indim),
#     #   SwitchNorm1d(indim),
#     #   nn.Sigmoid()
#     # )
#     # nn.init.constant_(self.att[3].bias, 0.0)
#     # nn.init.constant_(self.att[3].weight,0.0)
#
#
#
#
#     # self.p = nn.Parameter(torch.ones(1, middim, 1, 1))
#     # nn.init.constant_(self.p, 1.0)
#     # self.p = nn.Parameter(torch.ones(1))
#     # nn.init.constant_(self.p, 1e-5)
#     self.eps = eps  # *torch.ones(1).cuda()
#     # nn.init.uniform_(self.p,0,5)
#     # print(self.p)
#
#   def forward(self, x):
#
#     xdown = self.downconv(x)
#     filtered_out = F.conv2d(xdown, self.g, padding=(self.k - 1) // 2,
#                             groups=self.group)  # 卷积 (∑ipq Wpq.X i,j+p,k+q)
#
#     centered_image = xdown - filtered_out  # [:, :, mid:-mid, mid:-mid]  # Vijk
#     s_deviation = F.conv2d(centered_image.pow(2), self.g,
#                            padding=(self.k - 1) // 2,
#                            groups=self.group)   # ∑ipqWpq.v2 i,j+p,k+q
#
#     s_deviation = s_deviation.sqrt()
#     # s_deviation = s_deviation+self.eps
#     # if self.p > self.eps:
#     #     self.p = self.eps
#     s_deviation = s_deviation.clamp(min=self.eps)
#     highx = centered_image / s_deviation
#     highx = self.upconv(highx)
#
#     # sig = self.pool(highx).view(-1,self.indim)
#     # sig = self.att(sig).view(-1,self.indim,1,1)
#
#     x = highx + x
#
#     return x

# class lcn(nn.Module):
#   def __init__(self, indim=64, middim=32, ksize=3, eps=1e-6):
#     super(lcn, self).__init__()
#
#
#     middim = indim
#     self.group = middim
#     self.g = torch.Tensor(get_gaussian_filter((middim, middim//self.group, ksize, ksize),self.group)).cuda()  # 创建卷积核
#     self.k = ksize
#     self.middim=middim
#     self.indim=indim
#
#
#
#
#     self.p = nn.Parameter(torch.ones(1, middim, 1, 1))
#     nn.init.constant_(self.p, 0.0)
#     self.eps = eps  # *torch.ones(1).cuda()
#
#
#   def forward(self, x):
#
#     # xdown = self.downconv(x)
#     filtered_out = F.conv2d(x, self.g, padding=(self.k - 1) // 2,
#                             groups=self.group)  # 卷积 (∑ipq Wpq.X i,j+p,k+q)
#
#     centered_image = x - filtered_out  # [:, :, mid:-mid, mid:-mid]  # Vijk
#     s_deviation = F.conv2d(centered_image.pow(2), self.g,
#                            padding=(self.k - 1) // 2,
#                            groups=self.group)   # ∑ipqWpq.v2 i,j+p,k+q
#
#     s_deviation = s_deviation.sqrt()
#     # s_deviation = s_deviation+self.eps
#     s_deviation = s_deviation.clamp(min=self.eps)
#     highx = centered_image / s_deviation
#
#
#     # sig = self.pool(highx).view(-1,self.indim)
#     # sig = self.att(sig).view(-1,self.indim,1,1)
#
#     x = highx * self.p + x
#
#     return x
#

# class lcn(nn.Module):
#     def __init__(self,indim=64,ksize=1):
#         super(lcn, self).__init__()
#         self.g =  torch.Tensor(get_gaussian_filter((1, indim, ksize, ksize))).cuda()  # 创建卷积核
#         self.k = ksize
#         self.p = nn.Parameter(torch.ones(1,indim,1,1))
#         nn.init.constant_(self.p,0.0)
#         # nn.init.uniform_(self.p,0,5)
#         # print(self.p)
#
#     def forward(self, x):
#
#         # n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]  # (图片数、层数、x轴、y轴大小)
#
#         filtered_out = F.conv2d(x, self.g, padding=(self.k - 1) // 2)  # 卷积 (∑ipq Wpq.X i,j+p,k+q)
#
#         centered_image = x - filtered_out  # [:, :, mid:-mid, mid:-mid]  # Vijk
#
#         s_deviation = F.conv2d(centered_image.pow(2), self.g, padding=(self.k - 1) // 2).sqrt()  # ∑ipqWpq.v2 i,j+p,k+q
#
#         x = x + self.p * centered_image / (s_deviation + 1e-6)  # torch.Tensor(divisor).cuda()  # Yijk
#
#
#
#         return x

def get_gaussian_filter(kernel_shape,grp):
  x = np.zeros(kernel_shape, dtype='float64')

  # 二维高斯函数
  def gauss(x, y, sigma=1.5):
    Z = 2 * np.pi * sigma ** 2
    return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

  mid = np.floor(kernel_shape[-1] / 2.)  # 求出卷积核的中心位置(mid,mid)
  for c_idx in range(0, kernel_shape[0]):
    for kernel_idx in range(0, kernel_shape[1]):  # 遍历每一层
      for i in range(0, kernel_shape[2]):  # 遍历x轴
        for j in range(0, kernel_shape[3]):  # 遍历y轴
          x[c_idx, kernel_idx, i, j] = gauss(i - mid, j - mid)  / (grp*kernel_shape[2]*kernel_shape[3])




          # 计算出高斯权重

  return x #/ np.sum(x)


def get_avg_filter(kernel_shape):
  x = np.ones(kernel_shape, dtype='float64')
  return x / np.sum(x)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  """3x3 convolution with padding"""
  # original padding is 1; original dilation is 1
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride, dilation)
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

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    # original padding is 1; original dilation is 1
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride
    # self.uselcn = uselcn
    # print(self.uselcn)
    # if self.uselcn:
    #   self.lcn = lcn(indim=planes * 4,middim=(planes * 4)//2,ksize=3,eps=1e-5)

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

    # if self.uselcn:
    #   print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!2023012')
    #   out = self.lcn(out)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1):

    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64,  layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x


def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  # for key, value in state_dict.items():
  for key, value in list(state_dict.items()):
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict


def resnet18(pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
  return model


def resnet34(pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
  return model


def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    # model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])),strict=False)
  return model


def resnet101(pretrained=False, **kwargs):
  """Constructs a ResNet-101 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
      remove_fc(model_zoo.load_url(model_urls['resnet101'])))
  return model


def resnet152(pretrained=False, **kwargs):
  """Constructs a ResNet-152 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
      remove_fc(model_zoo.load_url(model_urls['resnet152'])))
  return model