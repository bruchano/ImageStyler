import torch
from torch.nn import ReflectionPad2d
from torch.nn.functional import interpolate
from torchvision import models
from collections import namedtuple
from random import shuffle



# '''
# the min input size is [1, 3, 16, 16] as the kernel size and stride reduce the height and width
# otherwise exception might caused as the input_size != output_size
# '''
#
#
# class Transformer(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Convlayer(in_channels=3, out_channels=32, kernel_size=9, stride=1)
#         self.inst1 = torch.nn.InstanceNorm2d(num_features=32, affine=True)
#         self.conv2 = Convlayer(in_channels=32, out_channels=64, kernel_size=3, stride=2)
#         self.inst2 = torch.nn.InstanceNorm2d(num_features=64, affine=True)
#         self.conv3 = Convlayer(in_channels=64, out_channels=128, kernel_size=3, stride=2)
#         self.inst3 = torch.nn.InstanceNorm2d(num_features=128, affine=True)
#         self.res1 = Residential(128)
#         self.res2 = Residential(128)
#         self.res3 = Residential(128)
#         self.res5 = Residential(128)
#         self.res4 = Residential(128)
#         self.upsample1 = UpsampleConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, upsample=2)
#         self.upsample2 = UpsampleConvLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample=2)
#         self.upsample3 = UpsampleConvLayer(in_channels=32, out_channels=3, kernel_size=3, stride=1)
#
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.inst1(self.conv1(x)))
#         x = self.relu(self.inst2(self.conv2(x)))
#         x = self.relu(self.inst3(self.conv3(x)))
#         x = self.res1(x)
#         x = self.res2(x)
#         x = self.res3(x)
#         x = self.res4(x)
#         x = self.res5(x)
#         x = self.relu(self.upsample1(x))
#         x = self.relu(self.upsample2(x))
#         x = self.relu(self.upsample3(x))
#         return x
#
#
# class Convlayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
#         super().__init__()
#         padding = kernel_size // 2
#         self.refl = torch.nn.ReflectionPad2d(padding)
#         self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#
#     def forward(self, x):
#         x = self.refl(x)
#         return self.conv(x)
#
#
# class Residential(torch.nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv1 = Convlayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3)
#         self.inst1 = torch.nn.InstanceNorm2d(in_channels)
#         self.conv2 = Convlayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3)
#         self.inst2 = torch.nn.InstanceNorm2d(in_channels)
#
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, x):
#         resident = x
#         x = self.relu(self.inst1(self.conv1(x)))
#         x = self.inst2(self.conv2(x))
#         return resident + x
#
#
# class UpsampleConvLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
#         super().__init__()
#         self.upsample = upsample
#         reflectpad = kernel_size // 2
#         self.reflectionpad = torch.nn.ReflectionPad2d(reflectpad)
#         self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, x):
#         if self.upsample:
#             x = torch.nn.functional.interpolate(x, scale_factor=self.upsample)
#         x = self.reflectionpad(x)
#         return self.relu(self.conv(x))
#
#
# class Vgg16(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super().__init__()
#         vgg_pretrained_features = models.vgg16(pretrained=True).features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(4):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4, 9):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(9, 16):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(16, 23):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(23, 31):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for parameter in self.parameters():
#                 parameter.requires_grad = False
#
#     def forward(self, x):
#         x = self.slice1(x)
#         h1 = x
#         x = self.slice2(x)
#         h2 = x
#         x = self.slice3(x)
#         h3 = x
#         x = self.slice4(x)
#         h4 = x
#         x = self.slice5(x)
#         h5 = x
#         vgg_output = namedtuple("Vgg_model", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
#         x = vgg_output(h1, h2, h3, h4, h5)
#         return x

a = torch.rand(1, 1, 3, 3)

print(a.shape[1] == 1)