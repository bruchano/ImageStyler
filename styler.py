import torch
from torch.nn import ReflectionPad2d
from torch.nn.functional import interpolate
from torchvision import models
from torchvision import transforms
from collections import namedtuple
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from random import shuffle


class Transformer(torch.nn.Module):
    '''
    the min input size is [1, 3, 16, 16] as the kernel size and stride reduce the height and width
    otherwise exception might caused as the input_size != output_size
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = Convlayer(in_channels=3, out_channels=32, kernel_size=9, stride=1)
        self.inst1 = torch.nn.InstanceNorm2d(num_features=32, affine=True)
        self.conv2 = Convlayer(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.inst2 = torch.nn.InstanceNorm2d(num_features=64, affine=True)
        self.conv3 = Convlayer(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.inst3 = torch.nn.InstanceNorm2d(num_features=128, affine=True)
        self.res1 = Residential(128)
        self.res2 = Residential(128)
        self.res3 = Residential(128)
        self.res4 = Residential(128)
        self.res5 = Residential(128)
        self.upsample1 = UpsampleConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, upsample=2)
        self.upsample2 = UpsampleConvLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample=2)
        self.upsample3 = UpsampleConvLayer(in_channels=32, out_channels=3, kernel_size=9, stride=1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.inst1(self.conv1(x)))
        x = self.relu(self.inst2(self.conv2(x)))
        x = self.relu(self.inst3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.upsample1(x))
        x = self.relu(self.upsample2(x))
        x = self.relu(self.upsample3(x))
        return x


class Convlayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.refl = torch.nn.ReflectionPad2d(padding)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.refl(x)
        return self.conv(x)


class Residential(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = Convlayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3)
        self.inst1 = torch.nn.InstanceNorm2d(in_channels, affine=True)
        self.conv2 = Convlayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3)
        self.inst2 = torch.nn.InstanceNorm2d(in_channels, affine=True)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        resident = x
        x = self.relu(self.inst1(self.conv1(x)))
        x = self.inst2(self.conv2(x))
        return resident + x


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        reflectpad = kernel_size // 2
        self.reflectionpad = torch.nn.ReflectionPad2d(reflectpad)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if self.upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        x = self.reflectionpad(x)
        return self.relu(self.conv(x))


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 31):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        h1 = x
        x = self.slice2(x)
        h2 = x
        x = self.slice3(x)
        h3 = x
        x = self.slice4(x)
        h4 = x
        x = self.slice5(x)
        h5 = x
        # vgg_output = namedtuple("Vgg_model", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        # x = vgg_output(h1, h2, h3, h4, h5)
        return h1


def load_image(img_path, scale=None):
    img = Image.open(img_path)
    if scale:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)))
    trans = transforms.ToTensor()
    img = trans(img).unsqueeze_(0)
    print(img.shape)
    return img


def save_image(img_tensor, save_path):
    trans = transforms.ToPILImage()
    img_tensor = img_tensor.squeeze()
    print(img_tensor.shape)
    print(img_tensor)
    img = trans(img_tensor)
    img.save(save_path)


def gram_matrix(tensor):
    '''
    return a 3 * 3 gram matrix
    '''
    _, f, h, w = tensor.shape
    tensor = tensor.reshape(_, f, h * w)
    tensor_transpose = tensor.transpose(1, 2)
    gram = tensor.bmm(tensor_transpose) / (f * h * w)
    return gram


def normalize_batch(batch):
    '''
    normalize the tensor with mean and std value of RBG
    '''
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


def train(style_image_path, dataset_file, scale=None, ver=1, trained_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = Transformer().to(device)
    if trained_model:
        transformer.load_state_dict(torch.load(trained_model))
    transformer.train()
    vgg = Vgg16(requires_grad=False)
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)

    style_img = load_image(style_image_path, scale=scale)
    features_style = vgg(normalize_batch(style_img))
    gram_style = gram_matrix(features_style)

    save_model_path = style_image_path[:-4] + "_model_ver_" + str(ver) + ".pth"
    BASEDIR = os.getcwd()
    dataset_file = os.path.join(BASEDIR, dataset_file)

    epoch = 1
    for e in range(epoch):
        transformer.train()
        list = os.listdir(dataset_file)
        shuffle(list)

        for i in range(100):
            optimizer.zero_grad()

            img_path = os.path.join(dataset_file, list[i])
            img = load_image(img_path, scale)
            img_original = img.to(device)
            if img_original.shape[1] != 3:
                continue

            img_stylized = transformer(img_original).clamp(0, 1)

            features_original = vgg(normalize_batch(img_original))
            features_stylized = vgg(normalize_batch(img_stylized))

            if features_original.shape != features_stylized.shape:
                features_stylized = features_stylized[:, :, :features_original.shape[2], :features_original.shape[3]]

            content_loss = mse_loss(features_stylized, features_original)

            # if img_stylized.shape != img_original.shape:
            #     img_stylized = img_stylized[:, :, :img_original.shape[2], :img_original.shape[3]]
            # content_loss = mse_loss(img_stylized, img_original)

            print("loop %3d content loss: %.3f" % ((i + 1), content_loss.item()))

            gram_stylized = gram_matrix(features_stylized)

            style_loss = 0
            style_weight = 10000
            for gm_y, gm_s in zip(gram_stylized, gram_style):
                style_loss += mse_loss(gm_y, gm_s)
            style_loss *= style_weight
            print("loop %3d style loss: %.3f" % ((i + 1), style_loss.item()))

            total_loss = content_loss + style_loss
            print("loop %3d total loss: %.3f" % ((i + 1), total_loss.item()))

            total_loss.backward()
            optimizer.step()
            print("loop %3d completed" % (i + 1))

        transformer.eval()
        torch.save(transformer.state_dict(), save_model_path)


def stylize(model_path, img_path, save_path, scale=1):
    stylizer = Transformer()
    stylizer.load_state_dict(torch.load(model_path))
    stylizer.eval()

    img = load_image(img_path, scale=scale)
    stylized_img = stylizer(img)
    save_image(stylized_img, save_path)


data_file = "train/photos"
style_image_path = "style_2.jpg"
trained_model = "style_2_model_ver_12.pth"
train(style_image_path, data_file, ver=13, trained_model=trained_model)

model_path = "style_2_model_ver_12.pth"
image_path = "bruno2.jpg"
save_path = image_path[:-4] + "_" + model_path[:-4] + ".jpg"
stylize(model_path, image_path, save_path, scale=3)

# vgg = Vgg16(requires_grad=False)
# a = torch.randn(1, 3, 40, 40)
# a = a.repeat(3, 1, 1, 1)
# x = vgg(a)
# print(x[0].shape)

# a = torch.randn(1, 3, 270, 400)
# T = Transformer()
# a = T(a)
# print(a.shape)

print("Done")

