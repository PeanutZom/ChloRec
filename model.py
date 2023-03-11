import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class DINCAEModel(nn.Module):
    #  模型
    def __init__(self, in_channels=9, out_channels=1, up_sampling_node='nearest'):
        super(DINCAEModel, self).__init__()
        # ENCODER
        self.ec_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.ec_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.ec_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.ec_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.ec_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        # DECODER
        self.dc_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.dc_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.dc_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.dc_conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.dc_conv5 = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False)

        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, image):
        # (batch_size=32) * (lat=192) * (lon=240) * (channel=6)
        ec_image_0 = image
        # ENCODER:
        ec_image_1 = self.max_pool(self.relu(self.ec_conv1(ec_image_0)))
        ec_image_2 = self.max_pool(self.relu(self.ec_conv2(ec_image_1)))
        ec_image_3 = self.max_pool(self.relu(self.ec_conv3(ec_image_2)))
        ec_image_4 = self.max_pool(self.relu(self.ec_conv4(ec_image_3)))
        ec_image_5 = self.max_pool(self.relu(self.ec_conv5(ec_image_4)))

        # DECODER:
        dc_image_0 = ec_image_5
        dc_image_1 = self.relu(
            self.dc_conv1(
                F.interpolate(dc_image_0, size=(ec_image_4.size()[2], ec_image_4.size()[3]), mode='nearest')))

        dc_image_2 = self.relu(
            self.dc_conv2(
                F.interpolate(dc_image_1 + ec_image_4, size=(ec_image_3.size()[2], ec_image_3.size()[3]), mode='nearest')))

        dc_image_3 = self.relu(
            self.dc_conv3(
                F.interpolate(dc_image_2 + ec_image_3, size=(ec_image_2.size()[2], ec_image_2.size()[3]), mode='nearest')))

        dc_image_4 = self.relu(
            self.dc_conv4(
                F.interpolate(dc_image_3 + ec_image_2, size=(ec_image_1.size()[2], ec_image_1.size()[3]), mode='nearest')))

        dc_image_5 = (
            self.dc_conv5(
                F.interpolate(dc_image_4 + ec_image_1, size=(ec_image_0.size()[2], ec_image_0.size()[3]), mode='nearest')))

        return dc_image_5

