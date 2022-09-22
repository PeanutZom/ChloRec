import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from partialconv2d import PartialConv2d


class DINCAEModel(nn.Module):
    #  模型
    def __init__(self, in_channels=8, out_channels=1, up_sampling_node='nearest'):
        super(DINCAEModel, self).__init__()
        # ENCODER
        self.ec_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.ec_conv2 = nn.Conv2d(16, 30, kernel_size=3, padding=1, bias=False)
        self.ec_conv3 = nn.Conv2d(30, 58, kernel_size=3, padding=1, bias=False)
        self.ec_conv4 = nn.Conv2d(58, 110, kernel_size=3, padding=1, bias=False)
        self.ec_conv5 = nn.Conv2d(110, 209, kernel_size=3, padding=1, bias=False)
        # DECODER
        self.dc_conv1 = nn.Conv2d(209, 110, kernel_size=3, padding=1, bias=False)
        self.dc_conv2 = nn.Conv2d(110, 58, kernel_size=3, padding=1, bias=False)
        self.dc_conv3 = nn.Conv2d(58, 30, kernel_size=3, padding=1, bias=False)
        self.dc_conv4 = nn.Conv2d(30, 16, kernel_size=3, padding=1, bias=False)
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





# --------------------------
# PConv-BatchNorm-Activation
# --------------------------
class PConvBNActiv(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, sample='none-3', activ='leaky', bias=False):
        super(PConvBNActiv, self).__init__()
        if sample == 'down-11':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=11, stride=2, padding=5, bias=bias, multi_channel = True)
        elif sample == 'down-9':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=9, stride=2, padding=4, bias=bias, multi_channel = True)
        elif sample == 'down-7':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=bias, multi_channel = True)
        elif sample == 'down-5':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=bias, multi_channel = True)
        elif sample == 'down-3':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias, multi_channel = True)
        else:
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias, multi_channel = True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, images, masks):
        images, masks = self.conv(images, masks)
        if hasattr(self, 'bn'):
            images = self.bn(images)
        if hasattr(self, 'activation'):
            images = self.activation(images)

        return images, masks

# ------------
# Double U-Net
# ------------
class PUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=1, up_sampling_node='nearest'):
        super(PUNet, self).__init__()
        self.freeze_ec_bn = False
        self.up_sampling_node = up_sampling_node
        self.ec_images_1 = PConvBNActiv(in_channels, 32, sample='down-7')
        self.ec_images_2 = PConvBNActiv(32, 64, sample='down-5')
        self.ec_images_3 = PConvBNActiv(64, 128, sample='down-3')
        self.ec_images_4 = PConvBNActiv(128, 256, sample='down-3')
        self.dc_images_4 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_images_3 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_images_2 = PConvBNActiv(64 + 32, 32, activ='leaky')
        self.dc_images_1 = PConvBNActiv(32 + in_channels, out_channels, bn=False, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, input_images, input_masks):
        ec_images = {}
        ec_images['ec_images_0'], ec_images['ec_images_masks_0'] = input_images, input_masks
        # => batch_size * 3 * 192 * 240
        ec_images['ec_images_1'], ec_images['ec_images_masks_1'] = self.ec_images_1(input_images, input_masks)
        # => batch_size * 16 * 96 * 120
        ec_images['ec_images_2'], ec_images['ec_images_masks_2'] = self.ec_images_2(ec_images['ec_images_1'], ec_images['ec_images_masks_1'])
        # => batch_size * 32 * 48 * 60
        ec_images['ec_images_3'], ec_images['ec_images_masks_3'] = self.ec_images_3(ec_images['ec_images_2'], ec_images['ec_images_masks_2'])
        # => batch_size * 64 * 24 * 30
        ec_images['ec_images_4'], ec_images['ec_images_masks_4'] = self.ec_images_4(ec_images['ec_images_3'], ec_images['ec_images_masks_3'])
        # => batch_size * 128 * 12 * 15
        # --------------
        # images decoder
        # --------------
        dc_images, dc_images_masks = ec_images['ec_images_4'], ec_images['ec_images_masks_4']
        for _ in range(4, 0, -1):
            ec_images_skip = 'ec_images_{:d}'.format(_ - 1)
            ec_images_masks = 'ec_images_masks_{:d}'.format(_ - 1)
            dc_conv = 'dc_images_{:d}'.format(_)
            dc_images = F.interpolate(dc_images, scale_factor=2, mode=self.up_sampling_node)
            dc_images_masks = F.interpolate(dc_images_masks, scale_factor=2, mode=self.up_sampling_node)
            dc_images = torch.cat((dc_images, ec_images[ec_images_skip]), dim=1)
            dc_images_masks = torch.cat((dc_images_masks, ec_images[ec_images_masks]), dim=1)
            dc_images, dc_images_masks = getattr(self, dc_conv)(dc_images, dc_images_masks)
        #outputs = self.tanh(dc_images)
        outputs = dc_images
        return outputs
