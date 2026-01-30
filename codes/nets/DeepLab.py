import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, kernel_size=3, input_channels=1):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = 1
        self.dropout_prob = 0.4
        self.conv_kernel_size = kernel_size
        self.conv_padding = kernel_size // 2  # 'same'

        # Backbone (Encoder ResNet)
        self.layer1 = self.res_block(self.input_channels, 64, stride=2)   # 1/2
        self.layer2 = self.res_block(64, 128, stride=2)                   # 1/4
        self.layer3 = self.res_block(128, 256, stride=2, dilation=1)      # 1/8
        self.layer4 = self.res_block(256, 512, stride=1, dilation=2)      # ASPP input (1/8)

        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = ASPP(512, 256)

        # Decoder (com skip do low-level feature)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(128, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_prob)
        )

        self.final = nn.Conv2d(256, self.n_classes, kernel_size=1)

    def res_block(self, in_channels, out_channels, stride=1, dilation=1):
        return ResidualBlock(in_channels, out_channels, stride, dilation,
                             kernel_size=self.conv_kernel_size,
                             padding=self.conv_padding,
                             dropout=self.dropout_prob)

    def next_multiple_of_8(self, n):
        return n if n % 8 == 0 else (n // 8 + 1) * 8

    def forward(self, x):
        B, C, H, W = x.shape

        H_pad = self.next_multiple_of_8(H)
        W_pad = self.next_multiple_of_8(W)

        pad_h = H_pad - H
        pad_w = W_pad - W

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        x1 = self.layer1(x)   # low-level feature
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # ASPP
        x_aspp = self.aspp(x4)
        x_aspp = F.interpolate(x_aspp, scale_factor=4, mode='bilinear', align_corners=False)

        # Decoder
        low_level = self.low_level_conv(x2)
        low_level_size = low_level.shape[-2:]

        x_aspp = F.interpolate(x_aspp, size=low_level_size, mode='bilinear', align_corners=False)
        x = torch.cat([x_aspp, low_level], dim=1)
        x = self.decoder(x)

        x = self.final(x)
        x = F.interpolate(x, size=(H_pad, W_pad), mode='bilinear', align_corners=False)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


# Residual Block básico usado no encoder
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 kernel_size=3, padding=1, dropout=0.4):
        super().__init__()
        self.same_channels = (in_channels == out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding 
                               =dilation,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)

        if not self.same_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        out += identity
        out = self.final_act(out)
        return out

# ASPP (Atrous Spatial Pyramid Pooling)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]

        feat1 = self.conv1(x)
        feat2 = self.conv6(x)
        feat3 = self.conv12(x)
        feat4 = self.conv18(x)

        img = self.image_pool(x)
        img = self.image_conv(img)
        img = F.interpolate(img, size=size, mode='bilinear', align_corners=False)

        out = torch.cat([feat1, feat2, feat3, feat4, img], dim=1)
        out = self.final_conv(out)
        return out

class DeepLabV3PlusEncoder(nn.Module):
    def __init__(self, net: DeepLabV3Plus):
        super().__init__()
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.aspp = net.aspp

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)   # low-level feature
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # ASPP
        x_aspp = self.aspp(x4)
        x_aspp = F.interpolate(x_aspp, scale_factor=4, mode='bilinear', align_corners=False)

        return x1, x2, x3, x4, x_aspp

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder: DeepLabV3PlusEncoder, net: DeepLabV3Plus, n_classes=1):
        super().__init__()
        self.encoder = encoder
        self.net = net
        
        # camadas extraídas diretamente da DeepLabV3Plus completa
        self.low_level_conv = net.low_level_conv
        self.decoder = net.decoder
        self.final = net.final

        self.next_multiple_of_8 = self.net.next_multiple_of_8
    
    def forward(self, x):

        B, C, H, W = x.shape
        H_pad = self.next_multiple_of_8(H)
        W_pad = self.next_multiple_of_8(W)
        pad_h = H_pad - H
        pad_w = W_pad - W
    
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
            
        # Encoder
        x1, x2, x3, x4, x_aspp = self.encoder(x)
        
        # Decoder
        low_level = self.low_level_conv(x2)
        low_level_size = low_level.shape[-2:]

        x_aspp = F.interpolate(x_aspp, size=low_level_size, mode='bilinear', align_corners=False)
        x = torch.cat([x_aspp, low_level], dim=1)
        x = self.decoder(x)

        x = self.final(x)
        x = F.interpolate(x, size=(H_pad, W_pad), mode='bilinear', align_corners=False)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x