import torch
import torch.nn as nn
import torch.nn.functional as F

class ResUnet(nn.Module):
    def __init__(self, kernel_size, input_channels=1):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = 1
        self.pool = nn.MaxPool2d(2)
        self.dropout_prob = 0.4
        self.conv_kernel_size = kernel_size
        self.conv_padding = self.conv_kernel_size // 2  # 'same'

        # Encoder
        self.enc1 = self.res_block(self.input_channels, 32)
        self.enc2 = self.res_block(32, 64)
        self.enc3 = self.res_block(64, 128)
        self.enc4 = self.res_block(128, 256)

        # Decoder
        self.up3 = self.up_block(256, 128)
        self.dec3 = self.res_block(256, 128)

        self.up2 = self.up_block(128, 64)
        self.dec2 = self.res_block(128, 64)

        self.up1 = self.up_block(64, 32)
        self.dec1 = self.res_block(64, 32)

        # Final layer
        self.final = nn.Conv2d(32, self.n_classes, kernel_size=1)

    # Residual block
    def res_block(self, in_channels, out_channels):
        return ResBlock(in_channels, out_channels,
                        kernel_size=self.conv_kernel_size,
                        padding=self.conv_padding,
                        dropout=self.dropout_prob)

    def up_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def next_multiple_of_8(self, n):
        if n % 8 == 0:
            return n
        return (n // 8 + 1) * 8

    def forward(self, x):
        B, C, H, W = x.shape

        H_pad = self.next_multiple_of_8(H)
        W_pad = self.next_multiple_of_8(W)

        pad_h = H_pad - H
        pad_w = W_pad - W

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Decoder
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.final(x)

        # remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x

# Bloco residual usado pela ResUNet
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.4):
        super().__init__()
        self.same_channels = (in_channels == out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        # Projeção 1x1 se número de canais mudar
        if not self.same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out)

        out += residual
        out = self.act2(out)

        return out