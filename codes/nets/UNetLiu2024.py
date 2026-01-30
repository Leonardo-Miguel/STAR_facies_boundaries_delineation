import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet77(nn.Module):
    def __init__(self, kernel_size=7, input_channels=1):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = 1
        self.pool = nn.MaxPool2d(2)
        self.dropout_prob = 0.4
        self.conv_kernel_size = kernel_size
        self.conv_padding = self.conv_kernel_size // 2  # 'same'

        # Encoder
        self.enc1 = self.conv_block(self.input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Decoder
        self.up2 = self.up_block(256, 128)
        self.dec2 = self.conv_block(256, 128)

        self.up1 = self.up_block(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Final layer
        self.final = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.conv_kernel_size, padding=self.conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.SELU(),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.conv_kernel_size, padding=self.conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.SELU(),
            nn.Dropout(self.dropout_prob)
        )

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
            x = F.pad(x, (0, pad_w, 0, pad_h)) # (left, right, top, bottom)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # Decoder
        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.final(x)
        
        # remoção do padding
        if pad_h > 0 or pad_w > 0: x = x[:, :, :H, :W]

        return x