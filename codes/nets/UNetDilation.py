import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetDilation(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = 1
        self.pool = nn.MaxPool2d(2)
        self.dropout_prob = 0.4
        self.dilation = 4
        self.kernel_size = 3
        self.padding = self.dilation * (self.kernel_size - 1) // 2  # mantém saída do mesmo tamanho

        # Encoder com convoluções dilatadas
        self.enc1 = self.conv_block_dilated(1, 32)
        self.enc2 = self.conv_block_dilated(32, 64)
        self.enc3 = self.conv_block_dilated(64, 128)

        # enc4
        self.enc4 = self.conv_block_dilated(128, 256)

        # Decoder
        self.up3 = self.up_block(256, 128)
        self.dec3 = self.conv_block_dilated(256, 128)

        self.up2 = self.up_block(128, 64)
        self.dec2 = self.conv_block_dilated(128, 64)

        self.up1 = self.up_block(64, 32)
        self.dec1 = self.conv_block_dilated(64, 32)

        # Camada final
        self.final = nn.Conv2d(32, self.n_classes, kernel_size=1)

    def conv_block_dilated(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
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

        # Padding automático para múltiplo de 8
        H_pad = self.next_multiple_of_8(H)
        W_pad = self.next_multiple_of_8(W)
        pad_h = H_pad - H
        pad_w = W_pad - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # enc4
        x4 = self.enc4(self.pool(x3))

        # Decoder
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.final(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


class UnetDilationEncoder(nn.Module):
    def __init__(self, unet: UnetDilation):
        super().__init__()
        self.enc1 = unet.enc1
        self.enc2 = unet.enc2
        self.enc3 = unet.enc3
        self.enc4 = unet.enc4
        self.pool = unet.pool

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        return x1, x2, x3, x4

class UnetDilationDecoder(nn.Module):
    def __init__(self, encoder: UnetDilationEncoder, unet: UnetDilation, n_classes=1):
        super().__init__()
        self.encoder = encoder
        self.unet = unet
        # camadas extraídas diretamente da UNet
        self.up3 = unet.up3
        self.dec3 = unet.dec3
        self.up2 = unet.up2
        self.dec2 = unet.dec2
        self.up1 = unet.up1
        self.dec1 = unet.dec1
        self.final = unet.final

        self.next_multiple_of_8 = self.unet.next_multiple_of_8
    
    def forward(self, x):

        B, C, H, W = x.shape
        H_pad = self.next_multiple_of_8(H)
        W_pad = self.next_multiple_of_8(W)
        pad_h = H_pad - H
        pad_w = W_pad - W
    
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
            
        # Encoder
        x1, x2, x3, x4 = self.encoder(x)
        
        # Decoder
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.final(x)

        # remoção do padding
        if pad_h > 0 or pad_w > 0: x = x[:, :, :H, :W]

        return x
