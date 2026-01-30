import torch
import torch.nn as nn
import torch.nn.functional as F

class DNFS(nn.Module):
    """
    Deep Neural Network for Facies Segmentation (DNFS)
    Adaptada de Lima et al. (2024)
    """

    def __init__(self, kernel_size, input_channels=1):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = 1
        self.pool = nn.MaxPool2d(2)
        self.dropout_prob = 0.1
        self.conv_kernel_size = kernel_size
        self.conv_padding = self.conv_kernel_size // 2  # 'same'

        # Encoder
        self.enc1 = self.conv_block(self.input_channels, 8)
        self.enc2 = self.conv_block(8, 16)
        self.enc3 = self.conv_block(16, 32)
        self.enc4 = self.conv_block(32, 64)
        self.bottleneck_layer = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

        # Decoder
        self.up4 = self.up_block(128, 64)
        self.dec4 = self.conv_block(128, 64)
        
        self.up3 = self.up_block(64, 32)
        self.dec3 = self.conv_block(64, 32)

        self.up2 = self.up_block(32, 16)
        self.dec2 = self.conv_block(32, 16)

        self.up1 = self.up_block(16, 8)
        self.dec1 = self.conv_block(16, 8)

        # Final layer
        self.final = nn.Conv2d(8, self.n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.conv_kernel_size, padding=self.conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.conv_kernel_size, padding=self.conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob)
        )

    def up_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def next_multiple_of_16(self, n):
        if n % 16 == 0:
            return n
        return (n // 16 + 1) * 16
                
    def forward(self, x):

        B, C, H, W = x.shape

        H_pad = self.next_multiple_of_16(H)
        W_pad = self.next_multiple_of_16(W)
    
        pad_h = H_pad - H
        pad_w = W_pad - W
    
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        bottleneck = self.bottleneck_layer(self.pool(x4))

        # Decoder
        x = self.up4(bottleneck)
        x = self.dec4(torch.cat([x, x4], dim=1))
        
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.final(x)
        
        # remoção do padding
        if pad_h > 0 or pad_w > 0: x = x[:, :, :H, :W]

        return x

'''
model = DNFS(kernel_size=3)
print(model)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total de parâmetros: {params:,}")

x = torch.randn(1, 1, 782, 1006)
y = model(x)
print(y.shape)  # → torch.Size([1, 1, 256, 256])
'''
'''
class DNFS(nn.Module):
    """
    Deep Neural Network for Facies Segmentation (DNFS)
    Adaptada de Lima et al. (2024)
    """

    def __init__(self, kernel_size, input_channels=1):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = 1
        self.pool = nn.MaxPool2d(2)
        self.dropout_prob = 0.1
        self.conv_kernel_size = kernel_size
        self.conv_padding = self.conv_kernel_size // 2  # 'same'

        # Encoder
        self.enc1 = self.conv_block(self.input_channels, 8)
        self.enc2 = self.conv_block(8, 16)
        self.enc3 = self.conv_block(16, 32)
        self.enc4 = self.conv_block(32, 64)

        # -------- latent (1x1 conv) --------
        self.latent = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

        self.up4 = self.up_block(128, 128)
        self.skip4 = nn.Conv2d(192, 64, kernel_size=1)
        self.dec4 = self.conv_block(64, 64)
        
        self.up3 = self.up_block(64, 64)
        self.skip3 = nn.Conv2d(96, 32, kernel_size=1)
        self.dec3 = self.conv_block(32, 32)

        self.up2 = self.up_block(32, 32)
        self.skip2 = nn.Conv2d(48, 16, kernel_size=1)
        self.dec2 = self.conv_block(16, 16)

        self.up1 = self.up_block(16, 16)
        self.skip1 = nn.Conv2d(24, 8, kernel_size=1)
        self.dec1 = self.conv_block(8, 8)

        # Final layer
        self.final = nn.Conv2d(8, self.n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.conv_kernel_size, padding=self.conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.conv_kernel_size, padding=self.conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

    def up_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def next_multiple_of_16(self, n):
        if n % 16 == 0:
            return n
        return (n // 16 + 1) * 16

    def forward(self, x):

        B, C, H, W = x.shape

        H_pad = self.next_multiple_of_16(H)
        W_pad = self.next_multiple_of_16(W)
    
        pad_h = H_pad - H
        pad_w = W_pad - W
    
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
       
        # -------- latent --------
        latent = self.latent(self.pool(x4))

        # -------- Decoder --------
        d4 = self.up4(latent)
        d4 = self.skip4(torch.cat([d4, x4], dim=1))
        d4 = self.dec4(d4)        

        d3 = self.up3(d4)
        d3 = self.skip3(torch.cat([d3, x3], dim=1))
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.skip2(torch.cat([d2, x2], dim=1))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.skip1(torch.cat([d1, x1], dim=1))
        d1 = self.dec1(d1)

        x = self.final(d1)
        
        # remoção do padding
        if pad_h > 0 or pad_w > 0: x = x[:, :, :H, :W]

        return x
'''
