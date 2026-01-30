import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, kernel_size, input_channels=1):
        super().__init__()
        # é 1 para problemas binários (uma classe de interesse), pois a loss aplica sigmoide internamente,
        # então a saída da rede deve ser logits com um canal apenas

        self.input_channels = input_channels
        self.n_classes = 1
        self.pool = nn.MaxPool2d(2)
        self.dropout_prob = 0
        self.conv_kernel_size = kernel_size
        self.conv_padding = self.conv_kernel_size // 2  # 'same'

        # Encoder
        self.enc1 = self.conv_block(input_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Decoder
        self.up3 = self.up_block(256, 256) # os blocos de upsampling mantêm a quantidade de canais, diferente da unet que dobrava para comportar os canais das skipt connections
        self.dec3 = self.conv_block(256, 128)

        self.up2 = self.up_block(128, 128)
        self.dec2 = self.conv_block(128, 64)

        self.up1 = self.up_block(64, 64)
        self.dec1 = self.conv_block(64, 32)

        # Final layer
        self.final = nn.Conv2d(32, self.n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.conv_kernel_size, padding=self.conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.conv_kernel_size, padding=self.conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
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
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Decoder
        x = self.up3(x4)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        x = self.final(x)
        
        # remoção do padding
        if pad_h > 0 or pad_w > 0: x = x[:, :, :H, :W]

        return x

class ConvNetEncoder(nn.Module):
    def __init__(self, convnet: ConvNet):
        super().__init__()
        self.enc1 = convnet.enc1
        self.enc2 = convnet.enc2
        self.enc3 = convnet.enc3
        self.enc4 = convnet.enc4
        self.pool = convnet.pool

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        return x1, x2, x3, x4

class ConvNetDecoder(nn.Module):
    def __init__(self, encoder: ConvNetEncoder, convnet: ConvNet, n_classes=1):
        super().__init__()
        self.encoder = encoder
        self.convnet = convnet

        self.up3 = convnet.up3
        self.dec3 = convnet.dec3
        self.up2 = convnet.up2
        self.dec2 = convnet.dec2
        self.up1 = convnet.up1
        self.dec1 = convnet.dec1
        self.final = convnet.final

        self.next_multiple_of_8 = self.convnet.next_multiple_of_8
    
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
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        x = self.final(x)

        # remoção do padding
        if pad_h > 0 or pad_w > 0: x = x[:, :, :H, :W]

        return x