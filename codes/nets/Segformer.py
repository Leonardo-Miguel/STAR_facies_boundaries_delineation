import torch
import torch.nn as nn
import torch.nn.functional as F
        
# EmbeddingImages — quebra a imagem em patches e projeta cada patch para um token (embedding).
class EmbeddingImages(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size 
        self.embed_dim = embed_dim

        # com cada convolução gera UM ÚNICO valor por patch, e saída é um vetor com tamanho (n_patches x 1 x embed_dim)
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x):

        B, C, H, W = x.shape

        # padding para múltiplos de patch_size
        H_pad = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        W_pad = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size

        pad_h = H_pad - H
        pad_w = W_pad - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
            
        # [B, embed_dim, H_pad/ps, W_pad/ps]
        x = self.proj(x)

        H_patches, W_patches = x.shape[2], x.shape[3]

        # [B, N_patches, embed_dim]
        n_patches = H_patches * W_patches
        x = x.flatten(2).transpose(1, 2)

        return x, n_patches, (H_patches, W_patches)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        # Normalização antes da atenção (pré-norm)
        self.norm1 = nn.LayerNorm(self.embed_dim)

        # Multi-Head Self-Attention
        # batch_first=True → entrada/saída no formato [batch, seq_len, embed_dim]
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        # Normalização antes do feed-forward (pré-norm)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # Feed-Forward Network (MLP) expandindo a dimensionalidade e depois reduzindo
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * self.mlp_ratio)), # projeção para dimensão maior
            nn.GELU(),                                                       # não linearidade suave
            nn.Dropout(self.dropout),                                        # regularização
            nn.Linear(int(self.embed_dim * self.mlp_ratio), self.embed_dim), # volta para embed_dim
            nn.Dropout(self.dropout),                                        # regularização
        )

    def forward(self, x):
        # ====== BLOCO DE SELF-ATTENTION ======
        # Normaliza a entrada antes de passar para a atenção
        norm_x = self.norm1(x)

        # Atenção: usa norm_x como query, key e value (self-attention)
        # Lembrando que incialmente K, Q e V são apenas CÓPIAS DO EMBEDDING
        x_attn = self.attn(norm_x, norm_x, norm_x)[0]

        # Adiciona conexão residual: saída da atenção + entrada original
        x = x + x_attn

        # ====== BLOCO FEED-FORWARD ======
        # Normaliza a saída antes de passar pela MLP
        norm_x = self.norm2(x)

        # Passa pela MLP (duas camadas lineares + GELU + dropout)
        x_ffn = self.mlp(norm_x)

        # Adiciona conexão residual: saída da MLP + entrada original
        x = x + x_ffn

        # Retorna a representação transformada
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Reduz H/W pela metade e aumenta canais
        self.reduction = nn.Conv2d(input_dim, output_dim, kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, C, H, W]
        return self.reduction(x)

class Segformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_size = 8
        self.n_classes = 1
        self.embed_dims = [32, 64, 128, 256]  # dims de cada stage
        self.depths = [1, 1, 1, 1]  # blocos por stage
        self.num_heads = 1
        self.mlp_ratio = 2
        self.dropout = 0.2

        # Patch embedding inicial
        self.patch_embed = EmbeddingImages(self.patch_size, self.embed_dims[0])

        # Encoder: múltiplos stages
        self.stages = nn.ModuleList()
        self.patch_mergings = nn.ModuleList()
        
        for i in range(len(self.embed_dims)):
            block = TransformerBlock(self.embed_dims[i], self.num_heads, self.mlp_ratio, self.dropout)
            self.stages.append(block)
            if i < 3:  # não precisa patch merging no último stage
                self.patch_mergings.append(PatchMerging(self.embed_dims[i], self.embed_dims[i+1]))

        # Decoder: upsample cada feature para a maior resolução
        self.decoder_convs = nn.ModuleList()
        for i, embed_dim in enumerate(self.embed_dims):
            feature = nn.ConvTranspose2d(embed_dim, self.n_classes, kernel_size=2**i, stride=2**i)
            self.decoder_convs.append(feature)
            

        self.bottleneck_conv = nn.Conv2d(sum(self.embed_dims), self.n_classes, kernel_size=1)  # fuse

    def forward(self, x):
        B, _, H_orig, W_orig = x.shape

        # Patch embedding inicial
        x, n_patches, (H_patches, W_patches) = self.patch_embed(x)
        #x = x + nn.Parameter(torch.zeros(1, n_patches, self.embed_dims[0], device=x.device))

        features = []

        for i, block in enumerate(self.stages):
            # Transformer blocks de um stage
            x = block(x)
            # Reshape para 2D
            H, W = H_patches, W_patches
            feat_2d = x.transpose(1, 2).reshape(B, self.embed_dims[i], H, W)
            features.append(feat_2d)

            # Patch merging para próximo stage
            if i < 3:
                x = self.patch_mergings[i](feat_2d)
                H_patches, W_patches = x.shape[2], x.shape[3]
                x = x.flatten(2).transpose(1, 2)

        # Decoder: upsample cada feature para H_orig/W_orig
        upsampled_feats = []
        for i, feat in enumerate(features):
            scale_factor = H_orig // feat.shape[2]
            upsampled = F.interpolate(feat, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            upsampled_feats.append(upsampled)

        x = torch.cat(upsampled_feats, dim=1)  # [B, sum_channels, H, W]
        x = self.bottleneck_conv(x)  # reduce to n_classes

        return x
