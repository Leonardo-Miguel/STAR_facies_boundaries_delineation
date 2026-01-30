import torch
import torch.nn as nn
import torch.nn.functional as F

# EmbeddingImages — quebra a imagem em patches e projeta cada patch para um token (embedding).
class EmbeddingImages(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size 
        self.embed_dim = embed_dim
        
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
        # Lembrand que incialmente K, Q e V são apenas CÓPIAS DO EMBEDDING
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

        # Retorna representação transformada
        return x


class Vit(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_size = 8
        self.n_classes = 1
        self.embed_dim = 96
        self.depth = 4 # é o número de blocos do transformer
        self.num_heads = 1
        self.mlp_ratio = 4
        self.dropout = 0.4
        self.pos_embed = None

        # patch embedding
        self.patch_embed = EmbeddingImages(self.patch_size, self.embed_dim)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout) for _ in range(self.depth)
        ])

        # Decoder para voltar ao espaço da imagem
        self.decoder_conv = nn.ConvTranspose2d(self.embed_dim, self.n_classes, kernel_size=self.patch_size, stride=self.patch_size)

        self.bottleneck_conv = nn.Conv2d(in_channels=self.depth, out_channels=self.n_classes, kernel_size=1)

    def forward(self, x):
        # guardar H/W originais para cortar depois do upsample
        B, _, H_orig, W_orig = x.shape

        # Patch embedding
        x, n_patches, (H_patches, W_patches) = self.patch_embed(x)  # [B, N_patches, D]

        # pos embed (recria se a grade mudar)
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != n_patches):
            # registrar como parâmetro treinável
            self.pos_embed = nn.Parameter(
                torch.zeros(1, n_patches, self.embed_dim, device=x.device, dtype=x.dtype)
            )

        # adiciona posição. É literalmente a soma do valor da posição á cada embedding, fazendo com que os embeddings do mesmo token/patch já tenha relação direta
        x = x + self.pos_embed  # [B, N_patches, D]

        # Transformer blocks
        all_blocks = []
        for blk in self.blocks:
            # OBS: x deve ser mantido no formato token [B, N_patches, D] durante todos os blocos, por isso a conversão para 2d é armazenada em outro tensor
            x = blk(x)            
            x_2d = x.transpose(1, 2).reshape(B, self.embed_dim, H_patches, W_patches)
            x_2d = self.decoder_conv(x_2d)
            all_blocks.append(x_2d)

        x = torch.cat(all_blocks, dim=1)

        # Decoder final: [B, n_classes, H_pad, W_pad]
        x = self.bottleneck_conv(x)
        
        # Remoção do padding: voltar ao tamanho original
        x = x[:, :, :H_orig, :W_orig]  # [B, n_classes, H_orig, W_orig]
      
        return x
