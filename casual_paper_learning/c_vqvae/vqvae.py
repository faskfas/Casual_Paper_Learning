# Modified from: https://github.com/SingleZombie/DL-Demos/blob/master/dldemos/VQVAE/model.py
import torch
import torch.nn as nn
from .pixelcnn import GatedBlock, GatedPixelCNN


# ------------------------------------------------------------------------------------------------
# VQ-VAE
# ------------------------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, input_dim:int):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(input_dim, input_dim, 1)

    def forward(self, x:torch.Tensor):
        # 形状始终为[bs, input_dim, h, w]
        # 由于模型比较简单，可以不使用batch_norm
        x_1 = self.conv1(self.relu(x)) 
        x_2 = self.conv2(self.relu(x_1))

        return x + x_2

class VQVAE(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_embedding:int):
        super().__init__()

        # input: [bs, input_dim, h, w]
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 4, 2, 1),  # -> [bs, hidden_dim, h/2, w/2]
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),  # -> [bs, hidden_dim, h/4, w/4]
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),  # 不变
            ResidualBlock(hidden_dim), ResidualBlock(hidden_dim)  # 不变
        )

        # 使用均匀分布初始化codebook
        self.vq_embedding = nn.Embedding(num_embedding, hidden_dim)
        self.vq_embedding.weight.data.uniform_(-1.0/num_embedding, 1.0/num_embedding)

        # input: [bs, hidden_dim, h/4, w/4]
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),  # 不变
            ResidualBlock(hidden_dim), ResidualBlock(hidden_dim),  # 不变
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),  # [bs, hidden_dim, h/2, w/2]
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim, 4, 2, 1)  # [bs, input_dim, h, w]
        )

        self.num_downsample = 2

    def forward(self, x:torch.Tensor):
        # [bs, input_dim, h, w] -> [bs, hidden_dim, h/4, w/4]
        ze = self.encoder(x)
        
        # 使用广播机制计算距离
        embedding = self.vq_embedding.weight.data 
        bs, c, h, w = ze.shape
        k, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, k, c, 1, 1)
        ze_broadcast = ze.reshape(bs, 1, c, h, w)
        """
        运算时都变成了: [bs, k, c, h, w]，我们看其中一个bs的[c, h, w]
        可以理解为，我们要把[h, w]中的每个通道向量c替换为codebook中与它最近的向量
        那就要计算通道向量和codebook中k个向量的距离，取最近的向量替换它
        把所有通道向量复制k份得到[k, c, h, w]，再与codebook计算距离，codebook也复制为[k, c, h, w](复制了h*w份)
        计算欧氏距离得到[k, c, h, w]，对c这个维度求和得到[k, h, w]，意思是[h, w]个特征向量和k个codebook向量的距离
        对k通道取min得到每个位置需要的codebook向量的索引，取出之后形状就变回了[h, w, c]
        转置一下就得到替换后的结果[c, h, w]
        """
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2) # -> [bs, k, c, h, w] -> [bs, k, h, w]
        nearest_neighbor = torch.argmin(distance, 1)  # -> [bs, h, w]
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)  # -> [bs, h, w, c] -> [bs, c, h, w]
        decoder_input = ze + (zq - ze).detach()  # stop gradient

        x_hat = self.decoder(decoder_input)

        return x_hat, ze, zq

    @torch.no_grad()
    def encode(self, x:torch.Tensor):
        ze = self.encoder(x)

        embedding = self.vq_embedding.weight.data 
        bs, c, h, w = ze.shape
        k, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, k, c, 1, 1)
        ze_broadcast = ze.reshape(bs, 1, c, h, w)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2) # -> [bs, k, c, h, w] -> [bs, k, h, w]
        nearest_neighbor = torch.argmin(distance, 1)  # -> [bs, h, w]
        
        return nearest_neighbor
        
    @torch.no_grad()
    def decode(self, discrete_latent:torch.Tensor):
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)

        return x_hat
    
    def get_latent_hw(self, input_shape):
        _, h, w = input_shape
        return (h // 2**self.num_downsample, w // 2**self.num_downsample)

# ------------------------------------------------------------------------------------------------
# Gated PixelCNN for VQ-VAE.
# ------------------------------------------------------------------------------------------------
class PixelCNNWithEmbedding(GatedPixelCNN):
    def __init__(self, n_blocks, hidden_dim, linear_dim, bn=True, color_level=256):
        # 这里我封装一下，原始代码里用的p，为了清晰，这里我用hidden_dim替代一下
        # color_level在这里就是num_embedding
        super().__init__(n_blocks, hidden_dim, linear_dim, bn, color_level)

        self.embedding = nn.Embedding(color_level, hidden_dim)
        self.block1 = GatedBlock('A', hidden_dim, hidden_dim, bn)

    def forward(self, x:torch.Tensor):
        # input: [bs, h, w]
        x = self.embedding(x)  # -> [bs, h, w, hidden_dim]
        x = x.permute(0, 3, 1, 2).contiguous()  # -> [bs, hidden_dim, h, w]

        return super().forward(x)  # -> [bs, color_level, h, w]
