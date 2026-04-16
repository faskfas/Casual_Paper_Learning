"""
Modified from:
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
"""
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from .vit import Transformer, ViT
import os
import matplotlib.pyplot as plt


class MAE(nn.Module):
    def __init__(self, *, encoder: ViT, decoder_dim, masking_ratio = 0.75, decoder_depth = 1, 
                 decoder_heads = 8, decoder_dim_head = 64):  # *表示它之后必须关键字传参
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.masking_ratio = masking_ratio

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]  # [num_patches, dim]

        # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.to_patch = encoder.to_patch_embedding[0]  # -> [bs, seq_len(h*w), patch_dim]
        # nn.LayerNorm(patch_dim) -> nn.Linear(patch_dim, dim) -> nn.LayerNorm(dim)
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])  # -> [bs, seq_len, dim]

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]  # patch_dim(p1*p2*c)

        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)  # [bs, seq_len, dim] -> [bs, seq_len, patch_dim]

    def forward(self, batch_imgs:torch.Tensor):
        batch_imgs = batch_imgs.to(self.device)
        
        # -----------------------------------------------------------------------------------------
        # patchify
        # -----------------------------------------------------------------------------------------
        # [bs, 3, H, W] -> [bs, seq_len, patch_dim]
        patches = self.to_patch(batch_imgs)
        batch, num_patches, *_ = patches.shape

        # -----------------------------------------------------------------------------------------
        # Embedding，并添加位置编码
        # -----------------------------------------------------------------------------------------
        # [bs, seq_len, patch_dim] -> [bs, seq_len, dim]
        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":  # 跳过cls token
            # NOTE: 这里源码报错了，源码是self.encoder.pos_embedding[:, 1:(num_patches + 1)]
            tokens += self.encoder.pos_embedding[1:(num_patches + 1)]  
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(self.device, dtype=tokens.dtype)

        # -----------------------------------------------------------------------------------------
        # 按照掩码比例进行掩码操作，并编码-解码
        # -----------------------------------------------------------------------------------------
        # 获取掩码和非掩码的索引
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = self.device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # 获取不需要掩码的token进行编码
        batch_range = torch.arange(batch, device = self.device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]  # -> [bs, unmasked_seq_len, dim]
        
        # 获取需要掩码的token用于计算重建损失
        masked_patches = patches[batch_range, masked_indices]
        
        # unmasked token的编码-解码
        encoded_tokens = self.encoder.transformer(tokens)
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)  # 位置还是原图像的位置

        # masked patches的编码-解码，掩码之后的token是一个统一的表示
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)  # -> [bs, masked_seq_len, dim]
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # 拼接二者
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=self.device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # 取出掩码token的解码结果，并unpatchify
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        # [bs, maeked_seq_len, dim] -> [bs, maeked_seq_len, patch_dim]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # -----------------------------------------------------------------------------------------
        # 计算重建损失
        # -----------------------------------------------------------------------------------------
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        return recon_loss
    
    @torch.no_grad()
    def vis_mask_and_unmask(self, batch_imgs:torch.Tensor, save_dir="./imgs/mask_vis"):
        self.eval()
        batch = batch_imgs.shape[0]
        os.makedirs(save_dir, exist_ok=True)
        batch_imgs = batch_imgs.to(self.device)

        patches = self.to_patch(batch_imgs)
        batch, num_patches, *_ = patches.shape

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = self.device).argsort(dim = -1)
        masked_indices = rand_indices[:, :num_masked]

        masked_patches = patches.clone()
        batch_idx = torch.arange(batch, device=self.device)[:, None]
        masked_patches[batch_idx, masked_indices] = 0.5

        *_, image_height, _ = batch_imgs.shape

        patches_to_img = lambda p: rearrange(
            p, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=image_height // self.encoder.patch_height,
            p1=self.encoder.patch_height,
            p2=self.encoder.patch_width,
            c=3
        )

        img_original = patches_to_img(patches)
        img_masked = patches_to_img(masked_patches)

        for i in range(batch):
            ori = img_original[i].cpu().permute(1, 2, 0).numpy()
            msk = img_masked[i].cpu().permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 2, figsize=(5, 3))
            fig.subplots_adjust(wspace=0.3)

            axes[0].imshow(ori)
            axes[0].set_title("origin")
            axes[0].axis('off')

            axes[1].imshow(msk)
            axes[1].set_title("masked")
            axes[1].axis('off')

            save_path = os.path.join(save_dir, f"mask_and_unmask_vis_{i}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
