from .unet import (
    DownBlock, Downsample,
    MiddleBlock,
    UpBlock, Upsample,
    Swish
)
import torch
import torch.nn as nn
from typing import Union, Tuple, List
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class VAEEncoder(nn.Module):
    def __init__(self, image_channels:int = 3, n_channels:int = 64,
                 ch_mults:Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn:Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks:int = 2):
        super().__init__()

        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        down = []
        out_channels = in_channels = n_channels

        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]

            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)
        self.mid = MiddleBlock(out_channels, n_channels * 4)
        self.out_channels = out_channels

    def forward(self, x:torch.Tensor):
        x = self.image_proj(x)

        for m in self.down:
            x = m(x)

        x = self.mid(x)

        return x

class VAEDecoder(nn.Module):
    def __init__(self, image_channels:int = 3, n_channels:int = 64,
                 ch_mults:Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn:Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks:int = 2, encoder_out_channels:int = 1024):
        super().__init__()

        in_channels = encoder_out_channels
        n_resolutions = len(ch_mults)

        self.mid = MiddleBlock(encoder_out_channels, n_channels * 4)

        up = []
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                # no skip connect
                up.append(UpBlock(in_channels-out_channels, out_channels, n_channels * 4, is_attn[i]))

            out_channels = in_channels // ch_mults[i]
            # no skip connect
            up.append(UpBlock(in_channels-out_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels

            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x:torch.Tensor):
        x = self.mid(x)

        for m in self.up:
            x = m(x)

        return self.final(self.act(self.norm(x)))
        
class VAE(nn.Module):
    def __init__(self, image_channels:int = 3, n_channels:int = 64,
                 ch_mults:Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn:Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks:int = 2, z_channels:int = 4):
        super().__init__()

        self.z_channels = z_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = VAEEncoder(image_channels, n_channels, ch_mults, is_attn, n_blocks)
        self.decoder = VAEDecoder(image_channels, n_channels, ch_mults, is_attn, n_blocks, self.encoder.out_channels)

        self.encoder_norm_out = nn.GroupNorm(num_groups=8, num_channels=self.encoder.out_channels)
        self.encoder_conv_out = nn.Conv2d(self.encoder.out_channels, 2*self.z_channels, kernel_size=3, padding=1)
        self.pre_quant_conv = nn.Conv2d(2*self.z_channels, 2*self.z_channels, kernel_size=1)

        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.encoder.out_channels, kernel_size=3, padding=1)

    def encode(self, x:torch.Tensor):
        x = self.encoder(x)
        x = self.encoder_norm_out(x)
        x = self.encoder_conv_out(x)
        output = self.pre_quant_conv(x)
        mean, logvar = torch.chunk(output, 2, dim=1)

        std = torch.exp(0.5*logvar)
        sample = mean + std*torch.randn(mean.shape).to(self.device)

        return sample, mean, logvar 
    
    def decode(self, z:torch.Tensor):
        z = self.post_quant_conv(z)
        decoder_input = self.decoder_conv_in(z)
        output = self.decoder(decoder_input)

        return output
    
    def forward(self, x:torch.Tensor):
        z, mean, logvar = self.encode(x)
        output = self.decode(z)

        return output, mean, logvar
    
    def recon_test(self, dataset_dir="./train", save_dir="./recon/vae", batch_size=2, img_size=128):
        self.eval()
        os.makedirs(save_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        pbar = tqdm(dataloader, desc="Generating compare image")
        with torch.no_grad():
            for bs, (imgs, _) in enumerate(pbar):
                imgs = imgs.to(device)

                recon, *_ = self(imgs)

                imgs = torch.clamp(imgs, -1., 1.)
                imgs = (imgs + 1) / 2
                recon = torch.clamp(recon, -1., 1.)
                recon = (recon + 1) / 2

                for i in range(imgs.shape[0]):
                    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
                    fig.subplots_adjust(wspace=0.3)

                    img_np = imgs[i].permute(1, 2, 0).cpu().numpy()
                    axes[0].imshow(img_np)
                    axes[0].set_title("origin", fontsize=14)
                    axes[0].axis('off')

                    recon_np = recon[i].permute(1, 2, 0).cpu().numpy()
                    axes[1].imshow(recon_np)
                    axes[1].set_title("recon", fontsize=14)
                    axes[1].axis('off')

                    save_path = os.path.join(save_dir, f"compare_batch{bs}_{i}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()

        print(f'[VAE]Compare images saved to: {save_dir}')
