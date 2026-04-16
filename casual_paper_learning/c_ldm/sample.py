from .models.ddpm_cfg import CFGDenoiseDiffusion
from .models.vae import VAE 
from torchvision.utils import save_image
from typing import Optional
import os
import torch


def sample(ddpm_cfg:CFGDenoiseDiffusion, vae:VAE, num_sample=4, save_dir="./sample", 
           class_label:Optional[int]=None, z_img_size=16, z_channels=4):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm_cfg.eps_model.to(device)
    ddpm_cfg.eps_model.eval()
    vae.to(device)
    vae.eval()

    with torch.no_grad():
        if class_label is not None:
            z = ddpm_cfg.sample_with_cfg(c=class_label, sample_num=num_sample, img_size=z_img_size, 
                                         img_channels=z_channels)
        else:
            z = ddpm_cfg.sample(sample_num=num_sample, img_size=z_img_size, img_channels=z_channels)

        z = z / 0.18215
        imgs = vae.decode(z)

        imgs = torch.clamp(imgs, -1., 1.)
        imgs = (imgs + 1) / 2

    for i in range(num_sample):
        img_name = f"sample_class{class_label}_{i}.png" if class_label is not None else f"sample_random_{i}.png"
        save_image(imgs[i], os.path.join(save_dir, img_name))

    print(f"[Sampler]Saved {num_sample} samples to {save_dir}")
