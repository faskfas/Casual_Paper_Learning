from ..models.ddpm_cfg import CFGDenoiseDiffusion
from ..models.unet import UNet
from ..models.vae import VAE 
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def train_ldm(pretrained_vae: VAE, ddpm_cfg: CFGDenoiseDiffusion, checkpoint_save_dir="./ckpts/ddpm_cfg/", 
              batch_size=2, lr=0.00002, epochs=5000, dataset_dir='./train', num_classes=2, img_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    pretrained_vae.to(device)
    ddpm_cfg.eps_model.to(device)
    pretrained_vae.eval()
    ddpm_cfg.eps_model.train()

    for param in pretrained_vae.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(ddpm_cfg.eps_model.parameters(), lr=lr)

    pbar = tqdm(range(epochs), desc="Training DDPM_CFG")
    for epoch in pbar:
        total_loss = 0.0

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                z, _, _ = pretrained_vae.encode(imgs)
                z = z * 0.18215

            optimizer.zero_grad()
            loss = ddpm_cfg.loss(x0=z, c=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix({"avg_loss": round(avg_loss, 6)})

        if (epoch + 1) % 2000 == 0:
            torch.save(
                ddpm_cfg.eps_model.state_dict(),
                os.path.join(checkpoint_save_dir, f"cfg_ddpm_unet_epoch_{epoch+1}.pth")
            )
            print(f'[trainer]Saved: cfg_ddpm_unet_epoch_{epoch+1}.pth')

    torch.save(
        ddpm_cfg.eps_model.state_dict(),
        os.path.join(checkpoint_save_dir, f"cfg_ddpm_unet_iter{epochs}.pth")
    )
    print(f'[trainer]Saved: cfg_ddpm_unet_iter{epochs}.pth')
