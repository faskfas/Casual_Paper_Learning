from ..models.lpips import LPIPS
from ..models.discriminator import Discriminator
from ..models.vae import VAE
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def train_vae(vae: VAE, img_size=128, checkpoint_save_dir="./ckpts/vae/", batch_size=2,
              lr=0.00001, epochs=2000, dataset_dir='./train', disc_weight=0.5, perceptual_weight=1,
              kl_weight=0.000005, autoencoder_acc_steps=1, disc_start=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    vae.to(device)
    vae.train()

    recon_criterion = nn.MSELoss()
    disc_criterion = nn.MSELoss()

    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)

    discriminator = Discriminator(im_channels=3).to(device)

    optimizer_g = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    step_count = 0

    pbar = tqdm(range(epochs), desc="Training VAE")
    for epoch in pbar:
        recon_losses = []
        kl_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []

        for data, _ in dataloader:
            step_count += 1
            x = data.to(device)

            output, mean, logvar = vae(x)

            # -------------------------------------------------------------------------------------------------
            # Optimize Generator
            # -------------------------------------------------------------------------------------------------
            # Reconstruction loss
            recon_loss = recon_criterion(output, x)

            # KL divergence loss: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            # kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.size(0)

            # Perceptual loss
            lpips_loss = torch.mean(lpips_model(output, x))

            # Total generator loss
            g_loss = recon_loss + kl_weight * kl_loss + perceptual_weight * lpips_loss

            # Adversarial loss(after disc_start steps)
            if step_count > disc_start:
                disc_fake_pred = discriminator(output)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones_like(disc_fake_pred))
                g_loss += disc_weight * disc_fake_loss
                gen_losses.append(disc_fake_loss.item())

            g_loss = g_loss / autoencoder_acc_steps
            g_loss.backward()

            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            perceptual_losses.append(lpips_loss.item())

            # -------------------------------------------------------------------------------------------------
            # Optimize Discriminator
            # -------------------------------------------------------------------------------------------------
            if step_count > disc_start:
                disc_fake_pred = discriminator(output.detach())
                disc_real_pred = discriminator(x)

                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                  torch.zeros_like(disc_fake_pred))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                  torch.ones_like(disc_real_pred))
                disc_loss = disc_weight * (disc_fake_loss + disc_real_loss) / 2
                disc_loss = disc_loss / autoencoder_acc_steps
                disc_loss.backward()
                disc_losses.append(disc_loss.item() * autoencoder_acc_steps)

                if step_count % autoencoder_acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()

            # Update generator
            if step_count % autoencoder_acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

        # Update any remaining gradients
        optimizer_g.step()
        optimizer_g.zero_grad()
        if step_count > disc_start:
            optimizer_d.step()
            optimizer_d.zero_grad()

        avg_recon = sum(recon_losses) / len(recon_losses)
        avg_kl = sum(kl_losses) / len(kl_losses)
        avg_perceptual = sum(perceptual_losses) / len(perceptual_losses)

        log_dict = {
            "recon": round(avg_recon, 4),
            "kl": round(avg_kl, 6),
            "perceptual": round(avg_perceptual, 4),
        }

        if len(disc_losses) > 0:
            log_dict["disc"] = round(sum(disc_losses) / len(disc_losses), 4)
            log_dict["gen"] = round(sum(gen_losses) / len(gen_losses), 4)

        pbar.set_postfix(log_dict)

        # Save checkpoint
        if (epoch + 1) % 2000 == 0 or epoch == epochs - 1:
            torch.save(
                vae.state_dict(),
                os.path.join(checkpoint_save_dir, f"vae_epoch_{epoch+1}.pth")
            )
            if step_count > disc_start:
                torch.save(
                    discriminator.state_dict(),
                    os.path.join(checkpoint_save_dir, f"disc_epoch_{epoch+1}.pth")
                )
            print(f'[trainer]Saved: vae_epoch_{epoch+1}.pth')

    # Save final checkpoint
    torch.save(vae.state_dict(), os.path.join(checkpoint_save_dir, f"vae_iter{epochs}.pth"))
    if step_count > disc_start:
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_save_dir, f"disc_iter{epochs}.pth"))
    print(f'[trainer]Saved: vae_iter{epochs}.pth')
