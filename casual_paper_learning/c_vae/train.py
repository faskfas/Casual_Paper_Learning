import torch
import torch.nn as nn
from .vae import VAE
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os


def loss_func(recon_x:torch.Tensor, x:torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor):
    # MSE loss
    MSE = nn.functional.mse_loss(recon_x, x, reduction="sum")

    # KL loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

class VAE_trainer:
    def __init__(self, vae:VAE, batch_size:int = 2, epochs:int = 50, lr:float = 1e-3, 
                 dataset_dir:str = './train', model_save_dir:str = './ckpts'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr 
        self.dataset_dir = dataset_dir
        self.model_save_dir = model_save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = vae.to(self.device)

        os.makedirs(self.model_save_dir, exist_ok=True)

    def train(self):
        self.vae.train()

        dataloader = self._load_dataset()
        optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)

        pbar = tqdm(range(self.epochs), desc="Training")

        for epoch in pbar:
            total_loss = 0.0

            for data, _ in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()

                recon_batch, mu, logvar = self.vae(data)
                loss = loss_func(recon_batch, data, mu, logvar)

                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            pbar.set_postfix({"avg_loss": round(avg_loss, 6)})

            if (epoch+1) % 1000 == 0:
                torch.save(
                    self.vae.state_dict(),
                    os.path.join(self.model_save_dir, f"vae_epoch_{epoch+1}.pth")
                )

                print(f'[trainer]Saved: vae_epoch_{epoch+1}.pth')

        torch.save(
            self.vae.state_dict(),
            os.path.join(self.model_save_dir, f"vae_iter{self.epochs}.pth")
        )   

        print(f'[trainer]Saved: vae_iter{self.epochs}.pth')

    def _load_dataset(self):
        # 归一化为range(-1, 1)
        transform = transforms.Compose([
            transforms.Resize((self.vae.img_size, self.vae.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.ImageFolder(root=self.dataset_dir, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        return dataloader
    