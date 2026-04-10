from .vqvae import VQVAE, PixelCNNWithEmbedding
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torchvision.utils import save_image


# --------------------------------------------------------------------------------------------------------------------------
# VQVAE training
# --------------------------------------------------------------------------------------------------------------------------
def load_dataset_vqvae(img_size = 64, dataset_dir = "./train", batch_size = 2):
        # 归一化为range(-1, 1)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

def train_vqvae(vqvae_model:VQVAE, img_size = 64, checkpoint_save_dir = "./ckpts/vqvae/", batch_size = 2, 
                lr = 1e-3, epochs = 100, l_w_embedding = 1, l_w_commitment = 0.25, dataset_dir = './train'):
    dataloader = load_dataset_vqvae(img_size, dataset_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    
    vqvae_model.to(device)
    vqvae_model.train()

    optimizer = torch.optim.Adam(vqvae_model.parameters(), lr)
    mse_loss = nn.MSELoss()

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        total_loss = 0

        for data, _ in dataloader:
            x = data.to(device)

            x_hat, ze, zq = vqvae_model(x)
            l_reconstruct = mse_loss(x, x_hat)  # 重建损失
            l_embedding = mse_loss(ze.detach(), zq)  # 码本学习损失
            l_commitment = mse_loss(ze, zq.detach())  # 承诺损失
            loss = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        pbar.set_postfix({"avg_loss": round(avg_loss, 6)})

        if (epoch+1) % 1000 == 0:
            torch.save(
                vqvae_model.state_dict(),
                os.path.join(checkpoint_save_dir, f"vqvae_epoch_{epoch+1}.pth")
            )

            print(f'[trainer]Saved: vqvae_epoch_{epoch+1}.pth')

    torch.save(
        vqvae_model.state_dict(),
        os.path.join(checkpoint_save_dir, f"vqvae_iter{epochs}.pth")
    )   

    print(f'[trainer]Saved: vqvae_iter{epochs}.pth')
    
def show_train_vqvae_recon(vqvae_model:VQVAE, dataset_dir = './train', save_res_dir="./vqvae_recon", img_size = 64, 
                           batch_size = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae_model.to(device)
    vqvae_model.eval()

    dataloader = load_dataset_vqvae(img_size, dataset_dir, batch_size)

    os.makedirs(save_res_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing VQVAE recon")

        batch_idx = 0
        for data, _ in pbar:
            x = data.to(device)
            bs = x.shape[0]

            # range(-1, 1) -> range(0, 1)
            x_hat, _, _ = vqvae_model(x)
            x_hat = torch.clamp(x_hat, -1.0, 1.0)
            x_hat = (x_hat + 1.0) / 2.0

            for i in range(bs):
                img_recon = x_hat[i].cpu()
                save_recon_path = os.path.join(save_res_dir, f"batch{batch_idx}_img{i}_recon.png")
                save_image(img_recon, save_recon_path)

                print(f'Save recon img at: {save_recon_path}')


# --------------------------------------------------------------------------------------------------------------------------
# PixelCNN training
# --------------------------------------------------------------------------------------------------------------------------
def train_pixelcnn(vqvae_model:VQVAE, pixelcnn_model:PixelCNNWithEmbedding, img_size = 64, 
                   checkpoint_save_dir = "./ckpts/pixelcnn/", batch_size = 2, lr = 1e-3, epochs = 100, 
                   dataset_dir = './train'):
    dataloader = load_dataset_vqvae(img_size, dataset_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    vqvae_model.to(device)
    pixelcnn_model.to(device)
    vqvae_model.eval()
    pixelcnn_model.train()

    optimizer = torch.optim.Adam(pixelcnn_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        total_loss = 0

        for data, _ in dataloader:
            x = data.to(device)

            with torch.no_grad():
                x = vqvae_model.encode(x)

            x_logits = pixelcnn_model(x)
            loss = loss_fn(x_logits, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        pbar.set_postfix({"avg_loss": round(avg_loss, 6)})

        if (epoch+1) % 1000 == 0:
            torch.save(
                pixelcnn_model.state_dict(),
                os.path.join(checkpoint_save_dir, f"pixelcnn_epoch_{epoch+1}.pth")
            )

            print(f'[trainer]Saved: pixelcnn_epoch_{epoch+1}.pth')

    torch.save(
        pixelcnn_model.state_dict(),
        os.path.join(checkpoint_save_dir, f"pixelcnn_iter{epochs}.pth")
    )   

    print(f'[trainer]Saved: pixelcnn_iter{epochs}.pth')    
