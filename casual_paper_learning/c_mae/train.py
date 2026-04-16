from .mae import MAE
from .vit import ViT
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


# --------------------------------------------------------------------------------------------------------------------------
# MAE pretraining encoder(ViT)
# --------------------------------------------------------------------------------------------------------------------------
def load_dataset_mae(img_size=224, dataset_dir="./train", batch_size=2):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader

def train_mae(mae_model: MAE, img_size=64, checkpoint_save_dir="./ckpts/", batch_size=2,
              lr=1e-4, epochs=100, dataset_dir='./train'):
    dataloader = load_dataset_mae(img_size, dataset_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    mae_model.to(device)
    mae_model.train()

    optimizer = torch.optim.AdamW(mae_model.parameters(), lr=lr, weight_decay=0.05)

    pbar = tqdm(range(epochs), desc="Training MAE")
    for epoch in pbar:
        total_loss = 0

        for data, _ in dataloader:
            x = data.to(device)

            loss = mae_model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        pbar.set_postfix({"avg_loss": round(avg_loss, 6)})

        if (epoch + 1) % 1000 == 0:
            torch.save(
                mae_model.encoder.state_dict(),
                os.path.join(checkpoint_save_dir, f"vit_epoch_{epoch+1}.pth")
            )
            print(f'[trainer]Saved: vit_epoch_{epoch+1}.pth')

    torch.save(
        mae_model.encoder.state_dict(),
        os.path.join(checkpoint_save_dir, f"vit_iter{epochs}.pth")
    )

    print(f'[trainer]Saved: vit_iter{epochs}.pth')


# --------------------------------------------------------------------------------------------------------------------------
# fine-tuning pretrained ViT to classify images
# --------------------------------------------------------------------------------------------------------------------------
def train_vit_finetune(vit_model: ViT, img_size=64, checkpoint_save_dir="./ckpts/", batch_size=2,
                       lr=1e-4, epochs=50, dataset_dir='./train'):
    dataloader = load_dataset_mae(img_size, dataset_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    vit_model.to(device)
    vit_model.train()

    optimizer = torch.optim.AdamW(vit_model.parameters(), lr=lr, weight_decay=0.05)
    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm(range(epochs), desc="Fine-tuning ViT")
    for epoch in pbar:
        total_loss = 0
        correct = 0
        total = 0

        for data, labels in dataloader:
            x = data.to(device)
            labels = labels.to(device)

            logits = vit_model(x)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = correct / total
        pbar.set_postfix({"avg_loss": round(avg_loss, 6), "acc": round(accuracy, 4)})

        if (epoch + 1) % 100 == 0:
            torch.save(
                vit_model.state_dict(),
                os.path.join(checkpoint_save_dir, f"vit_finetune_epoch_{epoch+1}.pth")
            )
            print(f'[trainer]Saved: vit_finetune_epoch_{epoch+1}.pth')

    torch.save(
        vit_model.state_dict(),
        os.path.join(checkpoint_save_dir, f"vit_finetune_iter{epochs}.pth")
    )

    print(f'[trainer]Saved: vit_finetune_iter{epochs}.pth')

