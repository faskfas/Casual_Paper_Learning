import torch 
import yaml
from .unet import UNet
from .ddpm_cfg import CFGDenoiseDiffusion
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import os
from tqdm import tqdm


def get_config(config_path):
    """Load yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
 
def get_unet(config):
    """Get UNet from config."""
    unet_config = config["unet"]
    unet = UNet(
        image_channels=unet_config["image_channels"],
        n_channels=unet_config["n_channels"],
        ch_mults=tuple(unet_config["ch_mults"]),
        is_attn=tuple(unet_config["is_attn"]),
        n_blocks=unet_config["n_blocks"],
        n_classes= unet_config["n_classes"],
        token_dropout_prob=unet_config["token_dropout_prob"]
    )

    return unet

def get_ddpm_cfg(config, load_pretrained= True):
    """Get DDPM from config."""
    device = torch.device(config["env"]["device"])

    unet = get_unet(config).to(device)
    if load_pretrained:
        model_path = Path(config["env"]["model_load_path"])
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=device)
            unet.load_state_dict(state_dict)

            print(f'Loaded checkpoint from {model_path}')
        else:
            print('Checkpoint {model_path} do not exists!')

    ddpm_cfg_config = config["ddpm_cfg"]
    ddpm_cfg = CFGDenoiseDiffusion(
        eps_model=unet,
        n_steps=ddpm_cfg_config["n_steps"],
        device=device,
        n_classes=config["unet"]["n_classes"]
    )

    return ddpm_cfg

class PenguinDataset(Dataset):
    def __init__(self, train_dataset_path, img_size, n_classes):
        super().__init__()

        self.img_size = img_size
        self.classes = [i for i in range(n_classes)]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.img_paths = [
            os.path.join(train_dataset_path, "0", os.listdir(os.path.join(train_dataset_path, "0"))[0]),
            os.path.join(train_dataset_path, "1", os.listdir(os.path.join(train_dataset_path, "1"))[0]),
        ]

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        label = idx % 2  # 0,1,0,1,0,1...
        img = Image.open(self.img_paths[label]).convert("RGB")
        img = self.transform(img)

        return img, label

class CFGDDPM_trainer:
    def __init__(self, ddpm_cfg, config, image_tensor=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.ddpm_cfg = ddpm_cfg
        self.train_config = config["train"]
        self.model_save_path = config["env"]["model_save_path"]

        self.img_size = config["ddpm_cfg"]["img_size"]
        self.batch_size = self.train_config["batch_size"]
        self.epochs = self.train_config["epochs"]
        self.lr = self.train_config["lr"]
        self.train_dataset_path = self.train_config["train_dataset_path"]
        self.n_classes = config["unet"]["n_classes"]

        os.makedirs(self.model_save_path, exist_ok=True)

        self.dataset = PenguinDataset(self.train_dataset_path, self.img_size, self.n_classes)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0
        )

        self.optimizer = torch.optim.Adam(
            self.ddpm_cfg.eps_model.parameters(), 
            lr=self.lr
        )


    def train(self):
        self.ddpm_cfg.eps_model.train()
        
        pbar = tqdm(range(self.epochs), desc="Training")
        for epoch in pbar:
            total_loss = 0.0

            for imgs, labels in self.dataloader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                loss = self.ddpm_cfg.loss(x0=imgs, c=labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(self.dataloader)
            pbar.set_postfix({"avg_loss": round(avg_loss, 6)})

            if (epoch + 1) % 10000 == 0:
                torch.save(
                    self.ddpm_cfg.eps_model.state_dict(),
                    os.path.join(self.model_save_path, f"cfg_ddpm_unet_epoch_{epoch+1}.pth")
                )

        torch.save(
            self.ddpm_cfg.eps_model.state_dict(),
            os.path.join(self.model_save_path, f"cfg_ddpm_unet_final.pth")
        )
