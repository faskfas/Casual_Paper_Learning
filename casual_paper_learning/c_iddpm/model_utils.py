import torch 
import yaml
from .unet import UNet
from .iddpm import IDDPM, ImportantSampler
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
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
        n_blocks=unet_config["n_blocks"]
    )

    return unet

def get_iddpm(config, load_pretrained= True):
    """Get IDDPM from config."""
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

    iddpm_config = config["iddpm"]
    iddpm = IDDPM(
        model=unet,
        n_steps=iddpm_config["n_steps"]
    )

    return iddpm 

class IDDPM_trainer:
    def __init__(self, iddpm: IDDPM, config, image_tensor=None):
        self.sampler = ImportantSampler(iddpm)
        self.iddpm = iddpm 
        self.train_config = config["train"]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_save_path = config["env"]["model_save_path"]

        self.img_size = config["iddpm"]["img_size"]
        self.batch_size = self.train_config["batch_size"]
        self.epochs = self.train_config["epochs"]
        self.lr = self.train_config["lr"]

        if not self.train_config["overfit_for_show"]:
            self.dataset = TensorDataset(image_tensor)
        else:
            img_path = self.train_config["train_dataset_path"]
            img = Image.open(img_path).convert("RGB")
            img_size = config["iddpm"]["img_size"]

            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            image_tensor = transform(img).unsqueeze(0)
            image_tensor = image_tensor.repeat(self.train_config["batch_size"], 1, 1, 1)
        
            self.dataset = TensorDataset(image_tensor)

        self.data_loader = DataLoader(self.dataset, batch_size=self.train_config["batch_size"], shuffle=True)

        self.optimizer = optim.AdamW(self.iddpm.pred_model.parameters(), lr=self.lr)

    def train(self):
        self.iddpm.pred_model.train()

        for epoch in tqdm(range(self.epochs), desc="Training"):
            for x0 in self.data_loader:
                x0 = x0[0].to(self.device)
                bs = x0.shape[0]

                noise = torch.randn_like(x0)
                timesteps, weights = self.sampler.sample(bs, self.device)
                xt = self.iddpm.q_sample(x0, timesteps, noise)

                losses = self.iddpm.training_losses(x0, xt, timesteps, noise)
                self.sampler.update_with_all_losses(timesteps, losses)
                loss = (losses*weights).mean()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (epoch+1) % 1000 == 0:
                print(f'[Epoch {epoch+1}] loss: {loss.item():.6f}')

        save_path = self.model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.iddpm.pred_model.state_dict(), save_path)


if __name__ == '__main__':
    config = get_config('iddpm_unet_config.yaml')
    IDDPM_model = get_iddpm(config, False)
