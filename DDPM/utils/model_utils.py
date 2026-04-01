import torch 
import yaml
from .unet import UNet
from .ddpm import DenoiseDiffusion
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import os


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

def get_ddpm(config, load_pretrained= True):
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

    ddpm_config = config["ddpm"]
    ddpm = DenoiseDiffusion(
        eps_model=unet,
        n_steps=ddpm_config["n_steps"],
        device=device
    )

    return ddpm 

class DDPM_trainer:
    def __init__(self, ddpm_model, device, config, image_tensor=None):
        self.train_config = config["train"]
        self.model_save_path = config["env"]["model_save_path"]

        self.ddpm_model = ddpm_model
        self.device = device

        if not self.train_config["overfit_for_show"]:
            self.dataset = TensorDataset(image_tensor)
        else: 
            img_path = self.train_config["train_dataset_path"]
            img = Image.open(img_path).convert("RGB")
            img_size = config["ddpm"]["img_size"]

            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            image_tensor = transform(img).unsqueeze(0)
            image_tensor = image_tensor.repeat(self.train_config["batch_size"], 1, 1, 1)
        
            self.dataset = TensorDataset(image_tensor)

        self.data_loader = DataLoader(self.dataset, batch_size=self.train_config["batch_size"], shuffle=True)

        self.optimizer = optim.Adam(self.ddpm_model.eps_model.parameters(), lr=self.train_config["lr"])

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.train_config["epochs"]

        self.ddpm_model.eps_model.train()

        for epoch in range(epochs):
            for data in self.data_loader:
                data = data[0].to(self.device)
                self.optimizer.zero_grad()
                loss = self.ddpm_model.loss(data)
                loss.backward()
                self.optimizer.step()
            
            if (epoch+1) % 5 == 0:
                print(f'[Epoch {epoch+1}] loss: {loss.item():.6f}')

        save_path = self.model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.ddpm_model.eps_model.state_dict(), save_path)


if __name__ == "__main__":
    config = get_config('./configs/ddpm_unet_config.yaml')
    DDPM_model_pretrained = get_ddpm(config, True)

    
