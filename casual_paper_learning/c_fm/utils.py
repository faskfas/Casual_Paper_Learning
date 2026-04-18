import yaml 
import torch 
from .unet import UNet


def get_config(config_path):
    """Load yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def get_unet(config, load_pretrained=False):
    """Get UNet from config."""
    unet_config = config["unet"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNet(
        image_channels=unet_config["image_channels"],
        n_channels=unet_config["n_channels"],
        ch_mults=tuple(unet_config["ch_mults"]),
        is_attn=tuple(unet_config["is_attn"]),
        n_blocks=unet_config["n_blocks"]
    ).to(device)

    if load_pretrained:
        model_path = unet_config["checkpoint_load_path"]
        state_dict = torch.load(model_path, map_location=device)
        unet.load_state_dict(state_dict)
        
        print(f'Loaded checkpoint from {model_path}')

    return unet
