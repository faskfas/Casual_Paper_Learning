import yaml
from .models.vae import VAE 
import torch
from .models.unet import UNet 
from .models.ddpm_cfg import CFGDenoiseDiffusion


def get_config(config_path):
    """Load yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def get_vae(config, load_pretrained=False):
    """Get VAE from config."""
    vae_config = config["vae"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE(
        image_channels=vae_config["image_channels"],
        n_channels=vae_config["n_channels"],
        ch_mults=vae_config["ch_mults"],
        is_attn=vae_config["is_attn"],
        n_blocks=vae_config["n_blocks"],
        z_channels=vae_config["z_channels"]
    ).to(device)

    if load_pretrained:
        vae.load_state_dict(torch.load(vae_config["checkpoint_load_path"]))
        print(f'[VAE]Loaded checkpoint from {vae_config["checkpoint_load_path"]}')

    return vae

def get_unet(config):
    """Get UNet from config."""
    unet_config = config["unet"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNet(
        image_channels=unet_config["image_channels"],
        n_channels=unet_config["n_channels"],
        ch_mults=tuple(unet_config["ch_mults"]),
        is_attn=tuple(unet_config["is_attn"]),
        n_blocks=unet_config["n_blocks"],
        n_classes= unet_config["n_classes"],
        token_dropout_prob=unet_config["token_dropout_prob"]
    ).to(device)

    return unet

def get_ddpm_cfg(config, load_pretrained=False):
    """Get DDPM from config."""
    ddpm_cfg_config = config["ddpm_cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = get_unet(config)
    if load_pretrained:
        unet.load_state_dict(torch.load(ddpm_cfg_config["checkpoint_load_path"]))
        print(f'[DDPM_CFG]Loaded checkpoint from {ddpm_cfg_config["checkpoint_load_path"]}')

    ddpm_cfg = CFGDenoiseDiffusion(
        eps_model=unet,
        n_steps=ddpm_cfg_config["n_steps"],
        device=device,
        n_classes=config["unet"]["n_classes"]
    )

    return ddpm_cfg
