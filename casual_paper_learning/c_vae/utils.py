import yaml 
from .vae import VAE
from .train import VAE_trainer
import torch


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
        latent_dim=vae_config["latent_dim"],
        img_size=vae_config["img_size"],
        input_dim=vae_config["input_dim"]
    ).to(device)

    if load_pretrained:
        vae.load_state_dict(torch.load(vae_config["checkpoint_load_path"]))
        print(f'[VAE]Loaded checkpoint from {vae_config["checkpoint_load_path"]}')

    return vae 

def get_trainer(config):
    """Get trainer from config."""
    trainer_config = config["trainer"]
    vae = get_vae(config)

    trainer = VAE_trainer(
        vae=vae,
        batch_size=trainer_config["batch_size"],
        epochs=trainer_config["epochs"],
        lr=trainer_config["lr"],
        dataset_dir=trainer_config["dataset_dir"],
        model_save_dir=trainer_config["checkpoint_save_dir"]
    )

    return trainer