import yaml 
from .vqvae import VQVAE, PixelCNNWithEmbedding
import torch 
from .pixelcnn import GatedPixelCNN


def get_config(config_path):
    """Load yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def get_vqvae(config, load_pretrained=False):
    """Get VQVAE from config."""
    vqvae_config = config["vqvae"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae = VQVAE(
        input_dim=vqvae_config["input_dim"],
        hidden_dim=vqvae_config["hidden_dim"],
        num_embedding=vqvae_config["num_embedding"]
    ).to(device)

    if load_pretrained:
        vqvae.load_state_dict(torch.load(vqvae_config["checkpoint_load_path"]))
        print(f'[VQVAE]Loaded checkpoint from {vqvae_config["checkpoint_load_path"]}')

    return vqvae

def get_pixelcnn_test(config, load_pretrained=False):
    """Get PixelCNN from config.(test)"""
    pixelcnn_config = config["pixel_cnn"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixelcnn = GatedPixelCNN(
        n_blocks=pixelcnn_config["n_blocks"],
        p=pixelcnn_config["p"],
        linear_dim=pixelcnn_config["linear_dim"]
    ).to(device)

    if load_pretrained:
        pixelcnn.load_state_dict(torch.load(pixelcnn_config["checkpoint_load_path"]))
        print(f'[PixelCNN]Loaded checkpoint from {pixelcnn_config["checkpoint_load_path"]}')

    return pixelcnn

def get_pixelcnn(config, load_pretrained=False):
    pixelcnn_config = config["pixel_cnn"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pixelcnn = PixelCNNWithEmbedding(
        n_blocks=pixelcnn_config["n_blocks"],
        hidden_dim=pixelcnn_config["hidden_dim"],
        linear_dim=pixelcnn_config["linear_dim"],
        bn=True,
        color_level=pixelcnn_config["num_embedding"]
    ).to(device)

    if load_pretrained:
        pixelcnn.load_state_dict(torch.load(pixelcnn_config["checkpoint_load_path"]))
        print(f'[PixelCNN]Loaded checkpoint from {pixelcnn_config["checkpoint_load_path"]}')

    return pixelcnn
