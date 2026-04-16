import yaml 
import torch
from .vit import ViT
from .mae import MAE


def get_config(config_path):
    """Load yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def get_vit(config, load_pretrained=[False, None]):
    """Get ViT from config."""
    vit_config = config["vit"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vit = ViT(
        image_size=vit_config["image_size"],
        patch_size=vit_config["patch_size"],
        num_classes=vit_config["num_classes"],
        dim=vit_config["dim"],
        depth=vit_config["depth"],
        heads=vit_config["heads"],
        mlp_dim=vit_config["mlp_dim"],
        dropout=vit_config["dropout"],
        emb_dropout=vit_config["emb_dropout"]
    ).to(device)

    load_pretrained, type = load_pretrained

    if load_pretrained:
        if type == "pretrained":
            path = vit_config["pretrained_checkpoint_load_path"]
        elif type == "finetuned":
            path = vit_config["finetuned_checkpoint_load_path"]

        vit.load_state_dict(torch.load(path))
        print(f'[ViT]Loaded checkpoint from {path}')

    return vit

def get_mae(config, load_pretrained=False):
    """Get MAE from config."""
    mae_config = config["mae"]
    vit = get_vit(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mae = MAE(
        encoder=vit,
        decoder_dim=mae_config["decoder_dim"],
        masking_ratio=mae_config["masking_ratio"],
        decoder_depth=mae_config["decoder_depth"],
        decoder_heads=mae_config["decoder_heads"],
        decoder_dim_head=mae_config["decoder_dim_head"]
    ).to(device)

    if load_pretrained:
        mae.load_state_dict(torch.load(mae_config["checkpoint_load_path"]))
        print(f'[MAE]Loaded checkpoint from {mae_config["checkpoint_load_path"]}')

    return mae
