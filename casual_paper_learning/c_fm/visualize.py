import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def visualize_sampling(unet, num_steps=50, img_size=64, device=None, save_dir="./visualize",
                       gif_duration=80, n_show=10, frame_interval=50):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)
    unet.eval()

    x0 = torch.randn(1, 3, img_size, img_size).to(device)
    dt = 1.0 / num_steps
    xt = x0

    frames = []         
    tensor_frames = []  

    with torch.no_grad():
        frames.append(_tensor_to_pil(xt))
        tensor_frames.append(xt.cpu().squeeze(0))

        for i in tqdm(range(num_steps), desc="Sampling for visualization"):
            t = torch.full((1,), i * dt, device=device)
            vt = unet(xt, t)
            xt = xt + vt * dt

            tensor_frames.append(xt.cpu().squeeze(0))
            if (i + 1) % frame_interval == 0 or i == num_steps - 1:
                frames.append(_tensor_to_pil(xt))

    gif_path = os.path.join(save_dir, "sampling_process.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif_duration,
        loop=0
    )

    indices = np.linspace(0, len(tensor_frames) - 1, n_show, dtype=int)
    show_tensors = torch.stack([tensor_frames[i] for i in indices])  # [n_show, C, H, W]

    grid = make_grid(show_tensors, nrow=n_show, normalize=True, value_range=(-1, 1))

    grid_path = os.path.join(save_dir, "sampling_process.png")
    save_image(grid, grid_path)


def _tensor_to_pil(t):
    t = t.detach().cpu().squeeze(0)  # [C, H, W]
    t = (t + 1) / 2.0
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)
