import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch 
from PIL import Image


def vis_img_change(origin_img_path, vis_interval, func, 
                   output_dir, gif_name, reverse=False):
    """Visualize how the image changes during the noising and denosing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / gif_name

    img = Image.open(origin_img_path).convert("RGB")
    # img = img.resize((256, 256))
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np / 127.5) - 1.0  # -> range(-1, 1)
    x0 = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    frames = []
    total_steps = 1000

    for t in range(0, total_steps, vis_interval):
        t_tensor = torch.tensor([t], dtype=torch.long, device=x0.device)

        with torch.no_grad():
            xt = func(x0, t_tensor)  # -> noised img

        # Restore to img 
        xt_np = xt.squeeze(0).permute(1, 2, 0).cpu().numpy()
        xt_np = (xt_np.clip(-1.0, 1.0) + 1.0) * 127.5
        xt_np = xt_np.astype(np.uint8)

        frames.append(Image.fromarray(xt_np))

    if not reverse:
        frames[0].save(
            fp=gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
    else:
        denoise_frames = frames[::-1]
        denoise_frames[0].save(
        fp=gif_path,
        save_all=True,
        append_images=denoise_frames[1:],
        duration=100,
        loop=0
    )

    print(f'GIF saved to: {gif_path}')
