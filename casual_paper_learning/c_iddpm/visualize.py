import numpy as np
from pathlib import Path
import torch 
from PIL import Image
from c_ddpm import vis_img_change


def vis_img_change(origin_img_path, vis_interval, func, get_cos_nosise_scheduler_func,
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
            xt = func(x0, t_tensor, get_cos_nosise_scheduler_func)  # -> noised img

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

def q_sample(x0: torch.Tensor, t: torch.Tensor, get_cos_nosise_scheduler_func) -> torch.Tensor:
    """
    Get xt from x0 after noising t steps.
    这个函数用于演示加噪和去噪的过程，具体实现以最终的实现类为主
    """
    # 噪声调度
    beta = get_cos_nosise_scheduler_func()
    alpha = 1.0 - beta  # \alpha_t
    alpha_bar = torch.cumprod(alpha, dim=0)  # \bar{\alpha_t}，即连乘

    eps = torch.randn_like(x0)  # 标准高斯噪声
    # 单个\sqrt{\bar{\alpha_t}}值 -> [bs, 1, 1 ,1] -> [bs, C, H, W](运算时广播)
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)  
    # 单个 1 - \sqrt{\bar{\alpha_t}} 值 -> [bs, 1, 1 ,1] -> [bs, C, H, W](运算时广播)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)

    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps

    return xt

def vis_img_change_cos(origin_img_path, vis_interval, get_cos_nosise_scheduler_func,
                       output_dir, gif_name):
    """
    Visualize how the image changes during the noising with cos noise scheduler.
    """
    vis_img_change(origin_img_path, vis_interval, q_sample, get_cos_nosise_scheduler_func,
                   output_dir, gif_name, False)
