from .vqvae import VQVAE, PixelCNNWithEmbedding
import torch 
from torchvision.utils import save_image
import torch.nn.functional as F
import os


def sample(vqvae_model:VQVAE, pixelcnn_model:PixelCNNWithEmbedding, sample_num = 4, 
           sample_save_dir = "./sample", img_size = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae_model = vqvae_model.to(device)
    pixelcnn_model = pixelcnn_model.to(device)
    vqvae_model.eval()
    pixelcnn_model.eval()

    os.makedirs(sample_save_dir, exist_ok=True)

    h, w = vqvae_model.get_latent_hw((sample_num, img_size, img_size))
    input_shape = (sample_num, h, w)
    x = torch.zeros(input_shape).to(device).to(torch.long)

    with torch.no_grad():
            for i in range(h):
                for j in range(w):
                    output = pixelcnn_model(x)  # -> [bs, num_embedding, h, w]
                    # -> [bs, num_embedding]，位置(i, j)的概率分布(softmax logits转换而来)
                    next_pixel_prob = F.softmax(output[:, :, i, j], dim=-1)
                    next_pixel = torch.multinomial(next_pixel_prob, num_samples=1)  # 采样 -> [bs, 1]
                    x[:, i, j] = next_pixel[:, 0]

    # range(-1, 1) -> range(0, 1)
    x_hat = vqvae_model.decode(x)
    x_hat = torch.clamp(x_hat, -1.0, 1.0)
    x_hat = (x_hat + 1.0) / 2.0

    for i in range(x_hat.shape[0]):
        img_recon = x_hat[i].cpu()
        save_recon_path = os.path.join(sample_save_dir, f"sample_img{i}.png")
        save_image(img_recon, save_recon_path)

        print(f'Save sample img at: {save_recon_path}')
