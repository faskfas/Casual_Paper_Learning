import torch
import torch.nn as nn
from torchvision.utils import save_image
import os


class VAE(nn.Module):
    def __init__(self, latent_dim:int = 128, img_size:int = 64, input_dim:int = 64):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.input_dim = input_dim
        self.down_size = img_size//16
        self.flatten_size = input_dim*8*self.down_size*self.down_size

        # [bs, 3, img_size, img_size], range(-1, 1) -> 
        self.encoder = nn.Sequential(   
            nn.Conv2d(3, input_dim, 4, 2, 1),  # -> [bs, input_dim, img_size/2, img_size/2]
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim*2, 4, 2, 1),  # -> [bs, input_dim*2, img_size/4, img_size/4]
            nn.ReLU(),
            nn.Conv2d(input_dim*2, input_dim*4, 4, 2, 1),  # -> [bs, input_dim*4, img_size/8, img_size/8]
            nn.ReLU(),
            nn.Conv2d(input_dim*4, input_dim*8, 4, 2, 1),  # -> [bs, input_dim*8, img_size/16, img_size/16]
            nn.ReLU(),
            nn.Flatten()  # -> [bs, input_dim*8*(img_size/16)*(img_size/16)]
        )

        # [bs, input_dim*8*(img_size/16)*(img_size/16)] -> [bs, latent_dim]
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # -> [bs, input_dim*8, img_size/16, img_size/16]
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (input_dim*8, self.down_size, self.down_size)),  # -> [bs, input_dim*8, img_size/16, img_size/16]
            nn.ConvTranspose2d(input_dim*8, input_dim*4, 4, 2, 1),  # -> [bs, input_dim*4, img_size/8, img_size/8]
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim*4, input_dim*2, 4, 2, 1),  # -> [bs, input_dim*2, img_size/4, img_size/4]
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim*2, input_dim, 4, 2, 1),  # -> [bs, input_dim, img_size/2, img_size/2]
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim, 3, 4, 2, 1),  # -> [bs, 3, img_size, img_size]
            nn.Tanh()  # -> range(-1, 1)  
        )

    def encode(self, x:torch.Tensor):
        feat = self.encoder(x)  # -> [bs, input_dim*8*(img_size/16)*(img_size/16)]

        # -> [bs, latent_dim] * 2
        return self.fc_mu(feat), self.fc_logvar(feat)
    
    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std  # -> [bs, latent_dim]
    
    def decode(self, z:torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)  # -> [bs, input_dim*8, img_size/16, img_size/16]

        return self.decoder(x)  # -> [bs, 3, img_size, img_size]
    
    def forward(self, x:torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar
    
    def print_forward(self, x:torch.Tensor):
        print(f'[input x size]: {x.size()}')

        mu, logvar = self.encode(x)
        print(f'[mu & logvar size]: {mu.size()}')

        z = self.reparameterize(mu, logvar)
        print(f'[reparameterized z size]: {z.size()}')

        output = self.decode(z)
        print(f'[after decoding, output size]: {output.size()}')

    def inference(self, sample_num:int = 4, save_dir:str = './samples'):
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            sample = torch.randn(sample_num, self.latent_dim).to(device)
            sample = self.decode(sample).cpu()

            for i in range(sample_num):
                save_image(
                    sample[i],
                    f"{save_dir}/sample_{i}.png",
                    normalize=False
                )

                print(f"Saved sample_{i}.png to {save_dir}")
