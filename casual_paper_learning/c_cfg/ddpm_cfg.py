"""
DDPM with CFG, modified from:
https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/ddpm
"""
import torch 
import torch.nn as nn
from typing import Tuple, Optional
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from tqdm import tqdm


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)  # 取出t对应的噪声调度、α或α的连乘
    
    # -> [bs, 1, 1, 1]，运算时广播为[bs, C, H, W]
    return c.reshape(-1, 1, 1, 1)  

class CFGDenoiseDiffusion:
    """
    ## Denoise Diffusion with CFG
    """
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, 
                 n_classes: int = 2, guidance_scale: int=1):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        * `n_classes` is the num of class labels
        """
        super().__init__()

        self.device = device
        self.n_classes = n_classes
        self.guidance_scale = guidance_scale
        
        # UNet作为噪声预测模型
        self.eps_model = eps_model

        # 生成噪声调度β
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # α = 1 - β
        self.alpha = 1. - self.beta
        # α的连乘
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # 总加噪步数
        self.n_steps = n_steps
        # σ^2即方差，就是对应的β
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}

        求出x_t的分布的均值和方差
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}

        求出x_0加噪t步后的图像x_t
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, c=None):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        # 不使用CFG
        c = torch.ones(t.shape[0], dtype=torch.long, device=self.device) * self.n_classes
        eps_theta = self.eps_model(xt, t, c, False)
        
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        
        # Sample，根据公式结合模型预测的噪声，求出的均值和方差，之后采样得到x_{t-1}
        return mean + (var ** .5) * eps
    
    def p_sample_with_cfg(self, xt: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """CFG单步采样"""
        eps_cond = self.eps_model(xt, t, c, False)
        force_drop_idx = torch.ones(c.shape[0], dtype=torch.long, device=self.device)
        eps_uncond = self.eps_model(xt, t, c, False, force_drop_idx)

        eps_theta = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        
         # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        
        # Sample，根据公式结合模型预测的噪声，求出的均值和方差，之后采样得到x_{t-1}
        return mean + (var ** .5) * eps


    def loss(self, x0: torch.Tensor, c: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        # 批梯度下降，一次处理batch_size个样本
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        # 训练模式下，进行类别嵌入时，会自动实现类别丢弃
        eps_theta = self.eps_model(xt, t, c, True)

        # MSE loss
        return F.mse_loss(noise, eps_theta)
    
    def sample(self, save_dir, sample_num=1, img_size=64, img_channels=3):
        """采样"""
        self.eps_model.eval()

        with torch.no_grad():
            x = torch.randn([sample_num, img_channels, img_size, img_size], device=self.device)

            for t_ in tqdm(range(self.n_steps), desc="Sampling without CFG"):
                t = self.n_steps - t_ - 1
                x  = self.p_sample(x, x.new_full((sample_num,), t, dtype=torch.long))
            
            # DDPM输入和输出在-1~1之间，在此之前需要转为0~1才能存为图像
            x = (x + 1) / 2.0
            x = x.clamp(0.0, 1.0)

            os.makedirs(save_dir, exist_ok=True)
            for i in range(sample_num):
                save_image(
                    x[i],
                    f"{save_dir}/sample_{i}.png",
                    normalize=False
                )

                print(f"Saved sample_{i}.png to {save_dir}")

    def sample_with_cfg(self, save_dir, c, sample_num=1, img_size=64, img_channels=3):
        """CFG采样"""
        self.eps_model.eval()

        with torch.no_grad():
            # 每一步采样，需要两次预测: 带条件和不带条件
            x = torch.randn([sample_num, img_channels, img_size, img_size], device=self.device)

            for t_ in tqdm(range(self.n_steps), desc=f"Sampling with CFG, class: {c}"): 
                t = self.n_steps - t_ - 1
                x = self.p_sample_with_cfg(x, x.new_full((sample_num,), t, dtype=torch.long), 
                                           x.new_full((sample_num,), c, dtype=torch.long))
                
            # DDPM输入和输出在-1~1之间，在此之前需要转为0~1才能存为图像
            x = (x + 1) / 2.0
            x = x.clamp(0.0, 1.0)

            os.makedirs(save_dir, exist_ok=True)
            for i in range(sample_num):  
                save_image(
                    x[i],
                    f"{save_dir}/sample_{i}.png",
                    normalize=False
                )

                print(f"Saved sample_{i}.png to {save_dir}")
