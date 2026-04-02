"""
Thanks to:
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
https://github.com/LittleNyima/code-snippets/blob/master/iddpm-tutorial/iddpm_training.py
"""
import numpy as np
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
from tqdm import tqdm
from .loss_utils import normal_kl, discretized_gaussian_log_likelihood
from PIL import Image
import os


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)  # 取出t对应的噪声调度、α或α的连乘
    
    # -> [bs, 1, 1, 1]，运算时广播为[bs, C, H, W]
    return c.reshape(-1, 1, 1, 1)

def get_cos_noise_scheduler(n_steps=1000, max_beta=0.999, s=0.008) -> np.ndarray:
    """
    余弦噪声调度
    
    :param n_steps: 总扩散步数
    :param max_beta: beta最大值
    :param s: 计算f(t)用到的平滑值s
    :return: 余弦噪声调度
    """
    # f(t) = cos((t/T+s)/(1+s) * (pi/2))^2，这里t/T就是t_norm
    # f(t)其实就是alpha_bar[t]
    ft = lambda t_norm: math.cos(((t_norm+s)/(1+s)) * (math.pi/2)) ** 2

    betas = []
    for i in range(n_steps):
        t1 = i/n_steps
        t2 = (i+1)/n_steps
        # beta_t = 1 - f(t+1)/t(t)，不能超过max_beta
        betas.append(min(1 - ft(t2)/ft(t1), max_beta))

    return torch.tensor(betas, dtype=torch.float32)

class ImportantSampler:
    """
    重要性采样实现
    """
    def __init__(self, iddpm_model, history_per_term=10, uniform_prob=0.001):
        """
        :param iddpm_model: 扩散模型
        :param history_per_term: 每个t保存最近的loss数
        :param uniform_prob: 采用均匀采样的概率
        """
        self.iddpm = iddpm_model
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob

        # 记录每个t产生的loss，以及每个t已记录的loss个数
        self._loss_history = np.zeros([self.iddpm.n_steps, self.history_per_term], 
                                      dtype=np.float64)
        self._loss_cnt = np.zeros([self.iddpm.n_steps], dtype=int)

    def weights(self):
        """更新每个t的权重"""
        if not self._warm_up():  # 若没有收集足够多的loss，均匀采样
            return np.ones([self.iddpm.n_steps], dtype=np.float64)
        
        # 求出每个t的loss^2的均值 -> [n_steps, 1]
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        # 归一化 weights
        weights /= np.sum(weights)
        # final_weights = (1-ε)*loss_weights + ε*uniform_weights
        weights *= 1 - self.uniform_prob  # (1-ε)*loss_weights
        weights += self.uniform_prob / len(weights)  # + ε*uniform_weights

        return weights

    def _warm_up(self):
        return (self._loss_cnt == self.history_per_term).all()
    
    def update_with_all_losses(self, ts, losses):
        """t的历史损失更新"""
        for t, loss in zip(ts, losses):
            if self._loss_cnt[t] == self.history_per_term:
                # 移除最早加入的loss
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_cnt[t]] = loss
                self._loss_cnt[t] += 1

    def sample(self, batch_size, device):
        """重要性采样"""
        # 概率 -> # [n_steps, 1]
        w = self.weights()  
        p = w / np.sum(w)

        # 按概率选择bs个t
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        
        # loss_weight = 1/(n_steps*p[t])，让频繁抽取到的t对应的loss权重降低
        # 用于后续给损失加权，防止模型偏向训练频繁采样的t
        loss_weights_np = 1 / (len(p) * p[indices_np])
        loss_weights = torch.from_numpy(loss_weights_np).float().to(device)
        
        return indices, loss_weights

class IDDPM(nn.Module):
    """
    IDDPM类
    """
    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.pred_model = model.to(self.device)  # output -> [bs, 2*image_channels, H, W]
        self.n_steps = n_steps
        
        """
        betas: 余弦噪声调度
        alphas: 1 - betas
        alphas_cumprod: alpha连乘
        alpha_cumprod_prev: 得到alpha_bar_{t-1}
        alpha_cumprod_next: 得到alpha_bar_{t+1}
        timesteps: 去噪步 T-1 -> 0
        """
        self.betas = get_cos_noise_scheduler(n_steps=self.n_steps).to(self.device)
        self.alphas = (1 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.concat((torch.ones(1).to(self.alphas_cumprod), self.alphas_cumprod[:-1])).to(self.device)
        self.alphas_cumprod_next = torch.concat((self.alphas_cumprod[1:], torch.zeros(1).to(self.alphas_cumprod))).to(self.device)
        self.timesteps = torch.arange(self.n_steps - 1, -1, -1).to(self.device)
    
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        求出x_t的分布的均值和方差
        """
        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alphas_cumprod, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alphas_cumprod, t)
        
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        求出x_0加噪t步后的图像x_t
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps
    
    def pred_mean_logvar(self, pred_noises: torch.Tensor, pred_vars: torch.Tensor,
                         batch_xt: torch.Tensor, t_inference: torch.Tensor):
        """
        根据(x_t, t)预测x_{t-1}的均值和标准差对数
        """
        # --------------------------------------------------------------------------------------------------
        # 参数预计算
        # --------------------------------------------------------------------------------------------------
        betas = self.betas.to(self.device)
        alphas = self.alphas.to(self.device)
        alphas_cumprod = self.alphas_cumprod.to(self.device)
        alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)

        sqrt_recip_alphas_cumprod = (1.0 / alphas_cumprod) ** 0.5
        sqrt_recipm1_alphas_cumprod = (1.0 / alphas_cumprod - 1.0) ** 0.5
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.concat((posterior_variance[1:2], posterior_variance[1:])))
        posterior_mean_coef1 = betas * alphas_cumprod_prev ** 0.5 / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas ** 0.5 / (1.0 - alphas_cumprod)


        # 计算均值，与DDPM相同，只不过写法不同而已
        x_0 = gather(sqrt_recip_alphas_cumprod, t_inference)*batch_xt - \
                gather(sqrt_recipm1_alphas_cumprod, t_inference)*pred_noises
        mean = gather(posterior_mean_coef1, t_inference)*x_0.clamp(-1, 1) + \
                gather(posterior_mean_coef2, t_inference)*batch_xt
        
        # 计算方差
        min_log = gather(posterior_log_variance_clipped, t_inference)  # 公式里的 \bar\beta
        max_log = gather(torch.log(betas), t_inference)  # \beta
        frac = (pred_vars + 1.0) / 2.0  # v(就是frac) = (model_out+1)/2
        log_variance = frac * max_log + (1.0 - frac) * min_log

        return mean, log_variance

    @torch.no_grad()
    def p_sample(self, batch_size: int, in_channels: int, img_size: int, save_dir="./imgs"):
        """
        从标准高斯噪声采样，并采样去噪
        """
        batch_xt = torch.randn(batch_size, in_channels, img_size, img_size, device=self.device)
        timesteps = self.timesteps.to(self.device)

        for t in tqdm(timesteps, desc="Sampling"):
            t_inference = batch_xt.new_full((batch_size,), t, dtype=torch.long)  # -> [bs]
            preds = self.pred_model(batch_xt, t_inference)
            pred_noises, pred_vars = torch.split(preds, in_channels, dim=1)

            mean, log_variance = self.pred_mean_logvar(pred_noises, pred_vars, batch_xt, t_inference)

            # 学习的方差
            if t > 0:
                stddev = torch.exp(0.5 * log_variance)  # -> 最终标准差
            else:
                # 最后一步去噪方差设为0
                stddev = torch.zeros_like(mean)

            epsilon = torch.randn_like(batch_xt)
            batch_xt = mean + stddev*epsilon

        # range (-1, 1) -> (0, 1)
        # permute: [bs, C, H, W] -> [bs, H, W, C]，从而正常保存图像
        batch_imgs = (batch_xt / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

        for i in range(batch_size):
            img = Image.fromarray((batch_imgs[i] * 255).astype('uint8'))
            img.save(os.path.join(save_dir, f"sample_{i}.png"))

        print(f'Saved all images at: {save_dir}')

        # return batch_imgs
    
    def get_true_log_var(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
        """
        根据x0, xt获得真实x_{t-1}的均值和标准差对数
        """
        # --------------------------------------------------------------------------------------------------
        # 参数预计算
        # --------------------------------------------------------------------------------------------------
        betas = self.betas.to(self.device)
        alphas = self.alphas.to(self.device)
        alphas_cumprod = self.alphas_cumprod.to(self.device)
        alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.concat((posterior_variance[1:2], posterior_variance[1:])))
        posterior_mean_coef1 = betas * alphas_cumprod_prev ** 0.5 / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas ** 0.5 / (1.0 - alphas_cumprod)


        posterior_mean = gather(posterior_mean_coef1, t) * x0 + gather(posterior_mean_coef2, t) * xt
        posterior_log_variance_clipped = gather(posterior_log_variance_clipped, t)

        return posterior_mean, posterior_log_variance_clipped
    
    def vlb_loss(self, pred_noises: torch.Tensor, pred_vars: torch.Tensor,
                 x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
        """
        IDDPM的L_vlb
        """
        pred_mean, pred_logvar = self.pred_mean_logvar(pred_noises, pred_vars, xt, t)
        true_mean, true_logvar = self.get_true_log_var(x0, xt, t)

        # KL散度损失 [bs, C, H, W] -> [bs]
        # 注: /log(2.0)，是为了把e为底变为x为底
        # log2(x) = log(x)/log(2)
        kl = normal_kl(true_mean, true_logvar, pred_mean, pred_logvar)
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / math.log(2.0)

        # 负对数似然损失 [bs, C, H, W] -> [bs]
        # nll = discretized_gaussian_log_likelihood(x0, pred_mean, pred_logvar * 0.5)
        nll = discretized_gaussian_log_likelihood(x0, means=pred_mean, log_scales=pred_logvar * 0.5)
        nll = nll.mean(dim=list(range(1, len(nll.shape)))) / math.log(2.0)

        # 最后一步去噪用NLL，其他则用KL
        results = torch.where(t == 0, nll, kl)

        return results
    
    def training_losses(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor,
                        noises: torch.Tensor, vlb_weight: float = 1e-3):
        _, channels, _, _ = x0.shape 
        pred = self.pred_model(xt, t)
        pred_noises, pred_vars = torch.split(pred, channels, dim=1)

        # L_simple，和DDPM一致
        l_simple = (pred_noises - noises) ** 2
        l_simple = l_simple.mean(dim=list(range(1, len(l_simple.shape))))
    
        # L_vlb
        l_vlb = self.vlb_loss(pred_noises, pred_vars, x0, xt, t)

        return l_simple + vlb_weight * l_vlb


if __name__ == "__main__":
    pass
