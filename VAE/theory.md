# VAE 损失理论推导
## 损失函数
我们的目标就是**最大化似然**$\log p_{\theta}(x)$，也就是在模型构建的分布下，$x$出现的概率最大化，接下来就进行一下变式

$$
\begin{aligned}
\log p_\theta(x)
&= \log \int_z p_\theta(x,z) dz \\
&= \log \int_z p_\theta(z)p_\theta(x|z) dz \\ 
&= \log \int_z q_\phi(z|x) \cdot \frac{p_\theta(z)p_\theta(x|z)}{q_\phi(z|x)} dz \\
&\ge \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(z)p_\theta(x|z)}{q_\phi(z|x)}\right]
\end{aligned}
$$

其中:
- $\phi$为编码器参数，$\theta$为解码器参数
- $\log p_{\theta}(x)$: 解码器构建的分布下，$x$出现的**合理性(概率)**，也是我们要**最大化**的目标，让模型重构的$\tilde{x}$尽可能接近$x$
- $q_{\phi}(z|x)$: 编码器根据输入$x$构建的潜在空间表示$z$的分布
- $p_{\theta}(z)$: $z$的先验分布，这里认为是标准正态分布$\mathcal{N} (0, I)$
- $p_{\theta}(x|z)$: 解码器根据输入$z$重构的$x$的分布

而$\mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(z)p_\theta(x|z)}{q_\phi(z|x)}\right]$称为 **ELBO(证据下界)**

进一步拆开，可以得到:

$$
\text{ELBO} = \mathbb{E}_{q_{\phi}(z|x)}\left[ \log p_{\theta}(x|z) \right] - \text{D}_{\text{KL}}\left( q_{\phi}(z|x) \parallel p_{\theta}(z) \right)
$$

最大化 **ELBO** 也可以最大化 $\log p_{\theta}(x)$，$\log p_{\theta}(x)$**难以直接求得，但是他的 ELBO 我们可以求出**；

给上面的式子取负数，就变成了需要最小化的损失:

$$
\mathcal{L} = - \mathbb{E}_{q_{\phi}(z|x)}\left[ \log p_{\theta}(x|z) \right] + \text{D}_{\text{KL}}\left( q_{\phi}(z|x) \parallel p_{\theta}(z) \right)
$$

其中:
- $- \mathbb{E}_{q_{\phi}(z|x)}\left[ \log p_{\theta}(x|z) \right]$: **重建损失**，就是让重建得到的$\tilde{x}$和原始的$x$尽可能相近
- $\text{D}_{\text{KL}}\left( q_{\phi}(z|x) \parallel p_{\theta}(z) \right)$: **KL 散度损失**，就是让我们的假设(**编码后的$z$符合标准正态分布**)和真实的后验分布$q_{\phi}(z|x)$(**编码器根据输入$x$编码得到的实际$z$符合的分布**)尽可能相似，这样我们才可以在标准正太分布中采样得到潜在空间的$z$解码得到有意义的图像$\tilde{x}$

在实际实现中，最终的损失函数如下:

$$
\mathcal{L} = \sum_{i=1}^N \left[
    \| \tilde{x}_i - x_i \|^2 + 
    \frac{1}{2} \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)
\right]
$$

其中$\mu_i, \sigma_i$为编码器$q_{\phi}(x_i)$输出得到的均值和方差

## 重参数化采样
因为编码器$q_{\phi}(x)$得到的是均值和方差，我们需要在它们构造的正态分布中采样，但是直接进行采样操作**是不可导的**，为了保证梯度的传递，我们需要利用**重参数化**技巧进行采样:

$$
z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)
$$

这样反向传播就能进行梯度回传了