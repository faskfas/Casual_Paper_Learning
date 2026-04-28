# DDPM 理论推导

## 马尔可夫链与 DDPM

简单而言，马尔科夫链就是说，**下一时刻的状态$x_{t+1}$仅依赖当前时刻的状态$x_t$**，而与前面所有时刻的状态无关，即:

$$
p(x_{t+1}|x_1, \dots, x_t) = p(x_{t+1}|x_t)
$$

在 DDPM 中，所谓的**加噪**和**去噪**过程都建模为了一个**马尔可夫链**，它认为，对图像$x_0 \sim q(x_0)$($q(x)$为所有图像样本组成的分布)逐渐添加高斯噪声之后，能够得到$x_t \sim \mathcal{N}(0, I)$，**并且这个过程可逆**，也就是说能够从标准高斯噪声中采样$x_t$，通过加噪的反向过程(即去噪)能够得到$\bar x_0$，且满足$\bar x_0 \sim q(x_0)$，也就是**去噪后的图像满足真实图像组成的分布，即它是一张有意义的图像**，接下来我们就具体介绍一下他是怎么建模**加噪和去噪**这两个过程的

## 向前加噪

假设我们有真实图像$x_0 \sim q(x_0)$，经过$T$次高斯噪声的添加，就得到了满足我们采样时用到的先验分布的$x_T$，即$x_T \sim \mathcal{N}(0, I)$，添加噪声的过程为:

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t|\sqrt{\alpha_t}x_{t-1}, \beta_t I) \tag{1}
$$

其中$\beta_t$称为**噪声调度**，它是随$t$单调递增的超参数，可以自由设置，$\alpha_t = 1 - \beta_t$，上式描述了以$x_{t-1}$为条件，$x_t$所有可能位置的概率密度。那么我们就能从$x_0 \sim q(x_0)$出发，循环上面的公式得到任意时刻$t$的中间状态$x_t$，你可能注意到了，**这个过程非常的低效，我们不能一步得到$x_t$，这会导致计算非常繁琐**，那么有没有一个**归纳公式**，能够直接得到$x_t$呢？下面我们就来推导一下$x_t$的归纳式

## 重参数化技巧推导归纳式(可跳)

依据(1)时定义的过程，使用**重参数化技巧**，我们从$x_{t-1}$得到$x_t$的过程如下:

$$
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

我们把$x_{t-1}$也拆开，得到:

$$
x_t = \sqrt{\alpha_t \alpha_{t-1}}x_{t-2} + 
\sqrt{\alpha_t \beta_{t-1}} \epsilon_{t-1} + 
\sqrt{\beta_t} \epsilon_t
$$

我们再拆$x_{t-2}$，得到:

$$
x_t = \sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}}x_{t-3} + 
\sqrt{\alpha_t \alpha_{t-1} \beta_{t-2}} \epsilon_{t-2} + 
\sqrt{\alpha_t \beta_{t-1}} \epsilon_{t-1} + 
\sqrt{\beta_t} \epsilon_t
$$

类似的过程，我们一直拆到只剩下$x_0$为止:

$$
x_t = \sqrt{\alpha_t \alpha_{t-1} \dots \alpha_1}x_0 + 
\sqrt{\beta_t} \epsilon_t + 
\sqrt{\alpha_t \beta_{t-1}} \epsilon_{t-1} + 
\cdots + 
\sqrt{\alpha_t \alpha_{t-1} \dots \alpha_2 \beta_1} \epsilon_1
$$

我们定义: $\bar \alpha_t = \alpha_1 \alpha_2 \dots \alpha_t$，代入得:

$$
x_t = \sqrt{\bar \alpha_t}x_0 + 
\sqrt{\beta_t} \epsilon_t + 
\sqrt{\alpha_t \beta_{t-1}} \epsilon_{t-1} + 
\cdots + 
\sqrt{\alpha_t \alpha_{t-1} \dots \alpha_2 \beta_1} \epsilon_1
$$

**因为所有的$\epsilon$都是独立从标准高斯分布中采样而来，那么我们可以把后面噪声项的和(记作$\epsilon_{\text{sum}}$)等价为从$\mathcal{N}(0, \beta_t + \alpha_t \beta_{t-1} + \dots + \alpha_t \alpha_{t-1} \dots \alpha_2 \beta_1)$中采样而来**，而我们定义$\beta_t = 1 - \alpha_t$，代入得:

$$
\begin{aligned}
\beta_t + \alpha_t \beta_{t-1} + \dots + \alpha_t \alpha_{t-1} \dots \alpha_2 \beta_1

&=  1 - \alpha_t + \alpha_t (1 - \alpha_{t-1}) + \alpha_t \alpha_{t-1}(1 - \alpha_{t-2}) + \dots + \alpha_t \alpha_{t-1} \dots \alpha_2 (1 - \alpha_1)

\\&= 1 - \alpha_t + \alpha_t - \alpha_t \alpha_{t-1} + \alpha_t \alpha_{t-1} - \alpha_t \alpha_{t-1} \alpha_{t-2} + \dots + \alpha_t \alpha_{t-1} \dots \alpha_2 - \alpha_t \alpha_{t-1} \dots \alpha_1
\end{aligned}
$$

你可以发现，**中间的所有项都抵消了，只剩下一头一尾**，那么:

$$
\begin{aligned}
\beta_t + \alpha_t \beta_{t-1} + \dots + \alpha_t \alpha_{t-1} \dots \alpha_2 \beta_1 &= 1 - \alpha_t \alpha_{t-1} \dots \alpha_1

\\&= 1 - \bar\alpha_t
\end{aligned}
$$

即:

$$
\epsilon_{\text{sum}} \sim \mathcal{N}(0, 1 - \bar\alpha_t)
$$

那么使用**重参数化采样**$\epsilon_{\text{all}}$得到:

$$
\epsilon_{\text{sum}} = \sqrt{1 - \bar\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

那么我们就得到了**归纳采样式**:

$$
x_t = \sqrt{\bar \alpha_t}x_0 + \sqrt{1 - \bar\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

## 反向去噪

我们已经能够从$x_0$出发，得到加噪任意时间步$t$的$x_t$了:

$$
x_t = \sqrt{\bar \alpha_t}x_0 + \sqrt{1 - \bar\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

那么反向的过程就是，从最终的$x_T \sim \mathcal{N}(0, I)$出发，经过$T$次反向去噪，逐渐得到$x_0$，在这一步中，**我们就引入一个参数为$\theta$的模型，他需要做到给定$x_t$，得到去噪一步后的$x_{t-1}$，这个过程我们记作$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}| \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$**，也就是说模型需要依据($x_t, t$)预测出$x_{t-1}$满足的高斯分布的均值和方差

那么我们训练模型，需要一个真实的参考均值和参考方差，它们怎么来？(**如果你还是只关心核心结论，那么下面的推导你也可以跳过**)其实是通过($x_0, x_t$)为条件的$x_{t-1}$的真实后验分布$q(x_{t-1}|x_0, x_t)$得到的，也就是描述了：假设知道了$x_t$对应的真实$x_0$，$x_{t-1}$的分布，我们把这个后验分布拆开:

$$
\begin{aligned}
q(x_{t-1}|x_0, x_t)
&= \frac{q(x_t|x_{t-1},x_0)\,q(x_{t-1}|x_0)}{q(x_t|x_0)}
\\&= \frac{q(x_t|x_{t-1})\,q(x_{t-1}|x_0)}{q(x_t|x_0)} & (马尔可夫链，只依赖上一状态)
\end{aligned}
$$

这三个$q$都表示向前加噪的过程，它们分别是:

$$
q(x_t|x_{t-1}) = \mathcal{N}\left(\sqrt{\alpha_t}x_{t-1},\;\beta_t I\right)

\\

q(x_{t-1}|x_0) = \mathcal{N}\left(\sqrt{\bar\alpha_{t-1}}x_0,\;(1-\bar\alpha_{t-1})I\right)

\\

q(x_t|x_0) = \mathcal{N}\left(\sqrt{\bar\alpha_t}x_0,\;(1-\bar\alpha_t)I\right)
$$

我们回顾一下高斯分布的公式: 

$$
\mathcal{N}(\mu, \sigma^2I) = 
\frac{1}{(2 \pi \sigma^2)^{d/2}} \exp (-\frac{1}{2 \sigma^2 }\parallel x - \mu \parallel^2)
$$

其中$d$为向量$x$的维度，那么我们就有:

$$
\mathcal{N}(\mu, \sigma^2I) \propto \exp (-\frac{1}{2 \sigma^2 }\parallel x - \mu \parallel^2) \tag{1}
$$

基于此，对于$q(x_t|x_{t-1},x_0)$和$q(x_{t-1}|x_0)$，我们有:

$$
q(x_t | x_{t-1}) \propto \exp\left( -\frac{\left\|x_t - \sqrt{\alpha_t}x_{t-1}\right\|^2}{2\beta_t} \right)

\\

q(x_{t-1} | x_0) \propto \exp\left( -\frac{\left\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0\right\|^2}{2(1 - \bar{\alpha}_{t-1})} \right)
$$

而对于$q(x_{t-1}|x_0, x_t)$中分母归一化项$q(x_t|x_0)$，它是一个与$x_{t-1}$无关的常数，所以我们有:

$$
q(x_{t-1}|x_0, x_t) \propto q(x_t|x_{t-1},x_0)\,q(x_{t-1}|x_0)
$$

即:

$$
q(x_{t-1} | x_0, x_t) \propto \exp\left( -\frac{1}{2} \left[ \frac{\left\|x_t - \sqrt{\alpha_t}x_{t-1}\right\|^2}{\beta_t} + \frac{\left\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0\right\|^2}{1 - \bar{\alpha}_{t-1}} \right] \right)
$$

我们继续回顾一下，拆开(1)式的平方项，我们可以得到:

$$
-\frac{1}{2 \sigma^2 }\parallel x - \mu \parallel^2 = 
-\frac{1}{2\sigma^2}\parallel x \parallel^2 + \frac{1}{\sigma^2}\mu^T x + C
$$

其中$C$为常数，若我们高斯分布式子的指数项有形如如下的式子:

$$
-\frac{1}{2}A \parallel x \parallel^2 + B^T x + C \tag{2}
$$

那么就有:

$$
A = \frac{1}{\sigma^2}, \quad B= \frac{\mu}{\sigma^2}
$$

那么为了直接“**读出**”$q(x_{t-1} | x_0, x_t)$的均值和方差，我们就需要配方成(2)这样的式子，**其实只需要关注二次项$\parallel x_{t-1} \parallel^2$和一次项$x_{t-1}$的系数，对应出A和B就可以了**，那么:

$$
A \parallel x_{t-1} \parallel^2 = \left(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar\alpha_{t-1}}\right) \parallel x_{t-1} \parallel^2
$$

$$
\begin{aligned}
B^T x_{t-1} &= \left[-\frac{2\sqrt{\alpha_t}}{\beta_t} x_t^T x_{t-1}
-\frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} x_0^T x_{t-1}\right] \cdot (-\frac{1}{2})

\\&= \left(
\frac{\sqrt{\alpha_t}}{\beta_t} x_t
+
\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} x_0
\right)^T x_{t-1}
\end{aligned}
$$

那么就有:

$$
\begin{aligned}                                            
\tilde\mu &= \frac{B}{A}
\\&= \frac{\frac{\sqrt{\alpha_t}}{\beta_t} x_t +
\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} x_0}{\frac{\alpha_t}{\beta_t} + 
\frac{1}{1-\bar{\alpha}_{t-1}}}
\\&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t + \sqrt{\bar{\alpha}_{t-1}}\beta_t 
x_0}{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}
\\&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t +
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0
\end{aligned}
$$

$$
\begin{aligned}
\tilde\sigma^2 &= \frac{1}{A}

\\&= \frac{1}{\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}}

\\&= \frac{\beta_t(1-\bar{\alpha}_{t-1})}{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}    

\\&= \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
\end{aligned}
$$

分母$\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t$是这样化简的:

$$
\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t = 
\alpha_t - \alpha_t\bar\alpha_{t-1} + \beta_t = 
\alpha_t - \bar\alpha_t + 1 - \alpha_t = 
1 - \bar\alpha_t
$$

我们发现后验参考方差只和我们的超参数相关，因此不需要预测方差，直接可以用于去噪；我们只需要预测出均值，尽可能让训练时的均值$\mu_\theta(x_t, t)$和后验参考均值$\tilde\mu$一致，那么训练函数就是:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim U(1, T), x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0, 1)} 
\parallel \mu_\theta(x_t, t) - \tilde\mu(x_t, t) \parallel_2^2
$$

注意在上式中，**我们的均值是$x_{t-1}$的的分布的均值**；因为我们是通过重参数化的方式从$x_0$直接得到$x_t$的，依据这个采样式: $x_t = \sqrt{\bar \alpha_t}x_0 + \sqrt{1 - \bar\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$，我们可以反解$x_0$:

$$
x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}
$$

把$x_0$带入到均值式子中并化解，可以有:

$$
\tilde\mu(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)
$$

那么我们模型的预测也可以写成类似的形式:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta \right)
$$

也就是说，我们只需要让模型预测出添加的噪声$\epsilon$就行了，那么我们就**得到了 DDPM 中真正的损失函数**:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim U(1, T), x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0, 1)} 
\parallel \epsilon_\theta(x_t, t) - \epsilon \parallel_2^2
$$
