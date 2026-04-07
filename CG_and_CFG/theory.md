# Classifier Guidance
**Classifier Guidance**(下文称 CG)，即分类器引导，可以实现让扩散模型生成想要的类别图像，这里简单介绍一下 CG 理论

在 DDPM 中，我们模型预测的噪声满足以下正比关系:

$$
\epsilon_\theta(x_t) \propto -\nabla_{x_t} \log p_\theta(x_t)
$$

更精确的形式是:

$$
\nabla_{x_t} \log p_\theta(x_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t)
$$

- $\sqrt{1-\bar\alpha_t}$就是标准差$\sigma_t$

其中$\log p_\theta(x_t)$指的是，真实的$x_0$加噪得到的$x_t$，**在模型预测的分布下的合理性**(即**似然**)，而式子$\nabla_{x_t} \log p_\theta(x_t)$称作分数函数(Score Function)，我们不管它是什么东西，先往下看

我们知道，DDPM 学习的是$p_\theta(x)$，那么加上条件后，结合贝叶斯公式，可以得到:

$$
p_\theta(x|c) = \frac{p_\theta(c|x) \cdot p_\theta(x)}{p(c)} 
$$

取对数后可得:

$$
\log p_\theta(x|c) = \log p_\theta(c|x) + \log p_\theta(x) - \log p(c)
$$

对x求梯度，变成分数函数的形式，可得:

$$
\nabla_x \log p_\theta(x|c) = \nabla_x \log p_\theta(c|x) + \nabla_x \log p_\theta(x)
$$

**注意**，因为$\log p(c)$跟x没关系，求梯度就变成0了

怎么理解呢？$\nabla_x \log p_\theta(x)$我们知道就是 DDPM，可以理解成模型构造对于x的合理分布的方向；$\nabla_x \log p_\theta(c|x)$就需要**额外引入一个分类器**来实现，那么它就是分类器的梯度，用于引导修正$\nabla_x \log p_\theta(x)$对于分布的构造，让他往条件 $c$ 的方向纠正

在实际控制引导的过程中，会引入一个类似温度系数的超参数(这里标为$w$)，用于控制引导的强弱，那么式子就变成了:

$$
\nabla_x \log p_\theta(x|c) = \nabla_x \log p_\theta(x) + w \cdot \nabla_x \log p_\theta(c|x)  
$$

把我们最开始的正比关系的式子带入，得到:

$$
\begin{cases}
\nabla_{x_t} \log p_\theta(x_t) = -\dfrac{1}{\sigma_t} \epsilon_\theta(x_t) \\[8pt]
\nabla_{x_t} \log p_\theta(x_t | c) = -\dfrac{1}{\sigma_t} \hat{\epsilon}_\theta(x_t | c)
\end{cases}
$$

代入得：
$$
-\frac{1}{\sigma_t} \hat{\epsilon}_\theta(x_t | c) = -\frac{1}{\sigma_t}\epsilon_\theta(x_t) + w \cdot \nabla_{x_t} \log p_\theta(c | x_t)
$$

两边同乘$(-\sigma_t)$得：
$$
\hat{\epsilon}_\theta(x_t | c) = \epsilon_\theta(x_t) - w \cdot \sigma_t \cdot \nabla_{x_t} \log p_\theta(c | x_t)
$$

那么我们就实现了带条件的噪声预测，从而控制模型生成图像的类别

# Classifier-Free Guidance

我们已经实现了使用分类器得梯度纠正扩散模型的生成，从而控制类别，但是需要训练一个额外的分类器，且会遇到分类器质量参差不齐的问题，为了改进，有人就引进了**不带分类器的类别引导(Classifier-Free Guidance，下文称 CFG)**，直接把类别嵌入到模型中进行训练，再在推理时进行引导，下面我们来进行一下推导

我们还是先把最开始得到的式子放在下面:

$$
\nabla_x \log p_\theta(x|c) = \nabla_x \log p_\theta(x) + w \cdot \nabla_x \log p_\theta(c|x) \tag{1}
$$

让$w=1$，得到下面的式子:

$$
\nabla_x \log p_\theta(x|c) = \nabla_x \log p_\theta(x) + \nabla_x \log p_\theta(c|x) \tag{2}
$$

换个写法:

$$
\nabla_x \log p_\theta(c|x) = \nabla_x \log p_\theta(x|c) - \nabla_x \log p_\theta(x) \tag{3}
$$

式子3带入式子1可以得到:

$$
\nabla_x \log p_\theta(x|c) = \nabla_x \log p_\theta(x) + w \cdot \left[ \nabla_x \log p_\theta(x|c) - \nabla_x \log p_\theta(x) \right] \tag{4}
$$

左边加上上标避免混淆，就是:

$$
\nabla_x \log p_\theta(x|c)^{\text{CFG}} = \nabla_x \log p_\theta(x) + w \cdot \left[ \nabla_x \log p_\theta(x|c) - \nabla_x \log p_\theta(x) \right] \tag{5}
$$

式子5就是我们最终的 CFG 引导公式，写成模型里的使用式子就是:

$$
\epsilon_\theta(x_t|c)^{\text{CFG}} = \epsilon_\theta(x_t|\emptyset) + w \cdot \left( \epsilon_\theta(x_t|c) - \epsilon_\theta(x_t|\emptyset) \right)
$$

$\emptyset$表示无条件控制的生成

直观上其实非常好理解，$\epsilon_\theta(x_t|c) - \epsilon_\theta(x_t|\emptyset)$就是一个**方向**，由无条件指向有条件，那么整体式子就是说，模型无条件生成的图像往有条件的方向进行修正，$w$控制着修正的强度，这就是 CFG 的理论了
