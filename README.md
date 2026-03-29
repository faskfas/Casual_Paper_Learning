# Casual Paper Learning

这是我个人在论文阅读过程中，遇到的一些经典架构(比如 Transformer)之后产生了疑惑，从而搜集了源码、博客的笔记，并在 jupyter notebook 上进行了一定的整理，相当于我**个人**的学习笔记，如果你看到了这个repo，也希望对你有一定的帮助

## 目录结构🤗
本项目的目录结构如下所示，每个文件夹对应了一个架构(比如 Transformer):

```text
Casual_Paper_Learning/
│
├── Transformer/
│   ├── imgs  # 架构图、运行结果等
│   └── Transformer.ipynb  # 对应的notebook
│   └── 其他的材料，如公式的详细推导可能也会单独附上
│
...
```

## 如何使用🧐
你可以使用原生的 Jupyter Notebook，在这里我是用 VSCode 的 Jupyter 拓展插件来预览和编辑ipynb文件

- step 1: 安装VSCode、Python相关拓展、Jupyter拓展，以及Ananconda

- step 2: 创建一个conda环境，为了在 notebook 中使用，还需要安装相关的**必要依赖包**:

```bash
conda create -n casual_paper_learning python=3.9
conda activate casual_paper_learning
pip install jupyter notebook ipykernel
```

- step 3: 打开 ipynb 文件，缺少的包直接 pip 即可，没有特殊版本要求，如:

```bash
pip install timm
```

之后打开 ipynb 文件，你就可以进行预览了，note 的开头一般都会给出**架构图**、**参考的博客链接(如知乎)**、**github源码仓库**、**原始论文链接**，本项目的笔记主要都是基于这三个参考信息进行整理(**主要是对架构进行理解**，其他信息比如训练可能会简略一些)，如果你还是有疑问，可以点击这三个链接转跳到对应的文章或源码进行参考

## 其他相关信息😊
在没有特殊事情的时候，我大概每周会整理1~2个架构，整理时不考虑依赖的顺序(比如会先整理 ViT，哪怕 Transformer 是它的基础)，因为看论文遇到什么就优先学习它，但是后续我都会尽量补充完全

**因为我也是初学者，因此难免会有遗漏或者疏忽，请大家谅解🙃**，希望能帮助到你
