# Sinusoidal Representation Networks (SIREN) With Some  extral experiment

Unofficial PyTorch implementation of Sinusodial Representation networks (SIREN) from the paper [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661). This repository is a PyTorch port of [this](https://github.com/titu1994/tf_SIREN) excellent TF 2.0 implementation of the same.

If you are using this codebase in your research, please use the following citation:
```
@software{aman_dalmia_2020_3902941,
  author       = {Aman Dalmia},
  title        = {dalmia/siren},
  month        = jun,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.1},
  doi          = {10.5281/zenodo.3902941},
  url          = {https://doi.org/10.5281/zenodo.3902941}
}
```

# Introduction
The compilation of this warehouse draws on Github, the code warehouse of many students.

My warehouse has the following differences:

1: A lot of comments are given for the code, which is very convenient for you to understand the meaning of the code

2: For the Siren module, combined with my understanding and previous work, I made some experiments, such as linking with Nerf's MLP network, although the effect is not satisfactory

本仓库的编写借鉴了Github这个很多同学的代码仓库。

我的这个仓库有以下几个不同点：

1：针对代码给了大量的注释，可以很方便你理解代码表达的意思

2：针对Siren模块，结合我的理解以及之前的工作，我做出了一些实验，比如与Nerf的MLP网络进行相关联动，尽管效果不如人意



Regarding the Implicit Neural Representations with Periodic Activation Functions paper, I wrote my own reading notes on it, including an introduction to some mathematical formulas. Welcome to watch

[(652条消息) Siren论文阅读笔记：Implicit Neural Representations with Periodic Activation Functions具有周期激活函数的隐式神经表示_出门吃三碗饭的博客-CSDN博客](https://blog.csdn.net/qq_40514113/article/details/131745437?spm=1001.2014.3001.5501)

# Quick Statart !
A partial implementation of the image inpainting task is available as the `train.py` and `eval.py` scripts.

Before training, remember to put the pictures to be trained under the data folder, and configure the corresponding path to read pictures

To run training:
```bash
$ python train.py
```

To run evaluation:
```bash
$ python eval.py
```

Weight files are made available in the repository under the `checkpoints` directory. It generates the following output after 5000 epochs of training with batch size 8192 while using only 10% of the available pixels in the image during training phase.

<img src="https://github.com/dalmia/siren/blob/master/images/celtic_spiral_knot.jpg?raw=true" height=100% width=100%>

# Tests
Tests are written using `unittest`. You can run any script under the `tests` folder.


### My results

[Siren_Study/plant_result.jpg at master · QinghongShao-sqh/Siren_Study (github.com)](https://github.com/QinghongShao-sqh/Siren_Study/blob/master/plant_result.jpg)



# Finaly


### If you have any questions about my project, please leave a comment!
### If my project can help you, I hope you can give me a star!
##The platform where I often move
### Bilibili(To update my paper sharing video) [出门吃三碗饭的个人空间_哔哩哔哩_bilibili](https://space.bilibili.com/38035003?spm_id_from=333.1007.0.0)
### CSDN (To update my blog)[(644条消息) 出门吃三碗饭的博客_CSDN博客-python,大学学习,复习笔记领域博主](https://blog.csdn.net/qq_40514113?spm=1000.2115.3001.5343)
### ZhiHu(To update my thesis notes and to receive counseling)[(2 封私信 / 50 条消息) 出门吃三碗饭 - 知乎 (zhihu.com)](https://www.zhihu.com/people/olkex)
### 公众号(need wechat，and You can buy some wacky items)  AI知识物语https://mp.weixin.qq.com/s/SL5QGtB1svkG_ac11OrR0Q
