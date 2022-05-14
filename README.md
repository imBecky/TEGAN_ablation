# DoubleBiGAN
DoubleBiGAN used for transfer learning in TensorFlow2.0.

Using spectral info solely.

This repository is still unfinished!

To do：

- [x] 优化model

- [x] 调整layers

- [x] 调整参数
  
  - [x] optimizer
  
- [ ] 使用软标签和带噪声的标签

  这一点在训练判别器时极为重要，使用硬标签(非0即1)训练出的判别器太无情了，要么通过，要么一棍子打死，没有半点道理可讲。我们可以在真实样本的标签1上加上(-0.2,0)的随机噪声，在假样本的标签0上加(0, 0.2)的随机噪声。软标签和带噪声的标签的加入可以在一定程度上解决生成网络梯度消失的问题。

