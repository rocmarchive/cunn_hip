# cifar.torch

The code achieves 92.45% accuracy on CIFAR-10 just with horizontal reflections.

Corresponding blog post: http://torch.ch/blog/2015/07/30/cifar.html

Accuracies:

 | No flips | Flips
--- | --- | ---
VGG+BN+Dropout | 91.3% | 92.45%
NIN+BN+Dropout | 90.4% | 91.9%

Would be nice to add other architectures, PRs are welcome!

## Steps to train Cifar10 Model

a) Data preprocessing:

```bash
 th -i provider.lua
```

b) Normalizing and saving the data

```lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)
```
Takes about 30 seconds and saves 1400 Mb file.

c) Training:

```bash
th train.lua --model vgg_bn_drop -s logs/vgg
```
