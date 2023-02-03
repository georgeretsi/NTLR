# Newton-based Trainable Learning Rate

This repository contains the code for our proposed meta-optimizer described in: G. Retsinas, G. Sfikas, P.P. Filntisis & P. Maragos, "Newton-based Trainable Learning Rate", under submission. The goal of this work is online adaptation of learning rate.

-------------------------------------------------------------------------
### Installation

```{bash}
git clone https://github.com/georgeretsi/NTLR
cd NTLR
```

Only three packages are needed:
```{bash}
pip install torch torchvision pandas
```
Developed and tested on:
pandas==1.5.1,
numpy==1.23.4 (is install along torch),
torch==1.12.1,
torchvision==0.13.1


### Using the proposed Meta-Optimizer



Our approach acts as a meta-optimizer that uses a typical optimizer as its input.
An example of defining an SGD-TLR optimizer is:

```python
from tlr_src.tlr_wrapper import TLR
...
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.8, dampening=.0, weight_decay=5e-4, nesterov=False)
optimizer = TLR(optimizer, batches_per_epoch=batches_per_epoch) 
...
```

The argument batches_per_epoch can be easily obtained by an typical Pytorch loader as:
```python
batches_per_epoch = len(train_loader)
```

Extra arguments of TLR are $p$ (meta_update - updates per epoch) and $\gamma$ (meta_gamma: hyper-parameter of the damped Newton approach):
```python
optimizer = TLR(optimizer, meta_gamma=.25, meta_update=.33, batches_per_epoch=batches_per_epoch)
```
Nonetheless, experimental evaluation showed robustness to initial learning rate and batch-size without these arguments needed to be fine-tuned. 

### Training and Configuration Options

We also provide a train example for MNIST and CIFAR10/100. The arguments to be configured are:
- dataset (str) : [mnist, cifar10, cifar100]
- model (str) : [mlp, wrnet_D_W], WideResNets cotrol their depth (D) and width (W) passed through their names, e.g. wrnet_16_4
- method (str) : [sgd, adam, sgd-tlr, adam-tlr] 
- scheduler (str) : [None, mstep]
- bsize (int) : batch-size
- lr (float) : (initial) learning rate
- gpu (int) : GPU id to run the experiment

Indicative possible configurations are listed below:
```{bash}
python train.py --method tlr-sgd --scheduler None --dataset cifar100 --model wrnet_16_8 --lr 0.01 --bsize 64 --epochs 120 --gpu 1
python train.py --method adam --scheduler mstep --dataset mnist --model mlp --lr 1e-9 --bsize 512 --epochs 60 --gpu 0
```

