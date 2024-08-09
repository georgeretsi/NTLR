from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

import os 

from models import  Wide_ResNet, MLP, ResNet
from tlr_src.tlr_wrapper import TLR

import pandas as pd

def count_parameters(model):
    return sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)

def train(epoch, model, loader, optimizer, gpu_id, log_interval=100):

    model.train()
    mloss = []

    report = []
    tloss = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = Variable(data.cuda(gpu_id)), Variable(target.cuda(gpu_id))

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        mloss += [loss.item()]
        loss.backward()

        optimizer.step()

        if (batch_idx+1) % log_interval == 0:
            # progress reporting!
            if ('tlr' in optimizer.__module__):
                lr = optimizer.get_lr()[0]
            else:
                lr = optimizer.param_groups[0]["lr"]
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}\tlr: {:.5f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), sum(mloss)/len(mloss), lr))

            mloss = np.asarray(mloss)

            report.append({"step": epoch-1 + batch_idx / len(loader),
                              "loss": mloss.mean(), "loss_std": mloss.std(),
                              "lr": lr})
            mloss = []

    return report

def test(epoch, model, loader, gpu_id):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        data, target = data.cuda(gpu_id), target.cuda(gpu_id)
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(loader.dataset)
    print('\n({}) - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return 100. * correct.item() / len(loader.dataset)

def evaluate_setting(dict, gpu_id, path='./results/'):

    if not os.path.isdir(path):
        os.makedirs(path)

    # setting
    dataset, model_name = dict["setting"]

    # optimizer + scheduler
    method, scheduler = dict["optimizer"]

    # hyper-params
    lr, bsize, epochs = dict["lr"], dict["bsize"], dict["epochs"]

    name = dataset + '_' + model_name + '_' + method + '_' + str(scheduler) + '_lr' + str(lr) + '_b' + str(bsize)

    # dataset selection, MNIST and CIFAR10/100 are supported here
    if dataset == 'mnist':
        mnist_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transforms)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transforms)
        nclasses = 10
    elif 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if dataset == 'cifar10':
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            nclasses = 10
        elif dataset == 'cifar100':
            trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            nclasses = 100
        else:
            print('not supported dataset')
    else:
        print('not supported dataset')


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

    if model_name == 'mlp':
        model = MLP(dropout=True)
    elif 'wrnet' in model_name:
        # wrnet_16_4
        # depth and width are extracted by name: e.g. 16 & 4
        mparams = model_name.split('_')
        model = Wide_ResNet(int(mparams[-2]), int(mparams[-1]), nclasses)

    model.cuda(gpu_id)

    # batches per epoch needed for TLR updates! 
    batches_per_epoch = len(train_loader)
    log_interval = int(.2 * len(train_loader))

    if 'tlr' in method:
        # base optimizer !! sgd or adam
        if 'sgd' in method:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.8, dampening=.0, weight_decay=5e-4, nesterov=False)
        elif 'adam' in method:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            print('currently not supported')
        optimizer = TLR(optimizer, batches_per_epoch=batches_per_epoch) 

    elif 'adam' in method:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif 'sgd' in method:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.8, dampening=.0, weight_decay=5e-4, nesterov=False)
    else:
        print('not defined optimizer')  

    if scheduler == 'None':
        scheduler = None
    
    if ('sgd' in method) or ('adam' in method):
        # several schedulers supported fort future exploration!
        if scheduler is not None:
            if scheduler == 'mstep':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5 * epochs), int(.75 * epochs)])
            elif scheduler == 'exp':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, .9)
            elif scheduler == 'wexp':   
                schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch + 1) / int(.1*epochs)),
                              torch.optim.lr_scheduler.ExponentialLR(optimizer, .92)]
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=[int(.1*epochs)])
            elif scheduler == 'cos':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            elif scheduler == 'rcos':
                tepochs = epochs/10
                schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, rr*tepochs) for rr in range(1,5)]
                milestones = [sum(range(1,rr+1))*tepochs for rr in range(1,4)]
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
            elif scheduler == 'wcos':   
                schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch + 1) / int(.1*epochs)),
                              torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(.9*epochs))]
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=[int(.1*epochs)])
            else:
                print('not defined scheduler')

    model_size = count_parameters(model)
    print('number of parameters: ' + str(model_size) + ' !!!!\n')


    train_report, acc_list = [], []
    for epoch in range(1, epochs + 1):
        report = train(epoch, model, train_loader, optimizer, gpu_id, log_interval)
        per_epoch_n = len(report)
        train_report += report
        if scheduler is not None:
            scheduler.step()
        if epoch % 1 == 0:
            acc = test(epoch, model, test_loader, gpu_id)
            acc_list += [acc]

        if epoch % 10 == 0:
            df_report = pd.DataFrame(train_report).join(pd.DataFrame({"acc": acc_list}, index=range(per_epoch_n-1,len(train_report),per_epoch_n)))
            df_report.to_csv(path + name + ".csv")

    df_report = pd.DataFrame(train_report).join(pd.DataFrame({"acc": acc_list}, index=range(per_epoch_n-1,len(train_report),per_epoch_n)))

    df_report.to_csv(path + name + ".csv")
    return df_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--bsize', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--method', type=str, default='sgd', help="Options : [sgd, adam, sgd=tlr, adam-tlr]")
    parser.add_argument('--scheduler', type=str, default=None, help="Options : [None, mstep, exp, cos]")
    parser.add_argument('--dataset', type=str, default='mnist', help="Options : [mnist, cifar10, cifar100]")
    parser.add_argument('--model', type=str, default='mlp', help="Options : [mlp, wrnet_D_W] (e.g. wrnet_16_4)")
    args = parser.parse_args()

    gpu_id = args.gpu

    exp_dict = {
        "setting": (args.dataset, args.model),
        "optimizer": (args.method, args.scheduler),
        "lr": args.lr,
        "bsize": args.bsize,
        "epochs": args.epochs,
    }

    evaluate_setting(exp_dict, gpu_id)
