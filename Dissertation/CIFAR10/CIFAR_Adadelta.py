'''
Project - ICASSP PAPER
Author  - Maosen Wang
Date    - 2021/10/01
'''
from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.distributions.distribution import Distribution
from torch.distributions import Categorical, Normal
from torch.distributions import constraints
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
#import wandb
#wandb.init(project="hebb_exp1-3")
import math
import random
import torch.nn.functional as F
import pandas as pd
from torch import nn


def relu(input):
    output = input.clamp(min=0)
    return output
def softmax(x):
    maxes = torch.max(x, dim=1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    result = x_exp/x_exp_sum
    return result

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)

def train(args, model, m, device, train_loader, optimizer, epoch, num_classes):
    #Create empty array for one-hot encoding
    y_onehot = torch.FloatTensor(args.batch_size, num_classes).to(device)
    fc1, fc2, t1 = model[0], model[1], model[2]
    correct = 0
    mmm = m.cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Unpacking this line from PyTorch, output = model(data)
        x = torch.flatten(data, 1)
        a1 = torch.mm(x, fc1)
        a1 = mmm(a1)
        output = softmax(torch.mm(relu(a1), fc2))
        # Calculate loss for SGD layer
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / 50000#len(train_loader)*args.batch_size
    print('\nTrain set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, 50000, accuracy))
    return accuracy
def valid(model, m, noise, device, validation_loader):
    correct = 0
    fc1, fc2 = model[0], model[1]
    mmm = m.cuda()
    noise = noise.cuda()
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            #Unpacking this line from PyTorch, output = model(data)
            x = torch.flatten(data, 1)
            x = x + noise
            x = torch.mm(x, fc1)
            x = mmm(x)
            x = torch.mm(relu(x), fc2)
            output = softmax(x)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Model output is {}", format(output))
    accuracy = 100. * correct / 10000#len(validation_loader)*args.batch_size
    print('\n Validation set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, 10000, accuracy))
    return accuracy
def main():
    saveresults=[[],[]]
    # Training settings
    parser = argparse.ArgumentParser(description='Hebbian Learning on CIFAR-10 dataset')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=450, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=float, default=10, metavar='M', help='Scheduler step size (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()
    #wandb.run.name = "Exp1.3-Batch=" + str(args.batch_size) + " LR=" + str(args.lr) + " Gamma=" + str(args.gamma) + " Epochs=" + str(args.epochs) + " Step=" + str(args.step_size)
    #wandb.run.save()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(transform_func),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    train_dataset = datasets.CIFAR10('/data/', train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10('/data/', train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    #Initialize weights
    fc1 = torch.zeros(3072, 2000, device=device, requires_grad=True)
    fc2 = torch.zeros(2000, 10, device=device, requires_grad=True)
    stdv = 1. / math.sqrt(fc1.size(1))
    fc1.data.uniform_(-stdv, stdv)
    stdv = 1. / math.sqrt(fc2.size(1))
    fc2.data.uniform_(-stdv, stdv)
    t1 = torch.zeros(3072, 2000, device=device, requires_grad=False)
    model = [fc1, fc2, t1]
    optimizer = optim.Adadelta(model, lr=args.lr)
    # optimizer = optim.SGD(model, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    #noise = torch.normal(0, 0.6, (100, 3072), device=device, requires_grad=False)
    noise = torch.zeros(100, 3072, device=device, requires_grad=False)
    m = nn.BatchNorm1d(2000, affine=False)

    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, m, device, train_loader, optimizer, epoch, num_classes=10)
        test_acc = valid(model, m, noise, device, test_loader)
        saveresults[0].append(train_acc)
        saveresults[1].append(test_acc)
        #wandb.log({"Train Accuracy": train_acc})
        #wandb.log({"Test Accuracy": test_acc})
        scheduler.step()
        #Plot weights
        if epoch % 40 == 0:
            #Create plt plot with subplots arranged in a #x_plots x #y_plots grid
            x_plots = 6
            y_plots = 6
            fig, axs = plt.subplots(x_plots, y_plots)
            fc1_weights = fc1.data.cpu().numpy()
            min_weight = np.min(fc1_weights)
            max_weight = np.max(fc1_weights)
            weight_range = max_weight - min_weight
            fc1_weights = (fc1_weights - min_weight) / weight_range
            print(fc1_weights.shape)
            print(min_weight)
            print(max_weight)
            #For each subplot, get input weights of that neuron, reshape into 3x32x32
            k=0
            for i in range(x_plots):
                for j in range(y_plots):
                    weights = fc1_weights[:,k].reshape(3,32,32)
                    # print(weights.shape)
                    weights = minmaxscaler(weights)
                    weights = np.moveaxis(weights, 0, -1)
                    # print(weights.shape)
                    # exit()
                    # axs[i,j].imshow(weights, cmap="gray")
                    axs[i,j].imshow(weights, cmap="bwr")
                    axs[i,j].axis('off')
                    k += 1
            #Print plot
            fig.savefig("Weights_epoch{}.png".format(epoch))

    for iii in range(20):
        value = (iii+1) * 0.3
        noise = torch.normal(0, value, (100, 3072), device=device, requires_grad=False)
        test_acc = valid(model, m, noise, device, test_loader)
        saveresults[0].append(test_acc)
        saveresults[1].append(test_acc)




    name=['Train Accuracy','Validation Accuracy']
    result = pd.DataFrame(list(zip(saveresults[0],saveresults[1])),columns=name)
    result.to_csv('Results.csv')

def transform_func(x):
	return F.pad(x.unsqueeze(0), (4,4,4,4),mode='reflect').squeeze()
if __name__ == '__main__':
    main()
