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
# import wandb
#wandb.init(project="hebb_exp2-3")
import math
import random
import torch.nn.functional as F
import pandas as pd
def relu(input):
    output = input.clamp(min=0)
    return output
def softmax(x):
    maxes = torch.max(x, dim=1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    result = x_exp/x_exp_sum
    return result
def train(args, model, device, train_loader, optimizer, epoch, num_classes):
    #Create empty array for one-hot encoding
    y_onehot = torch.FloatTensor(args.batch_size, num_classes).to(device)
    fc1, fc2, t1 = model[0], model[1], model[2]
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):




        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #Unpacking this line from PyTorch, output = model(data)
        x = torch.flatten(data, 1)
        #print(len(x))
        #print(len(x[0]))
        #x = x * 255
        #print(torch.amax(x))

        if epoch<200:

            a1 = torch.mm(x, fc1)
            output = softmax(torch.mm(a1, fc2))
            #Calculate loss for SGD layer
            loss = F.nll_loss(output, target)
            loss.backward()
            maxvv = torch.max(a1, 1)[0]
            poss = torch.max(torch.absolute(a1), 1)[1]
            newa1 = torch.zeros(len(a1), len(a1[0]))
            for i in range(0, len(poss)):
                newa1[i, poss[i]] = 0.01
            newa1 = newa1.cuda()

            fc1grad = torch.mm(torch.t(x), newa1)
            #fc1grad = torch.mm(torch.t(x), a1) / len(data)
            #fc1grad = torch.mm(torch.t(x), a1)
            #fc1grad = fc1grad - fc1



            newfc1 = torch.zeros(len(fc1grad), len(fc1grad[0]))
            newfc1 = newfc1.cuda()

            for i in range(0, len(poss)):
                newfc1[:,poss[i]] += fc1[:,poss[i]]


            fc1grad = fc1grad - newfc1




            #fc1grad = torch.clamp(fc1grad,min=0)

            nc = torch.amax(torch.absolute(fc1grad))  # 最大值
            fc1grad = fc1grad / nc
            fc1 = fc1 + fc1grad * 0.05
            model[0] = fc1

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        if epoch % 10 == 0:
            torch.save(fc1, '2000_Neurons.pt')






    accuracy = 100. * correct / 50000#len(train_loader)*args.batch_size
    print('\nTrain set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, 50000, accuracy))
    return accuracy
def valid(model, device, validation_loader):
    correct = 0
    fc1, fc2 = model[0], model[1]
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            #Unpacking this line from PyTorch, output = model(data)
            x = torch.flatten(data, 1)
            x = relu(torch.mm(x, fc1))
            x = torch.mm(x, fc2)
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
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 64)')
    #parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.5, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=float, default=4, metavar='M', help='Scheduler step size (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()
    #wandb.run.name = "Batch=" + str(args.batch_size) + " LR=" + str(args.lr) + " Gamma=" + str(args.gamma) + " Epochs=" + str(args.epochs) + " Step=" + str(args.step_size)
    #wandb.run.save()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    train_dataset = datasets.MNIST('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    #train_dataset = datasets.MNIST('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    #test_dataset = datasets.MNIST('../data/', train=False,download=True, transform=transforms.Compose([
            #transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    indices = range(len(train_dataset))
    indices_train = indices[:50000]
    indices_val = indices[50000:]
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices_train)#无放回地按照给定的索引列表采样样本元素
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=sampler_train,**kwargs)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=sampler_val,**kwargs)

    #Initialize weights
    # fc11 = np.random.normal(0.0, 1.0, (784, 100))
    # fc11 = torch.from_numpy(fc11)
    # fc1 = torch.ones(784, 100, device=device, requires_grad=False)
    # fc1 = torch.mul(fc1,fc11)

    fc1 = torch.normal(-0.5,1.0,(784,2000),device=device, requires_grad=False)
    #fc1 = torch.zeros(784, 100, device=device, requires_grad=False)
    fc2 = torch.zeros(2000, 10, device=device, requires_grad=True)
    #stdv = 1. / math.sqrt(fc1.size(1))
    #fc1.data.uniform_(-stdv, stdv)
    stdv = 1. / math.sqrt(fc2.size(1))

    fc2.data.uniform_(-stdv, stdv)

    fc1_weights = fc1.data.cpu().numpy()
    max_fc1 = np.amax(np.absolute(fc1_weights))

    print(max_fc1)
    t1 = torch.ones(784, 2000, device=device, requires_grad=False)*max_fc1*0.5
    print(t1[0][0])
    print('\n')

    model = [fc1, fc2, t1]
    optimizer = optim.Adadelta(model, lr=args.lr)
    #optimizer = optim.SGD(model, lr=args.lr, momentum=0.0005)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):




        train_acc = train(args, model, device, train_loader, optimizer, epoch, num_classes=10)
        t1_data = model[0].data.cpu().numpy()
        print(np.amax(np.absolute(t1_data)))


        #fc1_weights = fc1.data.cpu().numpy()
        #max_fc1 = np.amax(np.absolute(fc1_weights))
        #print(max_fc1)
        saveresults[0].append(train_acc)
        valid_acc = valid(model, device, validation_loader)
        saveresults[1].append(valid_acc)
        #wandb.log({"Train Accuracy": train_acc})
        #wandb.log({"Test Accuracy": test_acc})
        scheduler.step()
        #Plot weights
        if epoch % 1 == 0:
            #Create plt plot with subplots arranged in a #x_plots x #y_plots grid
            x_plots = 10
            y_plots = 10
            fig, axs = plt.subplots(x_plots, y_plots)
            fc1_weights = t1_data
            max_weight = np.amax(np.absolute(fc1_weights))
            #For each subplot, get input weights of that neuron, reshape into 28x28
            k=0
            for i in range(x_plots):
                for j in range(y_plots):
                    weights = fc1_weights[:,k].reshape(28,28)
                    axs[i,j].imshow(weights, cmap="bwr", vmin=-max_weight, vmax=max_weight)
                    axs[i,j].axis('off')
                    k += 1
            #Print plot
            plt.title("MaxWeight "+str(max_weight))
            fig.savefig("Weights_epoch{}.png".format(epoch))
    name=['Train Accuracy','Validation Accuracy']
    result = pd.DataFrame(list(zip(saveresults[0],saveresults[1])),columns=name)
    result.to_csv('Results.csv')
if __name__ == '__main__':
    main()