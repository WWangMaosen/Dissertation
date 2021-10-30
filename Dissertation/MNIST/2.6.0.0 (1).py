'''
Exp2.3  - Gradient sign reversed
Author  - Manas Gupta
Date    - 04/01/2021
Validation Set by Wang Maosen
Date    - 19/2/2021
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



from networkx.drawing.tests.test_pylab import plt

import math
import random
import torch.nn.functional as F
from collections import deque
import pandas as pd


def relu(input):
    output = input.clamp(min=0)
    return output


def softmax(x):
    maxes = torch.max(x, dim=1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    result = x_exp / x_exp_sum
    return result


def train(args, model, device, train_loader, optimizer, epoch, num_classes):
    # Create empty array for one-hot encoding
    def push_to_tensor_alternative(tensor, x):
        return torch.cat((tensor[1:], Tensor([x])))

    y_onehot = torch.FloatTensor(args.batch_size, num_classes).to(device)
    fc1, fc2, t1 = model[0], model[1], model[2]
    correct = 0
    # tensor = torch.Tensor(torch.zeros(784,2000).data,torch.zeros(784,2000).data,torch.zeros(784,2000).data,torch.zeros(784,2000).data,torch.zeros(784,2000).data).type('torch.cuda.FloatTensor')
    queue = deque()
    mask2 = torch.zeros(784, 2000, device=device, requires_grad=False)
    mask1 = torch.zeros(784, 2000, device=device, requires_grad=False)
    mask = torch.zeros(784, 2000, device=device, requires_grad=False)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Unpacking this line from PyTorch, output = model(data)
        x = torch.flatten(data, 1)
        a1 = relu(torch.mm(x, fc1)) 
        # a1 = torch.mm(x, fc1)
        # a2 = F.normalize(a1, p = 2 ,dim = 1)
        # a1 = F.normalize(a1, p = 1 ,dim = 1)
        # a1 = F.softsign(a1)

        output = softmax(torch.mm(a1, fc2))
        # Calculate loss for SGD layer
        loss = F.nll_loss(output, target)
        # loss = F.mse_loss(output, target)
        loss.backward()
        # Unpacking this line from PyTorch, backward(), and calculating and returning gradients for hidden layer
        # fc1.grad = torch.mm(torch.t(x), a1) / len(data) - t1
        # t = torch.mm(torch.t(x), a1) / len(data) + 0.1*torch.randn(784,2000).type('torch.cuda.FloatTensor')

        #t = torch.mm(torch.t(x), a1) / len(data)
        t = (torch.mm(torch.t(x), a1)-torch.mm(torch.mm(fc1,torch.t(a1)),a1)) / len(data)

        # t = F.normalize(t, p = 1 ,dim = 1)
        # fc1_ratio = torch.div(t,fc1)
        fc1.grad = t - t1
        standard = torch.tile(torch.sum(fc1.grad.data.abs(),dim = 0),(784,1))
        fc1_ratio = torch.div(fc1.grad, standard)
        #fc1_ratio = fc1_grad

        # Update only top 10% weights in hidden layer
        num_weights = torch.numel(fc1_ratio.data)
        #k = int(1 * num_weights)

        #select the LTD    mask3
        '''
        if (batch_idx + 1) % 10 == 0:  # LTD&LTP

            y = torch.flatten(mask2.data.detach().clone().abs())
            grad_threshold1 = torch.kthvalue(y, k).values
            mask1 = torch.ge(mask2.data.abs(), grad_threshold1).type('torch.cuda.FloatTensor')
            mask_zero = torch.zeros_like(fc1)
            mask_one = torch.ones_like(fc1)
            mask3 = mask1 * 2
            mask3 = torch.where(mask3 < 1, mask_one, mask3)
            mask3 = torch.where(mask3 > 1, mask_zero, mask3)
            mask2 = torch.zeros_like(fc1)

        if batch_idx >= 9:
            fc1_ratio *= mask3
        '''
        #selet the neurons with large value --mask4
        a1_ind = torch.argsort(a1, dim=1, descending=True)
        mask4 = torch.zeros(784, 2000, device=device, requires_grad=False)
        mask4[:,a1_ind[0:1000]] = 1

        #fc1_ratio *= mask4
        '''
        #select the large fc1_ratio and decrease the low
        y = torch.flatten(fc1_ratio.data.detach().clone().abs())
        k = int(0.995 * num_weights)
        k_down = int(0.99 * num_weights)
        if k == 0:
            k = 1
        grad_threshold_up = torch.kthvalue(y, k).values
        grad_threshold_down = torch.kthvalue(y, k_down).values

        mask_up = torch.zeros_like(fc1)
        mask_down = torch.zeros_like(fc1)
        #mask_up = torch.where(fc1_ratio.data.abs()>grad_threshold_up,torch.ones_like(fc1),mask_up)
        mask_up = torch.where(fc1_ratio.data.abs()>grad_threshold_up,torch.ones_like(fc1),mask_up)
        mask_down = torch.where(fc1_ratio.data.abs()>grad_threshold_down,torch.ones_like(fc1),mask_down)
        sum_up = torch.sum(mask_up)


        #fp = 0.9999
        mask_down -= mask_up
        
        #sum_down = torch.sum(mask_down)

        # 用加减法来代替乘法
        mask = mask_up
        #mask2 += mask_up

        #fc1_grad *= (mask + mask1)
        fc1.grad *= (mask - 0.4*mask_down)

        '''
        indx = torch.argmax(a1, dim=1)
        
        mask = torch.zeros_like(fc1.grad)
        for i in range(100):
          mask[:, indx[i]] = 1
          #i= indx[i]
          #y1 = torch.flatten(fc1.grad[:,i].data.detach().clone().abs())
          
          #grad_threshold_up = torch.median(fc1.grad[:,i])
          #mask_up = torch.zeros_like(fc1.grad[:,i])
          #mask_up = torch.where(fc1.grad[:,i].data.abs()>grad_threshold_up,torch.ones_like(fc1.grad[:,i]),mask_up)
          #fc1.grad[:,i] *= mask_up
        
        


        fc1.grad *= mask

        #anti_hebbian
        #k_num = 10
        #delta = 0.5
        #y_ind = torch.argsort(fc1.grad.data.detach().clone().abs(), dim=0, descending=True)
        #mask5 = torch.zeros(784, 2000, device=device, requires_grad=False)
        #mask5[y_ind[784-k_num:784,:], torch.arange(2000)] = 1.0  # y最后一行的数据，然后是[0-99的一个数列]，置为1？       //yl相当于把每一行的最大值置为1，第二大的值置为-delta，其他为0
        #mask5[y_ind[0:784-k_num], torch.arange(2000)] = delta

        #fc1.grad *= mask5


        fc1.grad = -fc1.grad  ## See the change here, this time we make Weight1=Weight1 + fc1.grad

        if batch_idx == 0:
            t1 = t
        if batch_idx > 0:
            t1 = 0.5 * t1 + 0.5 * t
        model[2] = t1

        # Take optimizer step
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / 50000  # len(train_loader)*args.batch_size
    max_weight5 = torch.max(fc1.grad.data.abs())
    print(' fc1:({:.5f})\n'.format(max_weight5))
    max_weight4 = torch.max(x.data.abs())
    print(' x:({:.5f})\n'.format(max_weight4))
    max_weight3 = torch.max(a1.data.abs())
    print(' a1:({:.5f})\n'.format(max_weight3))
    max_weight2 = torch.max(t.data.abs())
    print(' t1:({:.5f})\n'.format(max_weight2))
    max_weight1 = torch.max(fc1.data.abs())
    print('max_weight:({:.5f})\n'.format(max_weight1))
    max_fc1_ratio = torch.max(fc1_ratio.data.abs())
    print('max_weight:({:.5f})\n'.format(max_fc1_ratio))

    print('\nTrain set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, 50000, accuracy))
    return accuracy


def valid(model, device, validation_loader):
    correct = 0
    fc1, fc2 = model[0], model[1]
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            # Unpacking this line from PyTorch, output = model(data)
            x = torch.flatten(data, 1)
            x = torch.mm(x, fc1)
            x = relu(x)
            x = torch.mm(x, fc2)
            output = softmax(x)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Model output is {}", format(output))
    accuracy = 100. * correct / 10000  # len(validation_loader)*args.batch_size
    print('\n Validation set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, 10000, accuracy))
    return accuracy


def spars(array):
    weight = np.zeros((2000, 1))
    for i in range(200):
        ind = np.unravel_index(np.argmax(array, axis=None), array.shape)
        array[ind] = 0
        weight[ind] = 1
    return weight


def main():
    # Training settings
    saveresults = [[], []]
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=float, default=5, metavar='M', help='Scheduler step size (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}

    train_dataset = datasets.MNIST('/content/drive/MyDrive/data', train=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    # train_dataset = datasets.MNIST('/content/drive/MyDrive/data/', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_dataset = datasets.MNIST('../data/', train=False,download=True, transform=transforms.Compose([
    # transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    indices = range(len(train_dataset))
    indices_train = indices[:50000]
    indices_val = indices[50000:]
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices_train)  # 无放回地按照给定的索引列表采样样本元素
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler_train,
                                               **kwargs)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler_val,
                                                    **kwargs)

    # Initialize weights
    stdv = 1. / math.sqrt(2000)
    fc1 = torch.normal(0, stdv, (784, 2000), device=device, requires_grad=True)
    fc2 = torch.normal(0, stdv, (2000, 10), device=device, requires_grad=True)
    stdv = 1. / math.sqrt(fc1.size(1))
    # fc1.data.normal(-stdv, stdv)
    # fc1.data.uniform_(-0.1, 0.1)
    mask1 = torch.ones(784, 2000, device=device, requires_grad=False)
    mask2 = torch.zeros(784, 2000, device=device, requires_grad=False)

    t1 = torch.zeros(784, 2000, device=device, requires_grad=False)

    # stdv = 1. / math.sqrt(fc2.size(1))
    # fc2.data.uniform_(-stdv, stdv)

    model = [fc1, fc2, t1]
    optimizer = optim.Adadelta(model, lr=args.lr)
    # optimizer = optim.SGD(model, lr=args.lr, momentum=0.0005)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    train_list = []
    valid_list = []

    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch, num_classes=10)
        valid_acc = valid(model, device, validation_loader)

        train_list.append(train_acc)
        valid_list.append(valid_acc)
        saveresults[0].append(train_acc)
        saveresults[1].append(valid_acc)
        scheduler.step()
        # Plot weights
        if epoch % 5 == 0:
            # Create plt plot with subplots arranged in a #x_plots x #y_plots grid
            fc1_weights = fc1.data.cpu().numpy()
            fc2_weights = fc2.data.cpu().numpy()

            max_weight = np.amax(np.absolute(fc1_weights))
            x_plots = 50
            y_plots = 40
            fig, axs = plt.subplots(x_plots, y_plots)
            k = 0

            # weights2 = fc2_weights[:,m]
            # weight1 = spars(weights2).reshape(x_plots,y_plots)
            fig, axs = plt.subplots(x_plots, y_plots)
            k = 0
            for i in range(x_plots):
                for j in range(y_plots):
                    weights = fc1_weights[:, k].reshape(28, 28)

                    axs[i, j].imshow(weights, cmap="bwr", vmin=-max_weight, vmax=max_weight)
                    axs[i, j].axis('off')
                    k += 1
            # Print plot
            fig.set_size_inches(x_plots, y_plots)
            plt.title("MaxWeight " + str(max_weight))
            fig.savefig("Weights1_epoch{}.png".format(epoch))
            plt.clf()

            '''
            x1_plots = 5
            y1_plots = 2
            fig, axs = plt.subplots(x1_plots, y1_plots)
            k1=0
            weight2 = np.zeros((x_plots,y_plots))
            for i1 in range(x1_plots):
                for j1 in range(y1_plots):
                    weights2 = fc2_weights[:,k1]
                    weight1 = spars(weights2).reshape(x_plots,y_plots)
                    weight2 += weight1
                    # axs[i,j].imshow(weights, cmap="gray")
                    axs[i1,j1].imshow(weight2, cmap="gray")            
                    axs[i1,j1].axis('off')
                    k1 += 1

            #Print plot
            fig.set_size_inches(x_plots,y_plots)
            fig.savefig("Weights2_epoch{}.png".format(epoch))
            plt.clf()
            '''

    x = range(0, args.epochs)
    y1 = train_list
    y2 = valid_list
    plt.plot(x, y1, 'o-', color='red')
    plt.title('train accuracy(red) vs. valid accuracy(blue)')
    plt.ylabel('accuracy')
    plt.plot(x, y2, '.-', color='blue')
    plt.xlabel('epoches')
    plt.savefig("accuracy.png")

    # 结果的统计数据
    '''
    dic_accuracy = {'train_accuracy_max':max(train_list),'train_accuracy_mean':np.mean(train_list),'train_accuracy_median':np.median(train_list),
                'valid_accuracy_max':max(valid_list),'valid_accuracy_mean':np.mean(valid_list),'valid_accuracy_median':np.median(valid_list)}

    jsObj = json.dumps(dic_accuracy)

    fileObject = open(r'accuracy.json', 'w')

    fileObject.write(jsObj)

    fileObject.close()
    '''

    # 生成gif动图
    '''gif_images = []
    for i in range(0, 20):
      gif_images.append(imageio.imread("Weights_epoch{}.png".format(i)))   # 读取多张图片
    imageio.mimsave("accuracy.gif", gif_images, fps=1)'''

    name = ['Train Accuracy', 'Validation Accuracy']
    result = pd.DataFrame(list(zip(saveresults[0], saveresults[1])), columns=name)
    result.to_csv('Results.csv')


if __name__ == '__main__':
    main()
