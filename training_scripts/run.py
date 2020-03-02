# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Minahil Raza

import argparse
import os
import random
import configparser

import torch
from torch.nn.functional import cross_entropy, nll_loss
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from lenet import LeNet5
from alexnet import AlexNet
from vgg import VGG

parser = argparse.ArgumentParser(description='Brevitas Training')

parser.add_argument('--device', default='cuda', help='cuda or cpu')
parser.add_argument("--dataset", help="Dataset", default="MNIST")
parser.add_argument('--network', help= 'topology', default= 'Lenet')
parser.add_argument('--resume', '-r', action='store_true', help='resume training from checkpoint')
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")

# Hyper parameters

# Optimizer hyperparams
parser.add_argument("--batch_size", default=100, type=int, help="batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=1e-3, type=float, help="Weight decay")
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs")


args = parser.parse_args()

# Load Dataset
if args.dataset= 'MNIST':
        data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor()]))
        data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((28, 28)),
                      transforms.ToTensor()]))        

else if args.dataset= 'CIFAR10':
        data_train = CIFAR10('./data/cifar10',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()]))
        data_test =  CIFAR10('./data/cifar10',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
# you can add your dataset loader here

# use cuda
device= 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device == 'cpu':
    device= 'cpu'
# Load Model

if args.network = 'Lenet':
        model= LeNet5().to(device)
else if args.network = 'Alexnet':
        model= AlexNet().to(device)
else if args.network = 'VGG':
        model=VGG().to(device)


#resume training
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['accuracy']
    start_epoch = checkpoint['epoch']
else:
    best_acc = 0
    start_epoch = 0



# hyperparameters
batch_size= args.batch_size
epoch_num = args.epochs

optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)


train_loader = DataLoader(data_train, shuffle=True,batch_size= batch_size, num_workers=8)
test_loader = DataLoader(data_test, shuffle=True, batch_size= batch_size, num_workers=8)

def train(epoch):
    model.train() # set model in training mode (need this because of dropout)
    correct = 0
    loss    = 0

    for batch_id, (data, label) in enumerate(train_loader):
        data = data.to(device)
        target = label.to(device)
        
        optimizer.zero_grad()
        preds = model(data)
        loss = nll_loss(preds, target)
        loss.backward()
        optimizer.step()
        preds= preds.data.max(1)[1]
        correct += preds.eq(target.data).cpu().sum()

        if batch_id % 250 == 0:
          print("Batch #",batch_id,"  : Loss      =",loss.data.data)
    
    accuracy= 100. * float(correct )/float(len(train_loader.dataset))
    print('\nTraining set: Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(train_loader.dataset),
        accuracy))
         

def test(epoch):
    global best_acc
    model.eval() # set model in inference mode (need this because of dropout)
    correct = 0
    
    for data, target in test_loader:
        data = data.to(device) 
        target = target.to(device)
        
        output = model(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    accuracy = 100. * float(correct) /float(len(test_loader.dataset))
    print('Test set: Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(test_loader.dataset),
        accuracy))

    #save model
    if accuracy > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'accuracy': accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = accuracy

if args.evaluate = "True":
    test(1)        

else:

    for epoch in range(0, epoch_num):
        print("Starting Epoch %d" % epoch)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        train(epoch)
        test(epoch)
        scheduler.step()
    print("Best Accuracy = ", best_acc)


