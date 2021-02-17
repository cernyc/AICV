from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from Net import Net
import argparse
import numpy as np


def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''

    # Set model to train mode before each epoch
    model.train()

    # Empty list to store losses
    losses = []
    correct = 0

    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample

        # Push data/label to correct device
        data, target = data.to(device), target.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()

        # Do forward pass for current set of data
        output = model(data)

        # ======================================================================
        # Compute loss based on criterion
        loss = criterion(output, target)

        # Computes gradient based on final loss
        loss.backward()

        # Store loss
        losses.append(loss.item())

        # Optimize model parameters based on learning rate and gradient
        optimizer.step()

        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)

        # ======================================================================
        # Count correct predictions overall
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx + 1) * batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        float(np.mean(losses)), correct, (batch_idx + 1) * batch_size,
                                         100. * correct / ((batch_idx + 1) * batch_size)))
    return train_loss, train_acc


def test(model, device, test_loader):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''

    # Set model to eval mode to notify all layers.
    model.eval()

    losses = []
    correct = 0

    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            # Predict for data by doing forward pass
            output = model(data)

            # ======================================================================
            # Compute loss based on same criterion as training
            loss = F.cross_entropy(output, target, reduction='mean')

            # Append loss to overall test loss
            losses.append(loss.item())

            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # ======================================================================
            # Count correct predictions overall
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)

    # Initialize the model and send to device
    model = Net(FLAGS.mode).to(device)

    # Initialize the criterion for loss computation
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Initialize optimizer type
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-7)

    # Create transformations to apply to each data sample
    # Can specify variations such as image flip, color flip, random crop, ...
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    #import the CIFAR10 data
    dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
    test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())

    #Set validation size to 5000
    val_size = 5000
    #Define the training set size which is the data minus 5000
    train_size = len(dataset) - val_size
    #randomly split the data for validation and training
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    len(train_ds), len(val_ds)
    #Define the train and test loader
    train_loader = DataLoader(train_ds, 10, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, 10 * 2, num_workers=4, pin_memory=True)
    #Initialize variables needed for the report
    maxAcc = 0.0
    historyTrainLoss = []
    historyTrainAcc = []
    historyTestAcc = []
    historyTestLoss = []

    # Run training for n_epochs specified in config
    for epoch in range(1, FLAGS.num_epochs + 1):
        print("\nEpoch: ", epoch)
        train_loss, train_accuracy = train(model, device, train_loader,
                                           optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss, test_accuracy = test(model, device, test_loader)
        #append each epoch result to out report data
        historyTrainLoss.append(train_loss)
        historyTestLoss.append(test_loss)
        historyTrainAcc.append(train_accuracy)
        historyTestAcc.append(test_accuracy)

        if test_accuracy > maxAcc:
            best_accuracy = test_accuracy
    #Add the generated data to histograms
    plt.plot([range(1, FLAGS.num_epochs + 1)],[historyTrainLoss], 'ro')
    plt.plot([range(1, FLAGS.num_epochs + 1)], [historyTestLoss], 'bo')
    plt.ylabel('Train/Test Loss')
    plt.show()
    #Add the generated data to histograms
    plt.plot([range(1, FLAGS.num_epochs + 1)],[historyTrainAcc], 'ro')
    plt.plot([range(1, FLAGS.num_epochs + 1)], [historyTestAcc], 'bo')
    plt.ylabel('Train/Test Accuracy')
    plt.show()

    print("accuracy is {:2.2f}".format(best_accuracy))

    print("Training and evaluation finished")


if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=3,
                        help='Select mode between 1-3.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    run_main(FLAGS)

