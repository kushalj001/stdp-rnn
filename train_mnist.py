from typing import Sequence, Union, Optional, Any
from pathlib import Path
from PIL.Image import Image
from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import numpy as np
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.utils import AvalancheDataset
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize
from models import MnistRNN
from snn_utils import SleepRNNLayer
from avalanche.benchmarks import NCScenario, nc_benchmark
from torchvision.datasets import MNIST
import torchvision
from torchvision import transforms



def train(model, train_loader, train_dataset, criterion, optimizer, device):
    epochs = 5
    input_dim = seq_len = 28

    train_results = []
    train_losses = []
    train_accs = []
    model.train()
    for epoch in range(epochs):
        print(f"Starting epoch: {epoch+1}")
        train_loss = 0
        train_acc = 0
        for idx, batch in enumerate(train_loader):
            images, labels = batch
            images = images.view(-1, seq_len, input_dim).to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_acc += torch.sum((torch.argmax(preds,dim=-1)==labels)).item()
        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc/len(train_dataset))
        result = {
            "epoch": epoch,
            "train_acc": train_acc/len(train_dataset),
            "train_loss": train_loss/len(train_loader)
        }
        train_results.append(result)
    return train_results

# Train IMDB model with induced typos, not adv changes.
# Train the model with all the permutations at once: 180_000 images.
# PMNIST: train on 2 exp=> sleep=> evaluate on both the experiences.

def test(model, test_dataset, test_loader, criterion, device):
    test_loss = 0
    test_acc = 0
    input_dim = seq_len = 28

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.view(-1, seq_len, input_dim).to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            test_loss += loss.item()
            test_acc += torch.sum((torch.argmax(preds,dim=-1)==labels)).item()
        eval_result = {
            "test_acc": test_acc/len(test_dataset)
        }
    return eval_result
        


def create_sleep_input(train_stream, num_iterations):
    imgs = []
    for exp in train_stream:
        dataset = exp.dataset
        for ex in dataset:
            imgs.append(ex[0])
    # [120k, 1, 28, 28]
    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.squeeze(dim=1)
    # [120k, 28, 28]
    sleep_input = torch.mean(imgs, dim=[0,2])
    # [28]
    # average row value [0,2]?
    sleep_input = torch.tile(sleep_input, (num_iterations,1))
    sleep_input = sleep_input.cpu().detach().numpy()
    return sleep_input

if __name__ == "__main__":
    my_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST('data', train = True, download=True, transform=my_transform)
    test_dataset = torchvision.datasets.MNIST('data', train = False, download=True, transform=my_transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)
    input_dim = 28
    seq_len = 28
    rnn_hidden_dim = 256
    fc_dim = 128
    output_dim = 10 # num_classes
    dropout = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers = 1
    model = MnistRNN(input_dim, fc_dim, rnn_hidden_dim, output_dim, num_layers, dropout, device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_results = train(model, train_loader, train_dataset, criterion, optimizer, device)
    print("Training results with sequential training: ")
    print(train_results)
    test_results = test(model, test_dataset, test_loader, criterion, device)
    print("Test results before sleep: ")
    print(test_results)

