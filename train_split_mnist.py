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
from avalanche.benchmarks import NCScenario, nc_benchmark, SplitMNIST


def train(model, criterion, optimizer, train_stream, test_stream, device, epochs):
    input_dim = seq_len = 28
    train_results = []
    for exp in train_stream:
        print(f"Starting training experience:{exp.current_experience}")
        model.train()
        exp_train_dataset = exp.dataset
        exp_test_dataset = test_stream[exp.current_experience].dataset
        exp_train_loader = DataLoader(exp_train_dataset, batch_size=32, shuffle=True)
        exp_test_loader = DataLoader(exp_test_dataset, batch_size=32, shuffle=True)
        exp_train_losses = []
        exp_train_accs = []
        for epoch in range(epochs):
            exp_train_loss = 0
            exp_train_acc = 0
            for idx, batch in enumerate(exp_train_loader):
                images, labels, task_ids = batch
                images = images.view(-1, seq_len, input_dim).to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                exp_train_loss += loss.item()
                exp_train_acc += torch.sum((torch.argmax(preds,dim=-1)==labels)).item()
            exp_train_losses.append(exp_train_loss/len(exp_train_loader))
            exp_train_accs.append(exp_train_acc/len(exp_train_dataset))

        exp_test_loss = 0
        exp_test_acc = 0
        model.eval()
        with torch.no_grad():
            for batch in exp_test_loader:
                images, labels, task_ids = batch
                images = images.view(-1, seq_len, input_dim).to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                exp_test_loss += loss.item()
                exp_test_acc += torch.sum((torch.argmax(preds,dim=-1)==labels)).item()
            

        exp_result = {
            "exp_id": exp.current_experience,
            "exp_train_loss": exp_train_losses[-1],
            "exp_train_acc": exp_train_accs[-1],
            "exp_test_loss": exp_test_loss/len(exp_test_loader),
            "exp_test_acc": exp_test_acc / len(exp_test_dataset)
        }
        train_results.append(exp_result)
    return train_results

# Train IMDB model with induced typos, not adv changes.
# Train the model with all the permutations at once: 180_000 images.
# PMNIST: train on 2 exp=> sleep=> evaluate on both the experiences.

def test(model, test_stream, criterion, device):
    model.eval()
    test_results = []
    input_dim = seq_len = 28
    for exp in test_stream:
        exp_id = exp.current_experience
        exp_test_dataset = exp.dataset
        exp_test_loader = DataLoader(exp_test_dataset, batch_size=32, shuffle=True)
        exp_test_loss = 0
        exp_test_acc = 0
        with torch.no_grad():
            for idx, batch in enumerate(exp_test_loader):
                images, labels, task_ids = batch
                images = images.view(-1, seq_len, input_dim).to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                exp_test_loss += loss.item()
                exp_test_acc += torch.sum(torch.argmax(preds,dim=-1)==labels).item()
            
        exp_result = {
            "exp_id": exp_id,
            "exp_test_acc": exp_test_acc/len(exp_test_dataset),
            "exp_test_loss": exp_test_loss/len(exp_test_dataset)
        }
        test_results.append(exp_result)
    return test_results

def create_sleep_input(train_stream, num_iterations):
    imgs = []
    for exp in train_stream:
        dataset = exp.dataset
        for ex in dataset:
            imgs.append(ex[0])
    # [60k, 1, 28, 28]
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
    split_mnist = SplitMNIST(n_experiences=5, seed=1234)

    input_dim = 28
    seq_len = 28
    rnn_hidden_dim = 256
    fc_dim = 128
    output_dim = 10 # num_classes
    dropout = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers = 1
    epochs = 2
    model = MnistRNN(input_dim, fc_dim, rnn_hidden_dim, output_dim, num_layers, dropout, device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_stream = split_mnist.train_stream # dataset for all the exps
    test_stream = split_mnist.test_stream
    train_results = train(model, criterion, optimizer,train_stream, test_stream, device, epochs)
    print("Training results with sequential training: ")
    print(train_results)
    test_results = test(model, test_stream, criterion, device)
    print("Test results before sleep: ")
    print(test_results)

    params = {} # sleep params
    params['inc'] 		= 0.001			# Magnitude of weight increase upon STDP event
    params['dec'] 		= 0.0001 		# Magnitude of weight decrease upon STDP event
    params['max_rate'] 	= 32.			# Maximum firing rate of neurons in the input layer
    params['dt'] 		= 0.001			# temporal resolution of simulation
    params['decay'] 	= 0.999			# decay at each time step
    params['threshold'] = 1.0 			# membrane threshold
    params['t_refractory'] 	= 0.0			# Refractory period
    params['alpha_linear'] 	= [0.5,0.5,0.5]	# synaptic scaling factors for feedforward weights
    params['alpha_rec'] = [0.5]		# Synaptic scaling factor for recurrent weights
    params['beta'] 		= [6., 6.5, 7.5]	# Synaptic threhsold scaling factors
    layer_sizes = [28, 256, 128, 10] # TODO: make this dynamic
    num_iterations = 10000

    sleep_input = create_sleep_input(train_stream, num_iterations)
    sleep_rnn_model = SleepRNNLayer(layer_sizes, params)
    model = sleep_rnn_model.sleep(model, sleep_input, num_iterations)

    model = model.to(device)
    test_results_after_sleep = test(model, test_stream, criterion, device)
    print("Test accuracy after sleep: ")
    print(test_results_after_sleep)

