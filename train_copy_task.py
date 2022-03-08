import torch
import time
from data import copy_task_dataloader
from models import CopyLSTM
from utils import epoch_time
import numpy as np

def train_batch(model, criterion, optimizer, batch, device):
    idx, inp, outp = batch
    inp = inp.to(device)
    outp = outp.to(device)

    input_dim = inp.shape[2]
    input_seq_len = inp.shape[0]
    output_seq_len, batch_size, _ = outp.shape

    model.reset(batch_size)

    # [5, 32, 8]
    for i in range(input_seq_len):
        model(inp[i])

    preds = torch.zeros(outp.shape)
    model_init = torch.zeros(batch_size, input_dim)
    for i in range(output_seq_len):
        preds[i] = model(model_init)

    loss = criterion(preds, outp)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # get cost
    preds_binarized = preds.clone().data
    preds_binarized = (preds_binarized > 0.5).float()
    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(preds_binarized - outp.data))
    return loss.item(), cost.item() / batch_size


def evaluate_batch(model, criterion, batch, device):
    idx, inp, outp = batch

        
    inp = inp.to(device)
    outp = outp.to(device)

    input_dim = inp.shape[2]
    input_seq_len = inp.shape[0]
    output_seq_len, batch_size, _ = outp.shape

    model.reset(batch_size)
    for i in range(input_seq_len):
        model(inp[i])
    preds = torch.zeros(outp.shape)
    model_init = torch.zeros(batch_size, input_dim)
    for i in range(output_seq_len):
        preds[i] = model(model_init)

    loss = criterion(preds, outp)
    

    # get cost
    preds_binarized = preds.clone().data
    preds_binarized = (preds_binarized > 0.5).float()
    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(preds_binarized - outp.data))
    acc = torch.mean((preds_binarized == outp).float())
    return loss.item(), acc.item(), cost.item() / batch_size

    # get cost
    preds_binarized = preds.clone().data
    preds_binarized = (preds_binarized > 0.5).float()
    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(preds_binarized - outp.data))
    return loss.item(), cost.item() / batch_size


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = []
    costs = []
    for batch in train_loader:
        idx, _, _ = batch
        start_time = time.time()
        loss, cost = train_batch(model, criterion, optimizer, batch, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        losses.append(loss)
        costs.append(cost)
        
        if idx % 10 == 0:
            print(f"Train loss : {np.array(losses).mean()}| Time: {epoch_mins}m {epoch_secs}s")
            print(f"Bit error rate: {np.array(costs).mean()}")
            print("====================================================================================")

    
    return losses, costs


if __name__ == "__main__":

    ## Dataset parameters
    num_batches = 1000
    batch_size = 32
    seq_width = 8
    min_seq_len = 1
    max_seq_len = 20
    train_loader = copy_task_dataloader(num_batches=num_batches,
                                batch_size=batch_size,
                                seq_width=seq_width,
                                min_seq_len=min_seq_len,
                                max_seq_len=max_seq_len)
    
    ## model hyperparams
    input_dim = 9
    output_dim = 8
    rnn_hidden_dim = 128
    num_layers = 1
    device = torch.device("cpu")

    model = CopyLSTM(input_dim, num_layers, output_dim, rnn_hidden_dim, device)
    model.to(device)

    ## Train
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters())
    losses, costs = train_model(model, train_loader, criterion, optimizer, device)
        
        