import time
import torch
import numpy as np
import pandas as pd
from attack_utils import CustomModelWrapper
from models import IMDBClassifierRNN
import torchtext
from torchtext.legacy import data, datasets
import random
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from textattack import AttackArgs
from textattack.datasets import Dataset
from textattack import Attacker
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import PWWSRen2019, DeepWordBugGao2018, Pruthi2019
from snn_utils import SleepRNNLayer, create_sleep_input
from utils import binary_accuracy, clip_grads, epoch_time

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        clip_grads(model)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def fit(model, train_iterator, test_iterator, optimizer, criterion):
    epochs = 1
    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    for epoch in range(epochs):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, test_iterator, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    return train_losses, valid_losses, train_accs, valid_accs


if __name__ == "__main__":
    SEED = 1234
    review = data.Field(tokenize="spacy", tokenizer_language="en_core_web_sm", include_lengths=True, batch_first=True)
    sentiment = data.LabelField(dtype=torch.float, batch_first=True)

    train_data, test_data = datasets.IMDB.splits(text_field=review, label_field=sentiment)
    max_vocab_size = 25_000
    review.build_vocab(train_data, 
                        max_size=max_vocab_size, 
                        vectors="glove.6B.100d", 
                        unk_init=torch.Tensor.normal_
                    )
    sentiment.build_vocab(train_data)
    batch_size = 32
    device = torch.device("cpu")
    train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data), 
                                                            batch_size=batch_size, 
                                                            sort_within_batch=True, 
                                                            device=device)
    # classifier model args
    input_dim = len(review.vocab)
    emb_dim = 100
    rnn_hidden_dim = 256
    fc_dim = 128
    num_layers = 1
    output_dim = 1
    dropout = 0.5
    pad_idx = review.vocab.stoi[review.pad_token]
    unk_idx = review.vocab.stoi[review.unk_token]

    model = IMDBClassifierRNN(input_dim, emb_dim, fc_dim, rnn_hidden_dim, output_dim, pad_idx, num_layers, dropout, device)
    model.to(device)
    pretrained_embeddings = review.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[unk_idx] = torch.zeros(emb_dim)
    model.embedding.weight.data[pad_idx] = torch.zeros(emb_dim)
    model.embedding.weight.requires_grad = False


    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    
    train_losses, valid_losses, train_accs, valid_accs = fit(model, train_iterator, test_iterator,
                                                            optimizer, criterion)
    # attack
    model_wrapper = CustomModelWrapper(model, review, device)
    attack = Pruthi2019.build(model_wrapper)
    dataset = HuggingFaceDataset("imdb", None, "test", shuffle=True)
    attack_args = AttackArgs(num_examples=10, checkpoint_dir="checkpoints")
    attacker = Attacker(attack, dataset, attack_args)
    attack_results_before_sleep = attacker.attack_dataset()
    print(attack_results_before_sleep)


    # sleep
    # all sleep args
    params = {} # sleep params
    params['inc'] 		= 0.001			# Magnitude of weight increase upon STDP event
    params['dec'] 		= 0.0001 		# Magnitude of weight decrease upon STDP event
    params['max_rate'] 	= 32.			# Maximum firing rate of neurons in the input layer
    params['dt'] 		= 0.001			# temporal resolution of simulation
    params['decay'] 	= 0.999			# decay at each time step
    params['threshold'] = 1.0 			# membrane threshold
    params['t_ref'] 	= 0.0			# Refractory period
    params['alpha_ff'] 	= [0.5,0.5,0.5]	# synaptic scaling factors for feedforward weights
    params['alpha_rec'] = [0.5]		# Synaptic scaling factor for recurrent weights
    params['beta'] 		= [6., 6.5, 7.5]	# Synaptic threhsold scaling factors
    layer_sizes = [100, 256, 128, 1] # TODO: make this dynamic
    num_iterations = 10000
    word2idx = review.vocab.stoi
    vectors = review.vocab.vectors # TODO: embedding.vectors if not using glove
    sleep_input = create_sleep_input(model, vectors, word2idx, num_iterations)
    sleep_rnn_model = SleepRNNLayer(layer_sizes, params)
    model = sleep_rnn_model.sleep(model, sleep_input, num_iterations)


    # attack
    model_wrapper = CustomModelWrapper(model, review, device)
    attack = Pruthi2019.build(model_wrapper)
    attacker = Attacker(attack, dataset, attack_args)
    attack_results_after_sleep = attacker.attack_dataset()
    print(attack_results_after_sleep)

    
