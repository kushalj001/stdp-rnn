from copyreg import pickle
import time
import numpy as np
from models import CorruptedIMDBClassifierRNN
from nlp_utils import IMDBDataloader, build_word_vocab, corrupt_qwerty, corrupt_sentiment_masking, create_glove_matrix, create_word_embedding, get_df, load_data
from snn_utils import SleepRNNLayer
from utils import binary_accuracy, epoch_time
import torch
from torch import nn
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")


def train(model, train_loader, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for bi, batch in enumerate(train_loader):
        if bi % 100 == 0:
            print(f"Starting batch: {bi}")
        optimizer.zero_grad()
        
        text = batch["text"].to(device)
        labels = batch["labels"].to(device)
        text_lengths = batch["text_lengths"].to(device)
        
        predictions = model(text, text_lengths).squeeze(1)
#         print(predictions.dtype, predictions.shape)
#         print(labels.dtype, labels.shape)
        loss = criterion(predictions, labels.float())
        
        acc = binary_accuracy(predictions, labels.float())
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for bi, batch in enumerate(test_loader):

            text = batch["text"].to(device)
            labels = batch["labels"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, labels.float())
            
            acc = binary_accuracy(predictions, labels.float())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(test_loader), epoch_acc / len(test_loader)


def fit(model, optimizer, criterion, device):
    epochs = 1
    best_epoch = -1
    best_valid_acc = float('-inf')
    results = []
    for epoch in range(epochs):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')
        
        epoch_report = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "best_valid_acc": best_valid_acc,
        }
        results.append(epoch_report)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print(f"Best valid accuracy: {best_valid_acc}| Epoch: {best_epoch}")
    return results


def create_sleep_input(train_df, weights_matrix, word2idx):
    doc_embedding = np.zeros((100,))
    for idx, row in train_df.iterrows():
        review = row["review"]
        tokens = word_tokenize(review)
        for tok in tokens:
            idx = word2idx.get(tok, "<unk>")
            doc_embedding += weights_matrix[idx]
        doc_embedding /= len(tokens)
    doc_embedding /= len(train_df)
    sleep_input = torch.tile(torch.tensor(doc_embedding), (10000,1))
    sleep_input = sleep_input.cpu().detach().numpy()
    return sleep_input

if __name__ == "__main__":
    train_path = ".data/imdb/aclImdb/train/"
    test_path = ".data/imdb/aclImdb/test/"

    train_dataset = load_data(train_path)
    reviews = []
    for ex in train_dataset.data:
        reviews.append(ex["content"])


    word2idx, idx2word, word_vocab = build_word_vocab(reviews)
    train_df = get_df(train_dataset, word2idx)

    # TODO: add if else clauses depending on the corruption chosen in the command
    test_dataset = load_data(test_path)
    test_dataset = corrupt_qwerty(test_dataset, 50)
    # test_dataset = corrupt_sentiment_masking(test_dataset)
    # test_dataset = corrupt_remove_char(test_dataset, 30)

    test_df = get_df(test_dataset, word2idx)

    train_df["len"] = train_df["text_ids"].str.len()
    train_df = train_df.sort_values(by="len", ascending=False).drop(columns="len")
    test_df["len"] = test_df["text_ids"].str.len()
    test_df = test_df.sort_values(by="len", ascending=False).drop(columns="len")

    train_loader = IMDBDataloader(train_df, 32)
    test_loader = IMDBDataloader(test_df, 32)

    glove_dict = create_glove_matrix()
    weights_matrix, words_not_found = create_word_embedding(glove_dict, word_vocab)
    np.save('glove-imdb100d.npy',weights_matrix)


    device = torch.device("cuda")
    input_dim = len(word_vocab)
    emb_dim = 100
    rnn_hidden_dim = 256
    fc_dim = 128
    num_layers = 1
    output_dim = 1
    num_classes = 2
    dropout = 0.3
    pad_idx = 1

    model = CorruptedIMDBClassifierRNN(input_dim, emb_dim, fc_dim, rnn_hidden_dim, output_dim, pad_idx, num_layers, dropout, device)
    model.to(device)
    import torch.optim as optim
    unk_idx = 0
    pad_idx = 1
    model.embedding.weight.data[pad_idx] = torch.zeros(emb_dim)
    #model.embedding.weight.requires_grad = False
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    results = fit(model, optimizer, criterion, device)
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    model.load_state_dict(torch.load("best_model.pt"))
    
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

    weights_matrix = model.embedding.weight.data.cpu().detach().numpy()
    sleep_input = create_sleep_input(train_df, weights_matrix, word2idx)
    sleep_rnn_model = SleepRNNLayer(layer_sizes, params)
    model = sleep_rnn_model.sleep(model, sleep_input, num_iterations)

    model = model.to(device)
    print("Test accuracy after sleep: ")

    test_results_after_sleep = evaluate(model, test_loader, criterion, device)
    





