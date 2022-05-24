import torch
import torch.nn.functional as F
from data import (
    QuestionClassificationDataLoader, 
    build_word_vocab, 
    convert_to_dataframe, 
    load_data, preprocess, 
    text_to_ids
)
from models import QuestionClassifierRNN
from utils import create_glove_matrix, create_word_embedding, epoch_time
import time
import numpy as np

def train(model, optimizer, train_loader):
    
    print("Starting Training")
    train_loss = 0.
    train_acc = 0.
    model.train()
    
    for bi, batch in enumerate(train_loader):

        if bi % 100 == 0:
            print(f"Starting batch: {bi}")

        question_ids = batch['text'].to(device)
        labels = batch['labels'].to(device)
        qtn_lengths = batch["text_lengths"].to(device)
        preds = model(question_ids, qtn_lengths)
        loss = F.cross_entropy(preds, labels)
        
        train_loss += loss.item()
        train_acc += (torch.argmax(preds,dim=1)==labels).float().mean().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return train_loss/len(train_loader), train_acc/len(train_loader)
    
    

def validate(model, valid_loader):
    
    print("Starting validation")
    valid_loss = 0.
    valid_acc = 0.
    model.eval()
    
    for bi, batch in enumerate(valid_loader):

        if bi % 20 == 0:
            print(f"Starting batch: {bi}")

        question_ids = batch['text'].to(device)
        labels = batch['labels'].to(device)
        qtn_lengths = batch["text_lengths"].to(device)
        with torch.no_grad():
            
            preds = model(question_ids, qtn_lengths)
            loss = F.cross_entropy(preds, labels)
            
            valid_loss += loss.item()
            valid_acc += (torch.argmax(preds,dim=1)==labels).float().mean().item()
    
    return valid_loss/len(valid_loader), valid_acc/len(valid_loader)


if __name__ == "__main__":

    ## Load training data and preprocess it
    train_examples, train_labels, _ = load_data("train_5500.label.txt")
    valid_examples, valid_labels, _ = load_data("TREC_10.label.txt")
    train_examples = preprocess(train_examples)
    valid_examples = preprocess(valid_examples)
    word2idx, idx2word, word_vocab = build_word_vocab(train_examples)
    train_data = convert_to_dataframe(train_examples, train_labels)
    valid_data = convert_to_dataframe(valid_examples, valid_labels)
    train_text_ids = text_to_ids(train_examples, word2idx)
    valid_text_ids = text_to_ids(valid_examples, word2idx)
    train_data["text_ids"] = train_text_ids
    valid_data["text_ids"] = valid_text_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    ## Create dataloaders for training and validation set.
    train_loader = QuestionClassificationDataLoader(train_data, 32)
    valid_loader = QuestionClassificationDataLoader(valid_data, 32)


    ## Load static word embeddings for glove. Diehl et al train their own 64-dim word2vec
    ## from scratch.
    ## only have to do it on the first run to save the weights matrix
    glove_dict = create_glove_matrix()
    weights_matrix, words_not_found = create_word_embedding(glove_dict, word_vocab)
    np.save('glove-question100d.npy',weights_matrix)


    ## model hyperparameters
    emb_dim = 100
    rnn_hidden_dim = 256
    fc_dim = 128
    num_classes = 6
    device = torch.device("cpu")

    ## create a model
    model = QuestionClassifierRNN(emb_dim, fc_dim, rnn_hidden_dim, num_classes, device)
    model = model.to(device)


    ## train the model
    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")

        start_time = time.time()

        train_loss, train_acc = train(model, optimizer, train_loader)
        valid_loss, valid_acc = validate(model, valid_loader)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
        print(f"Epoch valid loss: {valid_loss}")
        print(f"Epoch train accuracy: {train_acc}")
        print(f"Epoch valid accuracy: {valid_acc}")
        print("====================================================================================")




