import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from data import QuestionClassificationDataLoader, build_word_vocab, convert_to_dataframe, load_data, preprocess, question_to_ids
from models import QuestionClassifier
from utils import create_glove_matrix, create_word_embedding, epoch_time
import time
import numpy as np

def train(model, optimizer, train_loader, device):
    
    print("Starting Training")
    train_loss = 0.
    train_acc = 0.
    model.train()
    
    for bi, batch in enumerate(train_loader):

        if bi % 100 == 0:
            print(f"Starting batch: {bi}")

        question_ids = batch['questions'].to(device)
        labels = batch['labels'].to(device)

        preds = model(question_ids)
        loss = F.cross_entropy(preds, labels)
        
        train_loss += loss.item()
        train_acc += (torch.argmax(preds,dim=1)==labels).float().mean().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return train_loss/len(train_loader), train_acc/len(train_loader)
    
    
def validate(model, valid_loader, device):
    
    print("Starting validation")
    valid_loss = 0.
    valid_acc = 0.
    model.eval()
    
    for bi, batch in enumerate(valid_loader):

        if bi % 20 == 0:
            print(f"Starting batch: {bi}")

        question_ids = batch['questions'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            
            preds = model(question_ids)
            loss = F.cross_entropy(preds, labels)
        
            valid_loss += loss.item()
            valid_acc += (torch.argmax(preds,dim=1)==labels).float().mean().item()
    
    return valid_loss/len(valid_loader), valid_acc/len(valid_loader)


if __name__ == "__main__":

    ## Load training data and preprocess it
    questions, coarse_labels, _ = load_data("train_4000.label.txt")
    questions = preprocess(questions)
    word2idx, idx2word, word_vocab = build_word_vocab(questions)
    data = convert_to_dataframe(questions, coarse_labels)
    question_ids = question_to_ids(questions, word2idx)
    data["question_ids"] = question_ids
    
    ## Split the data into training and validation sets.
    ## Create dataloaders for training and validation set.
    train_data, valid_data = train_test_split(data, test_size=0.2)
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
    projection_dim = 64
    rnn_hidden_dim = 32
    num_classes = 6
    device = torch.device("cpu")

    ## create a model
    model = QuestionClassifier(emb_dim, projection_dim, rnn_hidden_dim, num_classes, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    ## train the model
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")

        start_time = time.time()

        train_loss, train_acc = train(model, optimizer, train_loader, device)
        valid_loss, valid_acc = validate(model, valid_loader, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
        print(f"Epoch valid loss: {valid_loss}")
        print(f"Epoch train accuracy: {train_acc}")
        print(f"Epoch valid accuracy: {valid_acc}")
        print("====================================================================================")




