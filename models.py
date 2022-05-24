import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



class QuestionClassifierRNN(nn.Module):
    
    def __init__(self, emb_dim, fc_dim, rnn_hidden_dim, num_classes, device):
        super().__init__()
        self.device = device
        self.glove_embedding_layer = self.get_glove_embedding()
        # self.projection_layer = nn.Linear(in_features=emb_dim, out_features=projection_dim)
        self.rnn = nn.RNN(input_size=emb_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.fc = nn.Linear(in_features=rnn_hidden_dim, out_features=fc_dim)
        self.linear_layer = nn.Linear(in_features=fc_dim, out_features=num_classes)
    
    def get_glove_embedding(self):
        
        weights_matrix = np.load('glove-question100d.npy')
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device),freeze=False)

        return embedding
    
    def forward(self, text, text_lengths):
        # text = [bs, seq_len]
        
        embedded = F.relu(self.glove_embedding_layer(text))
        # embed = [bs, seq_len, emb_dim]

        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        # pack sequence to ignore padded positions while calculating hidden states.
        # projected = F.relu(self.projection_layer(embed))
        # projected = [bs, seq_len, projection_dim]
        
        packed_output, hidden = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # out = [bs, seq_len, rnn_hidden_dim]
        # hidden = [1, bs, rnn_hidden_dim]
        # output over padded tokens are zero tensors
        # out stacks the hidden states for all time-steps
        # hidden only contains the final hidden states 
        # we usually make predictions using the final hidden states.
        
        hidden = hidden.squeeze(0)
        # [bs, rnn_hidden_dim]
        
        logits = self.linear_layer(self.fc(hidden))
        # [bs, num_classes]
        
        return logits


class CopyLSTM(nn.Module):
    
    def __init__(self, input_dim, num_layers, output_dim, rnn_hidden_dim, device):
        
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=rnn_hidden_dim)
        self.linear = nn.Linear(rnn_hidden_dim, output_dim)
    
    def reset(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.rnn_hidden_dim).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.rnn_hidden_dim).to(self.device)
        self.state = (h,c)        
    
    def forward(self, x):
        # x = [bs, input_dim]
        
        x = x.unsqueeze(0)
        # x = [1, bs, input_dim]
        
        out, self.state = self.rnn(x, self.state)
        # out = [1, bs, rnn_hidden_dim]
        
        preds = self.linear(out)
        # [1, bs, output_dim]
        
        preds = torch.sigmoid(preds)
        
        return preds


class IMDBClassifierRNN(nn.Module):
    
    def __init__(self, input_dim, emb_dim, fc_dim, rnn_hidden_dim, output_dim, pad_idx, num_layers, dropout, device):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(input_size=emb_dim, hidden_size=rnn_hidden_dim, num_layers=num_layers, 
                          batch_first=True, bias=False, nonlinearity="relu")
        self.fc1 = nn.Linear(in_features=rnn_hidden_dim, out_features=fc_dim, bias=False)
        self.fc2 = nn.Linear(in_features=fc_dim, out_features=output_dim, bias=False)
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text = [bs, seq_len]
        
        embedded = F.relu(self.embedding(text))
        # embed = [bs, seq_len, emb_dim]

        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to("cpu"), batch_first=True)
        # pack sequence to ignore padded positions while calculating hidden states.
        # projected = F.relu(self.projection_layer(embed))
        # projected = [bs, seq_len, projection_dim]
        
        packed_output, hidden = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # out = [bs, seq_len, rnn_hidden_dim]
        # hidden = [1, bs, rnn_hidden_dim]
        # output over padded tokens are zero tensors
        # out stacks the hidden states for all time-steps
        # hidden only contains the final hidden states 
        # we usually make predictions using the final hidden states.
        
        out = output[:,-1,:]
        # [bs, rnn_hidden_dim]
        
        logits = self.fc2(F.relu(self.fc1(out)))
        # [bs, num_classes]
        
        return logits

class MnistRNN(nn.Module):

    def __init__(self, input_dim, fc_dim, rnn_hidden_dim, output_dim, num_layers, dropout, device):

        super().__init__()
        self.device = device
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_hidden_dim, num_layers=num_layers, 
                            batch_first=True, bias=False, nonlinearity="relu")
        self.fc1 = nn.Linear(in_features=rnn_hidden_dim, out_features=fc_dim, bias=False)
        self.fc2 = nn.Linear(in_features=fc_dim, out_features=output_dim, bias=False)
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [bs, seq_len, input_dim]

        out, hidden = self.rnn(x)
        # out = [bs, seq_len, hid_dim]

        out = out[:,-1,:]
        # [bs, hid_dim]

        logits = self.fc2(F.relu(self.fc1(out)))
        # [bs, num_classes]

        return logits
            
class CorruptedIMDBClassifierRNN(nn.Module):
    
    def __init__(self, input_dim, emb_dim, fc_dim, rnn_hidden_dim, output_dim, pad_idx, num_layers, dropout, device):
        super().__init__()
        self.device = device
        self.pad_idx = pad_idx
        self.embedding = self.get_glove_embedding()
        self.rnn = nn.RNN(input_size=emb_dim, hidden_size=rnn_hidden_dim, num_layers=num_layers, 
                          batch_first=True)
        self.fc1 = nn.Linear(in_features=rnn_hidden_dim, out_features=fc_dim)
        self.fc2 = nn.Linear(in_features=fc_dim, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def get_glove_embedding(self):
        
        weights_matrix = np.load('glove-imdb100d.npy')
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device),freeze=False, padding_idx=self.pad_idx)
        return embedding
    
    def forward(self, text, text_lengths):
        # text = [bs, seq_len]
        
        embedded = F.relu(self.embedding(text))
        # embed = [bs, seq_len, emb_dim]

        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to("cpu"), batch_first=True)
        # pack sequence to ignore padded positions while calculating hidden states.
        # projected = F.relu(self.projection_layer(embed))
        # projected = [bs, seq_len, projection_dim]
        
        packed_output, hidden = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # out = [bs, seq_len, rnn_hidden_dim]
        # hidden = [1, bs, rnn_hidden_dim]
        # output over padded tokens are zero tensors
        # out stacks the hidden states for all time-steps
        # hidden only contains the final hidden states 
        # we usually make predictions using the final hidden states.
        
        hidden = output[:,-1,:]
        # [bs, rnn_hidden_dim]
        
        logits = self.fc2(F.relu(self.fc1(hidden)))
        # [bs, num_classes]
        
        return logits

