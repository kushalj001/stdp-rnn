import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



class QuestionClassifier(nn.Module):
    
    def __init__(self, emb_dim, projection_dim, rnn_hidden_dim, num_classes, device):
        super().__init__()
        self.device = device
        self.glove_embedding_layer = self.get_glove_embedding()
        self.projection_layer = nn.Linear(in_features=emb_dim, out_features=projection_dim)
        self.rnn = nn.LSTM(input_size=projection_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.linear_layer = nn.Linear(in_features=rnn_hidden_dim, out_features=num_classes)
    
    def get_glove_embedding(self):
        
        weights_matrix = np.load('glove-question100d.npy')
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device),freeze=False)

        return embedding
    
    def forward(self, x):
        # x = [bs, seq_len]
        
        embed = F.relu(self.glove_embedding_layer(x))
        # embed = [bs, seq_len, emb_dim]
        
        projected = F.relu(self.projection_layer(embed))
        # projected = [bs, seq_len, projection_dim]
        
        out, (hidden, cell) = self.rnn(projected)
        # out = [bs, seq_len, rnn_hidden_dim]
        # hidden = [1, bs, rnn_hidden_dim]
        # out stacks the hidden states for all time-steps
        # hidden only contains the final hidden states 
        # we usually make predictions using the final hidden states.
        
        hidden = hidden.squeeze(0)
        # [bs, rnn_hidden_dim]
        
        logits = self.linear_layer(hidden)
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
        
        