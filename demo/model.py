"""
Written by KrishPro @ KP
"""

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_size, embed_size, vocab_size, n_layers, dropout = 0.5, output_size = 1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=dropout)
        
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(0.3)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, review):
        batch_size = review.size(0)
        
        embeded = self.embed(review) # (batch_size, seq_len, embed_size)
    
        lstm_out, _ = self.lstm(embeded) # (batch_size, seq_len, hidden_size)
        
        lstm_out = lstm_out.reshape(-1, self.hidden_size) # (batch_size * seq_len, hidden_size)
        
        linear = self.linear(lstm_out) # (batch_size * seq_len, 1) (output_size=1)
        
        linear = linear.reshape(batch_size, -1) # (batch_size, seq_len)
        
        output = linear[:,-1:] # (batch_size, 1) (taking last item on seq_len dim)
        
        output = self.sigmoid(output) # sigmoid
        
        return output
