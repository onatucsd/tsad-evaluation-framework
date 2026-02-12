import torch
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, seq_len, no_features, embedding_size, hidden_size= -1, num_layers=1, dropout=0):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features    # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size   # the number of features in the embedded points of the inputs' number of features
        self.dropout = dropout
        if hidden_size < 0:
            self.hidden_size = (2 * embedding_size) # The number of features in the hidden state h
        else:
            self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(
            input_size = no_features,
            hidden_size = embedding_size,
            num_layers = self.num_layers,
            dropout  = self.dropout,
            batch_first=True
        )
        
    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x, (hidden_state, cell_state) = self.LSTM(x)  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
        return last_lstm_layer_hidden_state
    
class Decoder(nn.Module):
    
    def __init__(self, seq_len, no_features, output_size, hidden_size=-1, num_layers=1, dropout=0):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.dropout = dropout
        if hidden_size < 0:
            self.hidden_size = (2 * no_features)
        else:
            self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(
            input_size = no_features,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout  = self.dropout,
            batch_first = True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out