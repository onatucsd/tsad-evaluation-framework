import torch
import torch.nn as nn
from layers.LSTM import Encoder, Decoder 

class Model(nn.Module):
    """
    LSTM Autoencoder
    """
    def __init__(self, configs, window_len=100, n_features=51, encoding_dim=64, 
                 hidden_size=-1, num_layers=1, dropout=0):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = window_len
        self.no_features = n_features
        self.embedding_dim = encoding_dim
        self.dropout = dropout

        # Encoder
        self.encoder = Encoder(self.seq_len, self.no_features, self.embedding_dim, hidden_size=hidden_size, 
                               num_layers=num_layers, dropout=dropout)
        # Decoder
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.no_features, hidden_size=hidden_size, 
                               num_layers=num_layers, dropout=dropout)
        
    def anomaly_detection(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out)
        squeezed_decoded = dec_out.squeeze()
        return squeezed_decoded
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out 
        return None
    
    #def encode(self, x):
    #    self.eval()
    #    encoded = self.encoder(x)
    #    return encoded
    
    #def decode(self, x):
    #    self.eval()
    #    decoded = self.decoder(x)
    #    squeezed_decoded = decoded.squeeze()
    #    return squeezed_decoded