import torch.nn as nn

class Model(nn.Module):
    
    """
    TCN AE Autoencoder 
    https://github.com/locuslab/TCN
    """
    
    def __init__(self, configs, num_inputs, seq_len=100, kernel_size=2,
                 n_channels=51, dropout=0):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.num_inputs = num_inputs
        self.encoder_layers = []
        num_levels = len(n_channels)

        # Encoder

        # Decoder