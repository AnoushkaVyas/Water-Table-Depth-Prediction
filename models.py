import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, class_size, dropout, rnn_type,dropout_bool):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.class_size = class_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout_bool=dropout_bool

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,     # rnn hidden unit
                num_layers=self.num_layers,       # number of rnn layer
                batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
            
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
            )

        elif self.rnn_type == 'ffnn':
            self.rnn = nn.Linear(self.input_size, self.hidden_size)

        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.hidden_size, self.class_size) # FC layer 

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            r_out, _ = self.rnn(x, (h0, c0))
            
        elif self.rnn_type=='ffnn':
            r_out = self.rnn(x)

        else:
            r_out, _ = self.rnn(x, h0)

        outs = []    # save all predictions

        if self.dropout_bool:
            for time_step in range(r_out.size(1)):    # calculate output for each time step
                outs.append(self.out(self.dropout((r_out[:, time_step, :]))))

        else:
            for time_step in range(r_out.size(1)):    # calculate output for each time step
                outs.append(self.out((r_out[:, time_step, :])))

        return torch.stack(outs, dim=1)
