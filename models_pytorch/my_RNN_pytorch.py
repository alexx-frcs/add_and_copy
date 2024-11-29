# Simple RNN implementation
import torch
import torch.nn as nn


class SimpleRNNCell(nn.Module):

    def __init__(self, num_units, num_in):
        super(SimpleRNNCell, self).__init__()
        self.num_units = num_units
        self.num_in = num_in
        self.hidden_size = num_units

        # Input -> Hidden
        self.w_ih = nn.Parameter(torch.Tensor(num_units, num_in))
        nn.init.xavier_uniform_(self.w_ih)

        # Hidden -> Hidden
        self.w_hh = nn.Parameter(torch.Tensor(num_units, num_units))
        nn.init.xavier_uniform_(self.w_hh)

        # Hidden bias
        self.b_h = nn.Parameter(torch.zeros(num_units))

    @property
    def state_size(self):
        return self.num_units
    
    @property
    def output_size(self):
        return self.num_units
    
    def forward(self, inputs, states):
        """
        Args:
            inputs (Tensor): Input tensor of shape [batch_size, num_in].
            states (Tensor): Previous state tensor of shape [batch_size, num_units].
        Returns:
            new_state (Tensor): New state tensor of shape [batch_size, num_units].
        """
        # Previous state
        state = states

        # Linear combination of input
        inputs_mul = torch.matmul(inputs, self.w_ih.t())

        # Linear combination of state
        state_mul = torch.matmul(state, self.w_hh.t())
        # New state
        # new_state = torch.tanh(inputs_mul + state_mul + self.b_h)
        new_state = inputs_mul + state_mul + self.b_h

        return new_state