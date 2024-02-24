import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVRRN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        # input layer
        self.input_layer = nn.Linear(input_size + hidden_size + latent_size, hidden_size + latent_size)

        # GRU cell for RNN
        self.rnn_cell = nn.GRUCell(hidden_size + latent_size, hidden_size + latent_size)

        # Mean and variance layers for latent distribution
        self.mean_layer = nn.Linear(hidden_size + latent_size, latent_size)
        self.var_layer = nn.Linear(hidden_size + latent_size, latent_size)

        # Output Layer
        self.output_layer = nn.Linear(hidden_size +latent_size, 1)

    def forward(self, x, h):
        # combine input and hidden state
        combined = self.input_layer(torch.cat((x,h),dim=1))

        # GRU call update
        hidden =self.rnn_cell(combined)

        # Mean and variance of latent distribution
        mean = self.mean_layer(hidden)
        var = F.softplus(self.var_layer(hidden))

        # Reparmeterization trick

        epsilon= torch.randn_like(mean)
        z = mean + epsilon* torch.sqrt(var)

        # combined hidden and latent states

        combined = hidden

        # Output layer
        output = self.output_layer(combined)

        return output, z, hidden




input_size = 10
hidden_size = 20
latent_size = 5

model = SimpleVRRN(input_size,hidden_size,latent_size)

x = torch.randn(1,input_size)
h = torch.randn(1,hidden_size+latent_size)
output ,z, h = model(x,h)

print(output.shape)


