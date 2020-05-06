import torch
import torch.nn as nn

## Dueling network
class Duel_net(nn.Module):
    def __init__(self, in_size, hidden_size, n_actions):
        super().__init__()

        # layer parameters
        self.in_size = in_size
        self.n_actions = n_actions

        self.lin_1 = nn.Linear(in_size, hidden_size)
        self.relu_1 = nn.LeakyReLU()
        self.v_lay = nn.Linear(hidden_size, 1)

        self.lin_2 = nn.Linear(in_size, hidden_size)
        self.relu_2 = nn.LeakyReLU()
        self.a_lay = nn.Linear(hidden_size, n_actions)

    def forward(self, input):
        V_vals = self.v_lay(self.relu_1(self.lin_1(input)))
        A_vals = self.a_lay(self.relu_2(self.lin_2(input)))
        A_vals = A_vals - torch.mean(A_vals, dim=1).unsqueeze(1)
        return V_vals + A_vals