from torch import nn

class MLP(nn.Module):
    r"""
    A simple 3-layer MLP applied independently to each timestep.

    Args:
        n_input: Number of input features per timestep.
        n_output: Number of output features per timestep.
        hidden_dim: Hidden dimension size for the intermediate layers.
        dropout: Dropout probability applied after each hidden linear layer.
    """
    def __init__(self, n_input, n_output, hidden_dim=512, dropout=0.05):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.layer3 = nn.Linear(hidden_dim, n_output)

    def forward(self, x, x_lens):
        r"""
        Args:
            x: Tensor of shape (B, 150, n_input)

        Returns:
            Tensor of shape (B, 150, n_output)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x,x_lens,None
