import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

class DivergenceFreeMLP(nn.Module):
    """
    MLP that outputs a divergence-free vector field using antisymmetric Jacobian trick.
    The final vector field is projected to the desired output dimension.
    """
    def __init__(self, seq_length, n_features, output_dim):
        super(DivergenceFreeMLP, self).__init__()
        self.input_dim = seq_length * n_features
        self.output_dim = output_dim

        # Base MLP to define the b(x) vector field
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, self.input_dim)  # output same dimension as input for Jacobian
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Final projection to required output dimension
        self.out_proj = nn.Linear(self.input_dim, self.output_dim)

        self.name = "DivergenceFreeMLP"

    def base_forward(self, x):
        # Assumes x is already flattened
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

    def forward(self, x):
        # x: (batch_size, n_features, seq_length)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1).requires_grad_(True)  # (B, D)

        outputs = []
        for i in range(batch_size):
            xi = x_flat[i]

            def single_b(input_vec):
                return self.base_forward(input_vec)

            b = single_b(xi)  # (input_dim,)
            Jb = jacobian(single_b, xi)  # (input_dim, input_dim)
            A = Jb - Jb.T  # (input_dim, input_dim)
            v = A.sum(dim=1)  # (input_dim,)

            v_out = self.out_proj(v)  # (output_dim,)
            outputs.append(v_out.unsqueeze(0))  # Preserve shape and grad

        return torch.cat(outputs, dim=0)  # (batch_size, output_dim)
