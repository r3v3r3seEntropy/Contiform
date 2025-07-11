# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
import math
import torch
from torch import nn
import torch.nn.functional as F
import math
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import with absolute imports
import tsrnn
from tsrnn import NETWORKS

# Try to import module components, fall back to local imports if needed
try:
    from physiopro.module.linear import ODELinear, InterpLinear
    from physiopro.module.positional_encoding import PositionalEncoding
except ImportError:
    # Fallback for local execution
    try:
        import sys
        import os
        # Get parent directory (Contiform root)
        parent_dir = os.path.dirname(current_dir)
        module_path = os.path.join(parent_dir, 'module')
        if module_path not in sys.path:
            sys.path.insert(0, module_path)
        
        # Try importing with dependencies check
        import torchcde
        import torchdiffeq
        from linear import ODELinear, InterpLinear
        from positional_encoding import PositionalEncoding
        print("Successfully loaded ODE modules")
    except ImportError as e:
        print(f"Warning: Could not import ODE modules: {e}")
        if 'torchcde' in str(e):
            print("Missing dependency: torchcde. Install with: pip install torchcde")
        elif 'torchdiffeq' in str(e):
            print("Missing dependency: torchdiffeq. Install with: pip install torchdiffeq")
        
        # Create dummy classes if modules not available
        class ODELinear(nn.Module):
            def __init__(self, d_model, d_out, args_ode):
                super().__init__()
                self.linear = nn.Linear(d_model, d_out)
            def forward(self, x, t):
                return self.linear(x)
        
        class InterpLinear(nn.Module):
            def __init__(self, d_model, d_out, args_ode):
                super().__init__()
                self.linear = nn.Linear(d_model, d_out)
            def forward(self, x, t):
                return self.linear(x)
                
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.d_model = d_model
            def forward(self, x):
                return x


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True, args_ode=None):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        assert d_model == n_head * d_k
        assert d_model == n_head * d_v

        self.w_qs = InterpLinear(d_model, n_head * d_k, args_ode)
        self.w_ks = ODELinear(d_model, n_head * d_k, args_ode)
        self.w_vs = ODELinear(d_model, n_head * d_v, args_ode)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, t, mask=None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)



        # Pass through the pre-attention projection
        # Handle ODE modules that return [B, T, T, K, D] by taking diagonal
        q_out = self.w_qs(q, t)
        k_out = self.w_ks(k, t)
        v_out = self.w_vs(v, t)
        
        # If ODE modules return 5D tensors, extract the diagonal (same time interactions)
        if len(q_out.shape) == 5:  # [B, T, T, K, D]
            # Take the diagonal elements where query time = key time
            q_out = q_out[:, torch.arange(len_q), torch.arange(len_q), :, :].squeeze(-2)  # [B, T, D]
        if len(k_out.shape) == 5:
            k_out = k_out[:, torch.arange(len_k), torch.arange(len_k), :, :].squeeze(-2)
        if len(v_out.shape) == 5:
            v_out = v_out[:, torch.arange(len_v), torch.arange(len_v), :, :].squeeze(-2)
            
        # Now reshape to separate heads
        q = q_out.view(sz_b, len_q, n_head, d_k)
        k = k_out.view(sz_b, len_k, n_head, d_k)
        v = v_out.view(sz_b, len_v, n_head, self.d_v)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

    def interpolate(self, q, k, v, t, qt, mask=None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        len_qt = qt.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs.interpolate(q, t, qt, mask=mask).view(sz_b, len_qt, -1, n_head, d_k)
        k = self.w_ks.interpolate(k, t, qt).view(sz_b, len_qt, -1, n_head, d_k)
        v = self.w_vs.interpolate(v, t, qt).view(sz_b, len_qt, -1, n_head, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.permute(0, 3, 1, 2, 4), k.permute(0, 3, 1, 2, 4), v.permute(0, 3, 1, 2, 4)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, _ = self.attention.interpolate(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_qt, -1)
        output = self.fc(output)

        return output


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward neural network."""

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # Standard scaled dot-product attention
        # q, k, v have shape: [batch, n_head, seq_len, d_k]
        d_k = q.size(-1)
        
        # Compute attention scores: [batch, n_head, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        # Apply attention to values: [batch, n_head, seq_len, d_k]
        output = torch.matmul(attn, v)
        
        return output, attn

    def interpolate(self, q, k, v, mask=None):
        # Standard scaled dot-product attention for interpolation
        d_k = q.size(-1)
        
        # Compute attention scores: [batch, n_head, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values: [batch, n_head, seq_len, d_k]
        output = torch.matmul(attn, v)
        
        return output, attn


class Encoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
            self,
            d_model,
            d_inner,
            n_layers,
            n_head,
            d_k,
            d_v,
            dropout,
            args,
            add_pe=False,
            normalize_before=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.pe = PositionalEncoding(d_model)
        self.add_pe = add_pe

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)])

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before, args=args
                )
                for _ in range(n_layers)
            ]
        )

    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, x, t, mask):
        """Encode event sequences via masked self-attention."""

        tem_enc = self.temporal_enc(t)
        enc_output = x
        if self.add_pe:
            enc_output = self.pe(enc_output)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(enc_output, t, mask=mask)
        return enc_output


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True, args=None):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before, args_ode=args
        )
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, x, t, mask=None):
        enc_output, enc_slf_attn = self.slf_attn(x, x, x, t, mask=mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


@NETWORKS.register_module("contiformer")
class ContiFormer(nn.Module):
    """A sequence to sequence model with attention mechanism."""

    def __init__(
            self,
            input_size: Optional[int] = None,
            d_model: Optional[int] = 256,
            d_inner: Optional[int] = 1024,
            n_layers: Optional[int] = 4,
            n_head: Optional[int] = 4,
            d_k: Optional[int] = 64,
            d_v: Optional[int] = 64,
            dropout: Optional[float] = 0.1,
            actfn_ode: Optional[str] = "softplus",
            layer_type_ode: Optional[str] = "concat",
            zero_init_ode: Optional[bool] = True,
            atol_ode: Optional[float] = 1e-3,  # Tighter tolerance as per checklist
            rtol_ode: Optional[float] = 1e-3,  # Tighter tolerance as per checklist
            method_ode: Optional[str] = "rk4",
            linear_type_ode: Optional[str] = "inside",
            regularize: Optional[bool] = False,  # No longer needed with divergence-free
            approximate_method: Optional[str] = "last",
            nlinspace: Optional[int] = 3,
            interpolate_ode: Optional[str] = "linear",
            itol_ode: Optional[float] = 1e-2,
            add_pe: Optional[bool] = False,
            normalize_before: Optional[bool] = False,
            max_length: int = 100,
            divergence_free: Optional[bool] = True,  # Enable divergence-free by default
    ):
        super().__init__()

        args_ode = {
            "actfn": actfn_ode,
            "layer_type": layer_type_ode,
            "zero_init": zero_init_ode,
            "atol": atol_ode,
            "rtol": rtol_ode,
            "method": method_ode,
            "regularize": False,  # Always False for divergence-free fields
            "approximate_method": approximate_method,
            "nlinspace": nlinspace,
            "linear_type": linear_type_ode,
            "interpolate": interpolate_ode,
            "itol": itol_ode,
            "divergence_free": divergence_free,  # Pass divergence-free flag
        }
        args_ode = AttrDict(args_ode)

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            args=args_ode,
            add_pe=add_pe,
            normalize_before=normalize_before,
        )

        self.d_model = d_model

        if input_size is not None:   # None for hidden state inputs
            self.linear = nn.Linear(input_size, d_model)
        else:
            self.linear = None

        # Add final linear layer for regression to single output
        self.output_layer = nn.Linear(d_model, 1)

        self.max_length = max_length
        self.__output_size = d_model
        self.__hidden_size = d_model

    def forward(self, x, t=None, mask=None):
        # Handle MagNav input shape: [batch, features, seq_len] -> [batch, seq_len, features]
        if len(x.shape) == 3 and x.shape[2] == self.max_length:
            x = x.transpose(1, 2)
        
        if t is None:   # default to regular time series
            t = torch.linspace(0, 1, x.shape[1]).to(x.device)
            t = t.unsqueeze(0).repeat(x.shape[0], 1)

        if self.linear is not None:
            x = self.linear(x)

        enc_output = self.encoder(x, t, mask)
        # Take the last time step's output and map to single prediction
        final_output = enc_output[:, -1, :]  # [batch, d_model]
        prediction = self.output_layer(final_output)  # [batch, 1]
        return prediction

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
