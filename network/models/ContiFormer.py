#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to path to import physiopro modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the ContiFormer components directly
import contiformer
import divfree

class ContiFormer(nn.Module):
    """
    ContiFormer adapted for MagNav training with divergence-free ODEs.
    This wrapper makes ContiFormer compatible with the MagNav training framework.
    """
    
    def __init__(self, seq_length, n_features, divergence_free=True, d_model=64, n_layers=3, n_head=4):
        """
        Initialize ContiFormer for MagNav.
        
        Arguments:
        - `seq_length`: number of time steps in input sequence
        - `n_features`: number of input features (magnetometer + flight dynamics)
        - `divergence_free`: whether to use divergence-free ODEs (default: True)
        - `d_model`: model dimension (default: 64, smaller for magnetometer data)
        - `n_layers`: number of transformer layers (default: 3)
        - `n_head`: number of attention heads (default: 4)
        """
        super(ContiFormer, self).__init__()
        
        self.seq_length = seq_length
        self.n_features = n_features
        self.divergence_free = divergence_free
        
        # Create the base ContiFormer with adapted parameters
        self.contiformer = contiformer.ContiFormer(
            input_size=n_features,
            d_model=d_model,
            d_inner=d_model * 2,  # Standard transformer scaling
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            dropout=0.1,
            divergence_free=divergence_free,  # Enable divergence-free ODEs
            atol_ode=1e-3,  # Tighter tolerance for real data
            rtol_ode=1e-3,
            method_ode="rk4",
            normalize_before=True,
            max_length=seq_length
        )
        
        # Output layer to predict single magnetometer value
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
        self.name = "ContiFormer"
        
    def forward(self, x):
        """
        Forward pass for MagNav training.
        
        Arguments:
        - `x`: input tensor of shape [batch_size, n_features, seq_length]
        
        Returns:
        - `output`: predicted magnetometer value [batch_size, 1]
        """
        # ContiFormer expects [batch_size, seq_length, n_features]
        x = x.transpose(1, 2)  # [batch_size, seq_length, n_features]
        
        # Get ContiFormer output
        # contiformer returns (full_sequence_output, last_hidden_state)
        _, last_hidden = self.contiformer(x)
        
        # Apply output layer to get final prediction
        output = self.output_layer(last_hidden)
        
        return output
    
    def check_divergence_free_property(self, device='cpu'):
        """
        Check if the ODEs in this model are divergence-free.
        """
        if not self.divergence_free:
            print("Model not configured for divergence-free ODEs")
            return False
            
        print(f"Checking divergence-free property of {self.name}...")
        
        # Check each encoder layer
        for i, layer in enumerate(self.contiformer.encoder.layer_stack):
            print(f"\nLayer {i+1}:")
            
            # Check keys ODE
            if hasattr(layer.slf_attn.w_ks, 'ode_func'):
                print("  Keys ODE:")
                divfree.check_divergence_free(layer.slf_attn.w_ks.ode_func, dims=self.contiformer.d_model, n=64)
            
            # Check values ODE  
            if hasattr(layer.slf_attn.w_vs, 'ode_func'):
                print("  Values ODE:")
                divfree.check_divergence_free(layer.slf_attn.w_vs.ode_func, dims=self.contiformer.d_model, n=64)
        
        return True
        
    def get_model_info(self):
        """
        Get information about the model configuration.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.name,
            'divergence_free': self.divergence_free,
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'd_model': self.contiformer.d_model,
            'n_layers': len(self.contiformer.encoder.layer_stack),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
        
        return info


class DivFreeContiFormer(ContiFormer):
    """
    Explicit divergence-free ContiFormer for easy identification.
    """
    def __init__(self, seq_length, n_features, **kwargs):
        kwargs['divergence_free'] = True
        super().__init__(seq_length, n_features, **kwargs)
        self.name = "DivFreeContiFormer"


class StandardContiFormer(ContiFormer):
    """
    Standard ContiFormer without divergence-free constraints for comparison.
    """
    def __init__(self, seq_length, n_features, **kwargs):
        kwargs['divergence_free'] = False
        super().__init__(seq_length, n_features, **kwargs)
        self.name = "StandardContiFormer" 