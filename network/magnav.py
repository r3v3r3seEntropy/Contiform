#!/usr/bin/env python3

import numpy as np
import torch

def rmse(predictions, targets, verbose=True):
    """
    Calculate Root Mean Square Error between predictions and targets.
    
    Arguments:
    - `predictions`: predicted values
    - `targets`: ground truth values  
    - `verbose`: whether to print result
    
    Returns:
    - `rmse_value`: RMSE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    mse = np.mean((predictions - targets) ** 2)
    rmse_value = np.sqrt(mse)
    
    if verbose:
        print(f'RMSE: {rmse_value:.2f}')
    
    return rmse_value

# Placeholder functions for compatibility
def create_TL_A(flux_x, flux_y, flux_z):
    """Placeholder for Tolles-Lawson A matrix creation"""
    return np.eye(3)  # Dummy implementation

def create_TL_coef(flux_x, flux_y, flux_z, mag, **kwargs):
    """Placeholder for Tolles-Lawson coefficient creation"""
    return np.zeros((3, 1))  # Dummy implementation

def apply_TL(mag, coef, A):
    """Placeholder for Tolles-Lawson application"""
    return mag  # Return unchanged for now

def igrf(lon, lat, h, date):
    """Placeholder for IGRF calculation"""
    # Return dummy magnetic field components
    Be = np.zeros_like(lon)
    Bn = np.zeros_like(lon)  
    Bu = np.zeros_like(lon)
    return Be, Bn, Bu 