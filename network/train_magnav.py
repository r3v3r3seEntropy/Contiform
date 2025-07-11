#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

import argparse
import warnings
import os
from datetime import datetime
import time
import math
import psutil

import magnav
from models.CNN import CNN, ResNet18
from models.RNN import LSTM, GRU
from models.MLP import MLP
# Import the ODE-based ContiFormer
from contiformer import ContiFormer
from se3_contiformer import SE3ContiFormerMagNav
from models.DivergencefreeMLP import DivergenceFreeMLP



#-----------------#
#----Functions----#
#-----------------#
    
    
def trim_data(data, seq_len):
    '''
    Delete part of the training data so that the remainder of the Euclidean division between the length of the data and the size of a sequence is 0. This ensures that all sequences are complete.
    
    Arguments:
    - `data` : data that needs to be trimmed
    - `seq_len` : lenght of a sequence
    
    Returns:
    - `data` : trimmed data
    '''
    if (len(data)%seq_len) != 0:
        data = data[:-(len(data)%seq_len)]
    else:
        pass
    
    return data
    
    
class MagNavDataset(Dataset):
    '''
    Transform Pandas dataframe of flights data into a custom PyTorch dataset that returns the data into sequences of a desired length.
    '''
    def __init__(self, df, seq_len, split, train_lines, test_lines,truth='IGRFMAG1'):
        '''
        Initialization of the dataset.
        
        Arguments:
        - `df` : dataframe to transform in a custom PyTorch dataset
        - `seq_len` : length of a sequence
        - `split` : data split ('train' or 'test')
        - `train_lines` : flight lines used for training
        - `test_lines` : flight lines used for testing
        - `truth` : ground truth used as a reference for training the model ('IGRFMAG1' or 'COMPMAG1')
        
        Returns:
        - None
        '''
        self.seq_len  = seq_len
        self.features = df.drop(columns=['LINE',truth]).columns.to_list()
        self.train_sections = train_lines
        self.test_sections = test_lines
        
        if split == 'train':
            
            # Create a mask to keep only training data
            mask_train = pd.Series(dtype=bool)
            for line in self.train_sections:
                mask = (df.LINE == line)
                mask_train = mask|mask_train
            
            # Split in X, y for training
            X_train = df.loc[mask_train,self.features]
            y_train = df.loc[mask_train,truth]
            
            # Trim data and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_train.to_numpy(),dtype=torch.float32),seq_len))
            self.y = trim_data(torch.tensor(np.reshape(y_train.to_numpy(),[-1,1]),dtype=torch.float32),seq_len)
            
        elif split == 'test':
            
            # Create a mask to keep only testing data
            mask_test = pd.Series(dtype=bool)
            for line in self.test_sections:
                mask = (df.LINE == line)
                mask_test = mask|mask_test
            
            # Split in X, y for testing
            X_test = df.loc[mask_test,self.features]
            y_test = df.loc[mask_test,truth]
            
            # Trim data and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_test.to_numpy(),dtype=torch.float32),seq_len))
            self.y = trim_data(torch.tensor(np.reshape(y_test.to_numpy(),[-1,1]),dtype=torch.float32),seq_len)

    def __getitem__(self, idx):
        '''
        Return a sequence for a given index.
        
        Arguments:
        - `idx` : index of a sequence
        
        Returns:
        - `X` : sequence of features
        - `y` : ground truth corresponding to the sequence
        '''
        X = self.X[:,idx:(idx+self.seq_len)]
        y = self.y[idx+self.seq_len-1]
        return X, y
    
    def __len__(self):
        '''
        Return the numbers of sequences in the dataset.
        
        Arguments:
        -None
        
        -Returns:
        -number of sequences in the dataset
        '''
        return len(torch.t(self.X))-self.seq_len


def make_training(model, epochs, train_loader, test_loader, scaling=['None']):
    '''
    PyTorch training loop with testing.
    
    Arguments:
    - `model` : model to train
    - `epochs` : number of epochs to train the model
    - `train_loader` : PyTorch dataloader for training
    - `test_loader` : PyTorch dataloader for testing
    - `scaling` : (optional) scaling parameters
    
    Returns:
    - `train_loss_history` : history of loss values during training
    - `test_loss_history` : history of loss values during testing
    '''
    # Different hyperparameters for ContiFormer
    if 'ContiFormer' in model.name or 'Transformer' in model.name:
        # ContiFormer needs different hyperparameters
        learning_rate = 0.0001  # Lower learning rate for transformers
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999))
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        print(f"Using ContiFormer optimization: AdamW with lr={learning_rate}, cosine annealing schedule")
    else:
        # Original settings for CNN/MLP
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
        lambda1 = lambda epoch: 0.9**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        print(f"Using standard optimization: Adam with lr=0.001, exponential decay")
    
    # Create batch and epoch progress bar
    batch_bar = tqdm(total=len(train)//BATCH_SIZE,unit="batch",desc='Training',leave=False, position=0, ncols=150)
    epoch_bar = tqdm(total=epochs,unit="epoch",desc='Training',leave=False, position=1, ncols=150)
    
    train_loss_history = []
    test_loss_history = []
    Best_RMSE = 9e9
    patience = 20  # Early stopping patience
    patience_counter = 0

    for epoch in range(epochs):

        #----Train----#

        train_running_loss = 0.

        # Turn on gradients computation
        model.train()
        
        batch_bar.reset()
        
        # Enumerate allow to track batch index and intra-epoch reporting 
        for batch_index, (inputs, labels) in enumerate(train_loader):
            
            # Put data to the desired device (CPU or GPU)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Make predictions for this batch
            predictions = model(inputs)

            # Compute the loss
            loss = criterion(predictions, labels)

            # Zero gradients of optimizer for every batch
            optimizer.zero_grad()

            # Calculate gradients
            loss.backward()
            
            # Gradient clipping for ContiFormer
            if 'ContiFormer' in model.name:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_running_loss += loss.item()
            
            # Update batch progess bar
            batch_bar.set_postfix(train_loss=train_running_loss/(batch_index+1),lr=optimizer.param_groups[0]['lr'])
            batch_bar.update()
        
        # Update learning rate
        scheduler.step()

        # Compute the loss of the batch and save it
        train_loss = train_running_loss / (batch_index + 1)
        train_loss_history.append(train_loss)

        #----Test----#

        test_running_loss = 0.
        preds = []
        
        # Disable layers specific to training such as Dropout/BatchNorm
        model.eval()
        
        # Turn off gradients computation
        with torch.no_grad():
            
            # Enumerate allow to track batch index and intra-epoch reporting
            for batch_index, (inputs, labels) in enumerate(test_loader):

                # Put data to the desired device (CPU or GPU)
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Make prediction for this batch
                predictions = model(inputs)
                
                # Save prediction for this batch
                preds.append(predictions.cpu())

                # Compute the loss
                loss = criterion(predictions, labels)

                # Gather data and report
                test_running_loss += loss.item()

        # Compute the loss of the batch and save it
        preds = np.concatenate(preds)

                # --- Handle Subset wrapper ---
        if isinstance(test, torch.utils.data.Subset):
            full_targets = torch.stack([test.dataset[i][1] for i in test.indices[SEQ_LEN:]])
        else:
            full_targets = test.y[SEQ_LEN:]

        # --- Truncate predictions and targets to matching length ---
        min_len = min(preds.shape[0], full_targets.shape[0])
        preds = preds[:min_len]
        full_targets = full_targets[:min_len]

        # --- Compute RMSE properly with scaling ---
        if scaling[0] == 'None':
            RMSE_epoch = magnav.rmse(preds, full_targets, False)
        elif scaling[0] == 'std':
            RMSE_epoch = magnav.rmse(preds * scaling[2] + scaling[1], full_targets * scaling[2] + scaling[1], False)
        elif scaling[0] == 'minmax':
            RMSE_epoch = magnav.rmse(
                scaling[3] + ((preds - scaling[1]) * (scaling[4] - scaling[3]) / (scaling[2] - scaling[1])),
                scaling[3] + ((full_targets - scaling[1]) * (scaling[4] - scaling[3]) / (scaling[2] - scaling[1])),
                False
            )



        test_loss = test_running_loss / (batch_index + 1)
        test_loss_history.append(test_loss)
        
        # Save best model
        if Best_RMSE > RMSE_epoch:
            Best_RMSE = RMSE_epoch
            Best_model = model
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

        # Update epoch progress bar
        epoch_bar.set_postfix(train_loss=train_loss,test_loss=test_loss,RMSE=RMSE_epoch,lr=optimizer.param_groups[0]['lr'])
        epoch_bar.update()
    print('\n')
    
    return train_loss_history, test_loss_history, Best_RMSE, Best_model


def Standard_scaling(df):
    '''
    Apply standardization (Z-score normalization) to a pandas dataframe except for the 'LINE' feature.
    
    Arguments:
    - `df` : dataframe to standardize
    
    Returns:
    - `df_scaled` : standardized dataframe
    '''
    df_scaled = (df-df.mean())/df.std()
    df_scaled['LINE'] = df['LINE']

    return df_scaled


def MinMax_scaling(df, bound=[-1,1]):
    '''
    Apply min-max scaling to a pandas dataframe except for the 'LINE' feature.

    Arguments:
    - `df` : dataframe to standardize
    - `bound` : (optional) upper and lower bound for min-max scaling
    
    Returns:
    - `df_scaled` : scaled dataframe
    '''
    df_scaled = bound[0] + ((bound[1]-bound[0])*(df-df.min())/(df.max()-df.min()))
    df_scaled['LINE'] = df['LINE']
    
    return df_scaled


def apply_corrections(df,mags_to_cor,diurnal=True,igrf=True):
    '''
    Apply IGRF and/or diurnal corrections on data.
    
    Arguments:
    - `df` : dataframe to correct
    - `mags_to_cor` : list of string of magnetometers to be corrected
    - `diurnal` : (optional) apply diunal correction (True or False)
    - `igrf` : (optional) apply IGRF correction (True or False)
    
    Returns:
    - `df_cor` : corrected dataframe
    '''
    mag_measurements = np.array(mags_to_cor)
    df_cor = df.copy()
    
    # Diurnal cor
    if diurnal == True:
        df_cor[mag_measurements] = df_cor[mag_measurements]-np.reshape(df_cor['DIURNAL'].values,[-1,1])
    
    # IGRF cor
    lat  = df_cor['LAT']
    lon  = df_cor['LONG']
    h    = df_cor['BARO']*1e-3
    date = datetime(2020, 6, 29) # Date on which the flights were made
    Be, Bn, Bu = magnav.igrf(lon,lat,h,date)

    if igrf == True:
        igrf_total = np.sqrt(Be**2+Bn**2+Bu**2)
        # Reshape to match the number of magnetometer columns being corrected
        igrf_correction = np.tile(igrf_total.reshape(-1, 1), (1, len(mag_measurements)))
        df_cor[mag_measurements] = df_cor[mag_measurements] - igrf_correction

    return df_cor


def create_synthetic_magnav_data():
    """
    Create synthetic magnetometer data for testing ContiFormer.
    """
    print("Creating synthetic MagNav data for testing...")
    
    n_samples = 10000
    t = np.linspace(0, 100, n_samples)
    
    # Synthetic magnetometer readings
    mag1 = 50000 + 100 * np.sin(0.1 * t) + 50 * np.random.randn(n_samples)
    mag2 = 50100 + 80 * np.cos(0.08 * t) + 40 * np.random.randn(n_samples)
    
    # Synthetic flight dynamics
    velocity_n = 50 + 10 * np.sin(0.05 * t) + np.random.randn(n_samples)
    velocity_v = 5 + 2 * np.sin(0.03 * t) + np.random.randn(n_samples)
    velocity_w = np.random.randn(n_samples)
    
    pitch = 2 * np.sin(0.02 * t) + 0.5 * np.random.randn(n_samples)
    roll = 1.5 * np.cos(0.025 * t) + 0.3 * np.random.randn(n_samples)
    azimuth = 180 + 20 * np.sin(0.01 * t) + np.random.randn(n_samples)
    
    # Create DataFrame
    data = {
        'TL_comp_mag4_cl': mag1,
        'TL_comp_mag5_cl': mag2,
        'V_BAT1': 24 + 0.5 * np.random.randn(n_samples),
        'V_BAT2': 24 + 0.5 * np.random.randn(n_samples),
        'INS_VEL_N': velocity_n,
        'INS_VEL_V': velocity_v,
        'INS_VEL_W': velocity_w,
        'CUR_IHTR': 2 + 0.2 * np.random.randn(n_samples),
        'CUR_FLAP': 1 + 0.1 * np.random.randn(n_samples),
        'CUR_ACLo': 0.5 + 0.05 * np.random.randn(n_samples),
        'CUR_TANK': 0.8 + 0.08 * np.random.randn(n_samples),
        'PITCH': pitch,
        'ROLL': roll,
        'AZIMUTH': azimuth,
        'BARO': 1000 + 100 * np.random.randn(n_samples),
        'LINE': np.concatenate([
            np.full(2000, 1001),
            np.full(2000, 1002), 
            np.full(2000, 1003),
            np.full(2000, 1004),
            np.full(2000, 1005)
        ]),
        'IGRFMAG1': mag1 + 10 * np.random.randn(n_samples)
    }
    
    return pd.DataFrame(data)


#------------#
#----Main----#
#------------#


if __name__ == "__main__":
    
    # Start timer
    start_time = time.time()
    
    # set seed for reproducibility
    torch.manual_seed(27)
    random.seed(27)
    np.random.seed(27)
    
    #----User arguments----#
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-d","--device", type=str, required=False, default='cpu', help="Which GPU to use (cuda or cpu), default='cpu'. Ex : --device 'cuda' ", metavar=""
    )
    parser.add_argument(
        "-e","--epochs", type=int, required=False, default=35, help="Number of epochs to train the model, default=35. Ex : --epochs 200", metavar=""
    )
    parser.add_argument(
        "-b","--batch", type=int, required=False, default=256, help="Batch size for training, default=256. Ex : --batch 64", metavar=""
    )
    parser.add_argument(
        "-sq","--seq", type=int, required=False, default=20, help="Length sequence of data, default=20. Ex : --seq 15", metavar=""
    )
    parser.add_argument(
        "--shut", action="store_true", required=False, help="Shutdown pc after training is done."
    )
    parser.add_argument(
        "-sc", "--scaling", type=int, required=False, default=1, help="Data scaling, 1 for standardization, 2 for MinMax scaling, 0 for no scaling, default=1. Ex : --scaling 1", metavar=''
    )
    parser.add_argument(
        "-cor", "--corrections", type=int, required=False, default=3, help="Data correction, 0 for no corrections, 1 for IGRF correction, 2 for diurnal correction, 3 for IGRF+diurnal correction. Ex : --corrections 3", metavar=''
    )
    parser.add_argument(
        "-tl", "--tolleslawson", type=int, required=False, default=1, help="Apply Tolles-Lawson compensation to data, 0 for no compensation, 1 for compensation. Ex : --tolleslawson 1", metavar=''
    )
    parser.add_argument(
        "-tr", "--truth", type=str, required=False, default='IGRFMAG1', help="Name of the variable corresponding to the truth for training the model. Ex : --truth 'IGRFMAG1'", metavar=''
    )
    parser.add_argument(
        "-ml", "--model", type=str, required=False, default='ContiFormer', help="Name of the model to use. Available models : 'MLP', 'CNN', 'ResNet18', 'LSTM', 'GRU', 'ContiFormer', 'ContiFormerSimple', 'DivFreeContiFormer', 'StandardContiFormer', 'SE3ContiFormer'. Ex : --model 'ContiFormer'", metavar=''
    )
    parser.add_argument(
        "-wd", "--weight_decay", type=float, required=False, default=0.001, help="Adam weight decay value. Ex : --weight_decay 0.00001", metavar=''
    )
    parser.add_argument(
        "--synthetic", action="store_true", required=False, help="Use synthetic data for testing when real flight data is not available."
    )
    
    # added right now for MLP 
    
    parser.add_argument(
        '--subset', type=int, default=None, help='Number of training samples to use (debug mode)'
    )
    parser.add_argument(
        '--testsubset', type=int, default=None, help='Limit test set size for faster evaluation'
    )


    
    args = parser.parse_args()
    
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch
    DEVICE     = args.device
    SEQ_LEN    = args.seq
    SCALING    = args.scaling
    COR        = args.corrections
    TL         = args.tolleslawson
    TRUTH      = args.truth
    MODEL      = args.model
    WEIGHT_DECAY = args.weight_decay
    
    # Check device availability and set appropriate device
    if DEVICE == 'cuda' and torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f'\nCurrently training on {torch.cuda.get_device_name(DEVICE)}')
    else:
        DEVICE = 'cpu'
        print('\nCurrently training on cpu.')

    #----Import data----#
    
    if args.synthetic:
        # Use synthetic data for testing
        df = create_synthetic_magnav_data()
        flights_num = [1]  # Single synthetic flight
        flights = {1: df}
        print(f'Synthetic data created. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    else:
        # Try to import real flight data (memory optimized - load fewer flights)
        try:
            flights = {}
            flights_num = [2,3]  # Reduced to 2 flights to save memory
            
            for n in flights_num:
                df = pd.read_hdf("./data/processed/Flt_data.h5", key=f'Flt100{n}')
                flights[n] = df
            
            print(f'Data import done (2 flights for memory efficiency). Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
        except (FileNotFoundError, KeyError) as e:
            print(f"Real flight data issue ({e}). Using synthetic data instead.")
            df = create_synthetic_magnav_data()
            flights_num = [1]
            flights = {1: df}
            print(f'Synthetic data created. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Slecting train/test lines----#
    
    if args.synthetic:
        # For synthetic data, simple split
        train_lines = [[1001, 1002, 1003]]
        test_lines = [[1004, 1005]]
    else:
        # Memory-optimized flight-based split (using flights 2,3 only)
        train_lines = [flights[2].LINE.unique().tolist()]
        test_lines  = [flights[3].LINE.unique().tolist()]

    #----Apply Tolles-Lawson (Skip for synthetic data)----#
    
    if not args.synthetic and TL == 1:
        # Get cloverleaf pattern data
        mask = (flights[2].LINE == 1002.20)
        tl_pattern = flights[2][mask]

        # filter parameters
        fs      = 10.0
        lowcut  = 0.1
        highcut = 0.9
        filt    = ['Butterworth',4]
        
        ridge = 0.025
        for n in tqdm(flights_num):

            # A matrix of Tolles-Lawson
            A = magnav.create_TL_A(flights[n]['FLUXB_X'],flights[n]['FLUXB_Y'],flights[n]['FLUXB_Z'])

            # Tolles Lawson coefficients computation
            TL_coef_2 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG2'],
                                          lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
            TL_coef_3 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG3'],
                                          lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
            TL_coef_4 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG4'],
                                          lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
            TL_coef_5 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG5'],
                                          lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)

            # Magnetometers correction
            flights[n]['TL_comp_mag2_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG2'].tolist(),(-1,1)), TL_coef_2, A)
            flights[n]['TL_comp_mag3_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG3'].tolist(),(-1,1)), TL_coef_3, A)
            flights[n]['TL_comp_mag4_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG4'].tolist(),(-1,1)), TL_coef_4, A)
            flights[n]['TL_comp_mag5_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG5'].tolist(),(-1,1)), TL_coef_5, A)

        print(f'Tolles-Lawson correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    else:
        print(f'Tolles-Lawson correction skipped. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')

    #----Apply IGRF and diurnal corrections (Skip for synthetic data)----#

    flights_cor = {}

    if TL == 1 and not args.synthetic:
        mags_to_cor = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl']
    else:
        mags_to_cor = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl']  # These exist in synthetic data
    
    if COR == 0 or args.synthetic:
        flights_cor = flights.copy()
        del flights
        print(f'No correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    elif COR == 1:
        for n in tqdm(flights_num):
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=False, igrf=True)
        del flights
        print(f'IGRF correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    elif COR == 2: 
        for n in tqdm(flights_num):
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=False)
        del flights
        print(f'Diurnal correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    elif COR == 3:
        for n in tqdm(flights_num):
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=True)
        del flights
        print(f'IGRF+Diurnal correction done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Select features----#
    
    # Always keep the 'LINE' feature in the feature list so that the MagNavDataset function can split the flight data
    features = [mags_to_cor[0],mags_to_cor[1],'V_BAT1','V_BAT2',
                    'INS_VEL_N','INS_VEL_V','INS_VEL_W','CUR_IHTR','CUR_FLAP','CUR_ACLo','CUR_TANK','PITCH','ROLL','AZIMUTH','BARO','LINE',TRUTH]
    
    dataset = {}
    
    for n in flights_num:
        dataset[n] = flights_cor[n][features]
    
    del flights_cor
    print(f'Feature selection done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Data scaling----#
    
    if SCALING == 0:
        df = pd.DataFrame()
        for flight in flights_num:
            df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
        
        # Save scaling parameters
        scaling = {}
        for n in range(len(test_lines)):
            scaling[n] = ['None']
        
    elif SCALING == 2:
        # Save scaling parameters
        bound = [-1,1]
        scaling = {}
        df = pd.DataFrame()
        for flight in flights_num:
            df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
        for n in range(len(test_lines)):
            mask = pd.Series(dtype=bool)
            for line in test_lines[n]:
                temp_mask = (df.LINE == line)
                mask = temp_mask|mask
            scaling[n] = ['minmax', bound[0], bound[1], df.loc[mask,TRUTH].min(), df.loc[mask,TRUTH].max()]
        del mask, temp_mask, df
        
        # Apply Min-Max sacling to the dataset
        for n in tqdm(flights_num):
            dataset[n] = MinMax_scaling(dataset[n], bound=bound)
        df = pd.DataFrame()
        for flight in flights_num:
            df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)

    elif SCALING == 1:
        # Save scaling parameters
        scaling = {}
        df = pd.DataFrame()
        for flight in flights_num:
            df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
        for n in range(len(test_lines)):
            mask = pd.Series(dtype=bool)
            for line in test_lines[n]:
                temp_mask = (df.LINE == line)
                mask = temp_mask|mask
            scaling[n] = ['std', df.loc[mask,TRUTH].mean(), df.loc[mask,TRUTH].std()]
        del mask, temp_mask, df
        
        # Apply Standard scaling to the dataset
        for n in tqdm(flights_num):
            dataset[n] = Standard_scaling(dataset[n])
        df = pd.DataFrame()
        for flight in flights_num:
            df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
    
    del dataset
    print(f'Data scaling done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Training----#
    
    train_loss_history = []
    test_loss_history = []
    RMSE_history = []
    
    # Cross validation with selected folds 
    for fold in range(len(train_lines)):
        
        print('\n--------------------')
        print(f'Fold number {fold}')
        print('--------------------\n')
        
        # Split to train and test
        # Split to train and test
        train = MagNavDataset(df, seq_len=SEQ_LEN, split='train', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)
        # Limit test data to subset (for debugging evaluation time)


        test  = MagNavDataset(df, seq_len=SEQ_LEN, split='test', train_lines=train_lines[fold], test_lines=test_lines[fold], truth=TRUTH)

        if args.testsubset is not None:
            test = torch.utils.data.Subset(test, range(min(args.testsubset, len(test))))
            print(f"Using test subset: {len(test)} samples")
            
        # Limit training data to subset (for debugging speed)
        if args.subset is not None:
            train = torch.utils.data.Subset(train, range(min(args.subset, len(train))))
            print(f"Using training subset: {len(train)} samples")


        # Dataloaders
        train_loader  = DataLoader(train,
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=0,
                               pin_memory=False)

        test_loader    = DataLoader(test,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)

        # Model
        if MODEL == 'MLP':
            model = MLP(SEQ_LEN,len(features)-2).to(DEVICE)
            model.name = model.__class__.__name__
        elif MODEL == 'CNN':
            model = CNN(SEQ_LEN,len(features)-2).to(DEVICE)
            model.name = model.__class__.__name__
        elif MODEL == 'DFMLP':
            model = DivergenceFreeMLP(seq_length=SEQ_LEN, n_features=len(features) - 2, output_dim=1).to(DEVICE)
            model.name = model.__class__.__name__
        elif MODEL == 'ResNet18':
            model = ResNet18().to(DEVICE)
            model.name = model.__class__.__name__
        elif MODEL == 'LSTM':
            num_LSTM    = 2
            hidden_size = [32,32]
            num_layers  = [3,1]
            num_linear  = 2
            num_neurons = [16,4]
            model = LSTM(SEQ_LEN, hidden_size, num_layers, num_LSTM, num_linear, num_neurons, DEVICE).to(DEVICE)
            model.name = model.__class__.__name__
        elif MODEL == 'GRU':
            model = GRU(SEQ_LEN,len(features)-2,32).to(DEVICE)
            model.name = model.__class__.__name__
        elif MODEL == 'ContiFormer':
            # Use ODE-based ContiFormer with proper parameters
            model = ContiFormer(
                input_size=len(features)-2,
                d_model=256,
                d_inner=1024,
                n_layers=4,
                n_head=4,
                d_k=64,
                d_v=64,
                dropout=0.1,
                max_length=SEQ_LEN,
                # ODE-specific parameters
                actfn_ode="softplus",
                layer_type_ode="concat",
                zero_init_ode=True,
                atol_ode=1e-3,
                rtol_ode=1e-3,
                method_ode="rk4",
                linear_type_ode="inside",
                regularize=False,
                approximate_method="last",
                nlinspace=3,
                interpolate_ode="linear",
                itol_ode=1e-2,
                divergence_free=True,  # Enable divergence-free vector fields
                add_pe=False,
                normalize_before=False
            ).to(DEVICE)
            model.name = MODEL
            print(f"\nCreated ODE-based ContiFormer")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        elif MODEL == 'SE3ContiFormer':
            model = SE3ContiFormerMagNav(SEQ_LEN, len(features)-2).to(DEVICE)
            print(f"\nCreated SE3ContiFormer (SE(3) equivariant + divergence-free)")
            print(f"Model info: {model.get_model_info()}")
            # Check SE(3) equivariance and divergence-free properties
            model.check_se3_equivariance(device=DEVICE)
            model.check_divergence_free_property(device=DEVICE)

        # Loss function
        criterion = torch.nn.MSELoss()

        # Training
        train_loss, test_loss, Best_RMSE, Best_model = make_training(model, EPOCHS, train_loader, test_loader, scaling[fold])
        
        # Save results from training
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        RMSE_history.append(Best_RMSE)
        
        if fold == 0:
            folder_path = f'models/New_{Best_model.name}_{scaling[0][0]}_TL{TL}_COR{COR}_{TRUTH}'
            os.makedirs(folder_path, exist_ok=True)
        torch.save(Best_model,folder_path+f'/{Best_model.name}_fold{fold}.pt')
    
    # Compute pre-processing+training time
    end_time = time.time()-start_time
    
    # Compute global perf over all folds
    perf_folds = sum(RMSE_history)/len(train_lines)
    
    # Print perf of training
    print('\n-------------------------')
    print('Performance for all folds')
    print('-------------------------')
    for n in range(len(test_lines)):
        print(f'Fold {n} | RMSE = {RMSE_history[n]:.2f} nT')
    print(f'Total  | RMSE = {perf_folds:.2f} nT')
    
    # Show performance graphs
    for fold in range(len(test_lines)):
        fig, ax = plt.subplots(figsize=[10,4])
        ax.plot(train_loss_history[fold], label='Train loss')
        ax.plot(test_loss_history[fold], label='Test loss')
        ax.set_title(f'Loss for fold {fold}')
        ax.set_xlabel('Epoch')
        plt.legend()
        plt.savefig(folder_path+f'/losses_fold{fold}')
    
    # Save parameters
    params = pd.DataFrame(columns=['seq_len','epochs','batch_size','training_time','model','divergence_free'])
    params.loc[0,'seq_len'] = SEQ_LEN
    params.loc[0,'epochs'] = EPOCHS
    params.loc[0,'batch_size'] = BATCH_SIZE
    params.loc[0,'training_time'] = end_time
    params.loc[0,'model'] = MODEL
    params.loc[0,'divergence_free'] = MODEL in ['ContiFormer', 'DivFreeContiFormer', 'SE3ContiFormer']
    params.to_csv(folder_path+f'/parameters.csv', index=False)
    
    # Empty GPU ram if using CUDA
    if torch.cuda.is_available() and DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    # Shutdown computer
    if args.shut == True:
        os.system("shutdown") 