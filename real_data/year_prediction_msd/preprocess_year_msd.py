#!/usr/bin/env python3
"""
Preprocess YearPredictionMSD dataset into MATLAB .mat format.
UCI ID: 203 (https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)

Dataset specifications:
  - Total samples: 515,345
  - Features (d = 90): 12 timbre average + 78 timbre covariance
  - Target (Y): Release year of audio track [1922, 2011]
  - Official Train split: First 463,715 samples
  - Official Test split:  Last 51,630 samples (avoids producer/artist effect)
  - Number of distributed machines: M = 32

Outputs:
  - year_msd_sub10.mat (1/10 subset: N_train = 46,372, N_test = 5,163 for quick testing)
  - year_msd_processed.mat (Full dataset: N_train = 463,715, N_test = 51,630)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.io import savemat

def preprocess(txt_path, out_dir, M=32, save_full=True, save_sub10=True):
    print(f"--> Reading {txt_path} ...")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")
    
    # Read CSV (YearPredictionMSD is comma-delimited)
    df = pd.read_csv(txt_path, header=None, dtype=np.float64)
    print(f"    Loaded raw data with shape: {df.shape}")
    
    total_samples, total_cols = df.shape
    assert total_cols == 91, f"Expected 91 columns (1 target + 90 features), got {total_cols}"
    assert total_samples == 515345, f"Expected 515,345 rows, got {total_samples}"
    
    # Target is column 0, features are columns 1..90
    Y_all = df.iloc[:, 0].values.reshape(-1, 1)
    X_all = df.iloc[:, 1:].values
    d = 90
    
    N_train_full = 463715
    N_test_full = 51630
    
    # -------------------------------------------------------------
    # 1. Generate 1/10 Subsample for Verification/Smoke Testing
    # -------------------------------------------------------------
    if save_sub10:
        print(f"\n--> Preparing 1/10 Subsample (M = {M}) ...")
        N_train_sub = 46372
        N_test_sub = 5163
        
        # Take first 1/10 from train and first 1/10 from test
        X_tr_sub = X_all[:N_train_sub, :].copy()
        Y_tr_sub = Y_all[:N_train_sub, :].copy()
        
        X_te_sub = X_all[N_train_full : N_train_full + N_test_sub, :].copy()
        Y_te_sub = Y_all[N_train_full : N_train_full + N_test_sub, :].copy()
        
        # Z-score standardization on training set statistics
        mu_sub = np.mean(X_tr_sub, axis=0)
        std_sub = np.std(X_tr_sub, axis=0)
        std_sub[std_sub < 1e-8] = 1.0
        
        X_tr_sub = (X_tr_sub - mu_sub) / std_sub
        X_te_sub = (X_te_sub - mu_sub) / std_sub
        
        # Partition train among M machines
        split_indices_sub = np.array_split(np.arange(N_train_sub), M)
        X_tr_cells_sub = np.empty((1, M), dtype=object)
        Y_tr_cells_sub = np.empty((1, M), dtype=object)
        
        for m in range(M):
            idx = split_indices_sub[m]
            X_tr_cells_sub[0, m] = X_tr_sub[idx, :]
            Y_tr_cells_sub[0, m] = Y_tr_sub[idx, :]
            
        mat_sub10_path = os.path.join(out_dir, 'year_msd_sub10.mat')
        sub10_dict = {
            'M': M,
            'd': d,
            'N_train': N_train_sub,
            'N_test': N_test_sub,
            'X_train_cells': X_tr_cells_sub,
            'Y_train_cells': Y_tr_cells_sub,
            'X_train_pooled': X_tr_sub,
            'Y_train_pooled': Y_tr_sub,
            'X_test_pooled': X_te_sub,
            'Y_test_pooled': Y_te_sub,
            'mu_X': mu_sub,
            'std_X': std_sub
        }
        print(f"    Saving 1/10 dataset to {mat_sub10_path} ...")
        savemat(mat_sub10_path, sub10_dict, do_compression=True)
        print(f"    Sub10 dataset created successfully. Train N={N_train_sub} ({N_train_sub//M} per machine), Test N={N_test_sub}")

    # -------------------------------------------------------------
    # 2. Generate Full Dataset
    # -------------------------------------------------------------
    if save_full:
        print(f"\n--> Preparing Full Dataset (M = {M}) ...")
        X_tr = X_all[:N_train_full, :].copy()
        Y_tr = Y_all[:N_train_full, :].copy()
        
        X_te = X_all[N_train_full:, :].copy()
        Y_te = Y_all[N_train_full:, :].copy()
        
        # Z-score standardization on training set statistics
        mu_full = np.mean(X_tr, axis=0)
        std_full = np.std(X_tr, axis=0)
        std_full[std_full < 1e-8] = 1.0
        
        X_tr = (X_tr - mu_full) / std_full
        X_te = (X_te - mu_full) / std_full
        
        # Partition train among M machines
        split_indices = np.array_split(np.arange(N_train_full), M)
        X_tr_cells = np.empty((1, M), dtype=object)
        Y_tr_cells = np.empty((1, M), dtype=object)
        
        for m in range(M):
            idx = split_indices[m]
            X_tr_cells[0, m] = X_tr[idx, :]
            Y_tr_cells[0, m] = Y_tr[idx, :]
            
        mat_full_path = os.path.join(out_dir, 'year_msd_processed.mat')
        full_dict = {
            'M': M,
            'd': d,
            'N_train': N_train_full,
            'N_test': N_test_full,
            'X_train_cells': X_tr_cells,
            'Y_train_cells': Y_tr_cells,
            'X_train_pooled': X_tr,
            'Y_train_pooled': Y_tr,
            'X_test_pooled': X_te,
            'Y_test_pooled': Y_te,
            'mu_X': mu_full,
            'std_X': std_full
        }
        print(f"    Saving Full dataset to {mat_full_path} ...")
        savemat(mat_full_path, full_dict, do_compression=True)
        print(f"    Full dataset created successfully. Train N={N_train_full} ({N_train_full//M} per machine), Test N={N_test_full}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_file = os.path.join(script_dir, 'YearPredictionMSD.txt')
    preprocess(txt_file, script_dir, M=32, save_full=True, save_sub10=True)
