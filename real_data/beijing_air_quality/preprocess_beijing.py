#!/usr/bin/env python3
"""
Preprocess Beijing Multi-Site Air Quality Dataset into MATLAB .mat format.
Creates:
  - 12 natural distributed partitions (one per station: M = 12)
  - Autoregressive and lagged meteorological features (d = 19)
  - Rigorous Temporal Split:
      * Train: 2013-03-01 to 2016-02-29 (3 full years)
      * Test:  2016-03-01 to 2017-02-28 (1 full future year, out-of-time validation)
  - Saves to real_data/beijing_air_quality/beijing_air_quality_processed.mat
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import savemat

# Wind direction mapping (16 compass points to radians)
COMPASS_DIRS = {
    'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
    'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
    'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
    'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5
}

def clean_and_build_features(df):
    # Sort strictly by time
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Linear interpolation for continuous sensor values with small gaps
    numeric_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    
    # Wind direction mapping
    df['wd'] = df['wd'].ffill().bfill()
    wd_deg = df['wd'].map(COMPASS_DIRS).fillna(0.0)
    wd_rad = np.radians(wd_deg)
    df['wd_sin'] = np.sin(wd_rad)
    df['wd_cos'] = np.cos(wd_rad)
    
    # Cyclical hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Lagged features (avoiding data leakage: all predictors for y_t are known at t-1 or earlier, except instantaneous meteorology)
    df['PM2.5_lag1'] = df['PM2.5'].shift(1)
    df['PM2.5_lag2'] = df['PM2.5'].shift(2)
    df['PM2.5_lag24'] = df['PM2.5'].shift(24)
    
    df['PM10_lag1'] = df['PM10'].shift(1)
    df['SO2_lag1'] = df['SO2'].shift(1)
    df['NO2_lag1'] = df['NO2'].shift(1)
    df['CO_lag1'] = df['CO'].shift(1)
    df['O3_lag1'] = df['O3'].shift(1)
    
    # Drop initial 24 rows with NaN lags
    df = df.iloc[24:].reset_index(drop=True)
    
    feature_cols = [
        'PM2.5_lag1', 'PM2.5_lag2', 'PM2.5_lag24',
        'PM10_lag1', 'SO2_lag1', 'NO2_lag1', 'CO_lag1', 'O3_lag1',
        'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
        'wd_sin', 'wd_cos',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    
    X = df[feature_cols].values
    y = df['PM2.5'].values
    dates = df['datetime'].values
    
    return X, y, dates, feature_cols

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'PRSA_Data_20130301-20170228')
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
    print(f"Found {len(csv_files)} station CSV files.")
    
    stations = []
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    feature_names = []
    
    split_date = pd.Timestamp('2016-03-01 00:00:00')
    
    for f in csv_files:
        df = pd.read_csv(f)
        station_name = df['station'].iloc[0]
        stations.append(station_name)
        
        X, y, dates, f_cols = clean_and_build_features(df)
        feature_names = f_cols
        
        train_mask = dates < split_date
        test_mask = dates >= split_date
        
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        
        X_train_list.append(X_tr)
        y_train_list.append(y_tr.reshape(-1, 1))
        X_test_list.append(X_te)
        y_test_list.append(y_te.reshape(-1, 1))
        
        print(f"Station: {station_name:<15} | Train rows: {len(X_tr)} | Test rows: {len(X_te)}")
        
    M = len(stations)
    d = len(feature_names)
    
    # Normalization (Standardize using global train statistics to preserve true distributed scale)
    all_X_tr = np.vstack(X_train_list)
    mu_X = np.mean(all_X_tr, axis=0)
    std_X = np.std(all_X_tr, axis=0)
    std_X[std_X == 0] = 1.0
    
    all_y_tr = np.vstack(y_train_list)
    mu_y = np.mean(all_y_tr)
    std_y = np.std(all_y_tr)
    
    X_train_norm_cells = np.empty((M,), dtype=object)
    y_train_cells = np.empty((M,), dtype=object)
    X_test_norm_cells = np.empty((M,), dtype=object)
    y_test_cells = np.empty((M,), dtype=object)
    
    for m in range(M):
        X_train_norm_cells[m] = (X_train_list[m] - mu_X) / std_X
        y_train_cells[m] = y_train_list[m]
        X_test_norm_cells[m] = (X_test_list[m] - mu_X) / std_X
        y_test_cells[m] = y_test_list[m]
        
    X_train_pooled = np.vstack(X_train_norm_cells)
    y_train_pooled = np.vstack(y_train_cells)
    X_test_pooled = np.vstack(X_test_norm_cells)
    y_test_pooled = np.vstack(y_test_cells)
    
    out_mat_path = os.path.join(script_dir, 'beijing_air_quality_processed.mat')
    mat_dict = {
        'X_train_cells': X_train_norm_cells,
        'Y_train_cells': y_train_cells,
        'X_test_cells': X_test_norm_cells,
        'Y_test_cells': y_test_cells,
        'X_train_pooled': X_train_pooled,
        'Y_train_pooled': y_train_pooled,
        'X_test_pooled': X_test_pooled,
        'Y_test_pooled': y_test_pooled,
        'stations': np.array(stations, dtype=object),
        'feature_names': np.array(feature_names, dtype=object),
        'M': M,
        'd': d,
        'mu_X': mu_X,
        'std_X': std_X,
        'mu_y': mu_y,
        'std_y': std_y
    }
    
    savemat(out_mat_path, mat_dict, do_compression=True)
    print(f"\n==> Successfully saved processed Beijing dataset to:\n    {out_mat_path}")
    print(f"    Total Train samples: {len(X_train_pooled)} (12 stations)")
    print(f"    Total Test samples:  {len(X_test_pooled)} (12 stations)")
    print(f"    Features (d):       {d}")
    print(f"    Stations (M):       {M}")

if __name__ == '__main__':
    main()
