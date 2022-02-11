import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

def binance_resample(df, period):
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number of trades']
    f = df[cols].ffill()
    f.columns = [x.lower() for x in f.columns]
    f.rename(columns={'number of trades': 'ntrades'}, inplace=True)
    f = f.resample(period).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'ntrades': 'sum'})
    f['volume'] = f['volume'].clip(f['volume'].quantile(0.05), f['volume'].quantile(0.995))
    return f.rename_axis('time')

def yf_resample(df, period):
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    f = df[cols].ffill()
    f.columns = [x.lower() for x in f.columns]
    f = f.resample(period).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    f['volume'] = f['volume'].clip(0, f['volume'].quantile(0.995))
    return f

def consecutive_true(x):
    return x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)

def annotate_direction(df, rec):
    # Peak/trough continuous annotation
    df['direction'] = 0
    previous_ex = 0
    previous_dir = 0
    for en, ex, direction in zip(rec.entry_idx, rec.exit_idx, np.sign(rec.pnl)):
        df.loc[previous_ex:en, 'direction'] = -previous_dir
        df.loc[en:ex, 'direction'] = direction
        previous_ex = ex
        previous_dir = direction

    df.loc[previous_ex:, 'direction'] = -direction
    df = df.query('direction != 0')
    return df

def pt_to_strength(pt, sigma=1):
    '''
    Converts peak/trough signals to strength by Gaussian filtering
    '''
    return pd.Series(gaussian_filter1d(pt.astype(float), sigma, mode='reflect'), index=pt.index)
