import pandas as pd
import wfdb
import neurokit2 as nk
from globals import INDEX_TYPE


pd.set_option('display.max_columns', None)
mb_artm_directory = './databases/mit-bih-arrhythmia-database-1.0.0/'
mb_nsrdb_directory = './databases/mit-bih-normal-sinus-rhythm-database-1.0.0/'
dataframes_directory = './Results/dataframes/'
figures_directory = './Results/figures/'


def load_record(record: int, files_directory=mb_artm_directory, **kwargs):
    record_path = f'{files_directory}{record}'
    record = wfdb.rdrecord(record_path, **kwargs)
    ann = wfdb.rdann(record_path, 'atr', **kwargs)
    return record, ann


def find_peaks(ecg, sampling_rate, method):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method= method)
    return info["ECG_R_Peaks"]


def rolling_idxmax(series : pd.Series, window_size : int, closed : str = 'both', center : bool = True, **kwargs)->pd.Series:
    """
    Calculate the index of the maximum value in a rolling window.
    """
    # TODO: implement it using vectorized operations
    return series.rolling(window_size, closed = closed, center = center, **kwargs).apply(lambda x: x.idxmax()).dropna(ignore_index = False).astype(INDEX_TYPE).reset_index().rename(columns = {0: 'local_max'})


def correct_peaks(ecg : pd.Series, ann_beat_indexes, fs, window_size = 0.28 + 0.120, **kwargs):
    """
    Correct peaks by finding local maxima around the annotated peaks.
    Some methods may fail to detect the correct peaks in the begining of the signal. 
    """

    # TODO: make it more efficient by finding maxima only around ann_beat_indexes
    ecg = pd.Series(ecg)

    df_beats = pd.DataFrame({'index' : ann_beat_indexes})

    # Find local maxima
    rolling_idxmax_df = rolling_idxmax(ecg, int(fs * window_size), **kwargs)
    
    df_beats =  df_beats.merge(rolling_idxmax_df, how = 'left', on = 'index')

    return df_beats