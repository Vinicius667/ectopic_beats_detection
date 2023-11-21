from typing import Any, Union

import neurokit2 as nk  # type: ignore
import numpy as np
import pandas as pd
import wfdb  # type: ignore
from numpy import typing as npt

from globals import INDEX_TYPE

mb_artm_directory = './databases/mit-bih-arrhythmia-database-1.0.0/'
mb_nsrdb_directory = './databases/mit-bih-normal-sinus-rhythm-database-1.0.0/'
dataframes_directory = './Results/dataframes/'
figures_directory = './Results/figures/'


def load_record(record_num: int, files_directory: str = mb_artm_directory, **kwargs: Any):
    record_path = f'{files_directory}{record_num}'
    record = wfdb.rdrecord(record_path, **kwargs)
    ann = wfdb.rdann(record_path, 'atr', **kwargs)
    return record, ann


def find_peaks(ecg: Union[npt.NDArray[np.float32], pd.Series], sampling_rate: int, method: str) -> npt.NDArray[np.int32]:
    signal, info = nk.ecg_peaks(
        ecg, sampling_rate=sampling_rate, method=method)
    return info["ECG_R_Peaks"]  # type: ignore


def correct_peaks(ecg: Union[npt.NDArray[np.float32], pd.Series], ann_beat_indexes: Union[npt.NDArray[np.int32], pd.Series], fs: int, window_size: float = 0.28 + 0.120, **kwargs: Any) -> pd.DataFrame:
    """
    Correct peaks by finding local maxima around the annotated peaks.
    Some methods may fail to detect the correct peaks in the begining of the signal.
    """

    # TODO: make it more efficient by finding maxima only around ann_beat_indexes
    ecg = pd.Series(ecg)

    df_beats = pd.DataFrame(data={'index': ann_beat_indexes})

    # Find local maxima
    rolling_idxmax_df = rolling_idxmax(
        series=ecg, window_size=int(fs * window_size), **kwargs)

    df_beats = df_beats.merge(rolling_idxmax_df, how='left', on='index')

    return df_beats


def rolling_idxmax(series: pd.Series, window_size: int, closed: str = 'both', center: bool = True, **kwargs: Any) -> pd.Series:
    """
    Calculate the index of the maximum value in a rolling window.
    """
    # TODO: implement it using vectorized operations

    series_rolling_idxmax: pd.Series = series.rolling(window_size,
                                                      closed=closed,  # type: ignore
                                                      center=center, **kwargs).apply(
        lambda x: x.idxmax()).dropna(ignore_index=False).astype(INDEX_TYPE).reset_index().rename(columns={0: 'local_max'})

    return series_rolling_idxmax
