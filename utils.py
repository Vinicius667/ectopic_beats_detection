from typing import Any, Dict, List, Tuple, Union

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from numpy import typing as npt
from plotly import graph_objects as go

from globals import (ANN_TYPE, BOOL_TYPE, ECG_TYPE, INDEX_TYPE, LIST_BEATS_1,
                     mb_artm_directory)


def load_record(record_num: int, files_directory: str = mb_artm_directory, **kwargs: Any) -> Tuple[wfdb.Record, wfdb.Annotation]:
    """Load a record from the specified directory.

    Args:
        record_num (int): record number
        files_directory (str, optional): directory where the record is stored. Defaults to mb_artm_directory.

    Returns:
        Tuple[wfdb.Record, wfdb.Annotation]: record and annotations
    """
    record_path = f'{files_directory}{record_num}'
    record = wfdb.rdrecord(record_path, **kwargs)
    ann = wfdb.rdann(record_path, 'atr', **kwargs)
    return record, ann


def find_peaks(ecg: Union[npt.NDArray[np.float32], pd.Series], sampling_rate: int, method: str) -> npt.NDArray[np.int32]:
    signal, info = nk.ecg_peaks(
        ecg, sampling_rate=sampling_rate, method=method)
    return info["ECG_R_Peaks"]  # type: ignore


def correct_peaks(ecg: Union[npt.NDArray[np.float32], pd.Series], ann_beat_indexes: Union[npt.NDArray[np.int32], pd.Series], fs: int, window_size: float = 0.2, **kwargs: Any) -> pd.DataFrame:
    """Correct peaks by finding local maxima around the annotated peaks. The correction is done by finding the local maxima in a window around the annotated peaks.

    Args:
        ecg (Union[npt.NDArray[np.float32], pd.Series]): ecg signal
        ann_beat_indexes (Union[npt.NDArray[np.int32], pd.Series]): annotated beat indexes
        fs (int): sampling rate (Hz)
        window_size (float, optional): window size for finding local maxima. Defaults to 0.2 (second)
        **kwargs: other arguments to be passed to pd.Series.rolling function

    Returns:
        pd.DataFrame: dataframe with corrected peaks with the following columns:
            index: all indexes of the annotated beats
            local_max: local maxima around the annotated beats
    """
    # TODO: make it more efficient by finding maxima only around ann_beat_indexes
    if not isinstance(ecg, pd.Series):
        ecg = pd.Series(ecg)

    df_beats = pd.DataFrame(data={'index': ann_beat_indexes})

    # Find local maxima
    rolling_idxmax_df = rolling_idxmax(
        series=ecg, window_size=int(fs * window_size), **kwargs)

    df_beats = df_beats.merge(rolling_idxmax_df, how='left', on='index')

    return df_beats


def rolling_idxmax(series: pd.Series, window_size: int, closed: str = 'both', center: bool = True, **kwargs: Any) -> pd.DataFrame:
    """Calculate the index of the maximum value in a rolling window.

    Args:
        series (pd.Series):
        window_size (int): _description_
        closed (str, optional): _description_. Defaults to 'both'.
        center (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    # TODO: implement it using vectorized operations

    series_rolling_idxmax = series.rolling(window_size,
                                           closed=closed,  # type: ignore
                                           center=center, **kwargs).apply(
        lambda x: x.idxmax()).dropna(ignore_index=False).astype(INDEX_TYPE).reset_index().rename(columns={0: 'local_max'})

    return series_rolling_idxmax


def create_compare_df(df_beats: pd.DataFrame, dict_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    methods = list(dict_results.keys())
    series_concat_index = pd.concat([df_beats.cor_peak_index] + [
                                    dict_results[method].local_max for method in methods], axis=0, ignore_index=True)

    df_comp_methods = pd.DataFrame(
        {'index': series_concat_index}).reset_index(drop=True)

    df_comp_methods['ann'] = pd.Series(df_comp_methods['index'].isin(
        df_beats.cor_peak_index), dtype=BOOL_TYPE)

    for method in methods:
        df_comp_methods[method] = pd.Series(df_comp_methods['index'].isin(
            dict_results[method].local_max), dtype=BOOL_TYPE)

    return df_comp_methods


def stringfy_processor(processor) -> str:
    if processor:
        return f'{processor[0]}|{processor[1]}'
    else:
        return 'None'


def processor(ecg: npt.NDArray[np.float32], *args: Any) -> npt.NDArray[np.float32]:
    print(args)
    process, params = args
    if process == 'detrend':
        ecg = nk.signal_detrend(ecg, **params)
    elif process == 'standardize':
        ecg = nk.standardize(ecg, **params)  # type: ignore

    else:
        raise ValueError(
            f'Invalid processor: {process}. Valid processors are: detrend, standardize')
    return ecg


def create_df_beats(record_num: int, total_time: int, offset: int, derised_anns: List = LIST_BEATS_1, signal_track: int = 0, *args: Any) -> Tuple[pd.DataFrame, pd.Series, int, int, int]:
    """
    Create a dataframe with all beats for a given record.
    """

    # Load record
    record, ann = load_record(record_num)
    fs = int(record.fs)  # type: ignore

    samples = int(total_time * fs)
    start_samples = int(offset * fs)
    end_samples = start_samples + samples

    # ECG signal
    ecg = record.p_signal[:, signal_track]  # type: ignore

    if args:
        ecg = processor(ecg, *args)

    ecg = pd.Series(ecg, dtype=ECG_TYPE)[
        start_samples:end_samples]

    ann_beat_indexes = pd.Series(ann.sample, dtype=INDEX_TYPE)
    ann_beat_symbols = pd.Series(ann.symbol, dtype=ANN_TYPE)

    # Mask for time window and derised annotations
    mask_derised_ann = ann_beat_symbols.isin(derised_anns)

    # We are only interested in samples in the time window
    mask_time_window = (ann_beat_indexes >= start_samples) & (
        ann_beat_indexes < end_samples)

    mask_used_ann = mask_time_window & mask_derised_ann

    # Apply mask
    ann_beat_indexes = ann_beat_indexes[mask_used_ann].reset_index(drop=True)
    ann_beat_symbols = ann_beat_symbols[mask_used_ann].reset_index(drop=True)

    df_beats = correct_peaks(ecg, ann_beat_indexes, fs)

    df_beats = df_beats.rename(columns={'index': 'peak_index', 'local_max': 'cor_peak_index'}).merge(
        pd.DataFrame({'peak_index': ann_beat_indexes, 'symbol': ann_beat_symbols}), on='peak_index', how='left', validate='one_to_one')

    # If the peak is not corrected, use the original peak index
    df_beats.loc[df_beats.cor_peak_index.isna(
    ), 'cor_peak_index'] = df_beats.peak_index

    return df_beats, ecg, start_samples, end_samples, fs


def create_dict_results(ecg: Union[npt.NDArray[np.float32], pd.Series], methods: Union[List, str], start_samples: int, end_samples: int, fs: int, discard_start_sec: int, discard_end_sec: int) -> Tuple[Dict[str, pd.DataFrame], int, int]:

    if isinstance(methods, str):
        methods = [methods]

    first_used_sample = start_samples + discard_start_sec * fs
    last_used_sample = end_samples - discard_end_sec * fs

    dict_results = {}
    for method in methods:
        method_beat_indexes = find_peaks(ecg, fs, method)
        # Fix index
        method_beat_indexes += start_samples
        df_method_beats = correct_peaks(ecg, method_beat_indexes, fs)

        # When the method fails to detect a peak, the index is set to NaN. We replace it with the original index.
        df_method_beats.loc[df_method_beats.local_max.isna(
        ), 'local_max'] = df_method_beats.local_max

        local_max = df_method_beats.local_max
        df_method_beats = df_method_beats[(
            local_max >= first_used_sample) & (local_max < last_used_sample)]
        # Store results in dict
        dict_results[method] = df_method_beats
    return dict_results, first_used_sample, last_used_sample


def plot_results(dict_results, df_beats, ecg, desired_anns, x_xis_factor=1):
    methods = list(dict_results.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ecg.index * x_xis_factor, y=ecg, name="ECG"))

    for desired_ann in desired_anns:
        # Get the samples of the desired annotations
        df_beats_desired = df_beats[df_beats.symbol == desired_ann]
        desired_peak_indexes = df_beats_desired.cor_peak_index

        # Plot the annotations
        fig.add_trace(go.Scatter(x=desired_peak_indexes * x_xis_factor,
                      y=ecg.loc[desired_peak_indexes], mode="markers", name=desired_ann, marker=dict(size=7, color="red")))

    for method in methods:
        df_method_beats = dict_results[method]
        peak_indexes = df_method_beats.local_max
        fig.add_trace(go.Scatter(x=peak_indexes * x_xis_factor,
                      y=ecg.loc[peak_indexes], mode="markers", name=method, marker=dict(size=7)))

        # Remove borders
    fig.update_layout(
        margin=dict(l=0, r=0, t=15, b=0),
        paper_bgcolor="white",
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="linear"
        )
    )

    return fig


def calculate_metrics(df_comp_methods, method) -> Tuple[np.float32, np.float32]:
    true_positive = ((df_comp_methods['ann'] == True) & (
        df_comp_methods[method]) == True).values.sum()
    true_negative = ((df_comp_methods['ann'] == False) & (
        df_comp_methods[method]) == False).values.sum()
    false_positive = ((df_comp_methods['ann'] == False) & (
        df_comp_methods[method] == True)).values.sum()
    false_negative = ((df_comp_methods['ann'] == True) & (
        df_comp_methods[method] == False)).values.sum()

    precision = true_positive / (true_positive + false_positive)
    # recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (true_positive +
                                                  true_negative + false_positive + false_negative)

    return precision, accuracy
