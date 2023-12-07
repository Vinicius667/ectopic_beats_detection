import os
import pickle
from typing import Any, Iterable, MutableMapping, Tuple, Union

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from numpy import typing as npt
from plotly import graph_objects as go

from globals import (ANN_TYPE, BOOL_TYPE, ECG_TYPE, INDEX_TYPE, LIST_BEATS_1,
                     mb_artm_directory)

DICT_RESULTS_TYPE = MutableMapping[str, pd.DataFrame]
METHODS_TYPE = Union[str, Iterable[str]]


class Processor:
    def __init__(self, processor_func: Union[str, None], *args, **kwargs):
        # sort kwargs by key
        kwargs = {k: v for k, v in sorted(
            kwargs.items(), key=lambda item: item[0])}
        args = sorted(args)

        self.processor_func = processor_func
        self.processor_name = '|'.join([str(processor_func)] +
                                       [str(param) for param in [args, kwargs] if param])
        self.args = args
        self.kwargs = kwargs


class Processors:
    def __init__(self, processors: Iterable[Processor]):
        self.processor_name = ';'.join(
            [processor.processor_name for processor in processors])
        self.processors = processors


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


def create_compare_df(df_beats: pd.DataFrame, dict_results: DICT_RESULTS_TYPE) -> pd.DataFrame:
    methods = list(dict_results.keys())

    for method in methods:
        if dict_results[method] is None:
            del dict_results[method]

    series_concat_index = pd.concat([df_beats.cor_peak_index] + [
                                    dict_results[method].local_max for method in methods], axis=0, ignore_index=True)  # type: ignore

    # Create column with all the indexes (annotations and methods results)
    df_comp_methods = pd.DataFrame(
        {'index': series_concat_index}).reset_index(drop=True)

    df_comp_methods = df_comp_methods.merge(
        df_beats[['cor_peak_index', 'symbol']], left_on='index', right_on='cor_peak_index', how='left')

    # Indexes present in annotations
    df_comp_methods['ann'] = pd.Series(df_comp_methods['index'].isin(
        df_beats.cor_peak_index), dtype=BOOL_TYPE)

    # Indexes present in each method
    for method in methods:
        df_comp_methods[method] = pd.Series(df_comp_methods['index'].isin(
            dict_results[method].local_max), dtype=BOOL_TYPE)  # type: ignore

    return df_comp_methods


def apply_processors(ecg: npt.NDArray[np.float32], processors: Union[Processor, Processors]) -> npt.NDArray[np.float32]:
    """Apply a processor to the ecg signal.

    Args:
        ecg (npt.NDArray[np.float32]): ecg signal

    Raises:
        ValueError: raised when an invalid processor is passed

    Returns:
        npt.NDArray[np.float32]: processed ecg signal
    """

    if isinstance(processors, Processor):
        processors = Processors([processors])

    for processor in processors.processors:
        args = processor.args
        kwargs = processor.kwargs
        processor_func = processor.processor_func

        if processor_func is None:
            continue
        elif processor_func == 'detrend':
            func = nk.signal_detrend
        elif processor_func == 'standardize':
            func = nk.standardize
        else:
            print(
                f'Invalid processor: {processor_func}. Valid processors are: detrend, standardize')
            continue
        ecg = func(ecg, *args, **kwargs)  # type: ignore

    return ecg


def create_df_beats(record_num: int, total_time: int, offset: int, derised_anns: Iterable[str] = LIST_BEATS_1, signal_track: int = 0, processor: Union[Processor, Processors] = Processor(None)) -> Tuple[pd.DataFrame, pd.Series, int, int, int]:
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
    if processor.processor_name:
        ecg = apply_processors(ecg, processor)
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


def create_dict_results(ecg: Union[npt.NDArray[np.float32], pd.Series], methods: METHODS_TYPE, start_samples: int, first_used_sample: int, last_used_sample: int, fs: int, discard_start_sec: int, discard_end_sec: int) -> DICT_RESULTS_TYPE:
    """Create a dictionary with the results of each method.

    Args:
        ecg (Union[npt.NDArray[np.float32], pd.Series]): ecg signal
        methods (Union[List, str]): methods to be used for peak detection
        first_used_sample (int): first sample used comparing the results
        last_used_sample (int): last sample used comparing the results
        fs (int): sampling rate (Hz)
        discard_start_sec (int): seconds to be discarded from the beginning
        discard_end_sec (int): seconds to be discarded from the end

    Returns:
        Tuple[Dict[str, Union[pd.DataFrame, None]], int, int]:
            dict_results: dictionary with the results of each method
    """

    if isinstance(methods, str):
        methods = [methods]

    dict_results = {}
    for method in methods:
        try:
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
        except (IndexError, ValueError, KeyError):
            print(f'Error in method: {method}')
            # dict_results[method] = None
    return dict_results


def plot_results(dict_results: MutableMapping, df_beats: pd.DataFrame, ecg: pd.Series, ecg_annotations: Iterable[Tuple[Iterable, MutableMapping[str, Any]]], x_xis_factor=1):
    methods = list(dict_results.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ecg.index * x_xis_factor, y=ecg, name="ECG"))

    for desired_anns, style in ecg_annotations:
        for desired_ann in desired_anns:
            # Get the samples of the desired annotations
            df_beats_desired = df_beats[df_beats.symbol == desired_ann]
            desired_peak_indexes = df_beats_desired.cor_peak_index

            style['name'] = desired_ann

            # Plot the annotations
            fig.add_trace(go.Scatter(x=desired_peak_indexes * x_xis_factor,
                                     y=ecg.loc[desired_peak_indexes], **style))

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


def calculate_metrics(df_comp_methods, methods: METHODS_TYPE) -> MutableMapping[str, Any]:
    if isinstance(methods, str):
        methods = [methods]
    dict_metrics = {}
    for method in methods:
        if method not in df_comp_methods.columns:
            continue
        mask_true_positive = ((df_comp_methods['ann'] == True) & (
            df_comp_methods[method]) == True)

        mask_true_negative = ((df_comp_methods['ann'] == False) & (
            df_comp_methods[method]) == False)

        mask_false_positive = ((df_comp_methods['ann'] == False) & (
            df_comp_methods[method] == True))

        mask_false_negative = ((df_comp_methods['ann'] == True) & (
            df_comp_methods[method] == False))

        quant_true_positive = mask_true_positive.values.sum()
        quant_true_negative = mask_true_negative.values.sum()
        quant_false_positive = mask_false_positive.values.sum()
        quant_false_negative = mask_false_negative.values.sum()

        precision = quant_true_positive / \
            (quant_true_positive + quant_false_positive)
        accuracy = (quant_true_positive + quant_true_negative) / (quant_true_positive +
                                                                  quant_true_negative + quant_false_positive + quant_false_negative)

        dict_metrics[method] = {
            'precision': precision,
            'accuracy': accuracy,
            'false_positive': df_comp_methods['symbol'][mask_true_positive].shape[0],
            'false_negative': df_comp_methods['symbol'][mask_false_negative].shape[0],
        }

    return dict_metrics


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def dict_multi_analysis_to_df(dict_multi_analysis: MutableMapping) -> pd.DataFrame:
    dict_multi_analysis_df = {
        'record_num': [],
        'processor': [],
        'method': [],
        'accuracy': [],
        'precision': [],
        'signal_track': [],
        'start_samples': [],
        'end_samples': [],
    }

    for record_num, dict_processor in dict_multi_analysis.items():
        for processor, dict_metrics in dict_processor.items():
            for method, metrics in dict_metrics.items():
                dict_multi_analysis_df['record_num'].append(record_num)
                dict_multi_analysis_df['processor'].append(processor)
                dict_multi_analysis_df['method'].append(method)
                dict_multi_analysis_df['accuracy'].append(metrics['accuracy'])
                dict_multi_analysis_df['precision'].append(
                    metrics['precision'])
                dict_multi_analysis_df['signal_track'].append(
                    metrics['signal_track'])
                dict_multi_analysis_df['start_samples'].append(
                    metrics['start_samples'])
                dict_multi_analysis_df['end_samples'].append(
                    metrics['end_samples'])

    df_multi_analysis = pd.DataFrame(dict_multi_analysis_df)
    return df_multi_analysis


def load_df_multi_analysis(dict_multi_analysis_files: Iterable[str]) -> pd.DataFrame:
    df_multi_analysis = pd.concat([dict_multi_analysis_to_df(load_pickle(
        file)) for file in dict_multi_analysis_files], axis=0, ignore_index=True)
    return df_multi_analysis


def check_in_df_multi_analysis(df_multi_analysis, record_num, processor, method, signal_track, start_samples, end_samples) -> bool:
    df = df_multi_analysis
    return df[
        (df['record_num'] == record_num) &
        (df['processor'] == processor) &
        (df['method'] == method) &
        (df['signal_track'] == signal_track) &
        (df['start_samples'] == start_samples) &
        (df['end_samples'] == end_samples)
    ].shape[0] > 0
