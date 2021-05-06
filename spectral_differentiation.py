#!/usr/bin/env python
# coding: utf-8
# spectral_differentiation.py

"""Functions for computing spectral differentiation."""

import numpy as np
import scipy.interpolate
import scipy.spatial
import scipy.signal
from numpy.core.multiarray import normalize_axis_index as _normalize_axis_index

from analysis import get_cells

TRUNCATION_TOLERANCE = 23


def join_axes(a, b, array):
    """Join adjacent axes ``a`` and ``b``."""
    a = _normalize_axis_index(a, array.ndim)
    b = _normalize_axis_index(b, array.ndim)
    # Check that the axes to be joined are adjacent
    if abs(a - b) != 1:
        raise np.AxisError(f"Axes to be joined must be adjacent; got {a} and {b}.")
    # Ensure that `a` is the earlier axis and `b` is the latter
    a, b = min(a, b), max(a, b)  # NOTE: Must be on one line
    # Compute the new shape
    return array.reshape(
        array.shape[:a] + (array.shape[a] * array.shape[b],) + array.shape[(b + 1) :]
    )


def split_axis(axis, a, b, array):
    """Split an axis into two adjacent axes of size ``a`` and ``b``."""
    axis = _normalize_axis_index(axis, array.ndim)
    if array.shape[axis] != a * b and a != -1 and b != -1:
        raise ValueError(
            f"Axis {axis} of size {array.shape[axis]} cannot be "
            f"split into axes of sizes {a} and {b}."
        )
    new_shape = array.shape[:axis] + (a, b) + array.shape[(axis + 1) :]
    return array.reshape(*map(int, new_shape))


def window(samples_per_window, data):
    """Split data into windows indexed by the third-to-last dimension.

    Assumes the data has shape (..., neuron, sample).

    Returns:
        An array with shape (..., window, neuron, sample).
    """
    num_windows = data.shape[-1] / samples_per_window
    if not float(num_windows).is_integer():
        raise ValueError("Data cannot be windowed evenly.")
    # Split last axis into windows and samples
    data = split_axis(-1, num_windows, samples_per_window, data)
    # Move window axis before neuron axis
    return np.moveaxis(data, -2, -3)


def to_log_spacing(frequencies, spectrum, frequency_axis=-1, precision=8):
    # Based on approach in https://stackoverflow.com/q/6670232/1085344
    log_frequencies = np.logspace(
        frequencies[0], np.log10(frequencies[-1]), num=len(frequencies), base=10
    )
    # Rounding is necessary to ensure interpolation bounds are respected,
    # since logspace introduces small floating-point errors
    log_frequencies = log_frequencies.round(precision)
    # Get cumulative spectrum
    df = frequencies[1] - frequencies[0]
    cumulative_spectrum = np.cumsum(spectrum, axis=frequency_axis) * df
    # Interpolate
    interpolate = scipy.interpolate.interp1d(
        frequencies, cumulative_spectrum, axis=frequency_axis, copy=False
    )
    interpolated_cumulative_spectrum = interpolate(log_frequencies)
    # Recover interpolated spectrum
    frequency_shape = [1] * spectrum.ndim
    frequency_shape[frequency_axis] = spectrum.shape[frequency_axis]
    log_spectrum = np.diff(
        interpolated_cumulative_spectrum, prepend=frequencies[0], axis=frequency_axis
    ) / np.diff(frequencies, axis=frequency_axis, prepend=(-df)).reshape(
        frequency_shape
    )
    assert np.allclose(
        spectrum.sum(),
        log_spectrum.sum(),
        atol=10 ** (-precision),
        rtol=10 ** (-precision),
    ), "Total power was not conserved!"
    return log_frequencies, log_spectrum


def spectral_states(sample_rate, window_length, data, axis=-1, log_frequency=False):
    """Compute the power spectrum for each window, then concatenate spectra for
    each neuron within each window.

    Assumes the data has shape (..., neuron, sample).

    Returns:
        An array containing spectra for each window, with shape
        (..., window, frequency).
    """
    # Check that data can be windowed with the given parameters
    samples_per_window = sample_rate * window_length
    if not float(samples_per_window).is_integer():
        raise ValueError(
            f"Invalid window length; implies fractional samples: {samples_per_window}"
        )
    samples_per_window = int(samples_per_window)
    data = window(samples_per_window, data)
    # Compute the FFT
    spectra = np.fft.rfftn(data, axes=(axis,))
    frequencies = np.fft.rfftfreq(samples_per_window, d=(1 / sample_rate))
    # Get the power spectrum
    spectra = np.square(np.abs(spectra))
    # Interpolate to log-spaced frequency bins if desired
    if log_frequency:
        frequencies, spectra = to_log_spacing(frequencies, spectra, frequency_axis=axis)
    # Concatenate power spectra of each neuron
    return frequencies, join_axes(-2, -1, spectra)


def differentiation(states, metric="euclidean"):
    """Return the differentiation among a set of states.

    Returns:
        A condensed pairwise distance matrix containing the Euclidean distances
        between the states.

    Note:
        From the NumPy docs for `pdist`::
            Returns a condensed distance matrix Y. For each i and j (where
            i<j<m), where m is the number of original observations, the metric
            dist(u=X[i], v=X[j]) is computed and stored in entry ij.
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)
    """
    # Compute the pairwise distances
    return scipy.spatial.distance.pdist(states, metric=metric)


def spectral_differentiation(
    data,
    sample_rate,
    window_length,
    metric,
    axis=-1,
    log_frequency=False,
    return_frequencies=False,
):
    """Return the spectral differentiation.

    Assumes data has shape (cell, sample).

    Returns:
        A condensed pairwise distance matrix containing the Euclidean
        distances between the spectra, concatenated across all cells, of
        non-overlapping windows of the traces.
    """
    frequencies, states = spectral_states(
        sample_rate, window_length, data, axis=axis, log_frequency=log_frequency
    )
    distances = differentiation(states, metric)
    if return_frequencies:
        return frequencies, distances
    return distances


def dff_differentiation_whole_stimulus(data, state_length, metric, fs, log_frequency):
    """Compute spectral differentiation of a set of traces."""
    samples_per_window = state_length * fs
    # Convert from seconds to samples
    trial_window = samples_per_window * int(data.shape[0] // samples_per_window)
    if not float(trial_window).is_integer():
        raise ValueError(f"Trial window length must be an integer; got {trial_window}")
    trial_window = int(trial_window)
    # Check we're not truncating too much
    if abs(data.shape[0] - trial_window) > TRUNCATION_TOLERANCE:
        raise ValueError(f"truncation tolerance exceeded: {data.shape}")
    # Truncate
    data = data[:trial_window]
    # Swap axes so it's (cell, sample)
    data = np.moveaxis(data, -1, -2)
    # Compute differentiation
    distances = spectral_differentiation(
        data,
        sample_rate=fs,
        window_length=state_length,
        metric=metric,
        log_frequency=log_frequency,
    )
    return np.median(distances)


def scale_spectra(S, window):
    """Scale a spectrum calculated from scipy.spectrogram to match the implementation using np.fft."""
    scaled = S.copy()

    # https://github.com/scipy/scipy/blob/v1.6.0/scipy/signal/spectral.py#L1802
    scaled *= window.sum() ** 2

    # https://github.com/scipy/scipy/blob/v1.6.0/scipy/signal/spectral.py#L1842
    if len(window) % 2:
        scaled[..., 1:] /= 2
    else:
        scaled[..., 1:-1] /= 2

    return scaled


def dff_differentiation_whole_stimulus_spectrogram(
    data,
    state_length,
    metric,
    fs,
    log_frequency,
    window,
    window_param,
    overlap,
    scale=True,
):
    """Return spectral differentiation using ``scipy.spectrogram``.

    Assumes data has shape (cell, sample).
    """
    nperseg = state_length * fs
    if not float(nperseg).is_integer():
        raise ValueError(
            f"state_length * fs must be integer: {state_length} * {fs} == {nperseg}"
        )
    nperseg = int(nperseg)

    # Get data in the shape (cell, sample) expected by SD functions
    assert data.ndim == 2, "Data is not 2D"
    data = data.transpose()
    assert (
        data.shape[0] < 400
    ), "Are there really more than 400 cells? Check dimensions!"

    if window_param is None:
        window = window
    else:
        window = (window, window_param)
    window = scipy.signal.get_window(window, nperseg)

    frequencies, times, spectra = scipy.signal.spectrogram(
        data,
        axis=-1,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=int(nperseg * overlap),
        detrend=False,
        return_onesided=True,
        scaling="spectrum",
        mode="psd",
    )

    # Reshape to (window, cell, frequency)
    spectra = np.moveaxis(spectra, -1, 0)

    if scale:
        spectra = scale_spectra(spectra, window)

    if log_frequency:
        frequencies, spectra = to_log_spacing(frequencies, spectra, frequency_axis=-1)

    # Concatenate spectra of each cell
    states = join_axes(-2, -1, spectra)

    # Compute pairwise distances
    distances = differentiation(states, metric)

    return np.median(distances)


def dff_differentiation_single_trial(
    group,
    state_length,
    metric,
    fs,
    log_frequency,
    window,
    window_param,
    overlap,
):
    # Only use cell columns
    data = group[get_cells(group)]
    # Pass on values
    data = data.values

    if window:
        return dff_differentiation_whole_stimulus_spectrogram(
            data,
            state_length=state_length,
            metric=metric,
            fs=fs,
            log_frequency=log_frequency,
            window=window,
            window_param=window_param,
            overlap=overlap,
        )
    else:
        return dff_differentiation_whole_stimulus(
            data,
            state_length=state_length,
            metric=metric,
            fs=fs,
            log_frequency=log_frequency,
        )


def compute_dff_differentiation(
    dff, state_length, metric, fs, log_frequency, window, window_param, overlap
):
    return dff.groupby(["stimulus", "trial"]).apply(
        dff_differentiation_single_trial,
        state_length=state_length,
        metric=metric,
        fs=fs,
        log_frequency=log_frequency,
        window=window,
        window_param=window_param,
        overlap=overlap,
    )
