#!/usr/bin/env python
# coding: utf-8
# load.py

"""Functions for loading, aligning, and preprocessing data."""

import itertools
from collections import Iterable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.signal
import yaml

import metadata
from metadata import METADATA, STIMULUS_METADATA

DATA_PATH = Path("data")
STIMULUS_PATH = DATA_PATH / "stim/stimuli/"

# Name of column indicating wall time
TIME = "t"


def load_yaml(path, **kwargs):
    """Load a YAML file at ``path``."""
    # Use SafeLoader if no loader is given
    kwargs = {"Loader": yaml.SafeLoader, **kwargs}
    with open(path, "rt") as f:
        return yaml.load(f, **kwargs)


def make_multiindex(session, index, name="frame"):
    """Make a multiindex with metadata from the given session.

    If the index already has a name, `name` is ignored and the index name is
    used.
    """
    mdata = metadata.for_session(session)
    if getattr(index, "name", False):
        name = index.name
    return pd.MultiIndex.from_arrays(
        [
            np.array([mdata["area"]] * len(index)),
            np.array([mdata["cre"]] * len(index)),
            np.array([mdata["session"]] * len(index)),
            index,
        ],
        names=["area", "cre", "session", name],
    )


# Stimuli
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load stimuli from disk
def load_stimulus_movie(name):
    """Load a stimulus file from disk.

    Ensures that the array is loaded as a float for accurate computation of
    FFT, etc.
    """
    filename = STIMULUS_METADATA.loc[name, "stimulus_filename"]
    if pd.isnull(filename):
        return None
    return np.load(STIMULUS_PATH / filename).astype(np.float64)


# Alignment data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


TIME_SYNC_FILE_PATTERN = DATA_PATH / "sync/{session}_time_synchronization.h5"


def load_alignment_data(key, session, dtype=int):
    f = h5py.File(str(TIME_SYNC_FILE_PATTERN).format(session=session), "r")
    data = np.array(f[key], dtype=dtype)
    has_correct_shape = (data.ndim == 2 and data.shape[-1] == 1) or (data.ndim == 1)
    assert has_correct_shape, f"Unexpected shape: {data.shape}"
    return data.reshape(-1)


def load_clock_alignment(session):
    # Mapping from 2p frame to clock time
    return load_alignment_data("twop_vsync_fall", session, dtype=float)


def load_eye_alignment(session):
    # Mapping from 2p frame to eye camera frame
    return load_alignment_data("eye_tracking_alignment", session)


def load_run_alignment(session):
    # Mapping from monitor vsync (and running wheel sample) to 2p frame
    return load_alignment_data("stimulus_alignment", session)


# Clock data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def load_clock(session):
    """Load rig clock data."""
    return np.array(load_clock_alignment(session), dtype=float)


def load_aligned_clock(session):
    """Load rig clock data in the 2p reference frame."""
    clock = load_clock(session)
    # The clock data is already in the 2p reference frame, so we don't need to
    # resample it.
    frame = np.arange(len(clock))
    index = make_multiindex(session, frame)
    return pd.DataFrame(data={TIME: pd.to_datetime(clock, unit="s")}, index=index)


# 2p data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


TWOP_FILE_PATTERN = DATA_PATH / "dff/dff_{session}.h5"


def _2p_dataframe(session, dataset):
    frames = np.arange(dataset.shape[-1])
    index = make_multiindex(session, frames)
    return pd.DataFrame(
        {"cell_{0}".format(i): dataset[i, :] for i in range(dataset.shape[0])},
        index=index,
    )


def load_dff(session):
    """Load dF/F data."""
    path = str(TWOP_FILE_PATTERN).format(session=session)
    f = h5py.File(path, "r")
    keys = list(f.keys())
    assert len(keys) == 1
    dataset = f[keys[0]]
    return _2p_dataframe(session, dataset)


# Event data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EVENT_FILE_PATTERN = DATA_PATH / "events/session_{session}"


def _load_events_single_cell(path):
    data = np.load(path)
    # Check that the data is as expected
    assert list(data.keys()) == ["events", "event_min_size"]
    assert data["event_min_size"] == 2
    assert data["events"].ndim == 2 and data["events"].shape[0] == 1
    # Flatten and return events
    return data["events"].reshape(-1)


def load_events(session):
    """Load events extracted from dF/F data."""
    path = Path(str(EVENT_FILE_PATTERN).format(session=session))
    cell_paths = map(
        str, sorted(path.glob("*"), key=lambda p: int(p.stem.split("_")[-1]))
    )
    dataset = np.array(list(map(_load_events_single_cell, cell_paths)))
    return _2p_dataframe(session, dataset)


# Stimuli presentation data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


STIMULUS_TABLE_FILE_PATTERN = DATA_PATH / "stim/stimulus_df_{session}.csv"


def load_stimulus(session, clock=False):
    """Load stimulus presentation data."""
    path = str(STIMULUS_TABLE_FILE_PATTERN).format(session=session)
    table = pd.read_csv(
        path,
        dtype={
            "movie": int,
            "trial": int,
            "start_frame": int,
            "end_frame": int,
            "num_frames": int,
        },
    ).rename(columns={"movie": "code"})
    table["stimulus"] = table["code"].map(
        {
            code: stimulus
            for stimulus, code in STIMULUS_METADATA["stimulus_code"].to_dict().items()
        }
    )
    if clock:
        clock = load_clock(session)
        table["start_t"] = clock[table["start_frame"]]
        table["end_t"] = clock[table["end_frame"]]
    return table


def load_aligned_stimulus(session):
    """Load stimulus presentation data in the 2p reference frame."""
    # Load the stimulus table
    table = load_stimulus(session)
    # Get a list of frames within the range described by the table
    start = table["start_frame"].min()
    end = table["end_frame"].max()
    frame = np.arange(start, end)
    # Initialize a mapping from frames to stimuli
    stimulus = pd.DataFrame(
        index=frame, data={"stimulus": "interstimulus", "trial": np.nan}
    )
    # Populate the mapping by filling in intervals for each stimulus with the
    # name
    for s in table["stimulus"].unique():
        this_stimulus = table[table["stimulus"] == s]
        for _, row in this_stimulus.iterrows():
            # Find frame indices corresponding to the interval from table
            idx = list(range(row["start_frame"], row["end_frame"]))
            # Check that each frame corresponds to exactly one stimulus by
            # ensuring that all the indices we found have not already been
            # assigned to another stimulus
            assert np.all(
                stimulus.loc[idx, "stimulus"] == "interstimulus"
            ), "Overlapping tags!"
            # Check that the number of frames matches the expected number from
            # the stimulus table
            expected_num_frames = row["num_frames"]
            actual_num_frames = len(idx)
            assert (
                actual_num_frames == expected_num_frames
            ), f"Expected {expected_num_frames} frames for '{s}', got {actual_num_frames}"
            # Assign the stimulus name to those indices
            stimulus.loc[idx, "stimulus"] = s
            # Assign the trial to thos indices
            stimulus.loc[idx, "trial"] = int(row["trial"])
    # Create multiindex with metadata
    stimulus.index = make_multiindex(session, stimulus.index)
    return stimulus


# Load running data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


RUN_FILE_PATTERN = DATA_PATH / "run/run_speed{session}.npy"


def load_run(session):
    """Load running velocity data."""
    return np.load(str(RUN_FILE_PATTERN).format(session=session))


def load_aligned_run(session):
    """Load running velocity data in the 2p reference frame."""
    run = load_run(session)
    alignment = load_run_alignment(session)
    # The alignment data is a mapping from vsync to 2p frame, but we want the
    # running velocity at each 2p frame, so we invert the mapping. Since it's
    # one-to-many, there's no unique inverse; we take the first vsync per frame.
    inverse_alignment = np.searchsorted(alignment, np.unique(alignment))
    # Some sessions have alignment data whose last running frame is not present
    # in the run data (e.g. session 742814901).
    if inverse_alignment[-1] == len(run):
        inverse_alignment = inverse_alignment[:-1]
    # Resample the running velocity
    run = run[inverse_alignment]
    # Get the corresponding 2p frames
    frame = alignment[inverse_alignment]
    index = make_multiindex(session, frame)
    return pd.DataFrame(data={"velocity": run}, index=index)


# Pupillometry data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


PUPIL_AREA_FILE_PATTERN = DATA_PATH / "eye/{session}_area.npy"


def parse_entry(entry):
    if isinstance(entry, int):
        return [entry]
    else:
        return list(map(int, entry.split(", ")))


EYE_TO_TWOP_MAP = dict(
    zip(
        itertools.chain.from_iterable(map(parse_entry, METADATA["all_ophys_sessions"])),
        itertools.chain.from_iterable(
            map(parse_entry, METADATA["all_ophys_experiments"])
        ),
    )
)
TWOP_TO_EYE_MAP = {value: key for key, value in EYE_TO_TWOP_MAP.items()}


def load_eye(session):
    """Load pupillometry data."""
    eye_session = TWOP_TO_EYE_MAP[session]
    pupil_path = str(PUPIL_AREA_FILE_PATTERN).format(session=eye_session)
    pupil = np.load(pupil_path)
    return pupil


def load_aligned_eye(session):
    """Load pupillometry data in the 2p reference frame."""
    # Load data
    pupil = load_eye(session)
    alignment = load_eye_alignment(session)
    # The eye-tracking alignment maps 2p frames to pupil frames, so we can index
    # the pupil data directly with the alignment to move into the 2p reference
    # frame.
    pupil = pupil[alignment]
    frame = np.arange(len(alignment))
    index = make_multiindex(session, frame)
    return pd.DataFrame(
        data={"pupil_area": pupil},
        index=index,
    )


# Preprocessing functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RUN_PREPROCESSING_PARAMS = load_yaml(DATA_PATH / "run/preprocessing_params.yaml")
EYE_PREPROCESSING_PARAMS = load_yaml(DATA_PATH / "eye/preprocessing_params.yaml")


def interpolate(signal, **kwargs):
    """Interpolate missing data with Pandas."""
    return pd.Series(signal).interpolate(**kwargs)


def mask_intervals(array, intervals, copy=False):
    """Mask the given intervals (inclusive)."""
    assert array.ndim == 1, f"Array must be 1D: got {array.shape}"
    indices = np.arange(array.shape[0])
    mask = np.zeros(array.shape[0], dtype=bool)
    for lower, upper in intervals:
        mask = mask | ((indices >= lower) & (indices <= upper))
    return np.ma.MaskedArray(data=array, mask=mask, copy=copy)


def erase_transients(signal, prominence, wlen, rel_height=0.5, return_peaks=False):
    """Find peaks in a signal and replace them with NaNs.

    Arguments:
        prominence: The minimal prominence for a peak to be identified. See
            `scipy.signal.find_peaks`.
        wlen: Controls the “locality” of the peak-finding. See
            `scipy.signal.find_peaks`.
    Keyword Arguments:
        rel_height: The relative height at which to evaluate the width of the peaks.
        return_peaks: Whether to return the indices of the peaks.

    Returns:
        The signal with the peaks removed.
    """
    # Transients to be erased will be marked with a mask
    signal = np.ma.MaskedArray(signal, fill_value=np.nan)
    # Interpolate through NaNs since `find_peaks` can't handle NaNs well
    interpolated = interpolate(signal.filled())
    # Find bad intervals based on peaks
    peaks, properties = scipy.signal.find_peaks(
        interpolated, prominence=prominence, wlen=wlen
    )
    prominence_data = (
        properties["prominences"],
        properties["left_bases"],
        properties["right_bases"],
    )
    _, _, left, right = scipy.signal.peak_widths(
        interpolated, peaks, rel_height=rel_height, prominence_data=prominence_data
    )
    intervals = np.stack([np.ceil(left), np.floor(right)], axis=1).astype(int)
    # Mask the bad intervals
    signal = mask_intervals(signal, intervals, copy=False)
    if return_peaks:
        return signal, peaks
    return signal


def iterated_erase_transients(signal, prominence, wlen, rel_height=0.5, n=None):
    """Apply ``n`` passes of ``erase_transients``."""
    if not isinstance(prominence, Iterable):
        prominence = [prominence] * n
    if not isinstance(wlen, Iterable):
        wlen = [wlen] * n
    if not isinstance(rel_height, Iterable):
        rel_height = [rel_height] * n
    assert (
        len(prominence) == len(wlen) == len(rel_height)
    ), "`prominence`, `wlen`, and `rel_height` must have the same length"
    for prom, wl, rh in zip(prominence, wlen, rel_height):
        signal = erase_transients(signal, prom, wl, rel_height=rh)
    return signal


def clean_signal(signal, prominence, wlen, **kwargs):
    """Clean by iteratively removing positive and negative peaks."""
    # Remove positive transients
    cleaned = iterated_erase_transients(signal, prominence, wlen, **kwargs)
    # Remove negative transients
    cleaned = -iterated_erase_transients(-cleaned, prominence, wlen, **kwargs)
    return cleaned


def normalize_pupil_diameter(data):
    """Compute and attach the normalized pupil diameter given session data.

    NOTE: Modifies the dataframe in place.
    """
    data["pupil_diameter"] = 2 * np.sqrt(data["pupil_area"] / np.pi)
    # Normalize by the maximum diameter within the block stimuli presentation
    # period
    block_frames = data.loc[
        data["stimulus_is_block"] == True
    ]  # Need `== True` to exclude NaNs
    maximum = block_frames["pupil_diameter"].max()
    data["normalized_pupil_diameter"] = data["pupil_diameter"] / maximum
    return data


def butterworth(cutoff, fs, order, btype="lowpass", output="ba", analog=False):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    return scipy.signal.butter(
        order, normalized_cutoff, btype=btype, output=output, analog=analog
    )


def describe_rows(df):
    """Describe a DataFrame rowwise.

    Does not include IQR statistics, which are expensive to compute rowwise.
    """
    return pd.DataFrame(
        dict(
            mean=df.mean(axis=1),
            std=df.std(axis=1),
            min=df.min(axis=1),
            max=df.max(axis=1),
        )
    )


def _2p_statistics(data):
    """Return frame-wise descriptive statistics of 2p data."""
    return describe_rows(data.drop(columns=[TIME]))


# API
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def load_preprocessed_clock(session):
    """Load and preprocess clock times."""
    return load_aligned_clock(session)


def load_preprocessed_dff(session):
    """Load and preprocess dF/F data."""
    # Load dF/F data
    dff = load_dff(session)
    # Attach clock time
    dff[TIME] = load_preprocessed_clock(session)
    return dff


def load_preprocessed_events(session):
    """Load and preprocess event data."""
    # Load dF/F data
    events = load_events(session)
    # Attach clock time
    events[TIME] = load_preprocessed_clock(session)
    return events


def load_preprocessed_stimulus(session):
    """Load and preprocess stimulus information."""
    data = load_aligned_stimulus(session)
    # Annotate with stimulus metadata
    data = data.merge(STIMULUS_METADATA, left_on="stimulus", right_index=True)
    return data


def load_preprocessed_run(session):
    """Load and preprocess running velocity."""
    run = load_aligned_run(session)
    # Remove artifacts
    # --------------------------------
    velocity = run["velocity"].values
    # Automatically clean artifacts
    velocity = clean_signal(velocity, **RUN_PREPROCESSING_PARAMS["clean_signal_params"])
    # Mask remaining manually-identified artifacts
    # NOTE: These values must be integer locations in the velocity timeseries,
    # since this is an `np.ndarray`, not a `pd.Series` (so it no longer has an
    # index). They are not dF/F frame indices.
    if session in RUN_PREPROCESSING_PARAMS["bad_samples"]:
        velocity = mask_intervals(
            velocity, RUN_PREPROCESSING_PARAMS["bad_samples"][session]
        )
    # Fill masked values with NaNs
    velocity = velocity.filled()
    # --------------------------------
    # Replace with cleaned signal
    run["velocity"] = velocity
    # Interpolate through NaNs
    run["velocity"] = run["velocity"].interpolate(limit_area="inside")
    # Apply lowpass filter
    b, a = butterworth(**RUN_PREPROCESSING_PARAMS["butterworth_params"])
    run["filtered_velocity"] = scipy.signal.filtfilt(b, a, run["velocity"])
    run["filtered_velocity"] = run["filtered_velocity"].interpolate(limit_area="inside")
    return run


def _preprocess_pupil(pupil, session):
    """Remove artifacts from pupillometry data."""
    index = pupil.index
    # Automatically remove artifacts
    pupil = clean_signal(
        pupil, **EYE_PREPROCESSING_PARAMS["pupil"]["clean_signal_params"][session]
    )
    # Mask remaining manually-identified artifacts
    pupil = mask_intervals(
        pupil, EYE_PREPROCESSING_PARAMS["pupil"]["bad_samples"][session]
    )
    # Fill masked values with NaN
    pupil = pupil.filled()
    # Convert back to a Series with the original index
    pupil = pd.Series(data=pupil, index=index)
    # Replace with cleaned signal
    return pupil


def load_preprocessed_eye(session):
    """Load and preprocess pupillometry data.

    NOTE: Normalized diameter is computed in ``load_data()``, not here, since
    it requires loading the stimulus presentation dataframe.
    """
    data = load_aligned_eye(session)
    # Preprocess pupil area
    data["pupil_area"] = _preprocess_pupil(data["pupil_area"], session)
    return data


def load_data(session):
    data = [
        load_preprocessed_stimulus(session),
        load_preprocessed_clock(session),
        load_preprocessed_run(session),
        load_preprocessed_eye(session),
    ]
    return pd.concat(data, axis="columns")


def load_preprocessed_data(session):
    """Load and perprocess all data from a session.

    dF/F is returned separately from other data, since there is a sample for
    each frame and each cell, and thus longform data would be ``ncells`` times
    larger.

    NOTE: Normalized pupil diameter is computed here, not in
    ``load_preprocessed_eye()``.

    Args:
        session (int): The ID of the session to retrieve.

    Returns:
        tuple[DataFrame]: Two DataFrames: the first contains dF/F data
        (sample x cell); the second contains all other session data.
    """
    dff = load_preprocessed_dff(session)
    data = load_data(session)
    # Compute descriptive statistics across cells (i.e., row-wise) for each
    # sub-frame of the dF/F data
    dff_stats = _2p_statistics(dff).add_prefix("dff_")
    data = pd.concat([data] + [dff_stats], axis="columns")
    # Attach metadata that isn't in the index
    mdata = metadata.for_session(session)
    for key in ["ncells", "start_time", "mouse_id"]:
        data[key] = mdata[key]
    # Normalize pupil diameter
    data = normalize_pupil_diameter(data)
    # Interpolate through missing velocity data again; this is necessary
    # because aligning wheel samples to 2p frames sometimes results in a
    # skipped frame
    data["velocity"] = data["velocity"].interpolate(limit_area="inside")
    data["filtered_velocity"] = data["filtered_velocity"].interpolate(
        limit_area="inside"
    )
    # Binary threshold for locomotion
    data["locomotion"] = (
        np.abs(data["filtered_velocity"])
        >= RUN_PREPROCESSING_PARAMS["locomotion_threshold"]
    )
    # Return dF/F traces and session data separately
    return (dff, data)
