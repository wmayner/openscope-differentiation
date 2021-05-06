#!/usr/bin/env python
# coding: utf-8
# metadata.py

"""Metadata for experiments and stimuli."""

import pandas as pd

# Nominal sampling rates
TWOP_SAMPLE_RATE = 30.0  # Hz
EYE_SAMPLE_RATE = 30.0  # Hz
RUN_SAMPLE_RATE = 60.0  # Hz
STIMULUS_SAMPLE_RATE = 30.0  # Hz


METADATA_FILENAME = "data/metadata.csv"
STIMULUS_METADATA_FILENAME = "data/stim/metadata.csv"


def _load_metadata():
    """Load the session metadata table."""
    # Drop unneeded columns
    df = pd.read_csv(METADATA_FILENAME, index_col=0)
    df.index.name = "session"
    df["area"] = pd.Categorical(
        df["area"], categories=["V1", "LM", "AL", "PM", "AM"], ordered=True
    )
    df["cre"] = pd.Categorical(
        df["cre"], categories=["Cux2", "Rorb", "Rbp4"], ordered=True
    )
    df["layer"] = pd.Categorical(
        df["layer"], categories=["L2/3", "L4", "L5"], ordered=True
    )
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    df.sort_index(inplace=True)
    return df


METADATA = _load_metadata()


def for_session(session):
    """Get metadata for a session as a dictionary."""
    return {**METADATA.loc[session].to_dict(), "session": session}


def _load_stimulus_metadata():
    """Load the stimulus metadata table."""
    df = pd.read_csv(STIMULUS_METADATA_FILENAME)
    df["stimulus"] = pd.Categorical(
        df["stimulus"], categories=df["stimulus"], ordered=True
    )
    df.set_index("stimulus", inplace=True)
    return df


STIMULUS_METADATA = _load_stimulus_metadata()
