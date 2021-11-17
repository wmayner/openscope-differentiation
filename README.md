## Environment

This project uses a [`conda`](https://docs.conda.io/en/latest/miniconda.html)
environment.

To create the environment on Linux, run the following commands from within the
top level of the repository:

```bash
conda env create -f environment.yml
conda activate openscope-differentiation
```

If you are using macOS or Windows, use `environment.cross-platform.yml` instead
of `environment.yml`.

## Data

The data are available from https://doi.org/10.5281/zenodo.4734580.

Once downloaded, they must be extracted into the top level of the repository:

```bash
tar -xzvf data.tar.gz
```

The `data` directory is organized as follows:

```
data
├── dff
│   ├── dff_*.h5    # dF/F traces for each session
├── eye
│   ├── *_area.npy                  # Pupil area for each session
│   └── preprocessing_params.yaml   # Preprocessing parameters for pupillometry data
├── events          # Detected L0 events for each cell in each session
│   └── session_*
│       └── event_cell_*.npz
├── metadata.csv    # Experiment metadata
├── pkl
│   └── pkl_*.pkl   # Stimulus presentation & behavioral data output from the optical physiology rig
├── run
│   ├── preprocessing_params.yaml   # Preprocessing parameters for locomotion data
│   ├── run_speed*.npy              # Locomotion velocity for each session
├── stim
│   ├── metadata.csv        # Stimulus metadata
│   ├── preview             # Stimulus movie files for viewing
│   ├── stimuli             # Stimulus movie files for presentation & analysis (unsigned 8-bit integer arrays)
│   ├── stimulus_df_*.csv   # Stimulus presentation table for each session
└── sync
    └── *_time_synchronization.h5   # Time synchronization data for alignment
```

## Project organization & reproduction

Figures and tables are generated in several notebooks, named accordingly.

Some of these depend on other notebooks that compute intermediate results and
write them to disk. Once the notebooks listed below have been run in that order,
the figures & statistical analyses can be reproduced by running their respective
notebooks.

| Notebook                    | Purpose                                                       |
| :-------------------------- | :------------------------------------------------------------ |
| `main.ipynb`                | Compute neurophysiological differentiation and decode stimuli |
| `stats.ipynb`               | Perform the main statistical analyses                         |
| `stimulus_properties.ipynb` | Compute stimulus differentiation & other stimulus properties  |

These in turn depend on several modules:

| Module                           | Purpose                                                |
| :------------------------------- | :----------------------------------------------------- |
| `load.ipynb`                     | Load and preprocess data & stimuli                     |
| `metadata.py`                    | Load metadata for experiments and stimuli              |
| `analysis.ipynb`                 | Analysis and plotting functions shared among notebooks |
| `spectral_differentiation.ipynb` | Compute spectral differentiation of timeseries         |
