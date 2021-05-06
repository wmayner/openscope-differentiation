## Environment

This project uses a [`conda`](https://docs.conda.io/en/latest/miniconda.html)
environment.

To create the environment on Linux, run the following commands from within the
top level of the repository:

```bash
conda env create -f environment.yml
conda activate openscope-differentiation
```

If you are using macOS, use `environment.cross-platform.yml` instead of
`environment.yml`.

## Data

The data are available from https://doi.org/10.5281/zenodo.4734581.

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
write them to disk. Once the notebooks listed below have been run, the figures &
statistical analyses can be reproduced by running their respective notebooks.

| Notebook                    | Purpose                                                                        |
| :-------------------------- | :----------------------------------------------------------------------------- |
| `main.ipynb`                | Compute neurophysiological differentiation from dF/F traces and decode stimuli |
| `stimulus_properties.ipynb` | Compute stimulus differentiation & other stimulus properties                   |
| `stats.ipynb`               | Perform the main statistical analyses                                          |

These in turn depend on several modules:

| Module                           | Purpose                                                |
| :------------------------------- | :----------------------------------------------------- |
| `load.ipynb`                     | Load and preprocess data & stimuli                     |
| `metadata.py`                    | Load metadata for experiments and stimuli              |
| `analysis.ipynb`                 | Analysis and plotting functions shared among notebooks |
| `spectral_differentiation.ipynb` | Compute spectral differentiation of timeseries         |
