#!/usr/bin/env python
# coding: utf-8
# analysis.py

"""Analysis and plotting functions shared among notebooks."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

from load import TIME
from metadata import METADATA, STIMULUS_METADATA

# Exclude the experiment with twop recording issues
SESSIONS = METADATA.loc[METADATA["valid"]].index

# Stimuli subsets
UNSCRAMBLED_SCRAMBLED = STIMULUS_METADATA.loc[
    STIMULUS_METADATA["stimulus_is_scrambled_pair"]
].index.tolist()
CONTINUOUS_NATURAL = STIMULUS_METADATA.loc[
    STIMULUS_METADATA["stimulus_is_continuous"]
    & (STIMULUS_METADATA["stimulus_type"] == "natural")
].index.tolist()
BLOCK_STIMULI = STIMULUS_METADATA.loc[
    STIMULUS_METADATA["stimulus_is_block"]
].index.tolist()


def get_cells(dataframe):
    """Return columns in the dataframe that start with 'cell'."""
    return [col for col in dataframe.columns if col.startswith("cell")]


def make_filename(params, prefix=None, suffix=None):
    return "__".join(
        ([prefix] if prefix else [])
        + [f"{param}-{value}" for param, value in params.items()]
        + ([suffix] if suffix else [])
    )


def savefig(fig, path, suffix=".svg", dpi=300, **kwargs):
    """Save a matplotlib plot."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    return path


def prepare_data_for_r(data, response):
    # NOTE: modifies data in place
    # Convert Categoricals to unordered since rpy2 has issues with them
    data["layer"] = pd.Categorical(data["layer"], categories=data["layer"].cat.categories, ordered=False)
    data["area"] = pd.Categorical(data["area"], categories=data["area"].cat.categories, ordered=False)
    # Convert Inf to NaN
    data.loc[
        np.isinf(data[response]),
        response
    ] = np.nan
    return data


def set_ax_size(width, height, fig=None, ax=None, aspect=False):
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    l, b, w, h = bounds = AxesDivider(ax).get_position()
    r = l + w
    t = b + h

    # Right (top) padding, fixed axes size, left (bottom) padding
    hori = [Size.Scaled(l), Size.Fixed(width), Size.Scaled(r)]
    vert = [Size.Scaled(t), Size.Fixed(height), Size.Scaled(b)]

    divider = Divider(fig, bounds, hori, vert, aspect=aspect)

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
    return ax


def compute_difference(
    data,
    col,
    a="natural",
    b="artificial",
    op="mean",
    groups=("stimulus type", "session"),
):
    nvalues = data[col].notnull().sum()
    print(f"Averaging over {nvalues} values")
    g = data.groupby(list(groups))[col]
    if op == "max":
        agg = g.max()
    elif op == "mean":
        agg = g.mean()
    else:
        raise ValueError("invalid aggregation operation")
    diffs = agg[a] - agg[b]
    return (
        pd.DataFrame(
            {
                "difference": diffs,
                a: agg[a],
                b: agg[b],
                "aggregation": op,
            }
        )
        .reset_index()
        .merge(METADATA, left_on="session", right_index=True)
    )


def boxplot(
    data,
    x=None,
    y=None,
    width=3,
    height=5,
    hue=None,
    hline=None,
    xrotation=0,
    color="#333",
    box=True,
    box_kwargs=None,
    swarm=True,
    swarm_kwargs=None,
    mean=False,
    mean_kwargs=None,
):
    if x is not None:
        data = data.sort_values(x)
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    if hline is not None:
        ax.axhline(y=hline, color="#999")
    if box:
        if box_kwargs is None:
            box_kwargs = {}
        box_kwargs = {
            "width": 0.4,
            "fliersize": 8,
            "color": "lightgrey" if swarm else color,
            **box_kwargs,
        }
        ax = sb.boxplot(x=x, y=y, data=data, ax=ax, **box_kwargs)
    if swarm:
        if swarm_kwargs is None:
            swarm_kwargs = {
                "s": 6,
            }
        swarm_kwargs = {"color": color, **swarm_kwargs}
        ax = sb.swarmplot(x=x, y=y, hue=hue, data=data, ax=ax, **swarm_kwargs)
    if mean:
        if mean_kwargs is None:
            mean_kwargs = {}
        ax = sb.pointplot(x=x, y=y, data=data, ax=ax, **mean_kwargs)
    if xrotation > 0:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation, ha="right")
    fig.tight_layout()
    return fig, ax


def faceted_plot(
    data,
    x=None,
    y=None,
    row=None,
    col=None,
    height=2.5,
    aspect=0.6,
    hue=None,
    color="#333",
    box=False,
    box_kwargs=None,
    mean=True,
    mean_kwargs=None,
    hline=None,
    margin_titles=True,
    hspace=0.20,
    wspace=0.0,
    **kwargs,
):
    if x is not None:
        data = data.sort_values(x)
    grid = sb.catplot(
        kind="swarm",
        x=x,
        y=y,
        hue=hue,
        data=data,
        row=row,
        col=col,
        height=height,
        aspect=aspect,
        margin_titles=margin_titles,
        color=color,
        **kwargs,
    )
    for ax, (idx, subset) in zip(grid.axes.flat, grid.facet_data()):
        if box:
            if box_kwargs is None:
                box_kwargs = {}
            box_kwargs = {
                "width": 0.4,
                "fliersize": 100,
                "color": "lightgrey" if swarm else color,
                **box_kwargs,
            }
            sb.boxplot(data=subset, x=x, y=y, hue=hue, ax=ax, **box_kwargs)
        if mean:
            if mean_kwargs is None:
                mean_kwargs = {}
            mean_kwargs = {
                "color": "lightgrey",
                # "ci": "sd",
                "ci": None,
                "errcolor": "0.75",
                "errwidth": 1.5,
                # "capsize": 0.05,
                "capsize": None,
                "dodge": False,
                **mean_kwargs,
            }
            sb.barplot(data=subset, x=x, y=y, hue=hue, ax=ax, **mean_kwargs)
        if idx[1] != 0:
            ax.yaxis.label.set_visible(False)
        if hline is not None:
            ax.axhline(y=hline, color="#999")
        # Remove default titles so we can set them later
        plt.setp(ax.texts, text="")
        # Remove categorical ticks
        ax.set_xticks([])
    # All but leftmost column
    for ax in grid.axes[:, 1:].flat:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    # First column except first row
    for ax in grid.axes[1:, 0].flat:
        ax.set_ylabel("")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    grid.fig.tight_layout()
    grid.fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return grid.fig, grid.axes, grid


def faceted_barplot(
    data,
    x=None,
    y=None,
    row=None,
    col=None,
    hue=None,
    logy=False,
    height=3.5,
    aspect=1.7,
    hspace=0.20,
    wspace=0.10,
    legend=True,
    margin_titles=True,
    **kwargs,
):
    grid = sb.catplot(
        kind="bar",
        row=row,
        col=col,
        data=data,
        x=x,
        y=y,
        hue=hue,
        height=height,
        aspect=aspect,
        margin_titles=margin_titles,
        legend=False,
        **kwargs,
    )
    for ax, (idx, subset) in zip(grid.axes.flat, grid.facet_data()):
        if idx[1] != 0:
            ax.yaxis.label.set_visible(False)
        # Remove default titles so we can set them later
        plt.setp(ax.texts, text="")
        if logy:
            ax.set_yscale("log")
    # All but leftmost column
    for ax in grid.axes[:, 1:].flat:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    # First column except first row
    for ax in grid.axes[1:, 0].flat:
        ax.set_ylabel("")
    # Legend
    if legend:
        grid.axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.25, 1))
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    grid.fig.tight_layout()
    grid.fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return grid.fig, grid.axes, grid


def _pivot_and_sort(data, x, y, value, transpose=False):
    data = data.pivot(index=y, columns=x, values=value)
    data = data.sort_index().T.sort_index().T
    if transpose:
        data = data.T
    return data


def heatmap(data, x, y, value, transpose=False, mask=None, **kwargs):
    plot_data = _pivot_and_sort(data, x, y, value, transpose=transpose)
    if mask:
        mask = _pivot_and_sort(data, x, y, mask, transpose=transpose)
    kwargs = {
        **dict(
            annot=True,
            fmt=".3f",
            vmin=0,
            vmax=1,
            cmap=sb.color_palette("vlag", as_cmap=True),
        ),
        **kwargs,
    }
    ax = sb.heatmap(
        data=plot_data,
        square=True,
        mask=mask,
        **kwargs,
    )
    ax.set_xlabel(ax.get_xlabel().replace("_", " "))
    ax.set_ylabel(ax.get_ylabel().replace("_", " "))
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    return ax


def pooled_std_dev(values, labels, a, b):
    counts = labels.value_counts()
    if len(counts) == 1:
        return values.std()
    assert len(counts) <= 2, f"only implemented for 2 factor levels; got {len(counts)}"
    n1, n2 = counts[a], counts[b]
    s1, s2 = values.groupby(labels).std()
    return np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))


def mean_difference(values, labels, a, b):
    """Compute the mean difference between 'a' and 'b'."""
    if a == b:
        return 0.0
    mean = values.groupby(labels).mean()
    return mean[a] - mean[b]


def cohens_d(data, value_col, label_col, a, b):
    """Compute Cohen's d."""
    # Select the two factors
    data = data.loc[data[label_col].isin([a, b])]

    values = data[value_col]
    labels = data[label_col]

    mean_diff = mean_difference(values, labels, a, b)
    pooled_std = pooled_std_dev(values, labels, a, b)

    return pd.Series(
        {
            "mean difference": mean_diff,
            "Cohen's d": (mean_diff / pooled_std) if pooled_std != 0.0 else 0.0,
        }
    )


def pairwise_cohens_d(data, value_col, label_col):
    """Compute Cohen's d between all pairs of factor levels."""
    levels = data[label_col].unique()
    return pd.DataFrame(
        [
            {
                **{
                    "a": a,
                    "b": b,
                },
                **cohens_d(data, value_col, label_col, a, b),
            }
            for a, b in product(levels, repeat=2)
        ]
    )


def pairwise_cohens_d_by_layer_area(data, value_col, label_col, order):
    pairwise_differences = []

    for area, layer in product(
        METADATA["area"].cat.categories, METADATA["layer"].cat.categories
    ):
        subset = data.loc[(data["area"] == area) & (data["layer"] == layer)]
        differences = pairwise_cohens_d(subset, value_col, label_col)
        differences["area"] = area
        differences["layer"] = layer
        pairwise_differences.append(differences)

    pairwise_differences = pd.concat(pairwise_differences)

    # Ensure everything is plotted in order
    pairwise_differences["a"] = pd.Categorical(
        pairwise_differences["a"],
        categories=order,
        ordered=True,
    )
    pairwise_differences["b"] = pd.Categorical(
        pairwise_differences["b"],
        categories=order,
        ordered=True,
    )

    return pairwise_differences.reset_index(drop=True)


def differences_heatmap(data, value_col=None, **kwargs):
    if value_col is None:
        raise ValueError("must provide `value_col`")
    # Pivot to square form and pass to heatmap
    data = data.pivot("a", "b", value_col)
    return sb.heatmap(
        data=data,
        **kwargs,
    )


def layer_area_heatmap(data, value_col, vmin=None, vmax=None):
    grid = sb.FacetGrid(
        data=data,
        row="layer",
        col="area",
        margin_titles=True,
        sharex=True,
        sharey=True,
    )

    if vmax is None:
        vmax = data[value_col].abs().max()
    if vmin is None:
        vmin = -vmax

    cbar_ax = grid.fig.add_axes([1.025, 0.15, 0.02, 0.7])

    grid.map_dataframe(
        differences_heatmap,
        value_col=value_col,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        cmap=sb.color_palette("vlag", as_cmap=True),
        cbar_ax=cbar_ax,
    )

    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    grid.set_xticklabels(rotation=45, horizontalalignment="right")
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    return grid


def bin_data(data, bin_length, agg="first", column=TIME):
    """Bin session data.

    Arguments:
        data: Data to bin.
        bin_length: Bin length, in seconds.

    Keyword Arguments:
        agg: The aggregation to perform.

    NOTE: No checks are performed to ensure that the bins do not span
          different stimuli. Events should be grouped by stimulus first.
    """
    # Convert to ms
    bin_length = pd.tseries.offsets.Milli(int(1_000 * bin_length))
    # Resample and aggregate
    return data.resample(bin_length, on=TIME).agg(agg)


def active_frames(data):
    """Return a boolean index for frames with at least one event."""
    # Ensure we're only considering event columns
    data = data.loc[:, get_cells(data)]
    # Sum over cells
    data = data.sum(axis="columns")
    return data > 0


def cleanup_grid(grid, hspace=0.1, wspace=0.1):
    # All but leftmost column
    for ax in grid.axes[:, 1:].flat:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlabel("")
    # First column except first row
    for ax in grid.axes[1:, 0].flat:
        ax.set_ylabel("")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    grid.fig.tight_layout()
    grid.fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return grid


def binarize(data):
    return data > 0
