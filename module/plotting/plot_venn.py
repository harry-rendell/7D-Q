import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from module.config import cfg
from module.plotting.common import savefigs
import matplotlib.pyplot as plt
from matplotlib_venn import venn3_unweighted, venn3_circles
import numpy as np
from venn import venn, pseudovenn

def plot_venn3(data, surveys=['sdss','ps','ztf'], save=False):
    """
    Plot a 3-way venn diagram of the survey data using matplotlib_venn
    """
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    total = np.any([data[s].values for s in surveys], axis=0).sum()

    v1 = venn3_unweighted(
        [set(data.index[data[s].values]) for s in surveys],
        set_labels=[s.upper() for s in surveys],
        set_colors=[cfg.FIG.COLORS.SURVEYS[s] for s in surveys],
        subset_label_formatter=lambda x: f"{(x/total):1.0%}",
        ax=ax,
        alpha=0.8
    )

    venn3_circles([1]*7, ax=ax, lw=0.5)
    if save:
        savefigs(fig, 'SURVEY-DATA-venn3_diagram', 'chap2')

def plot_venn4(data, surveys, save=False):
    """
    Plot a 4-way ellipse venn diagram of the survey data using pyvenn
    """
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    data = {s:set(data.index[data[s].values]) for s in surveys}

    ax = venn(data,
            cmap=[cfg.FIG.COLORS.SURVEYS[s] for s in surveys],
            alpha=0.4,
            fontsize=12,
            legend_loc="upper left",
            fmt="{size:,}",
            ax=ax)

    w1 = 0.2
    w2 = 0.4
    kwargs = {'fontsize':15,
              'ha':'center',
              'va':'center',
              'fontfamily':'serif'}

    ax.text(0.5-w2, 0.24, s=surveys[0].replace('a','s').upper(), **kwargs) # Hack to relabel SSA with SSS
    ax.text(0.5-w1, 0.82, s=surveys[1].upper(), **kwargs)
    ax.text(0.5+w1, 0.82, s=surveys[2].upper(), **kwargs)
    ax.text(0.5+w2, 0.24, s=surveys[3].upper(), **kwargs)
    ax.get_legend().remove()
    if save:
        savefigs(fig, 'SURVEY-DATA-venn4_diagram_counts', 'chap2')
