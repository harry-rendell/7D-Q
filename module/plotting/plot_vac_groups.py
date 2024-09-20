import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from module.config import cfg
import matplotlib.pyplot as plt
import matplotlib
from module.preprocessing.binning import calculate_groups
import numpy as np
import seaborn as sns

def plot_groups(x, bounds, plot=False, hist_kwargs={}, ax_kwargs={}):
    """
    Plot distribution of quasar property from VAC, and show groups.
    """
    groups, bounds_values = calculate_groups(x, bounds)
    for i in range(len(bounds)-1):
        print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],len(groups[i])))
        # print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],((bounds[i]<z_score)&(z_score<bounds[i+1])&(self.properties['mag_count']>2)).sum()))

    fig, ax = plt.subplots(1,1,figsize = (11,4.5))
    ax.hist(x, **hist_kwargs)
    for value, z in zip(bounds_values, bounds):
        ax.axvline(x=value, ymax=1, color = 'k', lw=0.5, ls='--')
        # ax.axvline(x=value, ymin=0.97, ymax=1, color = 'k', lw=0.5, ls='--') # If we prefer to have the numbers inside the plot, use two separate lines to make
        # a gap between text
        ax.text(x=value, y=1.01, s=r'${}\sigma$'.format(z), horizontalalignment='center', transform=ax.get_xaxis_transform())

    for i, value_centre in enumerate((bounds_values[1:] + bounds_values[:-1])/2):
        if i==0 or i==len(bounds)-2:
            color = 'k'
        else:
            color = 'w'
        ax.text(x=value_centre, y=0.2, s = f'$\mathit{{{i+1}}}$', horizontalalignment='center', transform=ax.get_xaxis_transform(), color=color, fontsize=18)

    ax.set(xlim=[bounds_values[0],bounds_values[-1]], **ax_kwargs)
    return fig

# def plot_groups_lambda_lbol(df, mask_dict, n_l=15, n_L=15, l_low=1000, l_high=5000, L_low=45.2, L_high=47.2):
#     # Create matplotlib polygon with vertices at the bin edges of Lbol_edges and lambda_edges
#     from matplotlib.collections import PatchCollection
#     from matplotlib.patches import Rectangle

#     lambda_edges = np.linspace(l_low, l_high, n_l)
#     Lbol_edges   = np.linspace(L_low, L_high, n_L)

#     # # create a series of 2d bins from the edges
#     # Lbol_bins = pd.cut(sigs['Lbol'], Lbol_edges, labels=False)
#     # lambda_bins = pd.cut(sigs['wavelength'], lambda_edges, labels=False)

#     # # masks = [(Lbol_bins == L).values & (lambda_bins == l).values for l,L in itertools.product(range(n-1), range(n-1))]
#     # masks_full = {(l,L):(Lbol_bins == L).values & (lambda_bins == l).values for l,L in itertools.product(range(n-1), range(n-1))}
#     # masks = {key:value for key,value in masks_full.items() if (value.sum() > 250) and (key[0] % 3 == 0) and (key[1] % 3 == 0)}

#     fig, ax = plt.subplots(1,1, figsize=(10,10))
#     binrange = np.array([[l_low-100, l_high+100], [L_low-0.5, L_high+0.5]])
#     sns.histplot(data=df.reset_index(), x='wavelength',y='Lbol', bins=100, cmap='Spectral_r', binrange=binrange, ax=ax)

#     vertices = [[lambda_edges[i], Lbol_edges[j]] for i,j in mask_dict.keys()]
#     squares = [Rectangle(vertex, width=(l_high-l_low)/(n_l-1), height=(L_high-L_low)/(n_L-1)) for vertex in vertices]
#     # add text to each square, showing the i, j indices

#     p = PatchCollection(squares, alpha=1, lw=2, ec='k', fc='none')
#     ax.add_collection(p)
#     ax.set(xlim=binrange[0], ylim=binrange[1], xlabel='wavelength', ylabel='Lbol')


#     for i, j in mask_dict.keys():
#         ax.text(lambda_edges[i]+0.5*(l_high-l_low)/(n_l-1), Lbol_edges[j]+0.5*(L_high-L_low)/(n_L-1), s=f'({i},{j})', ha='center', va='center', fontsize=8)

#     return fig

def plot_groups_lambda_prop(df, property_name, mask_dict, n_l=15, n_p=15, l_low=1000, l_high=5000, p_low=45.2, p_high=47.2, **kwargs):
    # Create matplotlib polygon with vertices at the bin edges of Lbol_edges and lambda_edges
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    lambda_edges = np.linspace(l_low, l_high, n_l)
    p_edges   = np.linspace(p_low, p_high, n_p)

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    binrange = np.array([[l_low-100, l_high+100], [p_low-0.5, p_high+0.5]])
    sns.histplot(data=df.reset_index(), x='wavelength',y=property_name, bins=100, cmap='Spectral_r', binrange=binrange, ax=ax)

    vertices = [[lambda_edges[i], p_edges[j]] for i,j in mask_dict.keys()]
    squares = [Rectangle(vertex, width=(l_high-l_low)/(n_l-1), height=(p_high-p_low)/(n_p-1)) for vertex in vertices]
    # add text to each square, showing the i, j indices

    p = PatchCollection(squares, alpha=1, lw=2, ec='k', fc='none')
    ax.add_collection(p)
    ax.set(xlim=binrange[0], ylim=binrange[1], xlabel='wavelength (Å)', ylabel=cfg.FIG.LABELS.PROP[property_name])
    ax.set(**kwargs)

    for i, j in mask_dict.keys():
        ax.text(lambda_edges[i]+0.5*(l_high-l_low)/(n_l-1), p_edges[j]+0.5*(p_high-p_low)/(n_p-1), s=f'({i},{j})', ha='center', va='center', fontsize=8)

    return fig

def plot_groups_lambda_prop_hardcoded(df, property_name, mask_dict, n_l=15, n_p=15, l_low=1000, l_high=5000, p_low=45.2, p_high=47.2, **kwargs):
    # Create matplotlib polygon with vertices at the bin edges of Lbol_edges and lambda_edges
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    lambda_edges = np.linspace(l_low, l_high, n_l)
    p_edges   = np.linspace(p_low, p_high, n_p)

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    if property_name == 'Lbol':
        binrange = np.array([[1000-100, 5000+100], [44.2, 47.7]])
    elif property_name == 'MBH':
        binrange = np.array([[1000-100, 5000+100], [7, 10.5]])
    elif property_name == 'nEdd':
        binrange = np.array([[1000-100, 5000+100], [-2.5, 0.5]])
    sns.histplot(data=df.reset_index(), x='wavelength',y=property_name, bins=100, cmap='Spectral_r', binrange=binrange, ax=ax)

    vertices = [[lambda_edges[i], p_edges[j]] for i,j in mask_dict.keys()]
    squares = [Rectangle(vertex, width=(l_high-l_low)/(n_l-1), height=(p_high-p_low)/(n_p-1)) for vertex in vertices]
    # add text to each square, showing the i, j indices

    p = PatchCollection(squares, alpha=1, lw=2, ec='k', fc='none')
    ax.add_collection(p)
    ax.set(xlim=binrange[0], ylim=binrange[1], xlabel='wavelength (Å)', ylabel=cfg.FIG.LABELS.PROP[property_name])
    ax.set(**kwargs)

    for i, j in mask_dict.keys():
        # ax.text(lambda_edges[i]+0.5*(l_high-l_low)/(n_l-1), p_edges[j]+0.5*(p_high-p_low)/(n_p-1), s=f'({i},{j})', ha='center', va='center', fontsize=8)
        ax.text(lambda_edges[i]+0.5*(l_high-l_low)/(n_l-1), p_edges[j]+0.5*(p_high-p_low)/(n_p-1), s=f'{j}', ha='center', va='center', fontsize=12)


    return fig