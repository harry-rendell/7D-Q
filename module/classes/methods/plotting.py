import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from module.config import cfg
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------

def plot_sf_moments_pm(self, key, bounds, save = False, t_max=3011, ztf=False):
    """
    Plots both positive and negative structure functions to investigate asymmetry in light curve.

    Parameters
    ----------
    key : str
            Property from VAC
    bounds : array of z scores to use


    Returns
    -------
    fig, ax, fig2, axes2, fig3, axes3: axes handles
    """
    fig, ax = plt.subplots(1,1,figsize = (16,8))
    fig2, axes2 = plt.subplots(2,1,figsize=(16,10))
    fig3, axes3 = plt.subplots(8,1,figsize = (16,50))
    label_range = {i:'{:.1f} < z < {:.1f}'.format(bounds[i],bounds[i+1]) for i in range(len(bounds)-1)}
    label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}
    # label_moment = ['mean', 'std', 'skew_stand', 'Excess kurtosis']
    label_moment = ['mean', 'Excess kurtosis']
    cmap = plt.cm.jet
    for i in range(8):
        dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.bin_dtdm(t_max = t_max, t_spacing='log', m_spacing='log', read_in=i+1, key=key, ztf=ztf)
        SF_n = (((m_bin_centres[:100]**2)*dms_binned[:,:100]).sum(axis=1)/dms_binned[:,:100].sum(axis=1))**0.5
        SF_p = (((m_bin_centres[100:]**2)*dms_binned[:,100:]).sum(axis=1)/dms_binned[:,100:].sum(axis=1))**0.5
        ax.plot(t_bin_chunk_centres, SF_p, label = label_range_val[i], lw = 0.5, marker = 'o', ls='-',  color = cmap(i/10))
        ax.plot(t_bin_chunk_centres, SF_n, label = label_range_val[i], lw = 0.5, marker = 'o', ls='--', color = cmap(i/10))
        ax.legend()
        ax.set(yscale='log', xscale='log')

        axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned[:,:100].sum(axis=1), alpha = 0.5, label = '-ve',bins = 19)
        axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned[:,100:].sum(axis=1), alpha = 0.5, label = '+ve',bins = 19)
        axes3[i].set(yscale='log')
        dms_binned_norm = np.zeros((19,200))
        moments = np.zeros(19)
        for j in range(19):
            dms_binned_norm[j],_= np.histogram(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[j], density=True);
            # print('number of -ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,:100].sum()))
            # print('number of +ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,100:].sum()))
        moments = calc_moments(m_bin_centres,dms_binned_norm)


        for idx, ax2 in enumerate(axes2.ravel()):
            ax2.plot(t_bin_chunk_centres, moments[idx], lw = 0.5, marker = 'o', label = label_range_val[i], color = cmap(i/10.0));
            # ax2.legend()
            ax2.set(xlabel='mjd', ylabel = label_moment[idx])
            ax2.axhline(y=0, lw=0.5, ls = '--')

    ax.set(xlabel='mjd', ylabel = 'structure function')
    if save:
        # fig.savefig('SF_{}.pdf'.format(key),bbox_inches='tight')
        fig2.savefig('moments_{}.pdf'.format(key),bbox_inches='tight')

    return fig, ax, fig2, axes2, fig3, axes3

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------

def plot_sf_moments(self, key, bounds, save = False, t_max=3011, ztf=False):
    fig, ax = plt.subplots(1,1,figsize = (16,8))
    fig2, axes2 = plt.subplots(2,1,figsize=(16,10))
    fig3, axes3 = plt.subplots(8,1,figsize = (16,50))
    label_range = {i:'{:.1f} < z < {:.1f}'.format(bounds[i],bounds[i+1]) for i in range(len(bounds)-1)}
    label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}
    # label_moment = ['mean', 'std', 'skew_stand', 'Excess kurtosis']
    label_moment = ['mean', 'Excess kurtosis']
    cmap = plt.cm.jet
    for i in range(8):
        dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.bin_dtdm(t_max = t_max, t_spacing='log', m_spacing='log', read_in=i+1, key=key, ztf=ztf)
        SF = (((m_bin_centres**2)*dms_binned).sum(axis=1)/dms_binned.sum(axis=1))**0.5
        ax.plot(t_bin_chunk_centres, SF, label = label_range_val[i], lw = 0.5, marker = 'o', ls='-',  color = cmap(i/10))
        ax.legend()
        ax.set(yscale='log', xscale='log')

        axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned.sum(axis=1), alpha = 0.5,bins = 19)
        axes3[i].set(yscale='log')
        dms_binned_norm = np.zeros((19,200))
        moments = np.zeros(19)
        for j in range(19):
            dms_binned_norm[j],_= np.histogram(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[j], density=True);
            # print('number of -ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,:100].sum()))
            # print('number of +ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,100:].sum()))
        moments = calc_moments(m_bin_centres,dms_binned_norm)


        for idx, ax2 in enumerate(axes2.ravel()):
            ax2.plot(t_bin_chunk_centres, moments[idx], lw = 0.5, marker = 'o', label = label_range_val[i], color = cmap(i/10.0));
            # ax2.legend()
            ax2.set(xlabel='mjd', ylabel = label_moment[idx])
            ax2.axhline(y=0, lw=0.5, ls = '--')

            # ax2.title.set_text(label_moment[idx])
    ax.set(xlabel='mjd', ylabel = 'structure function')
    if save == True:
        # fig.savefig('SF_{}.pdf'.format(key),bbox_inches='tight')
        fig2.savefig('moments_{}.pdf'.format(key),bbox_inches='tight')

    return fig, ax, fig2, axes2, fig3, axes3

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------

def plot_sf_ensemble(self, save = False):
    fig, ax = plt.subplots(1,1,figsize = (16,8))
    dms_binned_tot = np.zeros((8,19,200))
    for i in range(8):
        dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.bin_dtdm(t_max = 3011, t_spacing='log', m_spacing='log', read_in=i+1, key=key)
        dms_binned_tot[i] = dms_binned

    dms_binned_tot = dms_binned_tot.sum(axis=0)

    SF = (((m_bin_centres**2)*dms_binned_tot).sum(axis=1)/dms_binned_tot.sum(axis=1))**0.5
    ax.plot(t_bin_chunk_centres,SF, lw = 0.5, marker = 'o')
    ax.set(yscale='log',xscale='log')
    ax.set(xlabel='mjd',ylabel = 'structure function')
    if save == True:
        fig.savefig('SF_ensemble.pdf',bbox_inches='tight')

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------

def plot_series(df, uids, marker_dict={}, survey_dict={}, plt_color={}, sid=None, bands='r', show_outliers=False, figax=None, **kwargs):
    """
    Plot lightcurve of given objects

    Parameters
    ----------
    uids : array_like
            uids of objects to plot
    sid : int
            Only plot data from given survey
            1 = SSS_r1, 3 = SSS_r2, 5 = SDSS, 7 = PS1, 11 = ZTF
    """
    if np.issubdtype(type(uids),np.integer): uids = [uids]
    if figax is None:
        fig, axes = plt.subplots(len(uids),1,figsize = (25,3*len(uids)), sharex=True)
    else:
        fig, axes = figax
    if len(uids)==1:
        axes=[axes]
    for uid, ax in zip(uids,axes):
        single_obj = df.loc[uid].sort_values('mjd')
        for band in bands:
            single_band = single_obj[single_obj['band']==band]
            if sid is not None:
                # Restrict data to a single survey
                single_band = single_band[single_band['sid'].isin(sid)]
            for sid_ in single_band['sid'].unique():
                x = single_band[single_band['sid']==sid_]
                ax.errorbar(x['mjd'], x['mag'], yerr = x['magerr'], lw = 0.5, markersize = 3, marker = marker_dict[sid_], label = survey_dict[sid_]+' '+band, color = plt_color[band])
        
        if show_outliers:
            outlier_mask = single_obj['outlier'].values & single_obj['sid'].isin(sid)
            for band in bands:
                mask = single_obj['band']==band
                ax.scatter(single_obj['mjd'][outlier_mask & mask], single_obj['mag'][outlier_mask & mask], color = plt_color[band], marker="*", zorder=3, s=200)#, edgecolor='k', linewidths=1)

        # else:
        #     # Plot a single band
        #     if sid is not None:
        #         # Restrict data to a single survey
        #         single_obj = single_obj[single_obj['sid']==sid]
        #     for sid_ in single_obj['sid'].unique():
        #         x = single_obj[single_obj['sid']==sid_]
        #         ax.errorbar(x['mjd'], x['mag'], yerr = x['magerr'], lw = 0, elinewidth=0.7, marker = self.marker_dict[sid_], label = self.survey_dict[sid_]+' '+filtercodes, color = self.plt_color[filtercodes])
        #     mean = single_obj['mag'].mean()
        #     ax.axhline(y=mean, color='k', ls='--', lw=0.4, dashes=(50, 20))

            


            # """
            # requires having computed MAD previously
            # """
            # mjd, mag, MAD = single_obj.loc[single_obj['MAD']>0.25, ['mjd','mag', 'MAD']].values.T
            # ax.scatter(mjd, mag, s=100)
            # string = ', '.join(['{:.3f}' for _ in MAD]).format(*MAD)
            # # ax.text(0.02, 0.8, 'MAD max: {:.2f}'.format(np.max(MAD)), transform=ax.transAxes, fontsize=10)
            # ax.text(0.02, 0.8, string, transform=ax.transAxes, fontsize=10)
            # mu, std = self.df_grouped.loc[uid,['mag_mean','mag_std']].values.T
            # axis.axhline((mu-5*std),lw=0.5)
            # axis.axhline((mu+5*std),lw=0.5)
        # ax2 = ax.twinx()
        # ax2.set(ylim=np.array(ax.get_ylim())-mean, ylabel=r'$\mathrm{mag} - \overline{\mathrm{mag}}$')
        # ax2.invert_yaxis()

        ax.invert_yaxis()
        ax.set(xlabel='MJD', ylabel='mag', **kwargs)
        ax.text(0.02, 0.9, 'uid: {}'.format(uid), transform=ax.transAxes, fontsize=10)
    
    plt.subplots_adjust(hspace=0)
    
    return fig, axes

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------

def plot_series_bokeh(self, uids, survey=None, filtercodes=None):
    """
    Plot lightcurve of given objects using bokeh

    Parameters
    ----------
    uids : array_like
            uids of objects to plot
    catalogue : int
            Only plot data from given survey
    survey : 1 = SDSS, 2 = PS, 3 = ZTF

    """

    plots = []
    for uid in uids:
        single_obj = self.df.loc[uid]
        if survey is not None:
            single_obj = single_obj[single_obj['catalogue']==survey]
        p = figure(title='uid: {}'.format(uid), x_axis_label='mjd', y_axis_label='r mag', plot_width=1000, plot_height=400)
        for cat in single_obj['catalogue'].unique():
            mjd, mag = single_obj[single_obj['catalogue']==cat][['mjd','mag']].sort_values('mjd').values.T
            p.scatter(x=mjd, y=mag, legend_label=self.survey_dict[cat], marker=self.marker_dict_bokeh[cat], color=self.plt_color_bokeh[filtercodes])
            p.line   (x=mjd, y=mag, line_width=0.5, color=self.plt_color_bokeh[filtercodes])
        p.y_range.flipped = True
        plots.append(p)

    show(column(plots))

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------

def plot_property_distributions(self, prop_range_dict, n_width, n_bins = 250, separate_catalogues = True):
    """
    Parameters
    ----------
    prop_range_dict : dict
            dictionary of keys and their ranges to be used in histogram
    n_width : int
            width of subplot
    n_bins : int
            number of bins to use in histogram
    separate_catalogues : boolean
            if True, plot data from each survey separately.
    """
    m = -( -len(prop_range_dict) // n_width )
    fig, axes = plt.subplots(m, n_width,  figsize = (5*n_width,5*m))
    cat_label_dict = {1:'SDSS', 2:'PanSTARRS', 3:'ZTF'}
    for property_name, ax, in zip(prop_range_dict, axes.ravel()):
        if separate_catalogues == True:
            for cat, color in zip(self.cat_list,'krb'):
                self.properties[self.properties['catalogue']==cat][property_name].hist(bins = n_bins, ax=ax, alpha = 0.3, color = color, label = cat_label_dict[cat], range=prop_range_dict[property_name]);
            ax.legend()
        elif separate_catalogues == False:
            self.properties[property_name].hist(bins = 250, ax=ax, alpha = 0.3, range=prop_range_dict[property_name]);
        else:
            print('Error, seperate_catalogues must be boolean')
        ax.set(title = property_name)