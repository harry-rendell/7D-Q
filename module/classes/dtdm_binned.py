import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg
import matplotlib.pyplot as plt
from module.preprocessing.binning import bin_data
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.stats import linregress, chisquare
from module.preprocessing.binning import bin_data
from sklearn.mixture import GaussianMixture
from itertools import combinations_with_replacement

class dtdm_binned_class():
    """
    Class for processing and analysing dtdm files

    Parameters
    ----------
    obj : str
        qsos or calibStars
    name : str 
        name describing the survey pairs used to create the binned data. Eg sdss_ps or all
    label : str
        label to be used for plots
    n_bins_t : int
    n_bins_m : int
    n_bins_m2 : int
    t_max : int
    n_t_chunk : int
    steepness : float
    width : 
    leftmost_bin :
    verbose : bool
        If true, then print the number of dtdm counts in each ∆t bin
    subset :
    
    Notes from bin_data.calc_m_edges
    Produce a set of bins with max extent 'width'.
    Note that by keeping the steepness the same, the bin spacing is linear with width.
    if leftmost_bin is not None, then first bin will begin from leftmost_bin. This is used for dm2_de2 which has an asymmetrical distribution about zero. 
    qsos:  bins=248 for dm2_de2, leftmost_bin = -0.244
    stars: bins=235 for dm2_de2, leftmost_bin = -0.21

    """ 
    def __init__(self, obj, band, name, label, subset='', verbose=False):
        self.obj = obj
        self.band = band
        self.name = name
        self.label = label
        self.subset = subset
        self.read()
        self.stats(verbose)

    def read(self):
        """
        Read in binned data
        """

        with np.load(cfg.D_DIR + f'computed/{self.obj}/dtdm_binned/{self.subset}/binned_{self.band}.npz', allow_pickle=True, mmap_mode='r') as data:
            self.dts_binned = data['dts_binned']
            self.dms_binned = data['dms_binned']
            self.des_binned = data['des_binned']
            self.dcs_binned = data['dsids_binned']
            self.dm2_de2_binned = data['dm2_de2_binned']
            bin_dict = data['bin_dict'].item()

        def centres(edges):
            return (edges[1:] + edges[:-1])/2
        def widths(edges):
            return edges[1:] - edges[:-1]

        self.t_bin_edges = bin_dict['t_bin_edges']
        self.T_bin_edges = bin_dict['T_bin_edges']
        self.T_bin_centres = centres(self.T_bin_edges)
        self.n_bins_T = len(self.T_bin_centres)
        self.m_bin_edges = bin_dict['m_bin_edges']
        self.m_bin_centres = centres(self.m_bin_edges)
        self.m_bin_widths = widths(self.m_bin_edges)
        self.e_bin_edges = bin_dict['e_bin_edges']
        self.t_dict = bin_dict['T_dict']
        self.t_dict_latex = {key:'$'+string.replace('∆',r'\Delta ')+'$' for key, string in self.t_dict.items()}        
        self.m2_bin_edges = bin_dict['m2_bin_edges']
        self.m2_bin_widths = widths(self.m2_bin_edges)
        self.m2_bin_centres = centres(self.m2_bin_edges)
        self.width = bin_dict['width']

        self.t_dict_latex = {}
        for key, string in self.t_dict.items():
            split_label = string.replace('∆', r'\Delta ').split('<')
            # self.t_dict_latex[key] = f"${split_label[0]}\,\mathrm{{days}}<{split_label[1]}<{split_label[2]}\,\mathrm{{days}}$" # option 1, put days after each number
            self.t_dict_latex[key] = f"${split_label[0]}<{split_label[1]}<{split_label[2]}$ [days]" # option 2, [days] at the end

        # temporary hack to reformat dictionary.
        # self.t_dict = {key:f'{float(value.split("<t<")[0]):.0f}<∆t<{float(value.split("<t<")[1]):.0f}' for key, value in self.t_dict.items()}

    def plot_test(self):
        # text self.t_dict_latex
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        ax.set(xlabel=self.t_dict_latex[0], ylabel=self.t_dict_latex[1])

    def stats(self, verbose=False):
        if verbose:
            for i in range(self.n_bins_T):
                print('dtdm counts in {}: {:,}'.format(self.t_dict[i],self.dts_binned.sum(axis=1)[i]))

        N_dm = self.dms_binned.sum(axis=1)
        x   = self.m_bin_centres*self.dms_binned
        mean = x.sum(axis=1)/N_dm

        self.means = mean
        self.modes = self.m_bin_centres[ (self.dms_binned/self.m_bin_widths).argmax(axis=1) ]
        self.modes = np.where(abs(self.modes)>1, np.nan, self.modes)

        self.skew_1  = skew(x, axis=1)
        self.skew_2  = N_dm**0.5 * ((x-mean[:,np.newaxis])**3).sum(axis=1) * ( ((x-mean[:,np.newaxis])**2).sum(axis=1) )**-1.5

    def plot_means(self, ax, ls='-'):
        ax.errorbar(self.T_bin_centres, self.means, yerr=self.means*self.dms_binned.sum(axis=1)**-0.5, lw=0.5, marker='o', label=self.label)
         # ax.scatter(self.T_bin_centres, self.means, s=30, label=self.name, ls=ls)
         # ax.plot   (self.T_bin_centres, self.means, lw=1.5, ls=ls)

    def plot_modes(self, ax, ls='-'):
        ax.errorbar(self.T_bin_centres, self.modes, yerr=self.means*self.dms_binned.sum(axis=1)**-0.5, lw=0.5, marker='o', label=self.label)
         # ax.scatter(self.T_bin_centres, self.modes, s=30, label=self.name, ls=ls)
         # ax.plot   (self.T_bin_centres, self.modes, lw=1.5, ls=ls)

    def plot_sf_ensemble(self, figax=None):
        if figax is None:
            fig, ax = plt.subplots(1,1, figsize=(15,8))
        else:
            fig, ax = figax
        SF = (((self.m_bin_centres**2)*self.dms_binned).sum(axis=1)/self.dms_binned.sum(axis=1))**0.5
        SF[self.dms_binned.sum(axis=1)**-0.5 > 0.1] = np.nan # remove unphysically large SF values
        # Check errors below a right
        ax.errorbar(self.T_bin_centres, SF, yerr=self.dms_binned.sum(axis=1)**-0.5*self.means, lw = 0.5, marker = 'o', label = self.label)
        ax.set(yscale='log',xscale='log', xticks=[100,1000])
        ax.set(xlabel='∆t',ylabel = 'structure function')

        return figax, SF
    
    def plot_sf_ensemble_corrected(self, figax=None):
        if figax is None:
            fig, ax = plt.subplots(1,1, figsize=(15,8))
        else:
            fig, ax = figax
        SF = (((self.m_bin_centres**2)*self.dms_binned).sum(axis=1)/self.dms_binned.sum(axis=1))**0.5
        
        norm_cumsum_phot_err = self.des_binned.cumsum(axis=1)/self.des_binned.sum(axis=1)[:,np.newaxis]
        median_idx   = np.abs(norm_cumsum_phot_err-0.5).argmin(axis=1)
        median_phot_err = self.e_bin_edges[median_idx]
        
        SF = np.sqrt(SF**2 - 2*median_phot_err**2)
        SF[self.dms_binned.sum(axis=1)**-0.5 > 0.1] = np.nan # remove unphysically large SF values
        # Check errors below a right
        ax.errorbar(self.T_bin_centres, SF, yerr=self.dms_binned.sum(axis=1)**-0.5*self.means, lw = 0.5, marker = 'o', label = self.label)
        ax.set(yscale='log',xscale='log', xticks=[100,1000])
        ax.set(xlabel='∆t',ylabel = 'structure function')

        return figax, SF

    def plot_sf_ensemble_iqr(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(15,8))
        norm_cumsum = self.dms_binned.cumsum(axis=1)/self.dms_binned.sum(axis=1)[:,np.newaxis]
        lq_idxs  = np.abs(norm_cumsum-0.25).argmin(axis=1)
        uq_idxs  = np.abs(norm_cumsum-0.75).argmin(axis=1)
        SF = 0.74 * ( self.m_bin_centres[uq_idxs]-self.m_bin_centres[lq_idxs] ) * (( self.dms_binned.sum(axis=1) - 1 ) ** -0.5)
        SF[SF == 0] = np.nan

        ax.errorbar(self.T_bin_centres, SF, yerr=self.dms_binned.sum(axis=1)**-0.5*self.means, lw = 0.5, marker = 'o', label = self.label)
        ax.set(xticks=[100,1000], **kwargs)
        ax.set(xlabel='∆t',ylabel = 'structure function')
        return ax, SF
    
    def plot_sf_ensemble_asym(self, figax=None, color='b'):
        
        if figax is None:
            fig, ax = plt.subplots(1,1, figsize=(15,8))
        else:
            fig, ax = figax

        SF_n = (((self.m_bin_centres[:100]**2)*self.dms_binned[:,:100]).sum(axis=1)/self.dms_binned[:,:100].sum(axis=1))**0.5
        SF_p = (((self.m_bin_centres[100:]**2)*self.dms_binned[:,100:]).sum(axis=1)/self.dms_binned[:,100:].sum(axis=1))**0.5
        SF_n[self.dms_binned[:,:100].sum(axis=1)**-0.5 > 0.1] = np.nan
        SF_p[self.dms_binned[:,100:].sum(axis=1)**-0.5 > 0.1] = np.nan
        
        # Check errors below are right
        ax.errorbar(self.T_bin_centres, SF_n, yerr=self.dms_binned[:, :100].sum(axis=1)**-0.5*self.means, ls='--', color=color, lw = 0.5, marker = 'o', label = self.label + ' negative')
        ax.errorbar(self.T_bin_centres, SF_p, yerr=self.dms_binned[:, 100:].sum(axis=1)**-0.5*self.means, ls='-', color=color, lw = 0.5, marker = 'o', label = self.label + ' positive') 

        ax.set(yscale='log',xscale='log', xticks=[100,1000])
        ax.set(xlabel='∆t',ylabel = 'structure function')

        return figax, SF_n, SF_p
    
    def hist_dm(self, window_width, figax=None, 
                overlay_gaussian=False, overlay_lorentzian=False, overlay_exponential=False, overlay_gmm={}, overlay_diff=False, 
                cmap=plt.cm.cool, colors=['r','b','g','m'], alpha=1, n_skip=1, start=0, n_col=2, legend_height=10.65, verbose=False, scale=1):
        """
        Plot histograms of ∆m for each ∆t bin

        Parameters
        ----------
        window_width : float
            width of histogram
        figax : tuple
            tuple of fig, ax to overlay histograms on
        overlay_gaussian : bool
            fit a gaussian to the histogram
        overlay_lorentzian : bool
            fit a lorentzian to the histogram
        overlay_exponential : bool
            fit an exponential to the histogram
        overlay_gmm : dict
            dictionary of Gaussian Mixture Model parameters. Leave empty to omit GMM fit
            required keys:
                n_components : int
                overlay_components : bool
        overlay_diff : bool
            overlay the difference between the histogram and the fit

        ...
        """

        def gaussian(x,peak,x0):
            sigma = (2*np.pi)**-0.5*1/peak
            return peak*np.exp( -( (x-x0)**2/(2*sigma**2) ) )
        def lorentzian(x,gam,x0):
            return gam / ( gam**2 + ( x - x0 )**2) * 1/np.pi
        def exponential(x,peak,x0,exponent):
            return peak*np.exp(-np.abs(x-x0)*exponent)
        def normal_(x, sigma, mean):
            return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x-mean)**2 / (2 * sigma**2))
        def double_bin_size(edges, counts, n):
            """
            Not convinced this works properly
            """
            if n == 1:
                return edges, counts
            if n >= 8:
                n = 8
            # Double the size of bin edges and update counts
            updated_bin_edges = edges[::n]
            updated_bin_counts = counts[:-1:n] + counts[1::n]
            # Add the last bin edge and count if the original data had an even
            if len(edges) % 2 == 0:
                updated_bin_edges = np.append(updated_bin_edges, edges[-1])
                updated_bin_counts = np.append(updated_bin_counts, counts[-1])
            return updated_bin_edges, updated_bin_counts
        
        N = self.n_bins_T//n_skip

        if figax is None:
            fig, axes = plt.subplots(N//n_col-start,n_col,figsize = (14*scale,scale*3.1*(N/n_col-start))) # only intended to be used for 1 or 2 columns
            plt.subplots_adjust(wspace=0.1)
        else:
            fig, axes = figax
        n=1
        # stds     = np.zeros(self.n_bins_T)
        r2s_tot = []
        self.gmms = []
        edges = self.m_bin_edges
        text_height = 0.6
        for i, ax in enumerate(axes.T.ravel()):
            
            # on final iteration, set j to the last index
            if i == len(axes.T.ravel())-1:
                j = int(self.n_bins_T-1)
            else:
                j = int(i*n_skip + start)

            print(f'plotting: {j+1}/{self.n_bins_T}')

            counts = self.dms_binned[j]
            if counts.sum() == 0:
                continue
            r2s = []
            ax.set_title(self.t_dict_latex[j], fontsize=12, pad=1)

            # edges, counts = double_bin_size(edges, counts, 2**(i//8))
            # 
            m,_,_= ax.hist(edges[:-1], edges, weights = counts, alpha = alpha, density=True, 
                           label = self.label, color = cmap(0.3+j/20*0.5), range=[-window_width,window_width])
            
            ax.set(xlim=[-window_width,window_width], yticks=[])
            ax.set_xlabel(r'$\Delta m$', fontsize=12)
            # ax.axvline(x=modes[i], lw=0.5, ls='--', color='k')
            # ax.axvline(x=m_bin_centres[m.argmax()], lw=0.5, ls='--', color='r')
            if overlay_gaussian:
                try:
                    #Also make sure that bins returned from .hist match m_bin_edges : it is
                    x = np.linspace(-2,2,1000)

                    popt, _ = curve_fit(gaussian, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()]])
                    # ax.plot(x,gaussian(x, m.max(), self.m_bin_edges[:-1:n][m.argmax()]), label = 'gaussian') what is this for
                    ax.plot(x,gaussian(x, *popt), color=colors[1], label = 'Gaussian', lw=1.85, alpha=0.75)

                    # popt, _ = curve_fit(gaussian, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()]], sigma = 1/self.m_bin_widths)
                    # ax.plot(x,gaussian(x, *popt), label = 'gaussian_weighted')
                    
                    slope, intercept, r, p, stderr = linregress(m, gaussian(self.m_bin_centres, *popt))
                    chisq, p = chisquare(m, gaussian(self.m_bin_centres, *popt), ddof=0)
                    diff    = m - gaussian(self.m_bin_centres, *popt)
                    r2_gaussian = 1 -  (diff**2).sum() / ( (m - m.mean())**2).sum()
                    r2s.append(r2_gaussian)
                    if overlay_diff:
                        ax.plot(self.m_bin_centres, diff, label = 'diff')
                    # ax.text(0.05, text_height, r'Gaussian     $r^2$ = {:.5f}'.format(r2_gaussian), transform=ax.transAxes)
                    # ax.text(0.05, text_height-0.1, r'Gaussian linreg $r^2$ = {:.5f}'.format(r**2), transform=ax.transAxes)
                    # ax.text(0.05, text_height-0.2, r'Gaussian $\chi^2$ = {:.5f}'.format(chisq/m.sum()), transform=ax.transAxes)
    ######################### something wrong with chi squared
                    text_height -= 0.1
                except Exception as e:
                    if verbose:
                        print('unable to fit gaussian due to error:',e)

            if overlay_lorentzian:
                try:
                    x = np.linspace(-2,2,1000)

                    popt, _ = curve_fit(lorentzian, self.m_bin_edges[:-1:n], m, p0 = [1/m.max(),self.m_bin_edges[:-1:n][m.argmax()]])
                    ax.plot(x,lorentzian(x,popt[0],popt[1]), color = colors[2], label = 'Lorentzian', lw=2)
                    
                    # popt, _ = curve_fit(lorentzian, self.m_bin_edges[:-1:n], m, p0 = [1/m.max(),self.m_bin_edges[:-1:n][m.argmax()]], sigma = 1/self.m_bin_widths)
                    # ax.plot(x,lorentzian(x,popt[0],popt[1]), label = 'lorentzian weighted')
                    
                    diff    = m - lorentzian(self.m_bin_centres, *popt)
                    r2_lorentzian = 1 -  (diff**2).sum() / ( (m - m.mean())**2).sum() 
                    r2s.append(r2_lorentzian)
                    if overlay_diff:
                        ax.plot(self.m_bin_centres, diff, label = 'diff')
                    # ax.text(0.05, text_height, r'Lorentzian  $r^2$ = {:.5f}'.format(r2_lorentzian), transform=ax.transAxes)
                    text_height -= 0.1
                except Exception as e:
                    if verbose:
                        print('unable to fit lorentzian due to error:',e)


            if overlay_exponential:
                try:
                    # Also make sure that bins returned from .hist match m_bin_edges : it is
                    x = np.linspace(-2,2,1000)

                    popt, _ = curve_fit(exponential, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()], 1])
                    # ax.plot(x,exponential(x, m.max(), self.m_bin_edges[:-1:n][m.argmax()]), label = 'exponential') what is this for
                    ax.plot(x,exponential(x, *popt), color=colors[3], label = 'Exponential', lw=1.85, alpha=0.75)

                    # popt, _ = curve_fit(exponential, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()], 1], sigma = 1/self.m_bin_widths)
                    # ax.plot(x,exponential(x, *popt), label = 'exponential_weighted')
                    
                    diff    = m - exponential(self.m_bin_centres, *popt)
                    r2_exponential = 1 -  (diff**2).sum() / ( (m - m.mean())**2).sum() 
                    r2s.append(r2_exponential)
                    if overlay_diff:
                        ax.plot(self.m_bin_centres, diff, label = 'diff')
                    # ax.text(0.05, text_height, r'Expoxwnential $r^2$ = {:.5f}'.format(r2_exponential), transform=ax.transAxes)
                    text_height -= 0.1
                except Exception as e:
                    if verbose:
                        print('unable to fit exponential due to error:',e)

            if overlay_gmm:
                x = np.linspace(-2,2,1000)
                n_components = overlay_gmm['n_components']

                for tol in [1e-5, 1e-6, 1e-7]:
                    for i, n_components_ in enumerate(range(n_components, n_components+3)):
                        try:
                            gmm = GaussianMixture(n_components_, covariance_type='full', means_init=np.zeros(shape=(n_components_, 1)), max_iter=1000, tol=tol)
                            
                            N = 10  # Reduce the number of repeats by a factor of N
                            
                            # TODO: sample uniformly instead of np.repeat then use pymc
                            a = np.repeat(self.m_bin_centres, np.round(counts / (np.round(counts.min(), -2) * N), 0).astype(int))
                            gmm.fit(a.reshape(-1, 1))

                            # TODO: we should really add gmms using a dictionary in case some of the fits fail
                            self.gmms.append(gmm)
                            
                            individual_pdfs = (gmm.weights_ * normal_(x[:, np.newaxis], gmm.covariances_.flatten()**0.5, gmm.means_.flatten()))
                            combined_pdfs = individual_pdfs.sum(axis=1)
                            ax.plot(x, combined_pdfs, label='GMM', color=overlay_gmm['color'], lw=2.3, alpha=0.7, ls=['-','--','-.'][i])

                            if overlay_gmm['overlay_components']:
                                for k_ in range(n_components):
                                    label = 'GMM components' if k_==0 else None
                                    ax.plot(x, individual_pdfs[:,k_], color='k', lw=0.6, ls=(0,(5,3)), label=label)

                            # Set success flag to True and break out of both loops
                            success = True
                            break
                        except Exception as e:
                            if verbose:
                                print('Unable to fit GMM with tolerance', tol, 'and n_components', n_components_, 'due to error:', e)
                    
                    if success:
                        break
                    

            r2s_tot.append(r2s)
            # ax.text(0.05, 0.9, 'mean = {:.5f}'.format(self.means[i]), transform=ax.transAxes)
            # ax.text(0.05, 0.8, 'mode = {:.5f}'.format(self.modes[i]), transform=ax.transAxes)
            # ax.text(0.05, 0.7, 'skew  = {:.5f}'.format(self.skew_1[i]), transform=ax.transAxes)

            # a = {f'{b[0]}-{b[1]}':a[0]*a[1] for a,b in zip(combinations_with_replacement([3,5,7,11],2),combinations_with_replacement(['SSS','SDSS','PS','ZTF'],2))}
            a = {f'{b[0]}-{b[1]}':a[0]*a[1] for a,b in zip(combinations_with_replacement([3,5,7,11],2),combinations_with_replacement(['sss','sdss','ps','ztf'],2))}
            survey_fractions = ''
            for name, index in a.items():
                frac = self.dcs_binned[j][index]/self.dcs_binned[j].sum()
                if frac > 0.005:
                    survey_fractions += f'{name}: {frac*100:.0f}%\n'

            if self.label is not None:
                # If label is not None then assume we are going to be overlaying distributions
                survey_fractions = self.label + ':\n' + survey_fractions
            if self.obj=='qsos':
                ax.text(0.02, 0.96, survey_fractions, transform=ax.transAxes, verticalalignment='top', usetex=False, fontsize=11)
            else:
                ax.text(0.98, 0.96, survey_fractions, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', usetex=False, fontsize=11)

            # ax.axvline(x=self.means[i], lw=0.4, ls='-', color='b')
            # ax.axvline(x=self.modes[i], lw=0.4, ls='-', color='r')
            # ax.axvline(x=self.m_bin_centres[m.argmax()], lw=0.6, ls='-', color='r')
            ax.axvline(x=0, lw=1, ls='--', color='k')
        # add title
        # fig.suptitle('Quasar and Star $\Delta m$ distributions', fontsize=16, y=0.925)
        fig.suptitle('Quasar and Star $\Delta m$ distributions', fontsize=16, y=0.925)
        # put legend outside of subplots
        plt.legend(loc='upper right', bbox_to_anchor=(1.01, legend_height))
        plt.subplots_adjust(hspace=0.5)

        return fig, axes, np.array(r2s_tot).T

    def plot_gmm_variances(self):
        n = len(self.gmms)
        fig, axes = plt.subplots(n,1, figsize=(10, n*2), sharex=True, sharey=True)
        for i, ax in enumerate(axes.ravel()):
            weights =self.gmms[i].weights_
            variances = self.gmms[i].covariances_.flatten()**0.5
            ax.plot(weights, variances, 'o')
            ax.set(ylabel='Weight', title=self.t_dict_latex[i])
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.grid(visible=True, which='both', alpha=0.6)
            # plt.grid(visible=True, which='minor', alpha=0.2)
        # make title for figure
        axes[-1].set(xlabel='Standard deviation', xlim=(0, 1), ylim=(-0.1,1))
        fig.suptitle('GMM variances and weights', fontsize=16, y=0.9)
        return fig, axes

    def hist_de(self, window_width, overlay_gaussian=True, overlay_lorentzian=True, save=False):

        cmap = plt.cm.cool
        fig, axes = plt.subplots(self.n_bins_T,1,figsize = (15,3*self.n_bins_T))
        n=1
        stds     = np.zeros(self.n_bins_T)
        for i, ax in enumerate(axes):
            m,_,_= ax.hist(self.e_bin_edges[:-1], self.e_bin_edges[::n], weights = self.des_binned[i], alpha = 1, density=False, label = self.t_dict_latex[i], color = cmap(i/20.0));
            ax.set(xlim=[0,window_width], xlabel='∆σ')
            ax.axvline(x=0, lw=0.5, ls='--')
            ax.legend()

        if save:
            fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_{}_de_hist.pdf'.format(self.obj, self.obj, self.name), bbox_inches='tight')
            plt.close()

        return fig, ax

    def hist_dt(self, overlay_gaussian=True, overlay_lorentzian=True, save=False):

        cmap = plt.cm.cool
        fig, axes = plt.subplots(self.n_bins_T,1,figsize = (15,3*self.n_bins_T))
        n=1
        stds     = np.zeros(self.n_bins_T)
        for i, ax in enumerate(axes):
            m,_,_= ax.hist(self.t_bin_edges[:-1], self.t_bin_edges[::n], weights = self.dts_binned[i], alpha = 1, density=True, label = self.t_dict_latex[i], color = cmap(i/20.0));
            # ax.set(xlim=[self.T_bin_edges[i],self.T_bin_edges[i+1]], xlabel='∆t')
            ax.axvline(x=0, lw=0.5, ls='--')
            ax.legend()

        if save:
            fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_{}_dt_hist.pdf'.format(self.obj, self.obj, self.name), bbox_inches='tight')
            plt.close()

        return fig, ax
    
    def hist_dt_stacked(self, overlay_gaussian=True, overlay_lorentzian=True, save=False):
        fig, ax = plt.subplots(1,1, figsize = (15,4))
        cmap = plt.cm.cool
        n=1
        stds     = np.zeros(self.n_bins_T)
        for i in range(self.n_bins_T):
            m,_,_= ax.hist(self.t_bin_edges[:-1], self.t_bin_edges[::n], weights = self.dts_binned[i], alpha = 1, label = self.t_dict_latex[i], color = cmap(i/20.0));
        ax.axvline(x=0, lw=0.5, ls='--')
        ax.set(yscale='log')
        # ax.legend()

        if save:
            fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_{}_dt_hist.pdf'.format(self.obj, self.obj, self.name), bbox_inches='tight')
            plt.close()

        return fig, ax


class dtdm_key():
    def __init__(self, obj, name, label, key, n_bins_t, n_bins_m, n_bins_m2, t_max, n_t_chunk, steepness, width, leftmost_bin, verbose=False):
        self.obj = obj
        self.name = name
        self.label = label
        self.key = key

        self.n_t_chunk = n_t_chunk
        self.n_bins_t  = n_bins_t
        self.n_bins_m  = n_bins_m

        self.t_bin_edges, self.t_bin_chunk, self.t_bin_chunk_centres, self.m_bin_edges, self.m_bin_centres, self.m_bin_widths, self.e_bin_edges, self.t_dict, self.m2_bin_edges, self.m2_bin_widths, self.m2_bin_centres = bin_data(None, n_bins_t=n_bins_t, n_bins_m=n_bins_m, n_bins_m2=n_bins_m2, t_max=t_max, n_t_chunk=n_t_chunk, compute=False, steepness=steepness, width=width, leftmost_bin=leftmost_bin);

        self.width = width

        self.bounds_values = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/bounds_values.txt'.format(self.obj, self.key))
        self.label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],self.key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}

        self.read()
        # self.stats(verbose)

    def read(self):
        n = len([file for file in os.listdir(cfg.D_DIR + 'computed/archive/{}/binned/{}/dm/'.format(self.obj,self.key)) if file.startswith('dms')])
        self.n = n
        self.dts_binned = np.zeros((n, self.n_t_chunk, self.n_bins_t), dtype = 'int64')
        self.dms_binned = np.zeros((n, self.n_t_chunk, self.n_bins_m), dtype = 'int64')
        self.dm2_de2_binned =  np.zeros((n, self.n_t_chunk, self.n_bins_m), dtype = 'int64')
        self.des_binned = np.zeros((n, self.n_t_chunk, self.n_bins_m), dtype = 'int64')
        self.dcs_binned = np.zeros((n, self.n_t_chunk, 9), dtype = 'int64')

        for i in range(n):

            self.dts_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dt/dts_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint64')
            self.dms_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dm/dms_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint64')
            self.dm2_de2_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dm2_de2/dm2_de2_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint64')
            self.des_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/de/des_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint16')
            self.dcs_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dc/dcs_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint16')

    def stats(self, verbose=False):
        if verbose:
            for i in range(self.n_bins_T):
                print('dtdm counts in {}: {:,}'.format(self.t_dict[i],self.dts_binned.sum(axis=(0,-1))[i]))

        N_dm = self.dms_binned.sum(axis=-1)
        x   = self.m_bin_centres*self.dms_binned
        mean = x.sum(axis=-1)/N_dm

        self.means = mean
        self.modes = self.m_bin_centres[ (self.dms_binned/self.m_bin_widths).argmax(axis=-1) ]
        self.modes = np.where(abs(self.modes)>1, np.nan, self.modes)

        self.skew_1  = skew(x, axis=-1)
        self.skew_2  = N_dm**0.5 * ((x-mean[:,:,np.newaxis])**3).sum(axis=-1) * ( ((x-mean[:,:,np.newaxis])**2).sum(axis=-1) )**-1.5

    def plot_means(self, ax, ls='-'):
        for i in range(self.n):
            ax.errorbar(self.t_bin_chunk_centres, self.means[i], yerr=self.means[i]*(self.dms_binned[i].sum(axis=-1)**-0.5), lw=0.5, marker='o', label=self.label_range_val[i])
            # ax.scatter(self.t_bin_chunk_centres, self.means, s=30, label=self.name, ls=ls)
            # ax.plot   (self.t_bin_chunk_centres, self.means, lw=1.5, ls=ls)

    def plot_modes(self, ax, ls='-'):
        for i in range(self.n):
            ax.errorbar(self.t_bin_chunk_centres, self.modes[i], yerr=self.means[i]*(self.dms_binned[i].sum(axis=-1)**-0.5), lw=0.5, marker='o', label=self.label_range_val[i])
            # ax.scatter(self.t_bin_chunk_centres, self.modes, s=30, label=self.name, ls=ls)
            # ax.plot   (self.t_bin_chunk_centres, self.modes, lw=1.5, ls=ls)

    def plot_sf_ensemble(self, figax=None):
        if figax is None:
            fig, ax = plt.subplots(1,1, figsize=(15,8))
        else:
            fig, ax = figax

        for i in range(self.n):
            SF = (((self.m_bin_centres**2)*self.dms_binned[i]).sum(axis=-1)/self.dms_binned[i].sum(axis=-1))**0.5
            SF[self.dms_binned[i].sum(axis=-1)**-0.5 > 0.1] = np.nan
            # Check errors below a right
            ax.errorbar(self.t_bin_chunk_centres, SF, yerr=self.dms_binned[i].sum(axis=-1)**-0.5*self.means[i], lw = 0.5, marker = 'o', label=self.label_range_val[i])
        ax.set(yscale='log',xscale='log', xticks=[100,1000])
        ax.set(xlabel='∆t',ylabel = 'structure function')
        ax.legend()

        return figax, SF
    
    def plot_sf_dm2_de2(self, figax=None):
        if figax is None:
            fig, ax = plt.subplots(1,1, figsize=(15,8))
        else:
            fig, ax = figax
        
        for i in range(self.n):
            SF = (((self.m2_bin_centres**2)*self.dm2_de2_binned[i]).sum(axis=-1)/self.dm2_de2_binned[i].sum(axis=-1))
            # SF[self.dm2_de2_binned[i].sum(axis=-1)**-0.5 > 0.1] = np.nan
            # Check errors below a right
            ax.errorbar(self.t_bin_chunk_centres, SF, yerr=0, lw = 0.5, marker = 'o', label=self.label_range_val[i])
        ax.set(yscale='log',xscale='log', xticks=[100,1000])
        ax.set(xlabel='∆t',ylabel = 'structure function squared')
        ax.legend()
        
        return figax, SF
            
    def plot_sf_ensemble_iqr(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(15,8))
        else:
            fig, ax = figax
        norm_cumsum = self.dms_binned.cumsum(axis=-1)/self.dms_binned.sum(axis=-1)[:,np.newaxis]
        lq_idxs  = np.abs(norm_cumsum-0.25).argmin(axis=-1)
        uq_idxs  = np.abs(norm_cumsum-0.75).argmin(axis=-1)

        SF = 0.74 * ( self.m_bin_centres[uq_idxs]-self.m_bin_centres[lq_idxs] ) * (( self.dms_binned.sum(axis=-1) - 1 ) ** -0.5)
        SF[SF == 0] = np.nan

        ax.errorbar(self.t_bin_chunk_centres, SF, yerr=self.dms_binned.sum(axis=-1)**-0.5*self.means, lw = 0.5, marker = 'o', label = self.label)
        ax.set(yscale='log',xscale='log', xticks=[100,1000])
        ax.set(xlabel='∆t',ylabel = 'structure function')

        return figax, SF
    
    def hist_dm(self, window_width, overlay_gaussian=True, overlay_lorentzian=True, save=False):

        def gaussian(x,peak,x0):
            sigma = (2*np.pi)**-0.5*1/peak
            return peak*np.exp( -( (x-x0)**2/(2*sigma**2) ) )
        def lorentzian(x,gam,x0):
            return gam / ( gam**2 + ( x - x0 )**2) * 1/np.pi

        cmap = plt.cm.cool
        fig, axes = plt.subplots(self.n_bins_T,1,figsize = (15,3*self.n_bins_T))
        n=1
        stds     = np.zeros(self.n_bins_T)
        for i, ax in enumerate(axes):
            m,_,_= ax.hist(self.m_bin_edges[:-1], self.m_bin_edges[::n], weights = self.dms_binned[i], alpha = 1, density=True, label = self.t_dict[i], color = cmap(i/20.0));
            ax.set(xlim=[-window_width,window_width])
             # ax.axvline(x=modes[i], lw=0.5, ls='--', color='k')
             # ax.axvline(x=m_bin_centres[m.argmax()], lw=0.5, ls='--', color='r')

            if overlay_gaussian:
                #Also make sure that bins returned from .hist match m_bin_edges : it is
                x = np.linspace(-2,2,1000)

                try:
                    popt, _ = curve_fit(gaussian, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()]])
                    ax.plot(x,gaussian(x, m.max(), m_bin_edges[:-1:n][m.argmax()]), label = 'gaussian')

                    diff    = m - gaussian(self.m_bin_centres,popt[0],popt[1]) # residuals
                    ax.plot(self.m_bin_centres, diff, label = 'diff')

                    print('x0: {:.5f}'.format(popt[1]))
                except:
                    pass

            if overlay_lorentzian:
                x = np.linspace(-2,2,1000)

                try:
                    popt, _ = curve_fit(lorentzian, self.m_bin_edges[:-1:n], m, p0 = [1/m.max(),self.m_bin_edges[:-1:n][m.argmax()]])
                    ax.plot(x,lorentzian(x,popt[0],popt[1]), color = 'r', label = 'lorentzian')
                    diff    = m - lorentzian(self.m_bin_centres,popt[0],popt[1])
                    ax.plot(self.m_bin_centres, diff, label = 'diff')
                    print('x0: {:.5f}'.format(popt[1]))
                except:
                    pass

            ax.text(0.05, 0.9, 'mean: {:.5f}'.format(self.means[i]), transform=ax.transAxes)
            ax.text(0.05, 0.8, 'mode: {:.5f}'.format(self.modes[i]), transform=ax.transAxes)
            ax.text(0.05, 0.7, 'skew: {:.5f}'.format(self.skew_1[i]), transform=ax.transAxes)
            ax.axvline(x=self.means[i], lw=0.4, ls='-', color='b')
            ax.axvline(x=self.modes[i], lw=0.4, ls='-', color='r')
            ax.axvline(x=self.m_bin_centres[m.argmax()], lw=0.6, ls='-', color='r')

            ax.axvline(x=0, lw=0.5, ls='--')
            ax.legend()

        if save:
            fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_{}_dm_hist.pdf'.format(self.obj, self.obj, self.name), bbox_inches='tight')
            plt.close()

        return fig, ax