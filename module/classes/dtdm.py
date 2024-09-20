import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..config import cfg
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from module.preprocessing import data_io, parse

class dtdm_raw_analysis():
    """
    Class for analysing dtdm files
    """
    def __init__(self, obj, band, name, phot_str='clean'):
        self.obj = obj
        self.ID = 'uid' if (obj == 'qsos') else 'uid_s'
        self.band = band
        self.data_path = cfg.D_DIR + f'merged/{obj}/{phot_str}/dtdm_{band}/'
        self.name = name
        # sort based on filesize, then do ordered shuffle so that each core recieves the same number of large files
        if os.path.exists(self.data_path):
            fnames = [a for a in os.listdir(self.data_path) if re.match('dtdm_[0-9]{5,7}_[0-9]{5,7}.csv', a)]
            size=[]
            for file in fnames:
                size.append(os.path.getsize(self.data_path+file))
            self.fnames = [name for i in [0,1,2,3] for sizename, name in sorted(zip(size, fnames))[i::4]]
            self.fpaths = [self.data_path + fname for fname in self.fnames]

    def read(self, i=0, **kwargs):
        """
        Function for reading dtdm data
        """
        df = pd.read_csv(self.fpaths[i], index_col = self.ID, dtype = {self.ID: np.uint32, 'dt': np.float32, 'dm': np.float32, 'de': np.float32, 'sid': np.uint8}, **kwargs)
        self.df = parse.filter_data(df, bounds=cfg.PREPROC.dtdm_bounds[self.obj], dropna=True)

    def read_all(self, ncores=None, **kwargs):
        """
        Function for reading all dtdm data
        """
        if ncores is None: ncores = cfg.USER.N_CORES
        kwargs['basepath'] = self.data_path
        kwargs['ID'] = self.ID
        kwargs['dtypes'] = {self.ID: np.uint32, 'dt': np.float32, 'dm': np.float32, 'de': np.float32, 'sid': np.uint8}
        
        df = data_io.dispatch_reader(kwargs, max_processes=ncores)
        self.df = parse.filter_data(df, bounds=cfg.PREPROC.dtdm_bounds[self.obj], dropna=True)
    
    def read_key(self, key):
        """
        Read in the groups of uids for qsos binned into given key.
        """
        self.key = key
        path = cfg.D_DIR + 'computed/archive/{}/binned/{}/uids/'.format(self.obj, self.key)
        fnames = sorted([fname for fname in os.listdir(path) if fname.startswith('group')])
        self.groups = [pd.read_csv(path + fname, index_col=self.ID) for fname in fnames]
        self.n_groups = len(self.groups)
        self.bounds_values = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/bounds_values.txt'.format(self.obj, self.key))
        self.label_range_val = {i:r'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],cfg.FIG.LABELS.PROPv2[self.key],self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}

    def bin_de_2d(self, n_chunks, read=False):
        """
        2D binning into (de,dm) to see correlation between ∆m and ∆error and store them in:
        self.mean_tot
        self.std_tot
        self.median_tot

        Parameters
        ----------
        n_chunks : int
            number of files to read in
        read : bool
            If True, read in, if False, use current self.df

        """
        xbins = 100
        ybins = 100
        xlim   = (0,0.15)
        ylim   = (-0.2,0.2)
        self.de_edges = np.linspace(*xlim, xbins+1)
        self.dm_edges = np.linspace(*ylim, ybins+1)
        self.de_centres = (self.de_edges[1:]+self.de_edges[:-1])/2
        self.dm_centres = (self.dm_edges[1:]+self.dm_edges[:-1])/2
        self.total_counts = np.full((xbins,ybins),0, dtype='uint64')
        n_slices = 10
        self.mean_tot = np.zeros((n_slices,n_chunks))
        self.std_tot  = np.zeros((n_slices,n_chunks))
        self.median_tot  = np.zeros((n_slices,n_chunks))

        self.de_edges_stat = np.linspace(*xlim,n_slices+1)
        self.de_centres_stat = (self.de_edges_stat[1:]+self.de_edges_stat[:-1])/2

        for n in range(n_chunks):
            if read:
                self.read(n)
            counts = np.histogram2d(self.df['de'],self.df['dm'],range=(xlim,ylim), bins=(xbins, ybins))[0].astype('uint64')
            self.total_counts += counts

            std = []
            mean = []
            median = []
            for de1, de2 in zip(self.de_edges_stat[:-1], self.de_edges_stat[1:]):
                slice_ = self.df['dm'][(de1 < self.df['de']) & (self.df['de'] < de2)]
                std.append(slice_.std())
                mean.append(slice_.mean())
                median.append(slice_.median())

            self.mean_tot[:,n] = np.array(mean)
            self.std_tot[:,n]  = np.array(std)
            self.median_tot[:,n] = np.array(median)

    def bin_dt_2d(self, n_chunks, log_or_lin, read=False):
        """
        2D binning into (dt,dm2_de2), attempt at a 2D structure function

        Parameters
        ----------
        n_chunks : int
            number of files to read in
        read : b
            If True, read in, if False, use current self.df
        """
        xbins = 100
        ybins = 100
        xlim   = [0.9,23988.3/20]
        ylim   = [0.0001,0.6] # dm2_de2**0.5
        self.dt_edges = np.logspace(*np.log10(xlim), xbins+1)
        # self.dt_edges = np.linspace(*xlim, xbins+1) # for linear binning
        self.dm2_de2_edges = np.linspace(*ylim, ybins+1)
        self.dt_centres = (self.dt_edges[1:]+self.dt_edges[:-1])/2
        self.dm2_de2_centres = (self.dm2_de2_edges[1:]+self.dm2_de2_edges[:-1])/2
        self.total_counts = np.full((xbins,ybins),0, dtype='uint64')
        n_slices = 10
        self.mean_tot = np.zeros((n_slices,n_chunks))
        # self.std_tot  = np.zeros((n_slices,n_chunks))
        # self.median_tot  = np.zeros((n_slices,n_chunks))

        self.dt_edges_stat = np.linspace(*xlim,n_slices+1)
        self.dt_centres_stat = (self.dt_edges_stat[1:]+self.dt_edges_stat[:-1])/2

        for n in range(n_chunks):
            if read:
                self.read(n)
            boolean = self.df['dm2_de2']>0
            counts = np.histogram2d(self.df[boolean]['dt'],self.df[boolean]['dm2_de2']**0.5,range=(xlim,ylim), bins=(xbins, ybins))[0].astype('uint64')
            self.total_counts += counts

            # std = []
            mean = []
            # median = []
            for dt1, dt2 in zip(self.dt_edges_stat[:-1], self.dt_edges_stat[1:]):
                slice_ = self.df['dm2_de2']
                 # std.append(slice_.std())
                mean.append((slice_**0.5).mean())
                 # median.append(slice_.median())

            self.mean_tot[:,n] = np.array(mean)
            # self.std_tot[:,n]  = np.array(std)
            # self.median_tot[:,n] = np.array(median)

    def plot_dm_hist(self):
        n = len(self.de_edges_stat)-1
        fig, ax = plt.subplots(n, 1, figsize=(20,5*n))
        for i in range(n):
            de1, de2 = (self.de_edges_stat[i], self.de_edges_stat[i+1])
            slice_ = self.df['dm'][(de1 < self.df['de']) & (self.df['de'] < de2)]
            ax[i].hist(slice_, bins=101, range=(-0.5,0.5), alpha=0.4)
            ax[i].legend()

    def plot_dm2_de2_hist(self, figax, bins, **kwargs):
        n = 20
        if figax is None:
            fig, ax = plt.subplots(n,1, figsize=(18,5*n))
        else:
            fig, ax = figax
        mjds = np.linspace(0, 24000, n+1)
        for i, edges in enumerate(zip(mjds[:-1], mjds[1:])):
            mjd_lower, mjd_upper = edges
            boolean = (mjd_lower < self.df['dt']) & (self.df['dt']<mjd_upper)
            print(boolean.sum())
            ax[i].hist(self.df[boolean]['dm2_de2'], range=kwargs['xlim'], alpha=0.5, bins=bins, label='{:.2f} < ∆t < {:.2f}'.format(*edges))
            ax[i].set(xlabel='$(m_i-m_j)^2 - \sigma_i^2 - \sigma_j^2$', **kwargs) #title='Distribution of individual corrected SF values'
        ax.set(yscale='log')
        for i in range(n):
            de1, de2 = (self.de_edges_stat[i], self.de_edges_stat[i+1])
            slice_ = self.df['dm2_de2'][(de1 < self.df['de']) & (self.df['de'] < de2)]
            ax[i].hist(slice_, bins=101, range=(-0.5,0.5), alpha=0.4)
            ax[i].legend()

    def read_pooled_stats(self, log_or_lin, key=None, pooled_stats=None, mjd_edges=None):
        self.log_or_lin = log_or_lin
        self.key = key
        
        if (pooled_stats is None) & (mjd_edges is None):
            fpath = cfg.D_DIR + f'computed/{self.obj}/dtdm_stats/{key}/{self.log_or_lin}/{self.band}/'
            names = os.listdir(fpath)
            if key == 'all':
                self.pooled_stats = ({name[7:-4].replace('_',' '):np.loadtxt(fpath+name) 
                                    for name in names if name.startswith('pooled')})
                self.mjd_edges = np.loadtxt(fpath + 'mjd_edges.csv')
            else:
                self.bounds_values = np.loadtxt(fpath + 'bounds_values.csv')
                self.n_groups = len(self.bounds_values)-1
                self.pooled_stats = ({name[7:-6].replace('_',' '):np.array([np.loadtxt('{}{}_{}.csv'.format(fpath,name[:-6],i)) 
                                    for i in range(self.n_groups)])
                                    for name in names if name.startswith('pooled')})
                self.label_range_val = {i:r'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],cfg.FIG.LABELS.PROPv2[self.key],self.bounds_values[i+1]) for i in range(self.n_groups)}
                self.mjd_edges = np.array([np.loadtxt(fpath + f'mjd_edges_{i}.csv') for i in range(self.n_groups)])
        
        else:
            self.pooled_stats = pooled_stats
            self.mjd_edges = mjd_edges
        
        self.mjd_centres = (self.mjd_edges[..., :-1] + self.mjd_edges[..., 1:])/2

    def plot_stats(self, keys, figax, label=None, legend_loc='upper right', show_marker_bin_counts=False, plot_kwargs={},  **kwargs):
        if figax is None:
            fig, ax = plt.subplots(1,1, figsize=(10,6))
        else:
            fig, ax = figax
        if keys=='all':
            keys = list(self.pooled_stats.keys())[1:]
        
        color = plot_kwargs.pop('color') if 'color' in plot_kwargs else None

        if 'lw' not in plot_kwargs: plot_kwargs['lw'] = 1

        if label is None:
            label = ['{}, {}'.format(self.name,key) for key in keys]
        elif isinstance(label, str):
            label = [label]*len(keys)
        # Norm by log
        # normalised_bin_counts = np.log(self.pooled_stats['n']) + 10
        # Norm by total max
        # normalised_bin_counts = self.pooled_stats['n']/np.max(self.pooled_stats['n'])*1e4
        # Norm per time bin
        # normalised_bin_counts = self.pooled_stats['n']/self.pooled_stats['n'].sum(axis=0)*1e3
        # Norm per sqrt time bin (ie area propto counts)
        normalised_bin_counts = (self.pooled_stats['n']/self.pooled_stats['n'].sum(axis=0))**0.5*1e2
        error_norm = (2/self.pooled_stats['n'])**0.08
        for i, key in enumerate(keys):
            y = self.pooled_stats[key].copy()
            if key.startswith('SF'):
                y[y[:,0]<0] = np.nan
                y[y[:,0]==0] = np.nan
            if key.startswith('median'):
                y[y[:,0]==0] = np.nan
                y[abs(y[:,0])>0.3] *= 1
                y[:,1][(y[:,1])>0.15] = 0.15
            # else:
                # y[:,1] = y[:,1]**0.5 # NOTE: Non SF errors are actually variances as of 08/08/23. If extract_features is run since then, remove this line.

            # ax.errorbar(self.mjd_centres, y[:,0], yerr=y[:,1]**0.5, label='{}, {}'.format(key,self.name), color=color, lw=2.5) # square root this
            ax.errorbar(self.mjd_centres, y[:,0], yerr=y[:,1]*error_norm,
                        capsize=3,
                        marker='o',
                        ms=2,
                        elinewidth=0.4,
                        markeredgewidth=0.9,
                        color=color,
                        label=label[i],
                        **plot_kwargs)
            ax.scatter(self.mjd_centres, y[:,0], s=normalised_bin_counts,
                       
                       color=color)
    
            ax.set(xlabel='Rest frame time lag (days)')
        ax.grid(visible=True, which='major', alpha=0.5)
        ax.grid(visible=True, which='minor', alpha=0.2)
        ax.set(**kwargs)
        for handle in ax.legend(loc=legend_loc).legend_handles:
            try:
                handle.set_sizes([70])
            except:
                pass
        # ax.set(xlabel='$(m_i-m_j)^2 - \sigma_i^2 - \sigma_j^2$', title='Distribution of individual corrected SF values', **kwargs)
        return (fig,ax)
    
    def fit_stats(self, key, model_name, ax=None, least_sq_kwargs={}, plot_args={}, **kwargs):
        y, yerr = self.pooled_stats[key].T 
        if model_name == 'power_law':
            from module.modelling.fitting import fit_power_law
            # yerr = 10*np.ones(y.shape) # to fit without errors
            coefficient, exponent, pcov, model_values = fit_power_law(self.mjd_centres, y, yerr, **kwargs)
            label = rf'$\alpha \Delta t^{{\beta}}, \beta={exponent:.3f}\pm{pcov[1,1]**0.5:.3f}, \alpha={coefficient:.3f}\pm{pcov[0,0]**0.5/np.log(10):.3f}$'
            print(f'fitted power law: y = ({coefficient:.3f} ± {pcov[0,0]**0.5/np.log(10):.3f})*x^({exponent:.3f} ± {pcov[1,1]**0.5:.3f})')
            fitted_params = (coefficient, exponent)
        elif model_name == 'broken_power_law':
            from module.modelling.fitting import fit_broken_power_law
            # yerr = 10*np.ones(y.shape) # to fit without errors
            amplitude, break_point, index_1, index_2, pcov, model_values = fit_broken_power_law(self.mjd_centres, y, yerr, least_sq_kwargs=least_sq_kwargs, **kwargs)
            print(f'fitted broken power law:\ny = {amplitude:.2f}*x^{index_1:.2f} for x < {break_point:.2f}\ny = {amplitude:.2f}*x^{index_2:.2f} for x > {break_point:.2f}')
            label = 'broken power law'
            fitted_params = (amplitude, break_point, index_1, index_2)
        elif model_name == 'broken_power_law_minimize':
            from module.modelling.fitting import fit_minimize, cost_function
            from module.modelling.models import bkn_pow_smooth
            fitted_params, _, model_values = fit_minimize(bkn_pow_smooth, cost_function, self.mjd_centres, y, yerr, **kwargs)
            print(f'fitted broken power law:\ny = {fitted_params[0]:.2f}*x^{fitted_params[2]:.2f} for x < {fitted_params[1]:.2f}\ny = {fitted_params[0]:.2f}*x^{fitted_params[3]:.2f} for x > {fitted_params[1]:.2f}')
            label = 'broken power law'
        elif model_name == 'DRW SF':
            from module.modelling.fitting import fit_DRW_SF
            tau, SF_inf, pcov, model_values = fit_DRW_SF(self.mjd_centres, y, yerr, **kwargs)
            print(f'fitted DRW SF:\ntau = {tau:.2f}\nSF_inf = {SF_inf:.2f}')
            label = 'DRW SF'
            fitted_params = (tau, SF_inf)

        elif model_name == 'mod DRW SF':
            from module.modelling.fitting import fit_mod_DRW_SF
            tau, SF_inf, beta, pcov, model_values = fit_mod_DRW_SF(self.mjd_centres, y, yerr, **kwargs)
            print(f'fitted mod DRW SF:\ntau = {tau:.2f}\nSF_inf = {SF_inf:.2f}\nbeta = {beta:.2f}')
            label = 'mod DRW SF'
            fitted_params = (tau, SF_inf, beta)
            
        if ax is not None:
            ax.plot(*model_values, ls='-.', label=label, **plot_args, zorder=0)		
            ax.legend()

        return fitted_params

    def fit_stats_prop(self, key, model_name, ax=None, least_sq_kwargs={}, plot_args={}, **kwargs):
        fitted_params_ = []
        fitted_params_err = []
        for group_idx in range(self.n_groups):
            y, yerr = self.pooled_stats[key][group_idx].T
            x = self.mjd_centres[group_idx]
            if model_name == 'power_law':
                from module.modelling.fitting import fit_power_law
                # yerr = 10*np.ones(y.shape) # to fit without errors
                coefficient, exponent, pcov, model_values = fit_power_law(x, y, yerr, **kwargs)
                label = r'$\Delta t^{\beta}, \beta='+'{:.2f}'.format(exponent)+'$'
                fitted_params = (coefficient, exponent)			
            elif model_name == 'broken_power_law':
                from module.modelling.fitting import fit_broken_power_law
                # yerr = 10*np.ones(y.shape) # to fit without errors
                amplitude, break_point, index_1, index_2, pcov, model_values = fit_broken_power_law(x, y, yerr, least_sq_kwargs=least_sq_kwargs, **kwargs)
                # print(f'fitted broken power law:\ny = {amplitude:.2f}*x^{index_1:.2f} for x < {break_point:.2f}\ny = {amplitude:.2f}*x^{index_2:.2f} for x > {break_point:.2f}')
                label = 'broken power law'
                fitted_params = (amplitude, break_point, index_1, index_2)
            elif model_name == 'broken_power_law_minimize':
                from module.modelling.fitting import fit_minimize, cost_function
                from module.modelling.models import bkn_pow_smooth
                fitted_params, _, model_values = fit_minimize(bkn_pow_smooth, cost_function, x, y, yerr, **kwargs)
                # print(f'fitted broken power law:\ny = {fitted_params[0]:.2f}*x^{fitted_params[2]:.2f} for x < {fitted_params[1]:.2f}\ny = {fitted_params[0]:.2f}*x^{fitted_params[3]:.2f} for x > {fitted_params[1]:.2f}')
                label = 'broken power law'
            elif model_name == 'DRW SF':
                from module.modelling.fitting import fit_DRW_SF
                tau, SF_inf, pcov, model_values = fit_DRW_SF(x, y, yerr, **kwargs)
                # # print(f'fitted DRW SF:\ntau = {tau:.2f}\nSF_inf = {SF_inf:.2f}')
                label = 'DRW SF'
                fitted_params = (tau, SF_inf)

            elif model_name == 'mod DRW SF':
                from module.modelling.fitting import fit_mod_DRW_SF
                tau, SF_inf, beta, pcov, model_values = fit_mod_DRW_SF(x, y, yerr, **kwargs)
                # print(f'fitted mod DRW SF:\ntau = {tau:.2f}\nSF_inf = {SF_inf:.2f}\nbeta = {beta:.2f}')
                label = 'mod DRW SF'
                fitted_params = (tau, SF_inf, beta)
                
            if ax is not None:
                ax.plot(*model_values, lw=1.5, ls='-.', label=label, **plot_args)		
                # ax.legend()

            fitted_params_.append(fitted_params)
            fitted_params_err.append(np.diag(pcov))
        return np.array(fitted_params_), np.array(fitted_params_err)

    def plot_comparison_data(self, ax, name='macleod', **kwargs):
        # f = lambda x: 0.01*(x**0.443)
        # ax.plot(self.mjd_centres, f(self.mjd_centres), lw=0.5, ls='--', color='b', label='MacLeod 2012')
        if name=='macleod':
            x,y = pd.read_csv(cfg.W_DIR + 'assets/comparison_data/macleod2012_fig17.csv', comment='#').values.T
            ax.plot(x, y, label = 'Macleod et al. 2012', **kwargs)
        elif name=='caplar':
            dt, dm = pd.read_csv(cfg.W_DIR + 'assets/comparison_data/caplar2020_fig4.csv', comment='#').values.T
            dt = dt*365.25
            ax.scatter(dt, -dm, color='k', s=0.7, label='Caplar et al. 2020', **kwargs)
        elif name=='stone':
            dt, dm = pd.read_csv(cfg.W_DIR + 'assets/comparison_data/stone2022_fig10.csv', comment='#').values.T
            ax.plot(dt, dm, label='Stone et al. 2022', **kwargs)
            # ax.scatter(dt, dm, color='k', s=0.5, label='Stone et al. 2022', **kwargs)
        elif name=='devries':
            dt, sf = pd.read_csv(cfg.W_DIR + 'assets/comparison_data/devries2005_fig18.csv', comment='#').values.T
            # dt, sf = pd.read_csv(cfg.W_DIR + 'assets/comparison_data/devries2005_fig8.csv', comment='#').values.T
            dt = dt*365.25
            sf = 10**sf
            ax.plot(dt, sf, label='de Vries et al. 2005', **kwargs)
        elif name=='morganson':
            dt, sf = pd.read_csv(cfg.W_DIR + f'assets/comparison_data/morganson2014_fig6_{self.band}.csv', comment='#').values.T
            dt = dt*365.25
            ax.plot(dt, sf, label='Morganson et al. 2014', **kwargs)

    def plot_sf_from_drw_fits(self, ax, **kwargs):
        from module.assets import load_drw_mcmc_fits
        from eztao.carma import drw_sf
        def ensemble_drw(sigs, taus, lag):
            n = len(sigs)
            m = len(lag)
            sf2 = np.zeros((n, m))
            for i in range(n):
                true_drw = drw_sf(10**sigs[i], 10**taus[i])
                sf2[i,:] = true_drw(lag)**2
            sf = np.average(sf2, axis=0)**0.5
            return sf
        
        drw_fits = load_drw_mcmc_fits('gri')
        sigs, taus = drw_fits[['sig50', 'tau50']].values.T
        dt = np.logspace(0,4,100)
        ax.plot(dt, ensemble_drw(sigs, taus, dt), label='DRW SF')

    def plot_stats_property(self, keys, figax=None, macleod=False, fill_between=False, shift=False, **kwargs):
        if figax is None:
            fig, ax = plt.subplots(1,1, figsize=(10,7))
        else:
            fig, ax = figax
        if keys=='all':
            keys = list(self.pooled_stats.keys())[1:]
        
        # Norm by log
        # normalised_bin_counts = np.log(self.pooled_stats['n']) + 50
        # Norm by total max
        # normalised_bin_counts = self.pooled_stats['n']/np.max(self.pooled_stats['n'])*1e4
        # Norm per group
        normalised_bin_counts = self.pooled_stats['n']/self.pooled_stats['n'].sum(axis=0)*3e2 + 10
        # Norm per time bin
        # normalised_bin_counts = self.pooled_stats['n']/(self.pooled_stats['n'].sum(axis=1).reshape(-1,1))*2e3
        
        if fill_between:
            for key in keys:
                max_ = self.pooled_stats[key].max(axis=0)
                min_ = self.pooled_stats[key].min(axis=0)
                ax.fill_between(self.mjd_centres.max(axis=0), min_[:,0]-min_[:,1], max_[:,0]+max_[:,1], color='#ff7f0e', alpha=0.2,
                        edgecolor='#C3610C', lw=2)
            # Don't show errorbars if we are showing the fill_between
            elinewdith = 0
            markeredgewidth = 0
        else:
            elinewdith = 0.2
            markeredgewidth = 0.2

        self.key_centres = (self.bounds_values[1:] + self.bounds_values[:-1])/2

        for group_idx in range(self.n_groups):
            error_norm = (2/self.pooled_stats['n'][group_idx])**0.1
            for key in keys:
                y = self.pooled_stats[key][group_idx]
                if key.startswith('SF'):
                    y[y<0] = np.nan
                # mask = abs(y[:,0]-y[:,0].mean()) > 10*y[:,0].std() # Note this is a hack to remove the large values from the plot
                # y[mask, :] = np.nan,np.nan 
                color = cmap.gist_earth(group_idx/self.n_groups)
                x = self.mjd_centres[group_idx].copy()
                if shift:
                    x /= 10**(self.key_centres[group_idx, np.newaxis])
                ax.errorbar(x, y[:,0], yerr=y[:,1]*error_norm,
                            capsize=5,
                            lw=0.5,
                            elinewidth=elinewdith,
                            markeredgewidth=markeredgewidth)#, color=color) # square root this
                # make the size of the scatter points proportional to the number of points in the bin
                ax.scatter(x, y[:,0], s=normalised_bin_counts[group_idx],
                           alpha=0.6,
                           label=self.label_range_val[group_idx])#, color=color)
                ax.axvspan(0, 10, color='gray', alpha=0.01)

        if macleod:
            f = lambda x: 0.01*(x**0.443)
            ax.plot(self.mjd_centres.mean(), f(self.mjd_centres.mean()), lw=0.1, ls='--', color='b', label='MacLeod 2012')
            x,y = np.loadtxt(cfg.D_DIR + 'Macleod2012/SF/macleod2012.csv', delimiter=',').T
            ax.scatter(x, y, label = 'macleod 2012')

        ax.grid(visible=True, which='major', alpha=0.5)
        ax.grid(visible=True, which='minor', alpha=0.2)
       
        # Lines below are no longer needed since we now combine the legends from gri bands in the notebook
        # ax.legend()
        # for handle in plt.legend().legend_handles:
        #     try:
        #         handle.set_sizes([70])
        #     except:
        #         pass

        ax.set(xlabel='Rest frame time lag (days)')
        ax.set(**kwargs)

        return (fig,ax)

    def contour_de(self):
        fig, ax = plt.subplots(1,1, figsize=(20,10))

        ax.contourf(self.de_centres, self.dm_centres, self.total_counts.T, cmap='jet')

        plt.scatter(self.de_centres_stat, self.median_tot.mean(axis=1), color = 'b')
        for m in [-1,0,1]:
            plt.scatter(self.de_centres_stat,self.mean_tot.mean(axis=1)+self.std_tot.mean(axis=1)*m, color='k')
            plt.plot   (self.de_centres_stat,self.mean_tot.mean(axis=1)+self.std_tot.mean(axis=1)*m, color='k', lw=0.5)

    def contour_dt(self):
        fig, ax = plt.subplots(1,1, figsize=(20,10))

        ax.contourf(self.dt_centres, self.dm2_de2_centres, self.total_counts.T, levels=np.logspace(0,3.4,50), cmap='jet')
        # ax.set(xscale='log', yscale='log')
        # plt.scatter(self.dt_centres_stat, self.median_tot.mean(axis=1), color = 'b')
        # for m in [-1,0,1]:
        # 	plt.scatter(self.de_centres_stat,self.mean_tot.mean(axis=1)+self.std_tot.mean(axis=1)*m, color='k')
        # 	plt.plot   (self.de_centres_stat,self.mean_tot.mean(axis=1)+self.std_tot.mean(axis=1)*m, color='k', lw=0.5)