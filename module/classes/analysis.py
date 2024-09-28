from multiprocessing import Pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
# from bokeh.plotting import figure, output_notebook, show
# from bokeh.layouts import column
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, pairwise, binning, parse
from module.assets import load_grouped, load_grouped_tot, load_sets, load_coords, load_redshifts, load_vac
from .methods.plotting import plot_series as plot_series_
from module.classes.dtdm import dtdm_raw_analysis

def calc_moments(bins,weights):
    """
    Calculate mean and kurtosis
    """
    x = bins*weights
    z = (x-x.mean(axis=1)[:,np.newaxis])/x.std(axis=1)[:,np.newaxis]
    return x.mean(axis=1), (z**4).mean(axis = 1) - 3

class analysis():
    def __init__(self, obj, band):
        # Imported methods
        self.obj = obj
        self.ID = 'uid' if obj == 'qsos' else 'uid_s'
        self.band = band
        self.plt_color = {'u':'m', 'g':'g', 'r':'r', 'i':'k', 'z':'b'}
        # self.plt_color_bokeh = {'u':'magenta', 'g':'green', 'r':'red', 'i':'black', 'z':'blue'}
        self.marker_dict = {1:'^', 3:'v', 5:'D', 7:'s', 11:'o'}
        # self.marker_dict_bokeh = {1:'triangle', 3:'inverted_triangle', 5:'diamond', 7:'square', 11:'circle'}
        self.survey_dict = {3: 'SSS', 5:'SDSS', 7:'PS1', 11:'ZTF'}
        self.read_coords()

    def summary(self):
        """
        Run to create the following attributes:
        self.idx_uid : array_like
                unique set of uids
        self.uids_missing : array_like
                uids of objects which are present in DR14Q but not in observations
        self.n_qsos : int
                number of objects for which we have observations
        self.idx_cat : array_like
                list of surveys which contribute to observations (sdss=1, ps=2, ztf=3)
        """
        self.coords = pd.read_csv(cfg.D_DIR + f'catalogues/{self.obj}/{self.obj}_subsample_coords.csv', index_col=self.ID, comment='#')
        # Check which qsos we are missing and which we have, given a list
        uids_complete	 = self.coords.index
        
        self.idx_uid	  = self.df.index.unique()
        self.uids_missing = uids_complete[~np.isin(uids_complete,self.idx_uid)]
        self.n_qsos		   = len(self.idx_uid)
        # self.idx_cat	  = self.df['catalogue'].unique()

        print('Number of qsos with lightcurve: {:,}'.format(self.n_qsos))
        print('Number of datapoints in:\nSDSS: {:,}\nPS: {:,}\nZTF: {:,}'.format((self.df['catalogue']==5).sum(),(self.df['catalogue']==7).sum(),(self.df['catalogue']==11).sum()))

    def sdss_quick_look(self, uids):
        if np.issubdtype(type(uids),np.integer): uids = [uids]
        coords = self.coords.loc[uids].values
        for ra, dec in coords:
            print("https://skyserver.sdss.org/dr18/VisualTools/quickobj?ra={}&dec={}".format(ra, dec))
    
    def read_single_survey(self, survey, nrows=None, i=None, ncores=4, fnames=None, phot_str='clean', uids=None):
        kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
                'nrows': nrows,
                'basepath': cfg.D_DIR + f'surveys/{survey}/{self.obj}/{phot_str}/{self.band}_band', # we should make this path more general so it is consistent between surveys
                'ID': self.ID}

        self.df = data_io.dispatch_reader(kwargs, i=i, max_processes=ncores, concat=True, fnames=fnames, uids=uids)
        # self.df = self.df[np.any(self.df['band'].isin(self.band), axis=1)]
        # Remove objects with a single observation.
        mask_not_single_obs = self.df.index.duplicated(keep=False)
        print(f'Number of objects with single observation: {(~mask_not_single_obs).sum()}')
        self.df = self.df[mask_not_single_obs]

    def read_merged_photometry(self, nrows=None, i=None, ncores=4, fnames=None, phot_str='clean', uids=None):
        """
        Read in photometric data.
        This method will use ncores to load in all lightcurves located
            in basepath: '/data/merged/{object}/{band}_band}'
            and concatenate the result into a single DataFrame	
        
        Parameters
        ----------
        nrows   : dataframe
            Number of rows to read in per file
        fnames  : list
            list of names of files to read in
            if not specified, all files will be read in
        multiproc : bool
            Whether to use multiprocessing or not
            If false, the i'th file will be read in
        i : int
            which file to read in if not using multiproc, see above. 
        ncores : int
            Number of cores to use for parallel reading
            (only relevant if multiproc = True)

        Parameters
        ----------
        Example:
            >>> self.read_merged_photometry(nrows=1000, ncores=4, multiproc=True)
            will read in 1000 rows from each file in parallel using 4 cores and concatenate the result
            >>> self.read_merged_photometry(ncores=2, multiproc=True, fnames=['lc_000000_005000.csv','lc_020001_025000.csv','lc_055001_060000.csv'])
            will read in the specified files using 2 cores and concatenate the result
            >>> self.read_merged_photometry(multiproc=False, i=10)
            will read in the 10th file only
        """
        # Default to 4 cores
        # Use the path below for SSA
        # basepath = cfg.D_DIR + 'merged/{}/{}_band/with_ssa/'

        kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
                'nrows': nrows,
                'basepath': cfg.D_DIR + f'merged/{self.obj}/{phot_str}/', # we should make this path more general so it is consistent between surveys
                'ID': self.ID}

        self.df = data_io.dispatch_reader(kwargs, i=i, max_processes=ncores, concat=True, fnames=fnames, uids=uids)
        self.df = self.df[np.any([(self.df.band == b).values for b in self.band], axis=0)]
        # self.df = self.df[np.any(self.df['band'].isin(self.band), axis=1)]
        # Remove objects with a single observation.
        mask_not_single_obs = self.df.index.duplicated(keep=False)
        print(f'Number of objects removed with single observation: {(~mask_not_single_obs).sum()}')
        self.df = self.df[mask_not_single_obs]

    def calculate_dtdm(self, uids):
        """
        Calculate dtdm for a given set of uids
        """
        if np.issubdtype(type(uids),np.integer): uids = [uids]
        df_ = self.df[self.df.index.isin(uids)]
        self.dtdm = df_.groupby(self.ID, group_keys=False).apply(pairwise.calculate_dtdm, 'mjd_rf').set_index(self.ID)
        self.dtdm_uids = uids

    def calculate_stats(self):
        """
        Calculate statistics for a given set of uids
        """
        n_points = 10
        self.mjd_edges = binning.construct_T_edges(t_max=self.dtdm['dt'].max(), n_edges=n_points+1)
        kwargs = {'obj':self.obj,
                  'inner':False,
                  'features':['n', 'mean weighted a', 'mean weighted b', 'median a', 'median b', 'SF cwf a', 'SF c', 'SF', 'SF cwf p', 'SF cwf n', 'skewness', 'kurtosis', 'SF iqr'],
                  'n_points':n_points,
                  'mjd_edges': self.mjd_edges}
        
        n_processes = 4
        chunks = [group for _, group in self.dtdm.groupby(self.ID, group_keys=False)]
        self.mjd_centres = (self.mjd_edges[1:] + self.mjd_edges[:-1])/2
        all_pooled_stats = data_io.dispatch_function(pairwise.calculate_stats_looped, chunks=chunks, max_processes=n_processes, concat_output=False, **kwargs)
        self.all_pooled_stats = {uid:stats for uid, stats in zip(self.dtdm_uids, all_pooled_stats)}

    def plot_stats(self, keys, figax=None, label=None, legend_loc='upper right', plot_kwargs={}, **kwargs):
        for uid, pooled_stats in self.all_pooled_stats.items():
            self.pooled_stats = pooled_stats
            figax = dtdm_raw_analysis.plot_stats(self, keys=keys, figax=figax, label=f'uid: {uid}', legend_loc=legend_loc, plot_kwargs=plot_kwargs, **kwargs)
    
    def add_rest_frame_column(self):
        if 'mjd_rf' not in self.df.columns:
            if ~hasattr(self.df, 'redshift'):
                self.redshift = load_redshifts()
            self.df = self.df.join(self.redshift, how = 'left', on=self.ID)
            self.df['mjd_rf'] = self.df['mjd']/(1+self.df['redshift'])

    def read_coords(self):
        """
        Read in RA/Dec coords taken from DR16Q
        """
        self.coords = load_coords(self.obj)

    def read_grouped(self):
        """
        Read in precomputed grouped data
        """
        self.grouped_sdss, self.grouped_ps, self.grouped_ztf, self.grouped_ssa = load_grouped(self.obj, self.band, return_dict=False)
        self.grouped_tot = load_grouped_tot(self.obj, self.band)

    def read_vac(self, catalogue_name='dr16q_vac'):
        self.vac = load_vac(self.obj, catalogue_name=catalogue_name)

    def read_redshifts(self):
        if self.obj == 'calibStars':
            raise Exception('Stars have no redshift data')
        self.redshift = load_redshifts()

    def merge_with_catalogue(self):
        """
        Reduce self.df to intersection of self.df and catalogue.
        Compute summary() to reupdate idx_uid, uids_missing, n_qsos and idx_cat
        Create new DataFrame, self.properties, which is inner join of [df_grouped, vac] along uid.
        Add columns to self.properties:
                mag_abs_mean : mean absolute magnitude
                mjd_ptp_rf   : max ∆t in rest frame for given object

        Parameters
        ----------
        catalogue : DataFrame
                value added catalogue to be used for analysis
        remove_outliers : boolean
                remove objects which have values outside range specified in prop_range_any
        prop_range_any : dict
                dictionary of {property_name : (lower_bound, upper_bound)}

        """
        if ~hasattr(self, 'vac'):
            self.read_vac(self.obj)

        print(self.vac.index)
        self.df = self.df[self.df.index.isin(self.vac.index)]

        # Recalculate and print which qsos we are missing and which we have
        self.summary()
        self.properties = self.df_grouped.join(self.vac, how = 'inner', on=self.ID)
        
        #calculate absolute magnitude
        self.properties['mag_abs_mean'] = self.properties['mag_mean'] - 5*np.log10(3.0/7.0*self.redshifts*(10**9))
        self.properties['mjd_ptp_rf']   = self.properties['mjd_ptp']/(1+self.redshifts)

    def plot_series(self, uids, sid=None, bands='r', show_outliers=False, axes=None, **kwargs):
        fig, ax = plot_series_(self.df, uids, sid=sid, bands=bands, 
                                                       marker_dict=self.marker_dict, 
                                                       survey_dict=self.survey_dict,
                                                       plt_color=self.plt_color, 
                                                       show_outliers=show_outliers, 
                                                       axes=axes, 
                                                       **kwargs)
        return fig, ax 