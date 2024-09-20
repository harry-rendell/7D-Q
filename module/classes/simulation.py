import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.modelling.carma import generate_mock_dataset
from module.assets import load_grouped

class simulated():
    def __init__(self, seed=42):
        self.seed = seed

    # def generate_data_from_piecewise_exponential(self, uid, k1, k2, t0):
    #     noise = np.random.normal(0, 0.2, self.n)
    #     t = np.linspace(0, self.t_max, self.n)
    #     mag = piecewise_exponential(t, k1, k2, t0) + noise
    #     return pd.DataFrame(data={'uid': uid, 'mjd':t, 'mag':mag, 'magerr':0.01, 'sid':0})

    # def generate_lcs(self, n_obj):
    #     df_list = []
    #     k1 = np.random.normal(self.k1, 0.2, size=n_obj)
    #     k2 = np.random.normal(self.k2, 0.1, size=n_obj)
    #     t0 = np.random.normal(self.t0, 0.1, size=n_obj)
    #     for i in range(n_obj):
    #         df_list.append(self.generate_data_from_piecewise_exponential(i, k1[i], k2[i], t0[i]))
        
        # self.df = pd.concat(df_list, ignore_index=True).set_index('uid')

    def load_survey_features(self):
        survey_dict = load_grouped('qsos', bands='r')
        surveys = ['ssa', 'sdss', 'ps', 'ztf']
        self.survey_features = {s:{key:(survey_dict[s][key].mean(), survey_dict[s][key].std()) for key in ['mjd_min', 'mjd_max', 'n_tot']} for s in surveys}

    def generate_drw_lcs(self, n_obj, frac=0.1):
        if not hasattr(self, 'survey_features'):
            self.load_survey_features()
        # kwargs={'survey_features':None, 'band':'r', 'superposed_model':'piecewise_exponential', 'frac':0.1}
        kwargs = {'survey_features':self.survey_features, 'band':'r', 'frac':frac, 'seed':self.seed}
        uids = np.arange(1,n_obj+1)
        self.df = generate_mock_dataset(uids, kwargs).set_index('uid')
        self.uids = self.df.index.unique()

    def calculate_dtdm(self, uids=None):
        """
        Save (∆t, ∆m) pairs from lightcurves. 

        Parameters
        ----------
        uids : array_like
            uids of objects to be used for calcuation
        time_key : str
            either mjd or mjd_rf for regular and rest frame respectively
        Returns
        -------
        df : DataFrame
            DataFrame(columns=[self.ID, 'dt', 'dm', 'de', 'dm2_de2', 'dsid'])
        """
        if uids is None:
            uids = self.uids
        sub_df = self.df[['mjd', 'mag', 'magerr', 'sid']].loc[uids]
        dtdm_list = []
        for uid, group in sub_df.groupby('uid'):
            #maybe groupby then iterrows? faster?
            mjd_mag = group[['mjd','mag']].values
            magerr = group['magerr'].values
            sid	 = group['sid'].values
            n = len(mjd_mag)
            # dtdm defined as: ∆m = (m2 - m1), ∆t = (t2 - t1) where (t1, m1) is the first obs and (t2, m2) is the second obs.
            # Thus a negative ∆m corresponds to a brightening of the object
            unique_pair_indicies = np.triu_indices(n,1)

            dsid = sid*sid[:,np.newaxis]
            dsid = dsid[unique_pair_indicies]

            dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
            dtdm = dtdm[unique_pair_indicies]
            dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

            dmagerr = ( magerr**2 + magerr[:,np.newaxis]**2 )**0.5
            dmagerr = dmagerr[unique_pair_indicies]

            dm2_de2 = dtdm[:,1]**2 - dmagerr**2

            duid = np.full(int(n*(n-1)/2),uid,dtype='uint32')

            # collate data to DataFrame and append
            dtdm_list.append(pd.DataFrame({'uid':duid,'dt':dtdm[:,0],'dm':dtdm[:,1], 'de':dmagerr, 'dm2_de2':dm2_de2, 'dsid':dsid}))
            
        self.dtdms = pd.concat(dtdm_list, ignore_index=True).set_index('uid')
        
    def plot_series(self, uids, survey=None, axes=None, **kwargs):
        """
        Plot lightcurve of given objects

        Parameters
        ----------
        uids : array_like
                uids of objects to plot
        catalogue : int
                Only plot data from given survey
        survey : 1 = SSS_r1, 3 = SSS_r2, 5 = SDSS, 7 = PS1, 11 = ZTF

        """
        if np.issubdtype(type(uids),np.integer): uids = [uids]
        fig_ = None
        if axes is None:
            fig_, axes = plt.subplots(len(uids),1,figsize = (10,1.5*len(uids)), sharex=True)
        if len(uids)==1:
            axes=[axes]

        for uid, ax in zip(uids,axes):
            x = self.df.loc[uid].sort_values('mjd') #single obj
#             ax.errorbar(x['mjd'], x['mag'], yerr = x['magerr'], lw = 1, **kwargs)
            ax.plot(x['mjd'], x['mag'], lw = 1, **kwargs)
#             ax.invert_yaxis()
            ax.set(xlabel='MJD', ylabel='mag')
            # ax.text(0.02, 0.85, 'uid: {}'.format(uid), transform=ax.transAxes, fontsize=10)
            ax.invert_yaxis()
            
        plt.subplots_adjust(hspace=0.1)

        if fig_ is not None:
            return fig_, axes
        
    def calculate_drift(self, log_bins=False):
        drift = []
        if log_bins:
            mjd_edges = np.logspace(np.log10(self.dtdms['dt'].min()),np.log10(self.dtdms['dt'].max()),31)
        else:
            mjd_edges = np.linspace(0,self.dtdms['dt'].max(),31)
        mjd_centres = (mjd_edges[:-1] + mjd_edges[1:])/2
        for mjd_lower, mjd_upper in zip(mjd_edges[:-1], mjd_edges[1:]):
            subdf = self.dtdms[(mjd_lower < self.dtdms['dt']) & (self.dtdms['dt'] < mjd_upper)]
            n = len(subdf)
            drift.append( (subdf['dm'].sum()/n) )
        return mjd_centres, np.array(drift)
    
    def calculate_sf(self, log_bins=False):
        sf = []
        if log_bins:
            mjd_edges = np.logspace(np.log10(self.dtdms['dt'].min()),np.log10(self.dtdms['dt'].max()),31)
        else:
            mjd_edges = np.linspace(0,self.dtdms['dt'].max(),31)
        mjd_centres = (mjd_edges[:-1] + mjd_edges[1:])/2
        for mjd_lower, mjd_upper in zip(mjd_edges[:-1], mjd_edges[1:]):
            subdf = self.dtdms[(mjd_lower < self.dtdms['dt']) & (self.dtdms['dt'] < mjd_upper)]
            n = len(subdf)
            sf.append( ( (subdf['dm']**2).sum()/n )**0.5 )
        return mjd_centres, np.array(sf)
    
    def calculate_sf_asym(self):
        sf_p = []
        sf_n = []
        mjd_edges = np.logspace(np.log10(self.dtdms['dt'].min()),np.log10(self.dtdms['dt'].max()),31)
        mjd_centres = (mjd_edges[:-1] + mjd_edges[1:])/2
        for mjd_lower, mjd_upper in zip(mjd_edges[:-1], mjd_edges[1:]):
            subdf = self.dtdms[(mjd_lower < self.dtdms['dt']) & (self.dtdms['dt'] < mjd_upper)]
            n = len(subdf)
            sf_p.append( ( (subdf['dm'][subdf['dm']>0]**2).sum()/n )**0.5 )
            sf_n.append( ( (subdf['dm'][subdf['dm']<0]**2).sum()/n )**0.5 )
        return mjd_centres, np.array(sf_p), np.array(sf_n)