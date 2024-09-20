from .config import cfg
import pandas as pd
import numpy as np
import os
from .preprocessing import parse, color_transform

def load_grouped(obj, bands='gri', return_dict=True, clean=True, **kwargs):
    """
    Load grouped data for a given object and band.
    If return_dict = True, then return a dictionary of DataFrames.
    Otherwise, return in the order:
        sdss, ps, ztf, ssa, tot
    """
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    clean_str = 'clean' if clean else 'unclean'

    if len(bands) == 1:
        sdss = pd.read_csv(cfg.D_DIR + f'surveys/sdss/{obj}/{clean_str}/{bands}_band/grouped.csv',index_col=ID, **kwargs)
        ps   = pd.read_csv(cfg.D_DIR + f'surveys/ps/{obj}/{clean_str}/{bands}_band/grouped.csv',  index_col=ID, **kwargs)
        ztf  = pd.read_csv(cfg.D_DIR + f'surveys/ztf/{obj}/{clean_str}/{bands}_band/grouped.csv', index_col=ID, **kwargs)
        ssa  = pd.read_csv(cfg.D_DIR + f'surveys/ssa/{obj}/{clean_str}/{bands}_band/grouped.csv', index_col=ID, **kwargs)
    else:
        sdss = {b:pd.read_csv(cfg.D_DIR + f'surveys/sdss/{obj}/{clean_str}/{b}_band/grouped.csv', index_col=ID, **kwargs) for b in bands}
        ps   = {b:pd.read_csv(cfg.D_DIR + f'surveys/ps/{obj}/{clean_str}/{b}_band/grouped.csv',   index_col=ID, **kwargs) for b in bands}
        ztf  = {b:pd.read_csv(cfg.D_DIR + f'surveys/ztf/{obj}/{clean_str}/{b}_band/grouped.csv',  index_col=ID, **kwargs) for b in bands}
        ssa  = {b:pd.read_csv(cfg.D_DIR + f'surveys/ssa/{obj}/{clean_str}/{b}_band/grouped.csv',  index_col=ID, **kwargs) for b in bands}
    
    if return_dict:
        return {'sdss':sdss, 'ps':ps, 'ztf':ztf, 'ssa':ssa}
    else:
        return sdss, ps, ztf, ssa
    
def load_grouped_tot(obj, bands=None, clean=True, **kwargs):
    clean_str = 'clean' if clean else 'unclean'
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    if len(bands) == 1:
        tot  = pd.read_csv(cfg.D_DIR + f'merged/{obj}/{clean_str}/grouped_{bands}.csv', index_col=ID, **kwargs)
    else:
        tot = {b:pd.read_csv(cfg.D_DIR + f'merged/{obj}/{clean_str}/grouped_{b}.csv', index_col=ID, **kwargs) for b in bands}

    return tot

def load_sets(obj, band, **kwargs):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    return pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/sets/clean_{band}.csv', index_col=ID, comment='#', **kwargs)

def load_coords(obj, **kwargs):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    return pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/{obj}_subsample_coords.csv', index_col=ID, comment='#', **kwargs)

def load_redshifts(**kwargs):
    if 'usecols' in kwargs: kwargs['usecols'] += ['uid']
    return pd.read_csv(cfg.D_DIR + f'catalogues/qsos/dr14q/dr14q_redshift.csv', index_col='uid', **kwargs).squeeze()

def load_n_tot(obj, **kwargs):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    return pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/n_tot.csv', index_col=ID, **kwargs)

def load_drw_mcmc_fits(bands, bounds={'a':(0,0.01),'loc':(2,5),'scale':(0.1,1), 'z':(0.2,5), 'tau16':(0,5), 'tau50':(0,5), 'tau84':(0,6)}, dropna=True):
    # Load skewfit data
    ID = 'uid'
    vac = load_vac('qsos', usecols=['z','Lbol','MBH','nEdd'])
    drw_mcmc_fits = []
    for band in bands:
        # s = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/rest/{band}_sdss_ps_ztf_30.csv", index_col=ID)
        s = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/rest/{band}_all_0_best_phot.csv", index_col=ID)
        # s = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/rest/{band}_all_0.csv", index_col=ID)
        s['band'] = band
        vac['wavelength'] = color_transform.calculate_wavelength(band, vac['z'])
        s = s.join(vac, on=ID, how='left')
        drw_mcmc_fits.append(s)
    drw_mcmc_fits = pd.concat(drw_mcmc_fits).sort_index()
    if bounds:
        drw_mcmc_fits = parse.filter_data(drw_mcmc_fits, bounds=bounds, verbose=True, dropna=dropna)

    # Add extra columns
    drw_mcmc_fits['loglambda'] = np.log10(drw_mcmc_fits['wavelength']/3000)
    drw_mcmc_fits['logz'] = np.log10(drw_mcmc_fits['z'])
    drw_mcmc_fits['sigerr'] = (drw_mcmc_fits['sig84'] - drw_mcmc_fits['sig16'])/2
    drw_mcmc_fits['tauerr'] = (drw_mcmc_fits['tau84'] - drw_mcmc_fits['tau16'])/2
    drw_mcmc_fits['logmjd_ptp'] = np.log10(drw_mcmc_fits['mjd_ptp'])
    
    return drw_mcmc_fits

def load_all_features(bands, n_bins, bounds={'a':(0,0.01),'loc':(2,5),'scale':(0.1,1), 'z':(0.2,5), 'tau16':(0,5), 'tau50':(0,6), 'tau84':(0,7)}, dropna=True):
    ID = 'uid'
    obj = 'qsos'
    vac = load_vac('qsos', usecols=['z','Lbol','MBH','nEdd'])
    sf = []
    for band in bands:
        s = pd.read_csv(cfg.D_DIR + f'computed/{obj}/features/{band}/SF_{n_bins}_bins_all_pairs.csv', index_col=ID)
        # drw_mcmc_fits = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/rest/{band}_sdss_ps_ztf_30.csv", index_col=ID)
        drw_mcmc_fits = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/rest/{band}_all_0_best_phot.csv", index_col=ID)
        # drw_mcmc_fits = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/obs/{band}_sdss_ps_30.csv", index_col=ID)
        # drw_mcmc_fits = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/rest/{band}_all_0.csv", index_col=ID)
        drw_mcmc_fits = parse.filter_data(drw_mcmc_fits, bounds=bounds, verbose=True, dropna=dropna)
        s = s.join(drw_mcmc_fits, on=ID, how='inner')
        s['band'] = band
        vac['wavelength'] = color_transform.calculate_wavelength(band, vac['z'])
        s = s.join(vac, on=ID, how='left')
        sf.append(s)

    sf = pd.concat(sf).sort_index()
    
    # Add extra columns
    sf['loglambda'] = np.log10(sf['wavelength']/3000)
    sf['logz'] = np.log10(sf['z'])
    sf['sigerr'] = (sf['sig84'] - sf['sig16'])/2
    sf['tauerr'] = (sf['tau84'] - sf['tau16'])/2
    sf['logmjd_ptp'] = np.log10(sf['mjd_ptp'])
    return sf

def load_vac(obj, catalogue_name='dr16q_vac', **kwargs):
    """
    Load value-added catalogues for our quasar sample.
    Options of:
        dr12_vac
            Kozlowski 2016
            https://arxiv.org/abs/1609.09489
        dr14_vac
            Rakshit 2020
            https://arxiv.org/abs/1910.10395
        dr16q_vac : Preferred
            Shen 2022
            https://arxiv.org/abs/2209.03987
    """
    ID = 'uid'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    if obj == 'calibStars':
        raise Exception('Stars have no value-added catalogues')
    
    if catalogue_name == 'dr12_vac':
        # cols = z, Mi, L5100, L5100_err, L3000, L3000_err, L1350, L1350_err, MBH_MgII, MBH_CIV, Lbol, Lbol_err, nEdd, sdss_name, ra, dec, uid
        fpath = 'catalogues/qsos/dr12q/SDSS_DR12Q_BH_matched.csv'
        prop_range_all = {'Mi':(-30,-20),
                          'mag_mean':(15,23.5),
                          'mag_std':(0,1),
                          'redshift':(0,5),
                          'Lbol':(44,48),
                          'nEdd':(-3,0.5)}

    elif catalogue_name == 'dr14_vac':
        # cols = ra, dec, uid, sdssID, plate, mjd, fiberID, z, pl_slope, pl_slope_err, EW_MgII_NA, EW_MgII_NA_ERR, FWHM_MgII_NA, FWHM_MgII_NA_ERR, FWHM_MgII_BR, FWHM_MgII_BR_ERR, EW_MgII_BR, EW_MgII_BR_ERR, MBH_CIV, MBH_CIV_ERR, MBH, MBH_ERR, Lbol
        fpath = 'catalogues/qsos/dr14q/dr14q_spec_prop_matched.csv'
        prop_range_all = {'mag_mean':(15,23.5),
                          'mag_std':(0,1),
                          'redshift':(0,5),
                          'Lbol':(44,48)}

    elif catalogue_name == 'dr16q_vac':
        # cols = ra, dec, redshift_vac, Lbol, Lbol_err, MBH_HB, MBH_HB_err, MBH_MgII, MBH_MgII_err, MBH_CIV, MBH_CIV_err, MBH, MBH_err, nEdd, nEdd_err
        fpath = 'catalogues/qsos/dr16q/dr16q_vac_shen_matched.csv'
        prop_range_all = cfg.PREPROC.VAC_BOUNDS
    else:
        raise Exception('Unrecognised value-added catalogue')

    vac = pd.read_csv(os.path.join(cfg.D_DIR,fpath), index_col=ID, **kwargs)
    # vac = vac.rename(columns={'z':'redshift_vac'});
    if (catalogue_name == 'dr16q_vac') & ('nEdd' in vac.columns):
        # Note, in dr16q, bad nEdd entries are set to 0 (exactly) so we can remove those.
        vac['nEdd'] = vac['nEdd'].where((vac['nEdd']!=0).values)
    
    vac = parse.filter_data(vac, prop_range_all, dropna=False)

    return vac