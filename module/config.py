"""
This file uses easydict to set configurables which may be fetched in the module code
"""

from easydict import EasyDict as edict
__C = edict()
# Configurables may be fetched using cfg
cfg = __C

# imports
import numpy as np
import os

#------------------------------------------------------------------------------
# Path variables
#------------------------------------------------------------------------------
# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Working directory
__C.W_DIR = os.path.join(__C.ROOT_DIR, 'qso_photometry', '')

# Data directory
__C.D_DIR = os.path.join(__C.ROOT_DIR, 'data', '')

# Results directory.
__C.RES_DIR = os.path.join(__C.W_DIR, 'res', '')

# Path to thesis folder
__C.THESIS_DIR = os.path.join(__C.ROOT_DIR, 'thesis_hrb', '')

#------------------------------------------------------------------------------
# Configure environment variables
#------------------------------------------------------------------------------
os.environ['MPLCONFIGDIR'] = __C.RES_DIR
# Note, style sheets:
# https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html

#------------------------------------------------------------------------------
# User settings
#------------------------------------------------------------------------------
__C.USER = edict()

# Set below to True to use multiple cores during computationally intensive tasks.
# Single core is not currently well supported, may cause errors when setting this to False.
__C.USER.USE_MULTIPROCESSING = True
# Choose how many cores to use
__C.USER.N_CORES = 4


#------------------------------------------------------------------------------
# Data collection configurables
#------------------------------------------------------------------------------
__C.COLLECTION = edict()
# These dtypes should be checked against the datatypes provided when fetching lightcurve data
# float32 - float
# float64 - double

__C.SURVEY_LABELS = {'sdss':'SDSS',
                     'ps':'Pan-STARRS',
                     'ztf':'ZTF',
                     'ssa':'SuperCOSMOS'}

__C.SURVEY_LABELS_SHORT = {'sdss':'SDSS',
                           'ps':'PS',
                           'ztf':'ZTF',
                           'ssa':'SSS'}

# sss = 3
# sdss = 5
# ps = 7
# ztf = 11
# __C.SURVEY_LABELS_PAIRS = {9  :'sss-sss',
#                            15 :'sss-sdss',
#                            21 :'sss-ps',
#                            33 :'sss-ztf',
#                            25 :'sdss-sdss',
#                            35 :'sdss-ps',
#                            55 :'sdss-ztf',
#                            49 :'ps-ps',
#                            77 :'ps-ztf',
#                            121:'ztf-ztf'}

__C.COLLECTION.SDSS = edict()
# Datatypes
# mag converted from real[4] with similar precision
# magerr converted from real[4] with similar precision
__C.COLLECTION.SDSS.dtypes = {
                             **{'uid'    : np.uint32,  'uid_s'  : np.uint32,
                                'objID'  : np.uint64,  'mjd'    : np.float32,
                                'ra'     : np.float64, 'ra_ref' : np.float64,
                                'dec'    : np.float64, 'dec_ref': np.float64,
                                'get_nearby_distance': np.float32},
                             **{band + 'psf'   : np.float64 for band in 'ugriz'},
                             **{band + 'psferr': np.float64 for band in 'ugriz'},
                         }

__C.COLLECTION.PS = edict()
# Datatypes
__C.COLLECTION.PS.dtypes = {
                    'objID'     : np.uint64, 
                    'obsTime'   : np.float32, # converted from float[8] with reduced precision
                    'psfFlux'   : np.float64, # converted from float[8] with similar precision. Since this is flux we use double precision.
                    'psfFluxErr': np.float64, # converted from float[8] with similar precision.
                    'mjd'       : np.float32,
                    'mag'       : np.float32,
                    'magerr'    : np.float32,
                    'uid'       : np.uint32,
                    'uid_s'     : np.uint32
                         }

__C.COLLECTION.ZTF = edict()
# Datatypes
__C.COLLECTION.ZTF.dtypes = {
                    'oid'     : np.uint64, # note, uint32 is not large enough for ztf oids
                    'clrcoeff': np.float32,
                    'limitmag': np.float32,
                    'mjd'     : np.float32, # reduced from float64
                    'mag'     : np.float32,
                    'magerr'  : np.float32, 
                    'uid'     : np.uint32,
                    'uid_s'   : np.uint32
                         }

__C.COLLECTION.CALIBSTAR_dtypes = {
                            'ra'      : np.float64,
                            'dec'     : np.float64,
                            'n_epochs': np.uint32,
                            **{'mag_mean_'+b    : np.float32 for b in 'gri'},
                            **{'mag_mean_err_'+b: np.float32 for b in 'gri'}
                                }


# 'uid': np.uint32,
# 'uid_s':np.uint32,
# 'catalogue': np.uint8,
#------------------------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------------------------
__C.PREPROC = edict()

# Datatypes
__C.PREPROC.lc_dtypes = {'mjd'     : np.float32,
                         'mag'     : np.float32,
                         'mag_orig': np.float32,
                         'magerr'  : np.float32,
                         'uid'     : np.uint32,
                         'uid_s'   : np.uint32,
                         'sid'     : np.uint8}

__C.PREPROC.stats_dtypes = {'n_tot': np.uint16, # Increase this to uint32 if we think we will have more than 2^16 (65,536) observations for a single object
                            **{x:np.float32 for x in ['mjd_min','mjd_max','mjd_ptp',
                                                      'mag_min','mag_max','mag_mean','mag_med',
                                                      'mag_std',
                                                      'mag_mean_native','mag_med_native',
                                                      'mag_opt_mean','mag_opt_mean_flux','magerr_opt_std',
                                                      'magerr_max','magerr_mean','magerr_med']}}

__C.PREPROC.dtdm_dtypes = {'uid'	: np.uint32,
                           'uid_s' 	: np.uint32,
                           'dm' 	: np.float32,
                           'dm' 	: np.float32,
                           'de'		: np.float32,
                           'dm2_de2': np.float32,
                           'dsid'	: np.uint8}

# maybe not needed as the types stay consistent
__C.PREPROC.pairwise_dtypes = {'uid': np.uint32,
                               'dt' :np.float32,
                               'dm' :np.float32,
                               'de' :np.float32
                               }

# Bounds to be applied in when running calculate_stats_looped
__C.PREPROC.dtdm_bounds = {'qsos':       {'dm': (-5, 5), 'de': (1e-10, 2)},
                           'calibStars': {'dm': (-5, 5), 'de': (1e-10, 2)},
                           'sim':        {'dm': (-5, 5), 'de': (1e-10, 2)}}

# Limiting magnitudes
__C.PREPROC.LIMIT_MAG = edict()

# https://www.sdss4.org/dr16/imaging/other_info/
# 5σ limiting magnitudes
# I think these are single epoch?? SDSS doesn't really do much multiepoch imaging so it must be.
# This is just taken from the website and doesn't seem to appear in a paper. How to reference?
__C.PREPROC.LIMIT_MAG.SDSS = {
                            'u': 22.15,
                            'g': 23.13,
                            'r': 22.70,
                            'i': 22.20,
                            'z': 20.71
                            }

# https://outerspace.stsci.edu/display/PANSTARRS/PS1+FAQ+-+Frequently+asked+questions
# 5σ limiting magnitudes
__C.PREPROC.LIMIT_MAG.PS = {
                            # Below are the single epoch 5σ depths. Cite Chambers 2019
                            'g': 22.0,
                            'r': 21.8,
                            'i': 21.5,
                            'z': 20.9,
                            'y': 19.7

                            # Below are the 12 epochs stacked 5σ depths
                            # 'g': 23.3,
                            # 'r': 23.2,
                            # 'i': 23.1,
                            # 'z': 22.3,
                            # 'y': 21.4
                            }



# limitingmag can be fetched on a per-observation basis but below is an average
# Note, using limitmag to filter out observations may be biased as we are selectively removing
# 	dimmer observations.
# 5σ limiting magnitudes
__C.PREPROC.LIMIT_MAG.ZTF = {
                            'g': 20.8,
                            'r': 20.6,
                            'i': 19.9
                            }

# https://arxiv.org/abs/1607.01189, peacock_ssa
# 4σ limiting magnitudes, using the smaller of the two between UKST and POSS2.
__C.PREPROC.LIMIT_MAG.SSA = {
                                    # 4σ
                                    'g': 21.17,
                                    'r': 20.30,
                                    'i': 18.90
                                    }
                                    # 5σ
                                    # 'g': 20.26,
                                    # 'r': 19.78,
                                    # 'i': 18.38
# Table from paper above.
# Band    | 5σ    | 4σ    |
# --------|-------|-------|
# UKST  B | 20.79 | 21.19 |
# UKST  R | 19.95 | 20.30 |
# UKST  I | 18.56 | 19.94 |
# POSS2 B | 20.26 | 21.17 |
# POSS2 R | 19.78 | 20.35 |
# POSS2 I | 18.38 | 18.90 |

# Magnitude error threshold
__C.PREPROC.MAG_ERR_THRESHOLD = 0.198

# Bounds to use on parse.filter_data in average_nightly_observations.py when removing bad data.
__C.PREPROC.FILTER_BOUNDS = {'mag':(15,25),'magerr':(1e-10,2)}

__C.PREPROC.SURVEY_IDS =   {'ssa':3,
                            'sdss': 5,
                            'ps': 7,
                            'ztf': 11}

__C.PREPROC.VAC_BOUNDS = {'z':(0,5),
                          'redshift':(0,5),
                          'Lbol':(44,48),
                          'Lbol_err':(0,1),
                          'MBH_HB':(6,12),
                          'MBH_HB_err':(0,1),
                          'MBH_MgII':(6,12),
                          'MBH_MgII_err':(0,1),
                          'MBH_CIV':(6,12),
                          'MBH_CIV_err':(0,1),
                          'MBH':(6,12),
                          'MBH_err':(0,1),
                          'nEdd':(-3,1),
                          'nEdd_err':(0,2),
                          'Mi:':(-30,-20),
                          'mag_mean':(15,23.5)}

__C.PREPROC.MAX_DT = edict()
# Max ∆t for quasars and stars in rest frame, rounded up to the nearest integer.
# Calculated from mjd_ptp_rf.max() from clean/grouped_{band}.csv
__C.PREPROC.MAX_DT['REST'] = {'qsos':      {'g': 13794, 'r': 24765, 'i': 13056},
                              'calibStars':{'g':np.nan, 'r':np.nan, 'i':np.nan}}

# Do the same except with each black hole property

# Max ∆t for quasars and stars in observer frame frame, rounded up to the nearest integer
# Calculated from mjd_ptp.max() from clean/grouped_{band}.csv in grouped_analysis-NB.py
__C.PREPROC.MAX_DT['OBS']  = {'qsos':      {'g': 16513, 'r': 26702, 'i': 14698},
                              'calibStars':{'g': 15122, 'r': 26062, 'i': 12440},
                              'sim':       {'g': 25550, 'r': 25550, 'i': 25550}}
    
# Inner: using ∆m, ∆t pairs within surveys only.
__C.PREPROC.MAX_DT_INNER = edict()

__C.PREPROC.MAX_DT_INNER['REST']  = {'qsos':      {'g': 5108, 'r': 16141, 'i': 6320},
                                     'calibStars':{'g':np.nan, 'r':np.nan, 'i':np.nan}}

__C.PREPROC.MAX_DT_INNER['OBS']  = {'qsos':      {'g': 6181, 'r': 18010, 'i': 7721},
                                    'calibStars':{'g': 4377, 'r': 17504, 'i': 6942}}

__C.PREPROC.MAX_DT_COMBINED = {'outer':{'g': 15122, 'r': 26062, 'i': 13056},
                               'inner':{'g': 5108,  'r': 17504, 'i': 6942}}

# Max ∆t for quasars when splitting by black hole property, rounded up to the nearest integer
__C.PREPROC.MAX_DT_VAC = {'Lbol': {'g': [12896, 12735, 13077, 12919, 11698, 11964, 11268, 10467],
                                   'r': [23444, 22841, 23992, 22168, 22488, 21900, 19946, 18295],
                                   'i': [10631, 10624, 9652,  9677,  9998,  11184, 10280, 8603 ]},
                          'MBH':  {'g': [12896, 12735, 13077, 12919, 11698, 11964, 11268, 10467],
                                   'r': [23444, 22841, 23992, 22168, 22488, 21900, 19946, 18295],
                                   'i': [10631, 10624, 9652,  9677,  9998,  11184, 10280, 8603 ]},
                          'nEdd': {'g': [12896, 12735, 13077, 12919, 11698, 11964, 11268, 10467],
                                   'r': [23444, 22841, 23992, 22168, 22488, 21900, 19946, 18295],
                                   'i': [10631, 10624, 9652,  9677,  9998,  11184, 10280,  8603]}}

# error = p[0] * mag + p[1]
__C.PREPROC.SSA_ERROR_LINFIT = [0.0309916, -0.45708423]
#------------------------------------------------------------------------------
# Colour transformations
#------------------------------------------------------------------------------
__C.TRANSF = edict()
__C.TRANSF.SSA = edict()

# Form of transformation should be mag_native - mag_ref = c_n * color**n + ... + c_0
# where coefficients c are listed below

# https://arxiv.org/abs/1607.01189, peacock_ssa
# first coefficient is for highest order term
__C.TRANSF.SSA.PEACOCK = {'g_north': ('g-r',[-0.134, +0.078]),
                          'g_south': ('g-r',[-0.102, +0.058]),
                          'r2_north':('g-r',[+0.054, -0.012]),
                          'r2_south':('g-r',[+0.022, +0.002]),
                          'i_north': ('r-i',[+0.024, -0.008]),
                          'i_south': ('r-i',[+0.092, -0.022])}


# https://arxiv.org/abs/astro-ph/0701508, Ivezic2007_photometric_standardisation
# first coefficient is for highest order term
__C.TRANSF.SSA.IVEZIC = {'g_north':  ('g-r', [+0.2628, -0.7952, +1.0544, +0.0268]),
                         'g_south':  ('g-r', [+0.2628, -0.7952, +1.0544, +0.0268]),

                         'r1':       ('r-i', [-0.0107, +0.0050, -0.2689, -0.1540]),
                         'r2_north': ('r-i', [-0.0107, +0.0050, -0.2689, -0.1540]),
                         'r2_south': ('r-i', [-0.0107, +0.0050, -0.2689, -0.1540]),
                         
                         'i_north':  ('r-i', [-0.0307, +0.1163, -0.3341, -0.3584]),
                         'i_south':  ('r-i', [-0.0307, +0.1163, -0.3341, -0.3584])}


# __C.TRANSF.SSA.HRB = {'g_south':  ('g-r', [+0.23450781, +0.192816  ]),
#                       'r1':       ('g-r', [+0.20054183, -0.33739011]),
#                       'r2_south': ('g-r', [-0.06210134, -0.17076644]),
#                       'i_south':  ('r-i', [-0.07449726, -0.43398917]),
                      
#                       # ivezic
#                       'g_north':  ('g-r', [+0.2628, -0.7952, +1.0544, +0.0268]),
#                       'r2_north': ('r-i', [-0.0107, +0.0050, -0.2689, -0.1540]),
#                       'i_north':  ('r-i', [-0.0307, +0.1163, -0.3341, -0.3584])}

## Below is SSS -> SDSS
# __C.TRANSF.SSA.OWN = {'g_south': ('g-r', [0.234126365664, 0.192988066258]),
#                       'r1': ('g-r', [0.223064113088, -0.348041603371]),
#                       'r2_south': ('g-r', [-0.06663700799, -0.168667410171]),
#                       'i_south': ('r-i', [-0.094850522924, -0.42802724224]),
                      
#                       # ivezic
#                       'g_north':  ('g-r', [+0.2628, -0.7952, +1.0544, +0.0268]),
#                       'r2_north': ('r-i', [-0.0107, +0.0050, -0.2689, -0.1540]),
#                       'i_north':  ('r-i', [-0.0307, +0.1163, -0.3341, -0.3584])}

__C.TRANSF.SSA.OWN = {'g_south': ('g-r', [0.222067668122, 0.199906281913]),
                      'r1': ('g-r', [0.209817608231, -0.338134174909]),
                      'r2_south': ('g-r', [-0.091072654821, -0.150900424498]),
                      'i_south': ('r-i', [-0.125610015322, -0.435805256644]),
                      
                      'g_north': ('g-r', [0.222067668122, 0.199906281913]),
                      'r2_north': ('g-r', [0.209817608231, -0.338134174909]),
                      'i_north': ('r-i', [-0.125610015322, -0.435805256644])}

                    #   ivezic
                    #   'g_north':  ('g-r', [+0.2628, -0.7952, +1.0544, +0.0268]),
                    #   'r2_north': ('r-i', [-0.0107, +0.0050, -0.2689, -0.1540]),
                    #   'i_north':  ('r-i', [-0.0307, +0.1163, -0.3341, -0.3584])}

#------------------------------------------------------------------------------
# Analysis and results
#------------------------------------------------------------------------------
__C.RES = edict()






#------------------------------------------------------------------------------
# Figures
#------------------------------------------------------------------------------
__C.FIG = edict()

# Path to style files. Empty string at end ensures trailing slash
# This is no longer needed now that we have configured the MPLCONFIGDIR at the top of this file
# __C.FIG.STYLE_DIR = os.path.join(__C.RES_DIR, 'styles', '')

__C.FIG.COLORS = edict()

# Colour palettes for plots
__C.FIG.COLORS.PAIRED_BANDS = {'g':('#b2df8a', '#33a02c'),
                               'r':('#f98583', '#e31a1c'),
                               'i':('#c299D6', '#6a3d9a')}

__C.FIG.COLORS.BANDS = {'g':'#33a02c',
                        'r':'#e31a1c',
                        'i':'#6a3d9a'}

# __C.FIG.COLORS.SURVEYS = {'sdss':'#FBC599',
#                           'ps'  :'#DB5375',
#                           'ztf' :'#75B2E3',
#                           'ssa' :'#9C6AD2'}

# inverted
__C.FIG.COLORS.SURVEYS = {'ssa' :'#FBC599',
                          'ztf' :'#DB5375',
                          'ps'  :'#75B2E3',
                          'sdss':'#9C6AD2'}


__C.FIG.LABELS = edict()

__C.FIG.LABELS.PROP = {'Lbol':r'$\log_{10}( L_{\mathrm{bol}} \; [\mathrm{erg\;s}^{-1}])$',
                       'MBH' :r'$\log_{10}( M_{\mathrm{BH}}  /M_\odot)$',
                       'nEdd':r'$\log_{10}( L_{\mathrm{bol}} /L_{\mathrm{Edd}})$',
                       'z':r'$z$'}

__C.FIG.LABELS.PROPv2 = {'Lbol':r'$\log ( L_{\mathrm{bol}})$',
                         'MBH' :r'$\log ( M_{\mathrm{BH}}/M_\odot)$',
                         'nEdd':r'$\log ( n_{\mathrm{Edd}})$',
                         'z':r'$z$',
                         'wavelength':r'$\lambda$'}