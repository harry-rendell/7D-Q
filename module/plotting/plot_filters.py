import matplotlib.pyplot as plt
import pandas as pd
from ..config import cfg

def plot_filters(fig, ax, **kwargs):
	"""
    Plots SDSS, PanSTARRS, ZTF filters
	use fig, ax = plt.subplots(1,1, figsize=(10,5))
    Returns
    -------
	fig, ax, sdss, ps, ztf
	
    """
	wdir = cfg.D_DIR + 'surveys/filters/'
	sdss  = pd.read_csv(wdir + 'sdss.csv')
	ztf_g = pd.read_csv(wdir + 'raw/ztf/ztf_g.csv')
	ztf_r = pd.read_csv(wdir + 'raw/ztf/ztf_r.csv')
	ztf_i = pd.read_csv(wdir + 'raw/ztf/ztf_i.csv')
	ps    = pd.read_csv(wdir + 'ps.csv')


	# Colors
	colors = [cfg.FIG.COLORS.BANDS[c] for c in 'gri']

	# Getting everything into the same units
	sdss.loc[:, list('ugriz')] *= 100
	ps.loc[:, list('grizy')] *= 100
	ztf_g.loc[:,'lambda'] *= 10
	ztf_r.loc[:,'lambda'] *= 10
	ztf_i.loc[:,'lambda'] *= 10
	ps.loc[:,'lambda'] *= 10

	
	ztf_g.plot(x='lambda', y='g', ax=ax, ls='-', lw=1, ms=10, label=r''+cfg.SURVEY_LABELS['ztf'] + ' $g$', color = cfg.FIG.COLORS.BANDS['g'])
	ztf_r.plot(x='lambda', y='r', ax=ax, ls='-', lw=1, ms=10, label=r''+cfg.SURVEY_LABELS['ztf'] + ' $r$', color = cfg.FIG.COLORS.BANDS['r'])
	ztf_i.plot(x='lambda', y='i', ax=ax, ls='-', lw=1, ms=10, label=r''+cfg.SURVEY_LABELS['ztf'] + ' $i$', color = cfg.FIG.COLORS.BANDS['i'])

	ps  .plot(x='lambda', y=['g','r','i'], ax=ax, ls=(0, (3, 1, 1, 1)), ms=1, label=[r''+cfg.SURVEY_LABELS['ps'  ]+' $'+ b + '$' for b in 'gri'], color = colors)
	sdss.plot(x='lambda', y=['g','r','i'], ax=ax, ls='--', dashes=(5,1), ms=1, label=[r''+cfg.SURVEY_LABELS['sdss']+' $'+ b + '$' for b in 'gri'], color = colors)
# 	sdss.plot(x='lambda', y=['g','r','i','z'], ax=ax, ls='-' , ms=10, label=['sdss_'+b for b in 'griz'], color = list('grbk'))
# 	ps  .plot(x='lambda', y=['g','r','i','z'], ax=ax, ls='-.', ms=10, label=['ps_' + b for b in 'griz'], color = list('grbk'))

	ax.set(xlabel='Wavelength (Å)', ylabel='Transmission (%)', **kwargs);
	ax.get_legend().remove()
	ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1,1.22))
	ztf = [ztf_g, ztf_r, ztf_i]
	
	return sdss, ps, ztf