import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg
from module.plotting.common import savefigs
from module.preprocessing.parse import filter_data

class calib():
    def __init__(self, obj, band):
        self.obj = obj
        self.band = band
        self.ID = 'uid' if obj == 'qsos' else 'uid_s'
        
    def read(self):
        self.sdss = pd.read_csv(cfg.D_DIR + 'surveys/sdss/{}/unclean/{}_band/grouped.csv'.format(self.obj, self.band), index_col=self.ID)
        self.ps   = pd.read_csv(cfg.D_DIR + 'surveys/ps/{}/unclean/{}_band/grouped.csv'  .format(self.obj, self.band), index_col=self.ID)
        self.ztf  = pd.read_csv(cfg.D_DIR + 'surveys/ztf/{}/unclean/{}_band/grouped.csv' .format(self.obj, self.band), index_col=self.ID)
        self.sss  = pd.read_csv(cfg.D_DIR + 'surveys/ssa/{}/unclean/{}_band/grouped.csv' .format(self.obj, self.band), index_col=self.ID)
        # T
        # self.ps['mag_med_native'] = self.ps['mag_med']
    def join_colors(self):
        self.colors = pd.read_csv(cfg.D_DIR + 'computed/{}/colors_sdss.csv'.format(self.obj), index_col=self.ID)
        self.sdss = self.sdss.join(self.colors, on=self.ID, how='left')
        self.ps   = self.ps  .join(self.colors, on=self.ID, how='left')
        self.ztf  = self.ztf .join(self.colors, on=self.ID, how='left')
        self.sss  = self.sss .join(self.colors, on=self.ID, how='left')
        
    def calculate_offsets(self):
        offsets = self.colors
        offsets['ps-sdss_nat'] = self.ps ['mag_med'] - self.sdss['mag_med_native']
#         offsets['sdss-ztf_nat']= self.sdss['mag_med_native']-self.ztf['mag_med_native']
        offsets['ps-ztf_nat']  = self.ps  ['mag_med']-self.ztf['mag_med_native']
        offsets['ps-sss_nat']  = self.ps  ['mag_med']-self.sss['mag_med_native']
        
        offsets['ps-sdss'] = self.ps ['mag_med'] - self.sdss['mag_med']
#         offsets['sdss-ztf']= self.sdss['mag_med']-self.ztf['mag_med']
        offsets['ps-ztf']  = self.ps  ['mag_med']-self.ztf['mag_med']
        offsets['ps-sss']  = self.ps  ['mag_med']-self.sss['mag_med']
        
        offsets['mean_mag_sdss']  = self.sdss['mag_med']
        
        self.offsets = filter_data(offsets, percentiles=[0.01, 0.99], dropna=False)
        
    def sns_correlate(self, xname, yname, vmin, vmax, color='blue', limits=None, colorscale=None, g=None, save=False, **kwargs):
        """
        Seaborn jointplot with marginal distributions

        Parameters
        ----------
        xname : str
        yname : str
        vmin  : int
        vmax  : int
        color : str
        limits: list of tuples
        colorscale : str
        g : axis handle
        save : bool
        kwargs : dict, passed to ax.set()

        Returns
        -------
        value : type
            desc
        """
        data = self.offsets[[xname,yname]]

        # Setting up limits
        if limits is None:
            lims = lambda x, n: (x.mean()-n*x.std(), x.mean()+n*x.std())
            xbounds = lims(data[xname], 3) # set the window to +/- 3 sigma from the mean
            ybounds = lims(data[yname], 2.5)
        else:
            xbounds, ybounds = limits[0], limits[1]
        bounds={xname:xbounds, yname:ybounds}

        # Remove data outside limits
        data = data[((xbounds[0] < data[xname])&(data[xname] < xbounds[1])) & ((ybounds[0] < data[yname])&(data[yname] < ybounds[1]))]

        # Setting up colors
        marg_color = {'blue':'royalblue', 'red':'salmon'}[color]
        joint_color = {'blue':'Blues', 'red':'Reds'}[color]
        label_dict = {'qsos':'Quasars', 'calibStars':'Stars'}
        
        if g is None:
            g = sns.JointGrid(x=xname, y=yname, data=data, xlim=bounds[xname], ylim=bounds[yname], height=8)

        # If colorscale is 'log', use log scale and vmin, vmax as limits. Otherwise, default
        if colorscale == 'log':
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=vmin, vmax=vmax)
            vmin=None
            vmax=None
        else:
            norm = None
        from matplotlib import colormaps
        g = g.plot_joint(plt.hexbin, norm=norm, cmap='jet', vmin=vmin, vmax=vmax)#, gridsize=(200,200))
        g.ax_joint.set_facecolor(colormaps['jet'](0))
    #     g = sns.jointplot(x=xname, y=yname, data=data, xlim=bounds[xname], ylim=bounds[yname], height=6)
        g.ax_marg_x.hist(data[xname], bins=200, color=marg_color, alpha=0.5)
        g.ax_marg_y.hist(data[yname], bins=200, orientation='horizontal', density = True, color=marg_color, alpha=0.5)

        # Uncomment below to invert axis - needed when plotting absolute mag on y axis
    #     g.ax_joint.invert_yaxis()

        # Set axis labels and fontsize
        g.ax_joint.set_xlabel(kwargs['xlabel'], fontsize=25)
        g.ax_joint.set_ylabel(kwargs['ylabel'], fontsize=25)
        # Increase fontsize of ticks
        g.ax_joint.tick_params(axis='both', which='major', labelsize=20)
    #     g.ax_joint.set(**kwargs)

        # Uncomment below to add y=0 and x=0 dashed lines
        plt.axhline(y=0, lw=1.5, color='w', ls='--', dashes=(20,10))
    #     plt.axvline(x=0, lw=0.2, color='k', ls='--', dashes=(20,10))

        plt.grid(lw = 0.2, which='major')
        transf_string = 'untransformed' if yname.endswith('nat') else 'transformed'
        plt.text(0.03, 0.03, f'{label_dict[self.obj]} ({transf_string})', transform=g.ax_joint.transAxes, 
                                                             fontsize=22, 
                                                             color='white', 
                                                             ha='left', 
                                                             va='bottom')
                                                            #  bbox=dict(facecolor='white'))

        # Save
        if save:
            savefigs(g, f'colour_transf/RESIDUALS_2D_{self.obj}_{self.band}_{xname}_{yname}', 'chap2')

        # Return axis handle
        return g