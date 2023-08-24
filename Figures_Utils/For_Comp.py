import sys,os
import tarfile
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import SDFC.link as sdl
import NSSEA as ns
import NSSEA.plot as nsp
import NSSEA.models as nsm
import scipy.stats as sc
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples

import cftime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf
import matplotlib.patches as mplpatch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

from scipy.stats._morestats import _calc_uniform_order_statistic_medians

import seaborn as sns



def para_distribution_Comp(clim,climCXCB, pathOut, name,ci=0.05):
    #Cilm is Clim_CX here
    custom_params = {"axes.spines.bottom": True,"axes.spines.right": True,"axes.spines.left": True, "axes.spines.top":True}
    sns.set_theme(style="whitegrid",rc=custom_params)
    
    ofile=os.path.join( pathOut , name+"Comp_Distributions_Para.pdf" )
    pdf = mpdf.PdfPages( ofile )

    qcoef = clim.law_coef[:,1:,:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample" ).assign_coords( quantile = ["ql","qu","BE"] )
    qcoefX=climCXCB.law_coef[:,1:,:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample_MCMC" ).assign_coords( quantile = ["ql","qu","BE"] )

    ymin = float( (clim.law_coef ).min())
    ymax = float( (clim.law_coef ).max())
    delta = 0.1 * (ymax-ymin)
    ylim = (ymin-delta,ymax+delta)

    kwargs = {  "showmeans" : False , "showextrema" : False , "showmedians" : True }
    models=clim.data.model.values
    m='Multi_Synthesis'
    fig = plt.figure( figsize = ( 16 , 10 ) )

    for i in range(clim.n_coef):

        ax = fig.add_subplot(1,clim.n_coef,i+1)

        vplot = ax.violinplot( ((clim.law_coef) )[:,1:,:].loc[clim.law_coef.coef[i],:,m] , **kwargs )
        vplotc = ax.violinplot( ((climCXCB.law_coef) )[:,1:,:].loc[climCXCB.law_coef.coef[i],:,m]  , **kwargs )
    
        for pc in vplotc["bodies"]:
            pc.set_facecolor("red")
            pc.set_edgecolor("red")
            pc.set_alpha(0.5)

        for pc in vplot["bodies"]:
            pc.set_facecolor("blue")
            pc.set_edgecolor("blue")
            pc.set_alpha(0.3)


        for q in ["ql","qu","BE"]:
                a=ax.hlines( qcoefX[:,i,:].loc[q,m] , 1 - 0.1 , 1 + 0.1 , color = "red", label= "posterior" )
                ax.hlines( qcoef[:,i,:].loc[q,m] , 1 - 0.1 , 1 + 0.1 , color = "blue",label="prior" )
        ax.vlines( 1 , qcoefX[:,i,:].loc["BE",m] , qcoef[:,i,:].loc["BE",m] , color = "grey" )
        ax.set_xticks([1])
        q=qcoefX[:,i,:].loc["BE",m] - qcoef[:,i,:].loc["BE",m] 
        xticks = [ "{}".format(clim.ns_law.get_params_names(True)[i]) + " {}".format( "+" if np.sign(q) > 0 else "-" ) + r"${}$".format(float(np.sign(q)) * round(float(q),2)) ]
        ax.set_xticklabels( xticks , fontsize = 13 )
        if ((clim.law_coef) )[:,1:,:].loc[clim.law_coef.coef[i],:,m].min()*((clim.law_coef) )[:,1:,:].loc[clim.law_coef.coef[i],:,m].max()<0:
            ax.hlines( 0 , 1-0.3 , 1+0.3, color = "black" )
    ax.legend(["prior","posterior"],fontsize = 20)
    fig.set_tight_layout(True)

    fig.suptitle( " ".join(m.split("_")) , fontsize = 25 )
    plt.show()
    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()

