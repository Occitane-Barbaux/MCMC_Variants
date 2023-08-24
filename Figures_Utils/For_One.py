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



def para_time(clim,params,pathOut,name,ci=0.05,T=100):
	#Only works for clim with all models if used full ns_fit (Slow)
	#Ex: para_time(clim_CXCB_MHWG,paramsCXCB_MHWG,pathOut,name="Clim_CXCB_MHWG_"+dt_string)
	time = clim.time

	ofile=os.path.join( pathOut ,name+ "_Parametres_Dist.pdf" )
	pdf = mpdf.PdfPages( ofile )
	## Quantile
	l_params = [k for k in clim.ns_law.lparams]
	n_param  = len(l_params)
	qparams  = params[:,1:,:,:,:].quantile( [ ci / 2 , 1 - ci / 2 , 0.5 ] , dim = "sample" ).assign_coords( quantile = ["ql","qu","BE"] )
	#	if not clim.BE_is_median:
	#		qparams.loc["BE",:,:,:,:] = params.loc[:,"BE",:,:,:]

	ymin = params.min( dim = ["forcing","time","sample"] )
	ymax = params.max( dim = ["forcing","time","sample"] )
	xlim = [time.min(),time.max()]
	deltax = 0.05 * ( xlim[1] - xlim[0] )
	xlim[0] -= deltax
	xlim[1] += deltax
	deb=1850
	fin=2101
	models=clim.data.model.values
	for j in range(0,len(models)):
		m=models[j]
		
		fig = plt.figure( figsize = (n_param*10,15) )
		fig.suptitle( " ".join(m.split("_")) )
		gs = GridSpec(3, 3, figure=fig)
		gs.tight_layout(fig,pad=1.5,h_pad=10, w_pad=100,rect=[0, 0.0, 1, 2])
		#fig.tight_layout(rect=[0, 0.0, 1, 1])

		for i,p in enumerate(qparams.param):
			ax = fig.add_subplot( gs[0, i] )
			ax.plot( time , qparams.loc["BE",time,"F",p,m] , color = "red" )
			ax.fill_between( time , qparams.loc["ql",time,"F",p,m] , qparams.loc["qu",time,"F",p,m] , color = "red" , alpha = 0.5 )
			ax.set_ylim( (float(ymin.loc[p,m]),float(ymax.loc[p,m])) )
			ax.set_ylabel(str(p.values))
			ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f'{y:.2f}'))

		ax = fig.add_subplot( gs[1:2, 0] )
		Q_100=[sc.genextreme.ppf((1-1/T), -qparams.loc["BE",time,"F",qparams.param[2],m][i], qparams.loc["BE",time,"F",qparams.param[0],m][i],qparams.loc["BE",time,"F",qparams.param[1],m][i]) for i in list(range(0,fin-deb))]
		Q_100_u=[sc.genextreme.ppf((1-1/T), -qparams.loc["qu",time,"F",qparams.param[2],m][i], qparams.loc["qu",time,"F",qparams.param[0],m][i],qparams.loc["qu",time,"F",qparams.param[1],m][i]) for i in list(range(0,fin-deb))]
		Q_100_l=[sc.genextreme.ppf((1-1/T), -qparams.loc["ql",time,"F",qparams.param[2],m][i], qparams.loc["ql",time,"F",qparams.param[0],m][i],qparams.loc["ql",time,"F",qparams.param[1],m][i]) for i in list(range(0,fin-deb))]
		ax.plot( time ,Q_100, color = "red" )
		ax.fill_between( time , Q_100_l , Q_100_u , color = "red" , alpha = 0.5 )
		ax.set_ylabel("Niveau de retour Annuel " +str(T)+ " ans")
    

		#EQR
		#Ql_EQR,Qm_EQR,EQR_BE ,P_BE,EQR,Qu_EQR,EQR_S,EQR_P=Find_EQR_IC(m,params,T,T1,T2,ci,xlen=10000)
		#ax=fig.add_subplot( gs[1:2, 1:3] )
		#ax=sns.violinplot(EQR_S,orient="h",color = "red"  )
		#plt.setp(ax.collections, alpha=.5)
		#sns.boxplot(EQR_S, showfliers=False, showbox=False, whis=[2.5,97.5], orient='h')
		#plt.plot(EQR_BE, 0,'go' )
		#ax.set_ylabel("EQR " +str(T)+ " ans entre " + str(T1)+ " et "+str(T2)) 
		plt.show()
    
		pdf.savefig(fig)
		plt.close(fig)

	pdf.close()


def QQ_plot_Obs_IC(clim,params,Yo,Xo,pathOut,name,ci=0.05):
    ofile=os.path.join( pathOut , name+"_QQplot_GEV_GCM_Modeles_ICBoostrap.pdf" )
    pdf = mpdf.PdfPages( ofile )

    qparams  = params[:,1:,:,:,:].quantile( [ ci / 2 , 1 - ci / 2 , 0.5 ] , dim = "sample" ).assign_coords( quantile = ["ql","qu","BE"] )
    models=clim.data.model.values
    for j in range(0,len(models)):
        m=models[j]

        fig = plt.figure( figsize = (10,10) )
        ax = fig.add_subplot( 1 , 1 ,  1 )


        tY    = Yo.index
        X     = Xo.loc[tY]
        #X=X.to_pandas()
        residuals=[]
        for i in range(0,len(Yo)):
            shape=qparams.loc["BE",Yo.index[i],"F",'shape',m].values
            loc=qparams.loc["BE",Yo.index[i],"F",'loc',m].values
            scale=qparams.loc["BE",Yo.index[i],"F",'scale',m].values
        
            #residuals.append((1/shape)*np.log(1+shape*(Yo.iloc[i].values[0]-loc)/scale))
            residuals.append((Yo.iloc[i].values[0]-loc)/scale)
        sm.qqplot(np.asarray(residuals),dist=sc.genextreme,distargs=(-shape,), line="45",ax=ax)
        ax.set_ylabel("Sample Quantile for Yo (Obs)", fontsize = 20 )
    
    
        #Ajout IC 

        tirages=1000
        dist=sc.genextreme
        sparams=(-shape,)
        osm_uniform = _calc_uniform_order_statistic_medians(len(residuals))
        osm = dist.ppf(osm_uniform, *sparams)
        OSR=np.zeros(shape=(tirages, len(residuals)))
        for s in range(0,tirages):
            idx = np.random.choice( len(residuals), len(residuals) , replace = True )

            x_s= np.asarray(residuals)[idx]
            OSR[s] = np.sort(x_s)
    
    
        qu=np.quantile(OSR,1-ci / 2,axis=0)
        ql=np.quantile(OSR,ci / 2,axis=0)
        ax.plot(osm, qu,color="purple")
        ax.plot(osm, ql,color="purple")
    
        fig.suptitle( " ".join(m.split("_")) , fontsize = 25 )
        plt.show()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    
def para_distribution_MCMC(clim, pathOut, name,ci=0.05):
    #para_distribution_MCMC(clim_CXCB_MHWG, pathOut, name)
    custom_params = {"axes.spines.bottom": True,"axes.spines.right": True,"axes.spines.left": True, "axes.spines.top":True}
    sns.set_theme(style="whitegrid",rc=custom_params)
    
    ofile=os.path.join( pathOut , name+"Distributions_Para.pdf" )
    pdf = mpdf.PdfPages( ofile )

    qcoef = clim.law_coef[:,1:,:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample_MCMC" ).assign_coords( quantile = ["ql","qu","BE"] )

    ymin = float( (clim.law_coef ).min())
    ymax = float( (clim.law_coef ).max())
    delta = 0.1 * (ymax-ymin)
    ylim = (ymin-delta,ymax+delta)

    kwargs = {  "showmeans" : False , "showextrema" : False , "showmedians" : True }
    models=clim.data.model.values
    for j in range(0,len(models)):
        m=models[j]
    
       
        fig = plt.figure( figsize = ( 16 , 10 ) )

        for i in range(clim.n_coef):

            ax = fig.add_subplot(1,clim.n_coef,i+1)

            vplot = ax.violinplot( ((clim.law_coef) )[:,1:,:].loc[clim.law_coef.coef[i],:,m] , **kwargs )
            for pc in vplot["bodies"]:
                pc.set_facecolor("blue")
                pc.set_edgecolor("blue")
                pc.set_alpha(0.3)


            for q in ["ql","qu"]:
                ax.hlines( qcoef[:,i,:].loc[q,m] , 1 - 0.3 , 1 + 0.3 , color = "blue" )

            ax.set_xticks([1])
            q=qcoef.loc["BE",:,m][i]
            xticks = [ "{}".format(clim.ns_law.get_params_names(True)[i]) + " {}".format( "+" if np.sign(q) > 0 else "-" ) + r"${}$".format(float(np.sign(q)) * round(float(q),2)) ]
            ax.set_xticklabels( xticks , fontsize = 13 )
            if ((clim.law_coef) )[:,1:,:].loc[clim.law_coef.coef[i],:,m].min()*((clim.law_coef) )[:,1:,:].loc[clim.law_coef.coef[i],:,m].max()<0:
                ax.hlines( 0 , 1-0.3 , 1+0.3, color = "black" )

        fig.set_tight_layout(True)

        fig.suptitle( " ".join(m.split("_")) , fontsize = 25 )
        plt.show()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    
def Para_Correlation(clim, pathOut,name):
    #Para_Correlation(clim_CXCB_MHWG, pathOut,name)
    all_s=clim.law_coef[:,1:,0].assign_coords( quantile = "BE" ).to_pandas().T
    all_s['Best_Estimate']=0
    be=clim.law_coef[:,1:,:].quantile( 0.5 , dim = "sample_MCMC" ).assign_coords( quantile = "BE" ).to_pandas().T
    be['Best_Estimate']=1
    total=pd.concat([all_s,be])

    sns.pairplot(
    total ,hue='Best_Estimate',
     kind='reg'
    )
    plt.savefig(os.path.join( pathOut ,name+"_Parameters_Correlations.png"))
    plt.show(block=False)

