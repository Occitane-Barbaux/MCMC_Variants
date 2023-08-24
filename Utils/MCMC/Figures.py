import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import arviz #for ess univariate, install with conda install -c conda-forge arviz
from tabulate import tabulate
import matplotlib.backends.backend_pdf as mpdf
from matplotlib.gridspec import GridSpec
from statsmodels.graphics.tsaplots import plot_acf

import pandas as pd
import scipy.integrate as si
import scipy.stats as sc
import sys,os
sys.path.append(os.path.abspath("/home/barbauxo/Documents/Doctorat/03_Travail/2023_05 Calcul Multimodel/Propre/Utils"))

from multiESS import *


def Summary_run(draw,accept,q_l=0.05,q_h=0.95):
    n_mcmc_drawn=len(accept)
    accept_all= np.sum(accept) / n_mcmc_drawn
    if len(accept.shape)>1:
        #Hybrid
        accept_all= np.sum(accept) / (n_mcmc_drawn*draw.shape[1])
        accept_para=accept.sum(axis=0)/n_mcmc_drawn

    else:
        accept_all= np.sum(accept) / n_mcmc_drawn
        accept_para=[accept_all]*draw.shape[1]
        
        #add effective sample
    
    effective_samples_around=multiESS(draw, b='sqroot', Noffsets=10, Nb=None)
    idata = arviz.convert_to_inference_data(np.expand_dims(draw, 0))
    effective_samples_para=arviz.ess(idata).x.to_numpy()
    #add effective sample

    mean_para=np.mean(draw, axis = 0 )
    med_para=np.median(draw, axis = 0 )
    qlow_para=np.quantile(draw,q_l, axis = 0 )
    qhigh_para=np.quantile(draw,q_h, axis = 0 )
    return accept_all,accept_para,effective_samples_around,effective_samples_para,mean_para,med_para,qlow_para,qhigh_para


def Summary_run_table(pathOut,model_name,draw,accept,true_theta,MLE_theta,q_l=0.05,q_h=0.95,coef=['loc0' ,'loc1', 'scale0', 'scale1' ,'shape0'],show=True,Simulated=True):
	n_mcmc_drawn=len(accept)
	accept_all, accept_para,effective_samples_around,effective_samples_para,mean_para,med_para,qlow_para,qhigh_para=Summary_run(draw,accept,q_l=0.05,q_h=0.95)

	if show:
		print("Total acceptation rate is "+ str(accept_all))
		print("Total number of effective samples is "+ str(effective_samples_around))
	colnames=["Paramètre", "Valeur Sim","MLE","Q_"+str(q_l), "Médiane", "Moyenne","Q_"+str(q_h), "eff sample" ,"acceptation rate"]
	a = []#np.empty(shape=(draw.shape[1], 7))
	for i in range(draw.shape[1]):
		#add name
		a.append([coef[i], true_theta[i],MLE_theta[i],qlow_para[i],med_para[i],mean_para[i],qhigh_para[i],effective_samples_para[i],accept_para[i]])
		if Simulated:

			colnames=["Paramètre", "Valeur Sim","MLE","Q_"+str(q_l), "Médiane", "Moyenne","Q_"+str(q_h), "eff sample" ,"acceptation rate"]

		else:
			colnames=["Paramètre", "Prior","MLE","Q_"+str(q_l), "Médiane", "Moyenne","Q_"+str(q_h), "eff sample" ,"acceptation rate"]

        
	table=tabulate(a, headers=colnames, tablefmt='fancy_grid')
	if show:
		print(table)
    
	with open(os.path.join( pathOut ,"Table_"+model_name+".txt"), "w") as outf:
		outf.write("Total acceptation rate is "+ str(accept_all)+"\n")
		outf.write("Total number of effective samples is "+ str(effective_samples_around)+"\n")
		outf.write(table)
        
        
        
def Para_Runs(pathOut,draw,true_theta,MLE_theta,prior_mean,model_name,prior_law,coef=['loc0' ,'loc1', 'scale0', 'scale1' ,'shape0'],show=True):
    #Parameter Analysis for 1 run
    iteration=list(range(len(draw)))
    
    ofile=os.path.join( pathOut,"ParaComp_"+model_name+".pdf" )
    pdf = mpdf.PdfPages( ofile )


    for i in range(len(coef)):
        fig = plt.figure( figsize = (20,10) )
        gs = GridSpec(2, 2, figure=fig)
        ax = fig.add_subplot( gs[0, 0] )
        plt.plot(iteration,draw[:,i])
        plt.xlabel("Iterations")
        plt.ylabel(coef[i])
        if len(true_theta)>0:
        	plt.axhline(y = true_theta[i], color = 'r', linestyle = '-',label="True Val")
        	names=["Iterations","True theta","MLE","Prior mean"]
        else:
        	names=["Iterations","MLE","Prior mean"]
        plt.axhline(y = MLE_theta[i], color = 'm', linestyle = '-',label="MLE")
        plt.axhline(y = prior_mean[i], color = 'g', linestyle = '-',label="prior mean")
        ax.legend(names,fontsize = 15)
     
        #plt.show()
        
        ax = fig.add_subplot(gs[0, 1])
        
        prior_loc0=prior_law.rvs(10000)[:,i]
        ker_b=sc.gaussian_kde(prior_loc0)
        posterior_loc0=draw[:,i]
        ker_a=sc.gaussian_kde(posterior_loc0)
        def y_pts(pt):
            y_pt = min(ker_a(pt), ker_b(pt))
            return y_pt
        # Store overlap value.
        overlap = si.quad(y_pts,min(posterior_loc0),max(posterior_loc0)) 
        overlap_per=round(overlap[0]*100,2)        
        

        sns.kdeplot(np.array(prior_loc0),color='g',fill=True)
        sns.kdeplot(np.array(posterior_loc0),color='b')
        ax=sns.histplot(np.array(posterior_loc0),stat="density")
        if len(true_theta)>0:
            plt.axvline(x= true_theta[i], color = 'r', linestyle = '-')
        plt.axvline(x= MLE_theta[i], color = 'm', linestyle = '-')
        plt.axvline(x= prior_mean[i], color = 'g', linestyle = '-')
        plt.axvline(x= np.median(draw[:,i]), color='black',linestyle = '-')
        plt.xlim(min(posterior_loc0),max(posterior_loc0))
        ax.text(0.80, 0.98, "Overlap: \n"+str(overlap_per)+"%", ha="left", va="top", transform=ax.transAxes)
        #plt.show()

        #pdf.savefig(fig)
        #plt.close(fig) 
        ax = fig.add_subplot( gs[1, :])
        plot_acf(draw[:,i],lags=120,ax=ax )
        plt.ylim([-0.1,1.1])
        plt.title("Autocorrelation " +coef[i])
        if show:
            plt.show()
        pdf.savefig(fig)
        plt.close(fig)
    swarm_plot = sns.pairplot(pd.DataFrame(draw, columns = coef),corner=True,kind="kde")
    fig = swarm_plot.fig
    pdf.savefig(fig)
    pdf.close()
    
    
    
def Comparaison(model_name,Keep,pathOut,show=True):
	#Figures for a comparison 
	#create ESS Boxplots, Acceptance boxplot
	#and save the data
	ofile=os.path.join( pathOut+"ESS_Trans_"+model_name+".pdf" )
	pdf = mpdf.PdfPages( ofile )
	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="effective_samples_around")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Full ESS")
	if show:
    		plt.show()
	pdf.savefig(fig)
	plt.close(fig)

	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="effective_samples_para_loc0")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Location0 ESS")
	if show:
		plt.show()
	pdf.savefig(fig)
	plt.close(fig)

	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="effective_samples_para_loc1")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Location1 ESS")
	if show:
    		plt.show()
	pdf.savefig(fig)
	plt.close(fig)

	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="effective_samples_para_scale0")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Scale0 ESS")
	if show:
    		plt.show()
	pdf.savefig(fig)
	plt.close(fig)

	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="effective_samples_para_scale1")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Scale1 ESS")
	if show:
    		plt.show()
	pdf.savefig(fig)
	plt.close(fig)

	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="effective_samples_para_shape0")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Shape ESS")
	if show:
    		plt.show()
	pdf.savefig(fig)
	plt.close(fig)
    
	pdf.close()
	
	ofile=os.path.join( pathOut+"Acceptance_Trans_"+model_name+".pdf" )
	pdf = mpdf.PdfPages( ofile )
	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="Accept_all")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Full Acceptance Rate")
	if show:
    		plt.show()
	pdf.savefig(fig)
	plt.close(fig)

	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="accept_para_loc0")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Location0 Acceptance Rate")
	if show:
	    plt.show()
	pdf.savefig(fig)
	plt.close(fig)
	
	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="accept_para_loc1")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Location1 Acceptance Rate")
	if show:
		plt.show()
	pdf.savefig(fig)
	plt.close(fig)
	
	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="accept_para_scale0")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Scale0 Acceptance Rate")
	if show:
		plt.show()
	pdf.savefig(fig)
	plt.close(fig)
	
	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="accept_para_scale1")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Scale1 Acceptance Rate")
	if show:
		plt.show()
	pdf.savefig(fig)
	plt.close(fig)
	
	fig, ax = plt.subplots()
	sns.boxplot(data=Keep,x="Val", y="accept_para_shape0")
	plt.xlabel("Proportionnal Coefficients")
	plt.ylabel("Shape Acceptance Rate")
	if show:
		plt.show()
	pdf.savefig(fig)
	plt.close(fig)
    
	pdf.close()
	Keep_grouped=pd.DataFrame(Keep).groupby('Val').describe()
	Keep_grouped.to_csv(pathOut+"Results_Light_"+model_name+".csv")
	Keep.to_csv(pathOut+"Results_Full_"+model_name+".csv")
