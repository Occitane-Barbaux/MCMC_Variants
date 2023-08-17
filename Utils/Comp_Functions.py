import pandas as pd
import sys,os
import numpy as np
import time
sys.path.append(os.path.abspath("/home/barbauxo/Documents/Doctorat/03_Travail/2023_05 Calcul Multimodel/Propre/Utils"))

from multiESS import *
from Utils import *
from Transitions import *
from Preparation import *
from MCMC_Run import *
from Figures import *

def Run_Comp_Val(Trans_values,
	pathOut,
	model_name,
	prior_mean,
	prior_cov,
	true_theta=np.array([-0.3, 1.3,  0.5 ,  0.02, -0.25]),
	init=np.array([0,0,0,0,0]),
	tran_scale_G_O=np.array([1,1,1,1,1]),
	n_mcmc_drawn=10000,
	n_IC=1000,
	n_sim=37,
	n_features=5,
	TransitionAdapt=True,
	Within_Gibbs=True,
	Precision=""):
	##Used to compare result for several value of variance
	coef=['loc0' ,'loc1', 'scale0', 'scale1' ,'shape0']
	b=pd.DataFrame()
	Keep=pd.DataFrame()
	Trans0="MH"+Within_Gibbs*"_WGibbs"+TransitionAdapt*"Adaptative"+(1-TransitionAdapt)*"Fixed"+Precision
	for i in range(len(Trans_values)):
		#Pour chaque valeur du parametre
		Trans=Trans0+str(Trans_values[i])
		tran_scale_G=tran_scale_G_O*Trans_values[i]
		for k in range(n_IC):
			print((k+n_IC*i)/(n_IC*len(Trans_values)))
			X,Yo,MLE_theta=Simulate(n_sim,true_theta)
			prior_law=create_prior(prior_mean,prior_cov)
			sdlaw=create_likelihood(X,Yo)
			t0 = time.time()
			if Within_Gibbs:
				Cov_dis_full=[Cov_dis(i,prior_mean,prior_cov) for i in range(n_features)]
				draw, accept=MCMC_MH_W_Gibbs(prior_law,sdlaw,tran_scale_G,prior_mean,Cov_dis_full,init,n_mcmc_drawn,n_features,TransitionAdapt)
			else:
				draw, accept=MCMC_MH(prior_law,sdlaw,tran_scale_G,init,n_mcmc_drawn,n_features,TransitionAdapt)
			t1 = time.time()
			accept_all, accept_para,effective_samples_around,effective_samples_para,mean_para,med_para,qlow_para,qhigh_para=Summary_run(draw,accept,n_mcmc_drawn,q_l=0.05,q_h=0.95)
			c=pd.DataFrame()
			c['Accept_all']= [accept_all]
			c['effective_samples_around']=effective_samples_around
			for j in range(len(coef)):
				c["accept_para_"+coef[j]]=accept_para[j]
				c['effective_samples_para_'+coef[j]]=effective_samples_para[j]
				c['mean_para'+coef[j]]=mean_para[j]
			c['Time']=t1-t0
			c['Type']="Lambda_"+str(Trans)
			c['Val']=Trans_values[i]
			
    
			Keep=pd.concat([c,Keep],ignore_index=True)
	return(Keep)
			        		
def Run_Comp_Val_Data(Trans_values,
	pathOut,
	model_name,
	prior_mean,
	prior_cov,
	clim,Yo,
	init=np.array([0,0,0,0,0]),
	tran_scale_G_O=np.array([1,1,1,1,1]),
	n_mcmc_drawn=10000,
	n_IC=1000,
	n_features=5,
	TransitionAdapt=True,
	Within_Gibbs=True,
	Precision=""):
	##Used to compare result for several value of variance
	coef=['loc0' ,'loc1', 'scale0', 'scale1' ,'shape0']
	b=pd.DataFrame()
	Keep=pd.DataFrame()
	Trans0="MH"+Within_Gibbs*"_WGibbs"+TransitionAdapt*"Adaptative"+(1-TransitionAdapt)*"Fixed"+Precision
	for i in range(len(Trans_values)):
		#Pour chaque valeur du parametre
		Trans=Trans0+str(Trans_values[i])
		tran_scale_G=tran_scale_G_O*Trans_values[i]
		for k in range(n_IC):
			print((k+n_IC*i)/(n_IC*len(Trans_values)))
			#MLE theta
			s=clim.sample[k]
			X= clim.X.loc[Yo.index,s,"F","Multi_Synthesis"].values.squeeze()
			ns_law=nsm.GEV()
			ns_law.fit(Yo.values,X)
			MLE_theta=ns_law.get_params()
			prior_law=create_prior(prior_mean,prior_cov)
			sdlaw=create_likelihood(X,Yo.values)
			t0 = time.time()
			if Within_Gibbs:
				Cov_dis_full=[Cov_dis(i,prior_mean,prior_cov) for i in range(n_features)]
				draw, accept=MCMC_MH_W_Gibbs(prior_law,sdlaw,tran_scale_G,prior_mean,Cov_dis_full,init,n_mcmc_drawn,n_features,TransitionAdapt)
			else:
				draw, accept=MCMC_MH(prior_law,sdlaw,tran_scale_G,init,n_mcmc_drawn,n_features,TransitionAdapt)
			t1 = time.time()
			accept_all, accept_para,effective_samples_around,effective_samples_para,mean_para,med_para,qlow_para,qhigh_para=Summary_run(draw,accept,n_mcmc_drawn,q_l=0.05,q_h=0.95)
			c=pd.DataFrame()
			c['Accept_all']= [accept_all]
			c['effective_samples_around']=effective_samples_around
			for j in range(len(coef)):
				c["accept_para_"+coef[j]]=accept_para[j]
				c['effective_samples_para_'+coef[j]]=effective_samples_para[j]
				c['mean_para'+coef[j]]=mean_para[j]
			c['Time']=t1-t0
			c['Type']="Lambda_"+str(Trans)
			c['Val']=Trans_values[i]
			
    
			Keep=pd.concat([c,Keep],ignore_index=True)
	return(Keep)
	
	
def Run_Comp(Trans_values,Trans0,Trans_array,n_IC,pathOut):
	#Common init
	#Old version
	true_theta=np.array([-0.3, 1.3,  0.5 ,  0.02, -0.25])
	n_sim=37
	Info_prior= False
	show=False
	InitType="0"
	n_mcmc_drawn=10000
	coef=['loc0' ,'loc1', 'scale0', 'scale1' ,'shape0']
	b=pd.DataFrame()

	Keep=pd.DataFrame()


	model_name=file_name(n_sim,Info_prior,InitType,n_mcmc_drawn,dt_string="",Trans=Trans0,mcmcType="MCMC")

	for i in range(len(Trans_values)):
    		#Pour chaque valeur du parametre
    		Trans=Trans0+str(Trans_values[i])
    		tran_scale_G=Trans_array*Trans_values[i]
    		for k in range(n_IC):
        		#Tirages pour IC
        		print(k)
        		prior_law,sdlaw,init,prior_mean,MLE_theta=prepare_MCMC(n_sim,true_theta,Info_prior,InitType,show=show)
        		draw, accept=MCMC_Run(init,transition,tran_scale_G,prior_law,sdlaw,n_mcmc_drawn,n_features=5)
        		accept_all, accept_para,effective_samples_around,effective_samples_para,mean_para,med_para,qlow_para,qhigh_para=Summary_run(draw,accept,n_mcmc_drawn,q_l=0.05,q_h=0.95)
        		c=pd.DataFrame()
        		c['Accept_all']= [accept_all]
        		c['effective_samples_around']=effective_samples_around
        		for j in range(len(coef)):
            			c["accept_para_"+coef[j]]=accept_para[j]
            			c['effective_samples_para_'+coef[j]]=effective_samples_para[j]
            			c['mean_para'+coef[j]]=mean_para[j]
        		c['Type']="Lambda_"+str(Trans)
        		c['Val']=Trans_values[i]
    
        		Keep=pd.concat([c,Keep],ignore_index=True)
