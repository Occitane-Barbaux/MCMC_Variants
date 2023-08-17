import sys,os
import numpy as np

sys.path.append(os.path.abspath("/home/barbauxo/Documents/Doctorat/03_Travail/2023_05 Calcul Multimodel/Propre/Utils"))

from multiESS import *
from Utils import *
from Transitions import *
from Preparation import *
from MCMC_Run import *
from EQR import *
from Figures import *

def Full_Run_Analysis_simulated(dt_string,pathOut,model_name,prior_mean,prior_cov,true_theta=np.array([-0.3, 1.3,  0.5 ,  0.02, -0.25]),init=np.array([0,0,0,0,0]),tran_scale_G=np.array([1,1,1,1,1]),n_mcmc_drawn=10000,n_sim=37,n_features=5,TransitionAdapt=True,Within_Gibbs=True,show=True,initTrans=0.01,epsilon=0.01):
    #Create all Analysis File for 1 Run
    np.random.seed(int(dt_string))
    X,Yo,MLE_theta=Simulate(n_sim,true_theta)
    prior_law=create_prior(prior_mean,prior_cov)
    sdlaw=create_likelihood(X,Yo)
    if Within_Gibbs:
    	Cov_dis_full=[Cov_dis(i,prior_mean,prior_cov) for i in range(n_features)]
    	draw, accept=MCMC_MH_W_Gibbs(prior_law,sdlaw,tran_scale_G,prior_mean,Cov_dis_full,init,n_mcmc_drawn,n_features,TransitionAdapt)
    else:
    	draw, accept=MCMC_MH(prior_law,sdlaw,tran_scale_G,init,n_mcmc_drawn,n_features,TransitionAdapt,initTrans=initTrans,epsilon=epsilon)
    Summary_run_table(pathOut,model_name,draw,accept,n_mcmc_drawn,true_theta,MLE_theta,q_l=0.05,q_h=0.95,coef=['loc0' ,'loc1', 'scale0', 'scale1' ,'shape0'],show=show)
    Para_Runs(pathOut,draw,true_theta,MLE_theta,prior_mean,model_name,n_mcmc_drawn,prior_law,show=show)
    return draw,accept
    
    
    
def Full_Run_Analysis_Data(dt_string,
                           pathOut,
                           model_name,
                           prior_mean,
                           prior_cov,
                           X,Yo,
                           init=np.array([0,0,0,0,0]),
                           tran_scale_G=np.array([1,1,1,1,1]),
                           n_mcmc_drawn=10000,
                           n_features=5,
                           TransitionAdapt=True,
                           Within_Gibbs=True,
                           show=True,initTrans=0.01,epsilon=0.01):
	#Create all Analysis File for 1 Run
	np.random.seed(int(dt_string))
    
	#MLE theta
	ns_law=nsm.GEV()
	ns_law.fit(Yo.values,X)
	MLE_theta=ns_law.get_params()
	MLE_theta
	prior_law=create_prior(prior_mean,prior_cov)
	sdlaw=create_likelihood(X,Yo.values)
	if Within_Gibbs:
		Cov_dis_full=[Cov_dis(i,prior_mean,prior_cov) for i in range(n_features)]
		draw, accept=MCMC_MH_W_Gibbs(prior_law,sdlaw,tran_scale_G,prior_mean,Cov_dis_full,init,n_mcmc_drawn,n_features,TransitionAdapt)
	else:
		draw, accept=MCMC_MH(prior_law,sdlaw,tran_scale_G,init,n_mcmc_drawn,n_features,TransitionAdapt,initTrans=initTrans,epsilon=epsilon)
	true_theta=prior_mean
	Summary_run_table(pathOut,model_name,draw,accept,n_mcmc_drawn,true_theta,MLE_theta,q_l=0.05,q_h=0.95,coef=['loc0' ,'loc1', 'scale0', 'scale1' ,'shape0'],show=show,Simulated=False)
	true_theta=[]
	Para_Runs(pathOut,draw,true_theta,MLE_theta,prior_mean,model_name,n_mcmc_drawn,prior_law,show=show)
	return draw,accept
