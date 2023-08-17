import numpy as np
import scipy.stats as sc
import sys,os

sys.path.append(os.path.abspath("/home/barbauxo/Documents/Doctorat/03_Travail/2023_05 Calcul Multimodel/Propre/Utils"))

from Utils import *
from Transitions import *
from multiESS import *

def MCMC_MH(prior_law,sdlaw,tran_scale_G=np.array([1,1,1,1,1]),init=np.array([0,0,0,0,0]),n_mcmc_drawn=10000,n_features=5,TransitionAdapt=True,initTrans=0.01,epsilon=0.01):
	draw = np.zeros( (n_mcmc_drawn,n_features) )
	accept = np.zeros( n_mcmc_drawn , dtype = np.bool )



	draw[0,:]     = init
	lll_current   =  -sdlaw._negloglikelihood(draw[0,:])
	prior_current = prior_law.logpdf(draw[0,:]).sum()
	p_current     = prior_current + lll_current

	for i in range(1,n_mcmc_drawn):
		if TransitionAdapt:
			draw[i,:] = transition_adaptative(draw[i-1,:],i,draw[:(i-1),:],init=initTrans,epsilon=epsilon)
		else:
			draw[i,:] = transition(draw[i-1,:],tran_scale_G)
		## Likelihood and probability of new points
		lll_next=-sdlaw._negloglikelihood(draw[i,:])
		prior_next = prior_law.logpdf(draw[i,:]).sum()
		p_next     = prior_next + lll_next
    
        	## Accept or not ?
		p_accept = np.exp( p_next - p_current )
		#print(i)
		if np.random.uniform() < p_accept:
			lll_current   = lll_next
			prior_current = prior_next
			p_current     = p_next
			accept[i] = True
		else:
			draw[i,:]= draw[i-1,:]
			accept[i] = False

	return draw, accept

def MCMC_MH_W_Gibbs(prior_law,sdlaw,tran_scale_G=np.array([1,1,1,1,1]),prior_mean=np.array([0,0,0,0,0]),Cov_dis_full=np.array([0,0,0,0,0]),init=np.array([0,0,0,0,0]),n_mcmc_drawn=10000,n_features=5,TransitionAdapt=True):
	draw = np.zeros( (n_mcmc_drawn,n_features) )
	accept = np.zeros((n_mcmc_drawn,n_features) , dtype = np.bool )
	prior_current=np.zeros(n_features)
	draw[0,:]     = init
	lll_current   =  -sdlaw._negloglikelihood(draw[0,:])
	for j in range(n_features):
		Cond_mean= norm_cond(j,prior_mean,InfoCov=Cov_dis_full[j],Val_used=draw[0,:])
		Cond_cov=Cov_dis_full[j][0]
		Conditionnal_law    = sc.multivariate_normal( mean = Cond_mean , cov = Cond_cov , allow_singular = True )
		prior_current[j] = Conditionnal_law.logpdf(draw[0,j]).sum()
    
	#prior_current = prior_law.logpdf(draw[0,:]).sum()
	p_current     = prior_current[0] + lll_current
	prev_sigma=[0,0,0,0,0]
	for i in range(1,n_mcmc_drawn):
		draw[i,:]=draw[i-1,:]
        
		for j in range(n_features):#
			if TransitionAdapt:
				draw[i,j],prev_sigma[j] = transition_SCAM(x=draw[i,j] ,i=i,draw=draw[:(i),j],prev_sigma=prev_sigma[j])
			else:
				draw[i,j] = draw[i-1,j] + np.random.normal( size = 1 , scale = tran_scale_G[j] )
			## Likelihood and probability of new points
			lll_next   = - sdlaw._negloglikelihood(draw[i,:])
			Cond_mean= norm_cond(j,prior_mean,InfoCov=Cov_dis_full[j],Val_used=draw[i,:])
			Cond_cov=Cov_dis_full[j][0]
			Conditionnal_law    = sc.multivariate_normal( mean = Cond_mean , cov = Cond_cov , allow_singular = True )
			prior_next = Conditionnal_law.logpdf(draw[i,j]).sum()
			#prior_next = prior_law.logpdf(draw[i,:]).sum()
			p_current     = prior_current[j] + lll_current
			p_next     = prior_next + lll_next

			## Accept or not ?
			p_accept = np.exp( p_next - p_current )
			if np.random.uniform() < p_accept:
				lll_current   = lll_next
				prior_current[j]=prior_next
				p_current     = p_next
				accept[i,j] = True      
			else:
				draw[i,j] = draw[i-1,j]
				accept[i,j] = False
	return draw, accept    

