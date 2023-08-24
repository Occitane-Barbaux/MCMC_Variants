import numpy as np

import NSSEA.models as nsm
import scipy.stats as sc

def Simulate(n,true_theta):
    ###Simulate Data X et YO from known parameters theta for MCMC
    X=np.linspace(start=-1, stop=+1, num=n)
    ns_law=nsm.GEV()
    sdlaw=ns_law.sdlaw( method = "bayesian" )
    sdkwargs = ns_law._get_sdkwargs(X)
    sdlaw._rhs.lhs_.n_samples=len(X)


    ## Init LHS/RHS
    sdlaw._lhs.n_samples = X.size
    sdlaw._rhs.build(**sdkwargs)

    sdlaw.coef_ = true_theta
    ## Generate and Add Y
    Yo=sc.genextreme( loc = sdlaw.loc , scale = sdlaw.scale , c = - sdlaw.shape ).rvs()
    
    ns_law.fit(Yo,X)
    MLE_theta=ns_law.get_params()
    return X,Yo,MLE_theta

def create_prior(prior_mean,prior_cov):
	prior_law    = sc.multivariate_normal( mean = prior_mean , cov = prior_cov , allow_singular = True )
	return prior_law
	
def create_likelihood(X,Yo):
    ns_law=nsm.GEV()
    sdlaw=ns_law.sdlaw( method = "bayesian" )
    sdkwargs = ns_law._get_sdkwargs(X)
    sdlaw._rhs.lhs_.n_samples=len(X)


    ## Init LHS/RHS
    sdlaw._lhs.n_samples = X.size
    sdlaw._rhs.build(**sdkwargs)

    sdlaw.coef_ = [0,0,0,0,0]
    sdlaw._Y = Yo.reshape(-1,1)
    return sdlaw
