
import numpy as np

#Usefull Functions
def matrix_squareroot( M , disp = False ):##{{{
	"""
	NSSEA.matrix_squareroot
	=======================
	Method which compute the square root of a matrix (in fact just call scipy.linalg.sqrtm), but if disp == False, never print warning
	
	Arguments
	---------
	M   : np.array
		A matrix
	disp: bool
		disp error (or not)
	
	Return
	------
	Mp : np.array
		The square root of M
	"""
	Mh = scl.sqrtm( M , disp = disp )
	if not disp:
		Mh = Mh[0]
	return np.real(Mh)
##}}}

#Hybride
def norm_cond(i,prior_mean,InfoCov,Val_used):
    mu_2=np.delete(prior_mean,i,0)
    mu_1=prior_mean[i]
    val_theta_i=np.delete(Val_used,i,0)
    mu_dis=mu_1+InfoCov[1].dot(InfoCov[2].dot(val_theta_i-mu_2))
    return(mu_dis)

def Cov_dis(i,Mean_Prior,Cov_Prior):
    Sigma_11=Cov_Prior[i:i+1,i:i+1]

    Sigma_21=np.delete(Cov_Prior,i,0)[:,i]

    Sigma_12=np.delete(Cov_Prior,i,1)[i,:]

    Sigma_22=np.delete(np.delete(Cov_Prior,i,1),i,0)
    
    Inv_SIgna_22=np.linalg.inv(Sigma_22)
    
    Cov_dis=Sigma_11-Sigma_12.dot(Inv_SIgna_22.dot(Sigma_21))
    return([Cov_dis[0],Sigma_12,Inv_SIgna_22])
    
def file_name(n_sim,Info_prior,InitType,n_mcmc_drawn,dt_string,Trans=0.1,mcmcType="MCMC"):
    #Create file name using infos
    if Info_prior:
        prior="_PriorInfo"
    else:
        prior="_PriorNoInfo"
    if InitType=="0":
        #Initis only 0
        init ="_Init0"
    elif InitType=="randomPrior":
        #init is random from the prior
        init ="_InitRandomRVS"
    elif InitType=="Good":
        #init is the true values
        init ="_InitTrueVal"
    else:
        #Init is far away from the true value
        init ="_InitFalseVal"
    name=dt_string+"_"+mcmcType+prior+init+"_NSim"+str(n_sim)+"_Trans"+str(Trans)+"_NIt"+str(n_mcmc_drawn)
    return name
