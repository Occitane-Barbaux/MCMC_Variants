import numpy as np
import scipy.stats as sc
import sys,os
import xarray as xr
import NSSEA.models as nsm
import arviz #for ess univariate, install with conda install -c conda-forge arviz
#sys.path.append(os.path.abspath("/home/barbauxo/Documents/Doctorat/03_Travail/2023_05 Calcul Multimodel/Propre/Utils"))

sys.path.append(os.path.abspath("/home/barbauxo/Documents/Doctorat/03_Travail/2023_08 Clean Run/Utils/MCMC"))
from Utils import *
from Transitions import *
from multiESS import *
from Preparation import *
from Figures import *

from NSSEA.__tools import ProgressBar
from NSSEA.__covariates import GAM_FC
from NSSEA.__tools import matrix_positive_part
from NSSEA.__tools import matrix_squareroot
from NSSEA.__multi_model import MultiModel


def covariates_FC_GAM_pygam_light( clim , lX , XN , dof = 7 , verbose = False ):##{{{
	"""
	NSSEA.covariates_FC_GAM_pygam
	=============================
	
	Same parameters that NSSEA.covariates_FC_GAM, use pygam package.
	
	"""
	## Parameters
	models   = clim.model
	time     = clim.time
	n_model  = clim.n_model
	n_time   = clim.n_time
	samples  = clim.sample
	n_sample = clim.n_sample
	
	## verbose
	pb = ProgressBar( n_model  , "covariates_FC_GAM" , verbose )
	
	## Define output
	dX = xr.DataArray( np.zeros( (n_time,n_sample + 1,2,n_model) ) , coords = [time , samples , ["F","C"] , models ] , dims = ["time","sample","forcing","model"] )
	
	## Define others prediction variables
	time_C   = np.repeat( time[0] , n_time )
	
	## Main loop
	for X in lX:
		pb.print()
		model = X.columns[0]
		
		xn = XN.values[:,0]
		XF = np.stack( (time  ,xn) , -1 )
		XC = np.stack( (time_C,xn) , -1 )
		
		## GAM decomposition
		gam_model = GAM_FC( dof )
		gam_model.fit( np.stack( (X.index,XN.loc[X.index,0].values) , -1 ) , X.values )
		
		## prediction
		dX.loc[:,"BE","F",model] = gam_model.predict( XF )
		dX.loc[:,"BE","C",model] = gam_model.predict( XC )
		
		## Distribution of GAM coefficients
		gam_law = gam_model.error_distribution()
		coefs_  = gam_law.rvs(n_sample)
		
	
	clim.X = dX
	pb.end()
	return clim

def nslaw_fit_light( lY , clim , verbose = False ):
	"""
	NSSEA.nslaw_fit
	===============
	Fit non stationary parameters -Light Version - Only keep best estimate (Used for Light Multi synthesis)
	
	Arguments
	---------
	lY     : list
		List of models
	clim : NSSEA.Climatology
		Climatology variable
	verbose: bool
		Print or not state of execution
	
	Return
	------
	clim : NSSEA.climdef nslaw_fit_light( lY , clim , verbose = False ):
	"""
	## Parameters
	models      = clim.model
	n_models    = clim.n_model
	sample      = clim.sample
	n_sample    = clim.n_sample
	
	n_ns_params = clim.ns_law.n_ns_params
	ns_params_names = clim.ns_law.get_params_names()
	
	
	law_coef   = xr.DataArray( np.zeros( (n_ns_params,n_sample + 1,n_models) ) , coords = [ ns_params_names , sample , models ] , dims = ["coef","sample","model"] )
	
	pb = ProgressBar( n_models , "nslaw_fit" , verbose )
	for Y in lY:
		pb.print()
		model = Y.columns[0]
		tY    = Y.index
		X     = clim.X.loc[tY,"BE","F",model]
		
		law = clim.ns_law
		law.fit(Y.values,X.values)
		law_coef.loc[:,"BE",model] = law.get_params()
		
	
	clim.law_coef = law_coef
	pb.end()
	return clim

	
	
	

def infer_multi_model_light( clim , verbose = False ):
	"""
	NSSEA.infer_multi_model
	=======================
	Infer multi-model synthesis. A new model called "Multi_Synthesis" is added
	to "clim", synthesis of the model. The parameters are given
	in "clim.synthesis".
	Light Version in the calculated variance. Does not need the 1000 samples
	
	Arguments
	---------
	clim : [NSSEA.Climatology] Clim variable
	verbose  : [bool] Print (or not) state of execution
	
	Return
	------
	clim: [NSSEA.Climatology] The clim with the multi model synthesis with
	      name "Multi_Synthesis"
	
	"""
	
	pb = ProgressBar( 3 , "infer_multi_model" , verbose )
	
	## Parameters
	##===========
	n_time    = clim.n_time
	n_coef    = clim.n_coef
	n_sample  = clim.n_sample
	n_model   = clim.n_model
	sample    = clim.sample
	n_mm_coef = 2 * n_time + n_coef
	
	## Big matrix
	##===========
	mm_data                        = np.zeros( (n_mm_coef,n_sample + 1,n_model) )
	mm_data[:n_time,:,:]           = clim.X.loc[:,:,"F",:].values
	mm_data[n_time:(2*n_time),:,:] = clim.X.loc[:,:,"C",:].values
	mm_data[(2*n_time):,:,:]       = clim.law_coef.values
	pb.print()
	
	## Multi model parameters inference
	##=================================
	mmodel = MultiModel()
	pb.print()
	mmodel.mean = np.mean( mm_data[:,0,:] , axis = 1 )
	n_params_cov,n_sample_cov,n_models_cov = mm_data.shape
	SSM     = np.cov( mm_data[:,0,:] ) 
	mmodel.cov  = ( n_models_cov + 1 ) / n_models_cov * SSM
	
	
	## Generate sample
	##================
	name = "Multi_Synthesis"
	mm_sample = xr.DataArray( np.zeros( (n_time,n_sample + 1,2,1) ) , coords = [ clim.time , sample , clim.data.forcing , [name] ] , dims = ["time","sample","forcing","model"] )
	mm_params = xr.DataArray( np.zeros( (n_coef,n_sample + 1,1) )   , coords = [ clim.law_coef.coef.values , sample , [name] ]     , dims = ["coef","sample","model"] )
	
	mm_sample.loc[:,"BE","F",name] = mmodel.mean[:n_time]
	mm_sample.loc[:,"BE","C",name] = mmodel.mean[n_time:(2*n_time)]
	mm_params.loc[:,"BE",name]     = mmodel.mean[(2*n_time):]
	
	for s in sample[1:]:
		draw = mmodel.rvs()
		mm_sample.loc[:,s,"F",name] = draw[:n_time]
		mm_sample.loc[:,s,"C",name] = draw[n_time:(2*n_time)]
		mm_params.loc[:,s,name]     = draw[(2*n_time):]
	pb.print()
	
	
	## Add multimodel to clim
	##=======================
	data = xr.Dataset( { "X" : mm_sample , "law_coef" : mm_params } )
	clim.data = xr.concat( [clim.data,data] , dim = "model" , data_vars = "minimal" )
#	X        = xr.concat( [clim.X , mm_sample] , "model" )
#	law_coef = xr.concat( [clim.law_coef,mm_params] , "model" )
#	clim.data = xr.Dataset( { "X" : X , "law_coef" : law_coef } )
	
	## Add multimodel to xarray, and add to clim
	##==========================================
	index = [ "{}F".format(t) for t in clim.time ] + [ "{}C".format(t) for t in clim.time ] + clim.data.coef.values.tolist()
	dmm_mean  = xr.DataArray( mmodel.mean , dims = ["mm_coef"] , coords = [index] )
	dmm_cov   = xr.DataArray( mmodel.cov  , dims = ["mm_coef","mm_coef"] , coords = [index,index] )
	clim.data = clim.data.assign( { "mm_mean" : dmm_mean , "mm_cov" : dmm_cov } )
	
	pb.end()
	
	return clim
	
def MCMC_MH_Stop_ESS(prior_law,sdlaw,tran_scale_G=np.array([1,1,1,1,1]),init=np.array([0,0,0,0,0]),n_features=5,TransitionAdapt=True,initTrans=0.01,epsilon=0.01,n_sortie=100,burn_in=1000):
	n_mcmc_drawn =10000 #Max nb of iterations
	draw = np.zeros( (n_mcmc_drawn,n_features) )
	accept = np.zeros( n_mcmc_drawn , dtype = np.bool )



	draw[0,:]     = init
	lll_current   =  -sdlaw._negloglikelihood(draw[0,:])
	prior_current = prior_law.logpdf(draw[0,:]).sum()
	p_current     = prior_current + lll_current
    
	inMCMC=True
	i=0
	while inMCMC:
		i=i+1
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
		if (i>(2*burn_in))&((i%n_sortie)==0):
			#print(draw[burn_in:i,:])
			idata = arviz.convert_to_inference_data(np.expand_dims(draw[burn_in:i,:], 0))
			effective_samples_para=arviz.ess(idata).x.to_numpy()
			#ess=multiESS(draw[burn_in:i,:], b='sqroot', Noffsets=10, Nb=None)
			#print(str(i)+" iteration, ESS: "+str(ess))
			#print(str(min(effective_samples_para))+"  for iteration : "+ str(i))
			#print(effective_samples_para)
			if min(effective_samples_para) > n_sortie:
				inMCMC=False
				#print(min(effective_samples_para))            
				#print(i)  
	return draw[burn_in:i,:], accept[burn_in:i]

def draw_MCMC_MH(prior_law,sdlaw,tran_scale_G=np.array([1,1,1,1,1]),init=np.array([0,0,0,0,0]),n_features=5,TransitionAdapt=True,initTrans=0.01,epsilon=0.01,n_sortie=100,burn_in=1000):
    draw, accept=MCMC_MH_Stop_ESS(prior_law,sdlaw,tran_scale_G,init,n_features,TransitionAdapt,initTrans=initTrans,epsilon=epsilon,n_sortie=n_sortie,burn_in=burn_in)
    n_tirage=(len(draw)//n_sortie)
    return draw[0::n_tirage][:n_sortie]
    

def constrain_MCMC_MH_law_all( clim , Yo,pathOut,model_name,tran_scale_G=np.array([1,1,1,1,1]),TransitionAdapt=True,initTrans=0.01,epsilon=0.01,n_sortie=100,burn_in=1000,verbose=True , **kwargs ):##{{{
	clim2 = clim.copy()
	n_ns_params = clim.ns_law.n_ns_params
	sample    = clim.sample
	n_sample=len(sample)
	ns_params_names = clim.ns_law.get_params_names()
	pb = ProgressBar( clim.n_sample + 1 , "constrain_law" , verbose )
	
	n_features=clim.ns_law.n_ns_params
	init=np.array([0]*n_features)
	
	## Define prior
	prior_mean   = clim.data["mm_mean"][-clim.n_coef:].values
	prior_cov    = clim.data["mm_cov"][-clim.n_coef:,-clim.n_coef:].values
	prior_law    = sc.multivariate_normal( mean = prior_mean , cov = prior_cov , allow_singular = True )
	results=np.array([])
	sample_names =[s+"_"+str(i) for i in range(n_sortie) for s in sample[1:]]+["BE"]

	law_coef_bay   = xr.DataArray( np.zeros( (n_ns_params,(n_sample-1)*n_sortie + 1,1) ) , coords = [ ns_params_names , sample_names , ["Multi_Synthesis"] ] , dims = ["coef","sample_MCMC","model"] )
	
	#Illustration
	s="BE"
	X   = clim.X.loc[Yo.index,s,"F","Multi_Synthesis"].values.squeeze()
	#ns_law=nsm.GEV()
	ns_law=clim.ns_law
	ns_law.fit(Yo.values,X)
	MLE_theta=ns_law.get_params()
	sdlaw=create_likelihood(X,Yo.values)
	draw, accept=MCMC_MH_Stop_ESS(prior_law,sdlaw,tran_scale_G,init,n_features,TransitionAdapt,initTrans=initTrans,epsilon=epsilon,n_sortie=n_sortie,burn_in=burn_in)
	true_theta=prior_mean
	Summary_run_table(pathOut, model_name,draw,accept,true_theta,MLE_theta,q_l=0.05,q_h=0.95,coef=clim.ns_law.get_params_names(),show=False,Simulated=False)
	true_theta=[]
	Para_Runs(pathOut,draw,true_theta,MLE_theta,prior_mean,model_name,prior_law,coef=clim.ns_law.get_params_names(),show=False) 
	## And now MCMC loop
	for s in clim.sample[1:]:
		pb.print()
		X   = clim.X.loc[Yo.index,s,"F","Multi_Synthesis"].values.squeeze()
		#ns_law=nsm.GEV()
		ns_law=clim.ns_law
		ns_law.fit(Yo.values,X)
		MLE_theta=ns_law.get_params()
		sdlaw=create_likelihood(X,Yo.values)

		#draw = clim.ns_law.drawn_bayesian( Yo.values.squeeze() , X , n_mcmc_drawn , prior_law , min_rate_accept )
		draw=draw_MCMC_MH(prior_law,sdlaw,tran_scale_G,init,n_features,TransitionAdapt,initTrans,epsilon,n_sortie=n_sortie,burn_in=burn_in)
		law_coef_bay.loc[:,[s+"_"+str(i) for i in range(n_sortie)],"Multi_Synthesis"]=draw.T
        
		#clim.law_coef.loc[:,s,"Multi_Synthesis"] = draw

    
#	for s in range(len(results)):
#		law_coef_bay.loc[:,sample_names[s],"Multi_Synthesis"]=results[s]
	#clim2.n_sample=(n_sample-1)*n_sortie + 1
	#clim2.sample=sample_names
	clim2.law_coef=law_coef_bay
	clim2.law_coef.loc[:,"BE",:] = clim2.law_coef[:,1:,:].median( dim = "sample_MCMC" )
	clim2.BE_is_median = True
	
	pb.end()
	
	return clim2


def MCMC_MHWG_Stop_ESS(prior_law,sdlaw,tran_scale_G=np.array([1,1,1,1,1]),init=np.array([0,0,0,0,0]),n_features=5,TransitionAdapt=True,initTrans=0.01,epsilon=0.01,n_sortie=100,burn_in=1000):
	n_mcmc_drawn =100000 #Max nb of iterations
	draw = np.zeros( (n_mcmc_drawn,n_features) )
	accept = np.zeros((n_mcmc_drawn,n_features) , dtype = np.bool )



	draw[0,:]     = init
	lll_current   =  -sdlaw._negloglikelihood(draw[0,:])
	#prior_current=np.zeros(n_features)
	prior_current = prior_law.logpdf(draw[0,:]).sum()
	p_current     = prior_current + lll_current
	prev_sigma=np.array([0]*n_features)   
	inMCMC=True
	i=0
	while inMCMC:
		i=i+1
		draw[i,:]=draw[i-1,:]     
		for j in range(n_features):#
			if TransitionAdapt:
				draw[i,j],prev_sigma[j] = transition_SCAM(x=draw[i,j] ,i=i,draw=draw[:(i),j],prev_sigma=prev_sigma[j])
			else:
				draw[i,j] = draw[i-1,j] + np.random.normal( size = 1 , scale = tran_scale_G[j] )
			## Likelihood and probability of new points
			lll_next   = - sdlaw._negloglikelihood(draw[i,:])
			prior_next = prior_law.logpdf(draw[i,:]).sum()
			p_current     = prior_current + lll_current
			p_next     = prior_next + lll_next
			## Accept or not ?
			p_accept = np.exp( p_next - p_current )
			if np.random.uniform() < p_accept:
				lll_current   = lll_next
				prior_current = prior_next
				p_current     = p_next
				accept[i,j] = True      
			else:
				draw[i,j] = draw[i-1,j]
				accept[i,j] = False
		if i>(2*burn_in):
			#print(draw[burn_in:i,:])
			idata = arviz.convert_to_inference_data(np.expand_dims(draw[burn_in:i,:], 0))
			effective_samples_para=arviz.ess(idata).x.to_numpy()
			#ess=multiESS(draw[burn_in:i,:], b='sqroot', Noffsets=10, Nb=None)
			#print(str(i)+" iteration, ESS: "+str(ess))
			#print(str(min(effective_samples_para))+"  for iteration : "+ str(i))
			#print(effective_samples_para)
			if min(effective_samples_para) > n_sortie:
				inMCMC=False
				#print(min(effective_samples_para))            
				#print(i)  
	return draw[burn_in:i,:], accept[burn_in:i,:]

def draw_MCMC_MHWG(prior_law,sdlaw,tran_scale_G=np.array([1,1,1,1,1]),init=np.array([0,0,0,0,0]),n_features=5,TransitionAdapt=True,initTrans=0.01,epsilon=0.01,n_sortie=100,burn_in=1000):
	
	draw, accept=MCMC_MHWG_Stop_ESS(prior_law,sdlaw,tran_scale_G,init,n_features,TransitionAdapt,initTrans=initTrans,epsilon=epsilon,n_sortie=n_sortie,burn_in=burn_in)
	n_tirage=(len(draw)//n_sortie)
	return draw[0::n_tirage][:n_sortie]
    

def constrain_MCMC_MHWG_law_all( clim , Yo,pathOut,model_name,tran_scale_G=np.array([1,1,1,1,1]),TransitionAdapt=True,initTrans=0.01,epsilon=0.01,n_sortie=100,burn_in=1000,  verbose=True , **kwargs ):##{{{
	clim2 = clim.copy()
	n_ns_params = clim.ns_law.n_ns_params
	sample    = clim.sample
	n_sample=len(sample)
	ns_params_names = clim.ns_law.get_params_names()
	pb = ProgressBar( clim.n_sample + 1 , "constrain_law" , verbose )
	
	n_features=clim.ns_law.n_ns_params
	init=np.array([0]*n_features)
	## Define prior
	prior_mean   = clim.data["mm_mean"][-clim.n_coef:].values
	prior_cov    = clim.data["mm_cov"][-clim.n_coef:,-clim.n_coef:].values
	prior_law    = sc.multivariate_normal( mean = prior_mean , cov = prior_cov , allow_singular = True )
	results=np.array([])
	sample_names =[s+"_"+str(i) for i in range(n_sortie) for s in sample[1:]]+["BE"]

	law_coef_bay   = xr.DataArray( np.zeros( (n_ns_params,(n_sample-1)*n_sortie + 1,1) ) , coords = [ ns_params_names , sample_names , ["Multi_Synthesis"] ] , dims = ["coef","sample_MCMC","model"] )
	
	#Illustration
	s="BE"
	X   = clim.X.loc[Yo.index,s,"F","Multi_Synthesis"].values.squeeze()
	#ns_law=nsm.GEV()
	ns_law=clim.ns_law
	ns_law.fit(Yo.values,X)
	MLE_theta=ns_law.get_params()
	sdlaw=create_likelihood(X,Yo.values)
	draw, accept=MCMC_MHWG_Stop_ESS(prior_law,sdlaw,tran_scale_G,init,n_features,TransitionAdapt,initTrans=initTrans,epsilon=epsilon,n_sortie=n_sortie,burn_in=burn_in)
	true_theta=prior_mean
	Summary_run_table(pathOut,model_name,draw,accept,true_theta,MLE_theta,q_l=0.05,q_h=0.95,coef=clim.ns_law.get_params_names(),show=False,Simulated=False)
	true_theta=[]
	Para_Runs(pathOut,draw,true_theta,MLE_theta,prior_mean,model_name,prior_law,coef=clim.ns_law.get_params_names(),show=False)  
	
	## And now MCMC loop
	for s in clim.sample[1:]:
		pb.print()
		X   = clim.X.loc[Yo.index,s,"F","Multi_Synthesis"].values.squeeze()
		#ns_law=nsm.GEV()
		ns_law=clim.ns_law
		ns_law.fit(Yo.values,X)
		MLE_theta=ns_law.get_params()
		sdlaw=create_likelihood(X,Yo.values)

		#draw = clim.ns_law.drawn_bayesian( Yo.values.squeeze() , X , n_mcmc_drawn , prior_law , min_rate_accept )
		draw=draw_MCMC_MHWG(prior_law,sdlaw,tran_scale_G,init,n_features,TransitionAdapt,initTrans,epsilon,n_sortie=n_sortie,burn_in=burn_in)
		law_coef_bay.loc[:,[s+"_"+str(i) for i in range(n_sortie)],"Multi_Synthesis"]=draw.T
        
		#clim.law_coef.loc[:,s,"Multi_Synthesis"] = draw

    
#	for s in range(len(results)):
#		law_coef_bay.loc[:,sample_names[s],"Multi_Synthesis"]=results[s]
	#clim2.n_sample=(n_sample-1)*n_sortie + 1
	#clim2.sample=sample_names
	clim2.law_coef=law_coef_bay
	clim2.law_coef.loc[:,"BE",:] = clim2.law_coef[:,1:,:].median( dim = "sample_MCMC" )
	clim2.BE_is_median = True
	
	pb.end()
	
	return clim2
def build_params_along_time_fixed( clim , verbose = False ):##{{{
	"""
	NSSEA.extremes_stats
	====================
	Build trajectories of params along time
	Adaptated for MCMC sample changes
	
	Arguments
	---------
	clim : NSSEA.Climatology
		A clim variable
	verbose: bool
		Print state of execution or not
	
	Return
	------
	params : xr.DataArray
		An array containing params along time
	
	"""
	ns_law = clim.ns_law
	
	l_params = [k for k in clim.ns_law.lparams]
	xrdims   = ["time","sample","forcing","param","model"]
	xrcoords = [clim.time,clim.law_coef.sample_MCMC,["F","C"],l_params,clim.model]
	s_params = xr.DataArray( np.zeros( (clim.n_time,len(clim.law_coef.sample_MCMC),2,len(l_params),clim.n_model) ) , dims = xrdims , coords = xrcoords )
	
	
	pb = ProgressBar(  clim.n_model * (len(clim.law_coef.sample_MCMC)) , "build_params_along_time" , verbose = verbose )
	for m in clim.model:
		for s in s_params.sample:
			pb.print()
			s_X=str(s.values).split("_")[0]
			clim.ns_law.set_params( clim.law_coef.loc[:,s,m].values )
			for f in s_params.forcing:
				clim.ns_law.set_covariable( clim.X.loc[clim.time,s_X,f,m].values , clim.time )
				for p in l_params:
					s_params.loc[:,s,f,p,m] = clim.ns_law.lparams[p](clim.time)
	
	
	if verbose: pb.end()
	
	return s_params
