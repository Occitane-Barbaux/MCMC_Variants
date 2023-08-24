###Author Barbaux Occitane
### Script file for data preparation
import sys,os
import tarfile
import warnings

import numpy as np
import pandas as pd
import xarray as xr










def correct_miss( X , lo =  100 , up = 350 ):##{{{
#	return X
	mod = str(X.columns[0])
	bad = np.logical_or( X < lo , X > up )
	bad = np.logical_or( bad , np.isnan(X) )
	bad = np.logical_or( bad , np.logical_not(np.isfinite(X)) )
	if np.any(bad):
		idx,_ = np.where(bad)
		idx_co = np.copy(idx)
		for i in range(idx.size):
			j = 0
			while idx[i] + j in idx:
				j += 1
			idx_co[i] += j
		X.iloc[idx] = X.iloc[idx_co].values
	return X
##}}}

def load_models_CMIP6(pathInp,type_data):
    ## List of models X
    pathInpX= os.path.join(pathInp,"CMIP6/X",type_data)
    modelsX = [  "_".join(f.split("/")[-1][:-3].split("_")[-2:]) for f in os.listdir(pathInpX) ]
    modelsX.sort()

    ## List of models Y
    pathInpY= os.path.join(pathInp,"CMIP6/Y",type_data)
    modelsY = [ "_".join(f.split("/")[-1][:-3].split("_")[-2:]) for f in os.listdir(pathInpY) ]
    modelsY.sort()
    models = list(set(modelsX) & set(modelsY))
    models.sort()
    print(models)


    ## Load X and Y
    lX = []
    lY = []
    if type_data== "03_Post_treatment":
    	
    	for m in models:
		
        	## Load X
        
        	df   = xr.open_dataset( os.path.join( pathInpX , "full_Europe_tas_YearMean_ssp585_{}.nc".format(m) ) ,decode_times=True )
        	time = df.time["time.year"].values.astype(int)
        	X    = pd.DataFrame( df.tas.values.ravel() , columns = [m] , index = time )
        	
        	lX.append( correct_miss(X , lo =  -15 , up = 25))
    
        	## Load Y
        	df   = xr.open_dataset( os.path.join( pathInpY , "full_Tricastin_ssp585_{}.nc".format(m) ) ,decode_times=True  )
        	time = df.time["time.year"].values.astype(int)
        	Y    = pd.DataFrame( df.tasmax.values.ravel() , columns = [m] , 	index = time )
        	lY.append( correct_miss(Y, lo =  -15 , up = 25 ))
    else:
    	for m in models:

        	## Load X
        
        	df   = xr.open_dataset( os.path.join( pathInpX , "full_Europe_tas_YearMean_ssp585_{}.nc".format(m) ) ,decode_times=False )
        	time = df.time.values.astype(int)
        	X    = pd.DataFrame( df.tas.values.ravel() , columns = [m] , index = time )
    
        	lX.append( correct_miss(X) )
    
        	## Load Y
        	df   = xr.open_dataset( os.path.join( pathInpY , "full_Tricastin_ssp585_{}.nc".format(m) ) ,decode_times=False  )
        	time = df.time.values.astype(int)
        	Y    = pd.DataFrame( df.tasmax.values.ravel() , columns = [m] , 	index = time )
        	lY.append( correct_miss(Y) )
    return lX, lY,models
    

def load_obs(pathInp,type_data):
    ## Load Observations
    dXo = xr.open_dataset(os.path.join( pathInp ,"Observations",type_data,"Xo.nc"))
    Xo  = pd.DataFrame( dXo.tas.values.squeeze() , columns = ["Xo"] , index = dXo.time["time.year"].values )

    Xo #Deja en anomalies
    dYo = xr.open_dataset(os.path.join( pathInp ,"Observations",type_data,"Yo.nc"))
    Yo  = pd.DataFrame( dYo.TX.values.squeeze() , columns = ["Yo"] , index = dYo.time["time.year"].values )
    return Xo,Yo #en celsius
    

def Everyone_as_anomaly(Xo,Yo,lX,lY,time_reference):
    ## Anomaly from observations Y: Here Xo is already in anomaly. Reference period ?
    Xo -=Xo.loc[time_reference].mean()
    Yo=Yo[:-1] #2022 Abberrante
    bias = { "Multi_Synthesis" : Yo.loc[time_reference].mean().values }
    Yo -= bias["Multi_Synthesis"]
    ## Models in anomaly
	##==================
    for X in lX:
        X -= X.loc[time_reference].mean()
    for Y in lY:
        bias[str(Y.columns[0])] = Y.loc[time_reference].mean().values - 273.15 #biais en celsius
        Y -= Y.loc[time_reference].mean()
    return Xo,Yo,lX,lY,bias
    
def is_outlier(Y, m):
    Y_means=[]

    Y.index.name= 'Year'
    Y.index=pd.to_datetime(Y.index.astype(str), yearfirst=True)

    
    

    #freq_roll="3650D" #10years
    #freq_roll="7300D" #20 ans
    #freq_roll="1825D" #5 ans
    freq_roll="10950D" #30 ans
    mean_rolling_Y=Y.rolling(freq_roll,min_periods=1, center=True).mean()
    std_rolling_Y=Y.rolling(freq_roll,min_periods=1, center=True).std()
    #std_rolling_Y[std_rolling_Y<1.6]=1.66666
    #Supposant une loi normale car n grand (min 30, voir 25*30)
    #Probabilité 3eq= 99.7%
    #Probabilité 4eq= 99.99%
    #pas de prise en compte de la queue
    Y['Anomalies']=(abs(Y[m]-mean_rolling_Y[m])/std_rolling_Y[m]>4)*1
    return Y
