import numpy as np

def transition(x,tran_scale_G):
    # Fixed Transition Function For Metropolis-Hasting
    # Multivariate normal centered on 0
    # tran_scale_G give the covariance 
    #Ok for 1 or 5
    if len(tran_scale_G.shape)==1:
    	#(can be a 1d vector)
        tran_scale_G=np.diag(tran_scale_G*tran_scale_G)
    #(or a 5d matrix)
    x1=x+np.random.default_rng().multivariate_normal( mean=[0,0,0,0,0],cov = tran_scale_G )
    #Used to be np.random.normal(scale=tran_scale_G)
    return(x1)

#Adaptative MH
def transition_adaptative(x,i,draw,init=0.01,epsilon=0.01):
    # Adaptative Transition Function For Metropolis-Hasting
    # Adaptative Metropolis (Haario et al. 2001), Based on ([Craiu et Rosenthal, 2014, p. 189] 
    # Multivariate normal centered on 0
    # Use past draw to calculate covariance 
    if i<1000:
        sigma=np.identity(draw.shape[1])*(init)
        #pre period
    else:
        #Adaptative period
        sigma=np.cov(draw,rowvar=False)*(2.38*2.38/draw.shape[1])+np.identity(draw.shape[1])*(epsilon)
    
    
    x1=x+np.random.default_rng().multivariate_normal( mean=[0,0,0,0,0],cov = sigma )
    #Used to be np.random.normal(scale=tran_scale_G)
    return(x1)



def transition_SCAM(x ,i,draw,prev_sigma):
    # Adaptative Transition Function For Metropolis-Hasting Within Gibbs
    # Adaptative Metropolis (Haario et al. 2005), Based on ([Roberts et Rosenthal, 2009,]) 
    # Univariate normal centered on 0
    # Use past draw to calculate covariance 
    if i<1000:
        sigma=0.5
        #pre period
    elif i==1000:
        sigma=np.var(draw)*2.4+0.1
        #start adaptation
    else:
        #Adaptative period
        #could be faster, to be added
        #x_bar=np.mean(draw)
        #g_prev=(prev_sigma-0.01)/2.4
        #gt=(i-1)/i*g_prev+np.mean(draw[:(i-1)])**2+x**2/i-(i+1)/i*np.mean(draw)**2
        gt=np.var(draw)
        sigma=(2.4**2)*(gt+0.01)

    
    x1=x+np.random.normal( size = 1 , scale = sigma )
    #Used to be np.random.normal(scale=tran_scale_G)
    return(x1,sigma)

