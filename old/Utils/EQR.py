## EQR
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def prod_proba_z(z,para,X_T1_T2):
    
    X_T1_T2=X.loc[T1:(T2)]

    return np.prod(sc.genextreme.cdf(z, -para[4],scale=np.exp(para[2]+X_T1_T2*para[3]),loc=para[0]+X_T1_T2*para[1]))

def EQR_para(para,T,T1=2050,T2=2100,xlen=1000):
    #Find EQR knowing the 5 parameters
    X_T1_T2=X.loc[T1:(T2)]
    min_val_s=sc.genextreme.ppf((1-1/T), -para[4],scale=np.exp(para[2]+X_T1_T2.loc[T1]*para[3]),loc=para[0]+X_T1_T2.loc[T1]*para[1])
    max_val_s=sc.genextreme.ppf((1-1/T), -para[4],scale=np.exp(para[2]+X_T1_T2.loc[T2]*para[3]),loc=para[0]+X_T1_T2.loc[T2]*para[1])
    x = np.linspace(min_val_s,max_val_s ,xlen) #Valeurs a tester

    test_val_s=[prod_proba_z(z,para,X_T1_T2) for z in x]
    Eq_Reliability=(1-1/T)**(T2-T1+1) 
    idx=find_nearest(test_val_s, Eq_Reliability) 
    return [x[idx],test_val_s[idx],Eq_Reliability]
def EQR_Vector(para_v,T,T1=2050,T2=2100,xlen=1000):
    #EQR for a vector of draw
    return [EQR_para(para_v[i],T,T1,T2,xlen)[0][0] for i in range(len(para_v))]
    
    
    
