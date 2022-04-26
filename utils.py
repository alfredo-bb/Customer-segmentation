import numpy as np

def mahalanobis(x,data,cov=None):
    x_m_mu=x-np.mean(data)
    if not cov:
        cov=np.cov(data.values.T) 
    inv_cov=np.linalg.inv(cov)
    izq=np.dot(x_m_mu,inv_cov)
    mahal=np.dot(izq,x_m_mu.T)
    return mahal.diagonal()

#function to search for multivariable outliers