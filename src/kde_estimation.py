import torch
import math
import numpy as np

def compute_distances(x):
    '''
    Computes the distance matrix for the KDE Entropy estimation:
    - x (Tensor) : array of functions to compute the distances matrix from
    '''

    x_norm = (x**2).sum(1).view(-1,1)
    x_t = torch.transpose(x,0,1)
    x_t_norm = x_norm.view(1,-1)
    dist = x_norm + x_t_norm - 2.0*torch.mm(x,x_t)
    dist = torch.clamp(dist,0,np.inf)

    return dist

def KDE_IXY_estimation(y_logvar, y_mean):
    '''
    Computes the MI estimation of X and T. Parameters:
    - y_logvar (float) : log(var) of the representation variable 
    - y_mean (Tensor) : deterministic transformation of the input 
    '''

    n_batch, d = y_mean.shape
    var = torch.exp(y_logvar) + 1e-10 # to avoid 0's in the log

    # calculation of the constant
    normalization_constant = math.log(n_batch)

    # calculation of the elements contribution
    dist = compute_distances(y_mean)
    distance_contribution = - torch.mean(torch.logsumexp(input=- 0.5 * dist / var,dim=1))

    # mutual information calculation (natts)
    I_XY = n_batch * (normalization_constant + distance_contribution)

    return I_XY