import numpy as np 
import math
import torch

def get_conditional_entropy(a, b, avals = 2, bvals = 2):

    if torch.is_tensor(a):
        a = a.numpy()
    if torch.is_tensor(b):
        b = b.numpy()

    ab_emp = np.zeros((avals,bvals))
    b_emp = np.zeros(bvals)
    for j in range(bvals):
        b_emp[j] = (b==j).sum().astype(float)
        for i in range(avals):
            ab_emp[i,j] = ((a == i) & (b == j)).sum().astype(float)
    
    p_ab_emp = ab_emp / np.sum(ab_emp)
    p_b_emp = b_emp / np.sum(b_emp) 
    
    H_A_given_B_approx = - np.sum(p_ab_emp * np.log(p_ab_emp / p_b_emp)) / math.log(2)
    return H_A_given_B_approx


def get_entropy(a, avals = 2):

    if torch.is_tensor(a):
        a = a.numpy()

    a_emp = np.zeros(avals)
    for i in range(avals):
        a_emp[i] = (a == i).sum().astype(float) 
    
    p_a_emp = a_emp / np.sum(a_emp) 

    H_A_approx = - np.sum(p_a_emp * np.log(p_a_emp)) / math.log(2)
    return H_A_approx