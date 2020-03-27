import numpy as np 
import math

def get_conditional_entropy(a, b):

    ab_emp = np.zeros((2,2))
    ab_emp[0,0] = np.sum((a == 0) & (b == 0)).astype(float)
    ab_emp[0,1] = np.sum((a == 0) & (b == 1)).astype(float)
    ab_emp[1,0] = np.sum((a == 1) & (b == 0)).astype(float)
    ab_emp[1,1] = np.sum((a == 1) & (b == 1)).astype(float)

    p_ab_emp = ab_emp / np.sum(ab_emp)
    p_a_given_b_emp = ab_emp 
    p_a_given_b_emp[0] /= np.sum(ab_emp, 1)[0]
    p_a_given_b_emp[1] /= np.sum(ab_emp, 1)[1]

    H_A_given_B_approx = - np.sum(p_ab_emp * np.log(p_a_given_b_emp)) / math.log(2)
    return H_A_given_B_approx 

def get_entropy(a):

    a_emp = np.zeros(2) 
    a_emp[0] = np.sum(a == 0).astype(float)
    a_emp[1] = np.sum(a == 1).astype(float)
    
    p_a_emp = a_emp / np.sum(a_emp) 

    H_A_approx = - np.sum(p_a_emp * np.log(p_a_emp)) / math.log(2)
    return H_A_approx