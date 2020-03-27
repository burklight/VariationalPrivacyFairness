import torch 
import numpy as np 
import math

def get_accuracy(logits, target):

    logits = logits.view(-1)
    pred_target = torch.zeros_like(logits)
    pred_target[logits >= 0] = 1

    accuracy = torch.sum(pred_target == target).float() / target.shape[0]
    return accuracy

def get_accuracy_numpy(proba, target):

    pred_target = np.argmax(proba,1)

    accuracy = np.sum(pred_target == target).astype(float) / target.shape[0]
    return accuracy

def get_cross_entropy(proba, target):

    proba = proba[np.arange(len(target)),target.astype(int)]
    return - np.mean(target * np.log(proba + 1e-10)) / math.log(2)

def get_discrimination(logits, target, hidden):

    logits = logits.view(-1)
    pred_target = torch.zeros_like(logits)
    pred_target[logits >= 0] = 1

    pred_pos_s_1 = torch.sum((pred_target == 1) & (hidden == 1)).float()
    pred_pos_s_0 = torch.sum((pred_target == 1) & (hidden == 0)).float()
    s_1 = torch.sum(hidden == 1).float()
    s_0 = torch.sum(hidden == 0).float()

    discrimination = torch.abs(pred_pos_s_1 / (s_1 + 1e-10) - pred_pos_s_0 / (s_0+1e-10))
    return discrimination

def get_error_gap(logits, target, hidden):

    logits = logits.view(-1)
    pred_target = torch.zeros_like(logits)
    pred_target[logits >= 0] = 1

    s_1 = (hidden == 1)
    s_0 = (hidden == 0) 

    err_1 = torch.sum(pred_target[s_1] != target[s_1]).float() / (torch.sum(s_1.long()).float() + 1e-10)
    err_0 = torch.sum(pred_target[s_0] != target[s_0]).float() / (torch.sum(s_0.long()).float() + 1e-10)
    error_gap = torch.abs(err_1 - err_0)
    
    return error_gap

def get_eq_odds_gap(logits, target, hidden):

    t_1 = (target == 1)
    t_0 = (target == 0) 

    disc_t_1 = get_discrimination(logits[t_1], target[t_1], hidden[t_1])
    disc_t_0 = get_discrimination(logits[t_0], target[t_0], hidden[t_0])
    eq_odds_gap = torch.max(disc_t_1, disc_t_0)
    
    return eq_odds_gap
