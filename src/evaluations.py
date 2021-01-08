import sklearn.ensemble, sklearn.linear_model, sklearn.dummy
import metrics
import torch
import numpy as np

torch.manual_seed(2020)
np.random.seed(2020)

def evaluate_fair_representations(encoder, train_dataset, test_dataset, device, verbose = False, predictor_type='Linear'): 

    if encoder is not None:
        encoder = encoder.to('cpu').eval() 

    X, T, S = train_dataset.data, train_dataset.targets, train_dataset.hidden 
    if encoder is not None:
        Y, _ = encoder(torch.FloatTensor(X)) 
        Y = Y.detach().numpy()
    else: 
        Y = X

    if predictor_type == 'Linear':
        t_predictor = sklearn.linear_model.LogisticRegression() 
        s_predictor = sklearn.linear_model.LogisticRegression() 
    elif predictor_type == 'RandomForest':
        t_predictor = sklearn.ensemble.RandomForestClassifier()
        s_predictor = sklearn.ensemble.RandomForestClassifier()
    else: # Majority class 
        t_predictor = sklearn.dummy.DummyClassifier(strategy='prior')
        s_predictor = sklearn.dummy.DummyClassifier(strategy='prior')

    t_predictor.fit(Y, T) 
    s_predictor.fit(Y, S) 

    X, T, S = test_dataset.data, test_dataset.targets, test_dataset.hidden 
    if encoder is not None: 
        Y, _ = encoder(torch.FloatTensor(X))
        Y = Y.detach().numpy()
    else: 
        Y = X

    t_pred_prob = t_predictor.predict_proba(Y) 
    s_pred_prob = s_predictor.predict_proba(Y)

    accuracy = metrics.get_accuracy_numpy(t_pred_prob, T) 
    accuracy_s = metrics.get_accuracy_numpy(s_pred_prob, S) 
    discrimination = metrics.get_discrimination_numpy(t_pred_prob, T, S) 
    error_gap = metrics.get_error_gap_numpy(t_pred_prob, T, S)
    equalized_odds = metrics.get_eq_odds_gap_numpy(t_pred_prob, T, S)

    
    print(f'Accuracy ({predictor_type}): {accuracy}') if verbose else 0 
    print(f'Accuracy on S ({predictor_type}): {accuracy_s}') if verbose else 0 
    print(f'Discrimination ({predictor_type}): {discrimination}') if verbose else 0 
    print(f'Error gap ({predictor_type}): {error_gap}') if verbose else 0 
    print(f'Equalized odds gap ({predictor_type}): {equalized_odds}') if verbose else 0 

    if encoder is not None:
        encoder = encoder.to(device)
    return np.array([accuracy, accuracy_s, discrimination, error_gap, equalized_odds])

def evaluate_private_representations(encoder, train_dataset, test_dataset, device, verbose = False, 
    predictor_type='Linear', respects_MC = True): 

    if encoder is not None:
        encoder = encoder.to('cpu').eval() 

    X, S = train_dataset.data, train_dataset.hidden 
    if encoder is not None:
        if respects_MC:
            Y, _ = encoder(torch.FloatTensor(X)) 
        else: 
            Y, _ = encoder(torch.FloatTensor(X),torch.FloatTensor(S))
        Y = Y.detach().numpy()
    else: 
        Y = X

    if predictor_type == 'Linear':
        s_predictor = sklearn.linear_model.LogisticRegression() 
    elif predictor_type == 'RandomForest':
        s_predictor = sklearn.ensemble.RandomForestClassifier()
    else: # Majority class 
        s_predictor = sklearn.dummy.DummyClassifier(strategy='prior')

    s_predictor.fit(Y, S) 

    X, S = test_dataset.data, test_dataset.hidden 
    if encoder is not None: 
        if respects_MC:
            Y, _ = encoder(torch.FloatTensor(X))
        else: 
            Y, _ = encoder(torch.FloatTensor(X),torch.FloatTensor(S))
        Y = Y.detach().numpy()
    else: 
        Y = X

    s_pred_prob = s_predictor.predict_proba(Y)
    accuracy_s = metrics.get_accuracy_numpy(s_pred_prob, S) 
    
    print(f'Accuracy on S ({predictor_type}): {accuracy_s}') if verbose else 0 

    if encoder is not None:
        encoder = encoder.to(device)
    return accuracy_s