import numpy as np
import scipy.optimize as optim
import utils
from lfr import *
import warnings 
import numba.errors
import sklearn.ensemble, sklearn.linear_model, sklearn.dummy
import metrics

warnings.filterwarnings('ignore', category=numba.errors.NumbaWarning)

experiment = 1

def retrieve_data(dataset):
    X, S, Y = dataset.data, dataset.hidden, dataset.targets
    idx_plus = S==1
    idx_minus = S==0
    return X, S, Y, idx_plus, idx_minus

def calculate_representations(X, idx_plus, idx_minus, params, D, K):

    N_plus, N_minus = len(idx_plus), len(idx_minus)

    # Recover the parameters
    alpha0 = params[:D]
    alpha1 = params[D : 2 * D]
    w = params[2 * D : (2 * D) + K]
    v = np.matrix(params[(2 * D) + K:]).reshape((K, D))

    # Calculate the representations
    dists_plus = distances(X[idx_plus], v, alpha1, N_plus, K)
    dists_minus = distances(X[idx_minus], v, alpha0, N_minus, K)
    M_nk_plus = M_nk(dists_plus, N_plus, K)
    M_nk_minus = M_nk(dists_minus, N_minus, K)

    return np.concatenate((M_nk_plus, M_nk_minus))

def order(X, S, Y, idx_plus, idx_minus):

    X_ = np.concatenate((X[idx_plus], X[idx_minus]))
    S_ = np.concatenate((S[idx_plus], S[idx_minus]))
    Y_ = np.concatenate((Y[idx_plus], Y[idx_minus]))

    return X_, S_, Y_

def evaluate_representations(Y_train, S_train, Z_train, Y_test, S_test, Z_test, predictor_type='Linear', verbose=True):

    if predictor_type == 'Linear':
        y_predictor = sklearn.linear_model.LogisticRegression() 
        s_predictor = sklearn.linear_model.LogisticRegression() 
    elif predictor_type == 'RandomForest':
        y_predictor = sklearn.ensemble.RandomForestClassifier()
        s_predictor = sklearn.ensemble.RandomForestClassifier()
    else: # Majority class 
        y_predictor = sklearn.dummy.DummyClassifier(strategy='prior')
        s_predictor = sklearn.dummy.DummyClassifier(strategy='prior')

    y_predictor.fit(Z_train, Y_train) 
    s_predictor.fit(Z_train, S_train) 

    y_pred_prob = y_predictor.predict_proba(Z_test) 
    s_pred_prob = s_predictor.predict_proba(Z_test)

    accuracy = metrics.get_accuracy_numpy(y_pred_prob, Y_test) 
    accuracy_s = metrics.get_accuracy_numpy(s_pred_prob, S_test) 
    discrimination = metrics.get_discrimination_numpy(y_pred_prob, Y_test, S_test) 
    error_gap = metrics.get_error_gap_numpy(y_pred_prob, Y_test, S_test)
    equalized_odds = metrics.get_eq_odds_gap_numpy(y_pred_prob, Y_test, S_test)

    print(f'Accuracy ({predictor_type}): {accuracy}') if verbose else 0 
    print(f'Accuracy on S ({predictor_type}): {accuracy_s}') if verbose else 0 
    print(f'Discrimination ({predictor_type}): {discrimination}') if verbose else 0 
    print(f'Error gap ({predictor_type}): {error_gap}') if verbose else 0 
    print(f'Equalized odds gap ({predictor_type}): {equalized_odds}') if verbose else 0 

    return np.array([accuracy, accuracy_s, discrimination, error_gap, equalized_odds])

if experiment == 1: 
    trainset, testset = utils.get_adult()
    name = 'adult'
elif experiment == 2: 
    trainset, testset = utils.get_compas()
    name = 'compas'

# Get data
X_train, S_train, Y_train, idx_train_plus, idx_train_minus = retrieve_data(trainset)
X_test, S_test, Y_test, idx_test_plus, idx_test_minus = retrieve_data(testset)
D = X_train.shape[1]

# Initialize the parameters and bounds for scipy optimize
A_x, A_y, A_z = 1e-4, 0.1, 500
K = 10
params = np.random.uniform(size=D * 2 + K + D * K)
bounds = []
for i in range(len(params)):
    if i < D * 2 or i >= D * 2 + K:
        bounds.append((None, None))
    else: 
        bounds.append((0,1))
iterations = 150000


# Apply L-BFGS optimization
final_params = optim.fmin_l_bfgs_b(LFR, x0=params, epsilon=1e-5, args=(X_train[idx_train_plus], 
    X_train[idx_train_minus], Y_train[idx_train_plus], Y_train[idx_train_minus], K, A_x, A_y, A_z, 0),
    bounds = bounds, approx_grad=True,  maxfun=iterations,  maxiter=iterations)

# Save the parameters 
np.save('../results/logs/LFR/'+name+'_params',final_params[0])

# Load the parameters
params = np.load('../results/logs/LFR/'+name+'_params.npy')

# Get the representations and order the data for the evaluation
_, _, Z_train_plus, Z_train_minus = LFR(params, X_train[idx_train_plus], X_train[idx_train_minus], 
    Y_train[idx_train_plus], Y_train[idx_train_minus], K, A_x, A_y, A_z, 1)
_, _, Z_test_plus, Z_test_minus = LFR(params, X_test[idx_test_plus], X_test[idx_test_minus], 
    Y_test[idx_test_plus], Y_test[idx_test_minus], K, A_x, A_y, A_z, 1)
Z_train = np.concatenate((Z_train_plus, Z_train_minus))
Z_test = np.concatenate((Z_test_plus, Z_test_minus))
X_train, S_train, Y_train = order(X_train, S_train, Y_train, idx_train_plus, idx_train_minus)
X_test, S_test, Y_test = order(X_test, S_test, Y_test, idx_test_plus, idx_test_minus)

# Evaluate the representations 
metrics_lin = evaluate_representations(Y_train, S_train, Z_train, Y_test, S_test, Z_test, 'Linear')
metrics_rf = evaluate_representations(Y_train, S_train, Z_train, Y_test, S_test, Z_test, 'RandomForest')
np.save('../results/logs/LFR/'+name+'_lin',metrics_lin)
np.save('../results/logs/LFR/'+name+'_rf',metrics_rf)