from utils import get_mnist, get_adult, get_us_cens, weight_init
from variational_privacy_fairness import VPAF
import torch
import numpy as np
import sklearn.ensemble
from metrics import get_accuracy_numpy, get_cross_entropy
from mutual_information import get_entropy, get_conditional_entropy
from visualization import plot_vs_figure, plot_figure
from mine import MINE

torch.manual_seed(2020)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment = 3

# Experiment 1: Privacy on modified MNIST 

if experiment == 1:

    figsdir = '../results/images/modified_mnist/'

    input_type = 'image' 
    representation_type = 'image'
    output_type = 'image'
    problem = 'privacy'
    input_dim = (3,28,28)
    eval_rate = 5
    beta = 10.0

    trainset, testset = get_mnist()

    variational_PF_network = VPAF(
        input_type=input_type, representation_type=representation_type, output_type=output_type, problem=problem, beta=beta, input_dim=input_dim, 
    ).to(device)
    variational_PF_network.fit(trainset,testset,batch_size=2048,eval_rate=eval_rate)

elif experiment == 2:

    figsdir = '../results/images/adult/'

    input_type = 'vector'
    representation_type = 'vector'
    output_type = 'single_class'
    problem = 'fairness'
    input_dim = 13
    eval_rate = 100
    epochs = 75
    batch_size = 512
    representation_dim = 8

    trainset, testset = get_adult()

    N = 25
    gammas = np.linspace(1,30,N)
    IXY = np.zeros(N)
    IYT_given_S = np.zeros(N)
    accuracy = np.zeros(N) 
    discrimination = np.zeros(N)
    accuracy_s_gap = np.zeros(N)
    eq_odds_gap = np.zeros(N)
    err_gap = np.zeros(N)
    ISY = np.zeros(N)

    HS = get_entropy(testset.hidden)
    HT_given_S = get_conditional_entropy(testset.targets, testset.hidden)

    accuracy_s_dominant_class = np.sum(testset.hidden == 0).astype(float) / len(testset.hidden)

    for i, gamma in enumerate(gammas):

        # Train the network 
        variational_PF_network = VPAF(
            input_type=input_type, representation_type=representation_type, output_type=output_type, problem=problem, \
                gamma=gamma, input_dim=input_dim, representation_dim=representation_dim
        ).to(device)
        variational_PF_network.apply(weight_init)
        variational_PF_network.train()
        variational_PF_network.fit(trainset,testset,batch_size=batch_size,epochs=epochs,eval_rate=eval_rate,verbose=False)

        # Evaluate the network performance 
        variational_PF_network.eval()
        IXY[i], IYT_given_S[i], accuracy[i], discrimination[i], err_gap[i], eq_odds_gap[i] = variational_PF_network.evaluate(testset,True,'')

        # Evaluate the network performance against an adversary
        X, S = trainset.data, trainset.hidden
        Y, Y_mean = variational_PF_network.encoder(torch.FloatTensor(X).to(device))
        random_forest = sklearn.ensemble.RandomForestClassifier()
        random_forest.fit(Y.cpu().detach().numpy(), S)
        X, S = testset.data, testset.hidden 
        Y, Y_mean = variational_PF_network.encoder(torch.FloatTensor(X).to(device))
        S_pred_proba = random_forest.predict_proba(Y.cpu().detach().numpy()) 
        accuracy_s_gap[i] = np.abs(get_accuracy_numpy(S_pred_proba, S) - accuracy_s_dominant_class) 
        mine_network = MINE(1,representation_dim, hidden_size=100, moving_average_rate=0.1).to(device)
        print('MINE calculations...')
        ISY[i] = mine_network.train(S, Y.cpu().detach().numpy(), batch_size = 2*batch_size, n_iterations=int(1e4), n_verbose=-1, n_window=10, save_progress=-1)
        print(f'ISY: {ISY[i]}')
    
    plot_figure(IXY,IYT_given_S, r'$I(X;Y)$', r'$I(Y;T|S)$', figsdir, 'ixy_vs_iyt_given_s_adult')
    plot_figure(discrimination,accuracy, 'Discrimination', 'Accuracy', figsdir, 'accuracy_vs_discrimination_adult')
    plot_figure(gammas,accuracy,r'$\gamma$','Accuracy', figsdir, 'gammas_vs_accuracy')
    plot_figure(gammas,discrimination, r'$\gamma$','Discrimination',figsdir, 'gammas_vs_discrimination')
    plot_figure(gammas,eq_odds_gap,r'$\gamma$','Eq. Odds Gap', figsdir, 'gammas_vs_eq_odds')
    plot_figure(gammas,err_gap,r'$\gamma$','Error Gap', figsdir,'gammas_vs_err_gap')
    plot_figure(gammas,accuracy_s_gap,r'$\gamma$',r'Accuracy Gap (on $\S$)', figsdir,'gammas_vs_accuracy_s')
    plot_figure(gammas,ISY,r'$\gamma$',r'I(S;Y)',figsdir,'gammas_vs_ISY')

elif experiment == 3:

    figsdir = '../results/images/us_cens'

    input_type = 'vector'
    representation_type = 'vector'
    output_type = 'regression'
    problem = 'privacy'
    input_dim = 14
    eval_rate = 100
    epochs = 5
    batch_size = 512
    representation_dim = 8

    trainset, testset = get_us_cens()

    N = 25
    betas = np.linspace(0,50,N)
    IXY = np.zeros(N)
    H_X_given_SY = np.zeros(N)
    accuracy_s_gap = np.zeros(N)
    ISY = np.zeros(N)

    accuracy_s_dominant_class = np.sum(testset.hidden == 0).astype(float) / len(testset.hidden)

    for i, beta in enumerate(betas):

        # Train the network 
        variational_PF_network = VPAF(
            input_type=input_type, representation_type=representation_type, output_type=output_type, problem=problem, \
                beta=beta, input_dim=input_dim, representation_dim=representation_dim, output_dim=18,
        ).to(device)
        variational_PF_network.apply(weight_init)
        variational_PF_network.train()
        variational_PF_network.fit(trainset,testset,batch_size=batch_size,epochs=epochs,eval_rate=eval_rate,verbose=False)

        # Evaluate the network performance 
        variational_PF_network.eval()
        IXY[i], H_X_given_SY[i] = variational_PF_network.evaluate(testset,True,'')

        # Evaluate the network performance against an adversary
        X, S = trainset.data, trainset.hidden
        Y, Y_mean = variational_PF_network.encoder(torch.FloatTensor(X).to(device))
        random_forest = sklearn.ensemble.RandomForestClassifier()
        random_forest.fit(Y.cpu().detach().numpy(), S)
        X, S = testset.data, testset.hidden 
        Y, Y_mean = variational_PF_network.encoder(torch.FloatTensor(X).to(device))
        S_pred_proba = random_forest.predict_proba(Y.cpu().detach().numpy()) 
        accuracy_s_gap[i] = np.abs(get_accuracy_numpy(S_pred_proba, S) - accuracy_s_dominant_class) 
        mine_network = MINE(1,representation_dim, hidden_size=100, moving_average_rate=0.1).to(device)
        print('MINE calculations...')
        ISY[i] = mine_network.train(S, Y.cpu().detach().numpy(), batch_size = 2*batch_size, n_iterations=int(1e4), n_verbose=-1, n_window=10, save_progress=-1)
        print(f'ISY: {ISY[i]}')
    
    plot_figure(IXY,-H_X_given_SY, r'$I(X;Y)$', r'$-H(X|S,Y)$', figsdir, 'ixy_vs_ixy_given_s_uscens')
    plot_figure(ISY, IXY, r'$I(S;Y)$', r'$I(X;Y)$',figsdir,'isy_vs_ixy_uscens')
    plot_figure(betas,accuracy_s_gap,r'$\beta$',r'Accuracy Gap (on $S$)', figsdir,'betas_vs_accuracy_s_uscens')
    plot_figure(betas,ISY,r'$\beta$',r'I(S;Y)',figsdir,'betas_vs_ISY_uscens')
    plot_figure(betas,IXY,r'$\beta$',r'I(X;Y)',figsdir,'betas_vs_IXY_uscens')
    plot_figure(betas,-H_X_given_SY,r'$\beta$',r'-H(X|S,Y)',figsdir,'betas_vs_IXY_given_S_uscens')
    
