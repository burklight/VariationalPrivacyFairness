import utils 
import variational_privacy_fairness
import torch 
import numpy as np 
import sklearn.ensemble, sklearn.linear_model 
import metrics 
import mutual_information 
import visualization 
import mine 
import evaluations
import os

torch.manual_seed(2020)
np.random.seed(2020)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

args = utils.get_args()
experiment = args.experiment 

# Experiment 1: Generate image on the Colored MNIST dataset

if experiment == 1:

    figsdir = '../results/images/colored_mnist/example/'
    os.makedirs(figsdir, exist_ok=True)

    input_type = 'image' 
    representation_type = 'image' 
    output_type = ['image']
    output_dim = [(3,28,28)]
    problem = 'privacy'
    input_dim = (3,28,28)
    eval_rate = 500 # evaluate on last epoch
    beta = 1 
    epochs = 501 
    batch_size = 2048

    trainset, testset = utils.get_mnist() 

    network = variational_privacy_fairness.VPAF(
        input_type=input_type, representation_type=representation_type, output_type=output_type,
        problem=problem, beta=beta, input_dim=input_dim, output_dim=output_dim
    ).to(device)
    network.apply(utils.weight_init)
    network.train()
    network.fit(trainset,testset,epochs=epochs, batch_size=batch_size, 
        eval_rate=eval_rate, figs_dir=figsdir)

# Experiment 2: Fairness on the Adult dataset 

elif experiment == 2: 

    figsdir = '../results/images/adult/fairness/'
    logsdir = '../results/logs/adult/fairness/'
    os.makedirs(figsdir, exist_ok=True)
    os.makedirs(logsdir, exist_ok=True)

    input_type = 'vector'
    representation_type = 'vector'
    output_type = ['binary']
    output_dim = [1]
    problem = 'fairness'
    input_dim = 13
    eval_rate = 1000
    epochs = 150
    batch_size = 1024
    representation_dim = 2
    verbose=False

    trainset, testset = utils.get_adult()
    # Get information of the data 
    X, T, S = trainset.data, trainset.targets, trainset.hidden  
    random_chance_T = max(np.sum(T == 1) / len(T), 1.0 - np.sum(T == 1) / len(T))  
    random_chance_S = max(np.sum(S == 1) / len(S), 1.0 - np.sum(S == 1) / len(S))
    original_data_metrics_lin = evaluations.evaluate_fair_representations(None, trainset, testset, device, verbose=True, predictor_type='Linear')
    original_data_metrics_rf = evaluations.evaluate_fair_representations(None, trainset, testset, device, verbose=True, predictor_type='RandomForest')
    original_data_metrics_dummy = evaluations.evaluate_fair_representations(None, trainset, testset, device, verbose=True, predictor_type='Dummy')
    np.save(logsdir+'random_chance_t',random_chance_T)
    np.save(logsdir+'random_chance_s',random_chance_S)
    np.save(logsdir+'original_data_metrics_lin',original_data_metrics_lin)
    np.save(logsdir+'original_data_metrics_rf',original_data_metrics_rf)
    np.save(logsdir+'original_data_metrics_dummy',original_data_metrics_dummy)

    N = 30
    betas = np.logspace(0,np.log10(50),N)
    IXY = np.zeros(N)
    IYT_given_S = np.zeros(N)
    metrics_lin = np.zeros((N, 5))
    metrics_rf = np.zeros((N, 5))
    ISY = np.zeros(N)

    for i, beta in enumerate(betas):

        # Train the network 
        network = variational_privacy_fairness.VPAF(
            input_type=input_type, representation_type=representation_type, output_type=output_type, 
            problem=problem, beta=beta, input_dim=input_dim, representation_dim=representation_dim, 
            output_dim=output_dim
        ).to(device)
        network.apply(utils.weight_init)
        network.train()
        network.fit(trainset, testset, batch_size=batch_size,
            epochs=epochs, eval_rate=eval_rate, verbose=verbose)

        # Evaluate the representations performance
        network.eval()
        IXY[i], IYT_given_S[i] = network.evaluate(testset,True,'')
        metrics_lin[i] = evaluations.evaluate_fair_representations(network.encoder, trainset, testset, device, verbose=True, predictor_type='Linear')
        metrics_rf[i] = evaluations.evaluate_fair_representations(network.encoder, trainset, testset, device, verbose=True, predictor_type='RandomForest')
        X, T, S = testset.data, testset.targets, testset.hidden  
        Y, Y_mean = network.encoder(torch.FloatTensor(X).to(device)) 
        mine_network = mine.MINE(1,representation_dim, hidden_size=100, moving_average_rate=0.1).to(device)
        print('MINE calculations...')
        ISY[i] = mine_network.train(S, Y.detach().cpu().numpy(), batch_size = 2*batch_size, n_iterations=int(5e4), n_verbose=-1, n_window=100, save_progress=-1)
        print(f'ISY: {ISY[i]}')
    
    np.save(logsdir+'betas',betas)
    np.save(logsdir+'IXY',IXY)
    np.save(logsdir+'ITY_given_S',IYT_given_S)
    np.save(logsdir+'ISY',ISY)
    np.save(logsdir+'metrics_lin',metrics_lin)
    np.save(logsdir+'metrics_rf',metrics_rf)

# Experiment 2: Fairness on the Colored MNIST dataset 

elif experiment == 3:

    figsdir = '../results/images/colored_mnist/fairness/'
    logsdir = '../results/logs/colored_mnist/fairness/'
    os.makedirs(figsdir, exist_ok=True)
    os.makedirs(logsdir, exist_ok=True)

    input_type = 'image'
    representation_type = 'vector'
    output_type = ['classes']
    output_dim = [10]
    problem = 'fairness'
    input_dim = (3,28,28)
    s_dim = 1
    eval_rate = 10000
    epochs = 250
    batch_size = 1024
    s_type = 'classes'
    representation_dim = 2
    prior_type = 'Gaussian'
    verbose=False

    trainset, testset = utils.get_mnist('fairness')

    N = 30
    betas = np.logspace(0,np.log10(50),N)
    IXY = np.zeros(N)
    IYT_given_S = np.zeros(N)
    ISY = np.zeros(N)

    for i, beta in enumerate(betas):

        # Train the network 
        network = variational_privacy_fairness.VPAF(
            input_type=input_type, representation_type=representation_type, output_type=output_type, 
            problem=problem, beta=beta, input_dim=input_dim, representation_dim=representation_dim, 
            output_dim=output_dim, prior_type=prior_type
        ).to(device)
        network.apply(utils.weight_init)
        network.train()
        network.fit(trainset, testset, batch_size=batch_size,
            epochs=epochs, eval_rate=eval_rate, verbose=verbose)

        # Evaluate the representations performance
        network.eval()
        IXY[i], IYT_given_S[i] = network.evaluate(testset,True,'')
        X, T, S = testset.data, testset.targets, testset.hidden  
        Y, Y_mean = network.encoder(torch.FloatTensor(X).to(device)) 
        mine_network = mine.MINE(1,representation_dim, hidden_size=100, moving_average_rate=0.1).to(device)
        print('MINE calculations...')
        ISY[i] = mine_network.train(S, Y.detach().cpu().numpy(), batch_size = 2*batch_size, n_iterations=int(5e4), n_verbose=-1, n_window=100, save_progress=-1)
        print(f'ISY: {ISY[i]}')
    
    np.save(logsdir+'betas',betas)
    np.save(logsdir+'IXY',IXY)
    np.save(logsdir+'ITY_given_S',IYT_given_S)
    np.save(logsdir+'ISY',ISY)

# Experiment 4: Privacy on the Adult Dataset 

elif experiment == 4:

    figsdir = '../results/images/adult/privacy/'
    logsdir = '../results/logs/adult/privacy/'
    os.makedirs(figsdir, exist_ok=True)
    os.makedirs(logsdir, exist_ok=True)

    input_type = 'vector'
    representation_type = 'vector'
    output_type = ['regression','classes','regression','classes','classes','classes','classes','classes','classes','regression','regression','regression','classes']
    output_dim = [1,9,1,16,16,7,15,6,5,1,1,1,42]
    problem = 'privacy'
    input_dim = 13
    s_dim = 1
    eval_rate = 1000
    epochs = 150
    batch_size = 1024
    representation_dim = 2
    prior_type = 'Gaussian'
    verbose=False

    N = 30
    gammas = np.logspace(0,np.log10(50),N)
    IXY = np.zeros(N)
    H_X_given_SY = np.zeros(N)
    ISY = np.zeros(N)

    trainset, testset = utils.get_adult('privacy')

    for i, gamma in enumerate(gammas):

        # Train the network 
        network = variational_privacy_fairness.VPAF(
            input_type=input_type, representation_type=representation_type, output_type=output_type, 
            problem=problem, gamma=gamma, input_dim=input_dim, representation_dim=representation_dim, 
            output_dim=output_dim, prior_type=prior_type
        ).to(device)
        network.apply(utils.weight_init)
        network.train()
        network.fit(trainset, testset, batch_size=batch_size,
            epochs=epochs, eval_rate=eval_rate, verbose=verbose)

        # Evaluate the representations performance
        network.eval()
        IXY[i], H_X_given_SY[i] = network.evaluate(testset,True,'')
        X, T, S = testset.data, testset.targets, testset.hidden  
        Y, Y_mean = network.encoder(torch.FloatTensor(X).to(device)) 
        mine_network = mine.MINE(1,representation_dim, hidden_size=100, moving_average_rate=0.1).to(device)
        print('MINE calculations...')
        ISY[i] = mine_network.train(S, Y.detach().cpu().numpy(), batch_size = 2*batch_size, n_iterations=int(5e4), n_verbose=-1, n_window=100, save_progress=-1)
        print(f'ISY: {ISY[i]}')
    
    np.save(logsdir+'gammas',gammas)
    np.save(logsdir+'IXY',IXY)
    np.save(logsdir+'H_X_given_SY',H_X_given_SY)
    np.save(logsdir+'ISY',ISY)

# Experiment 5: Privacy on the Colored MNIST Dataset 

elif experiment == 5:

    figsdir = '../results/images/colored_mnist/privacy/'
    logsdir = '../results/logs/modified_mnist/privacy/'
    os.makedirs(figsdir, exist_ok=True)
    os.makedirs(logsdir, exist_ok=True)

    input_type = 'image'
    representation_type = 'vector'
    output_type = ['image']
    output_dim = [(3,28,28)]
    problem = 'privacy'
    input_dim = (3,28,28)
    s_dim = 1
    eval_rate = 1000
    epochs = 500
    s_type = 'classes'
    batch_size = 1024
    representation_dim = 2
    prior_type = 'Gaussian'
    verbose = True

    trainset, testset = utils.get_mnist()

    N = 30
    gammas = np.logspace(0,np.log10(50),N)
    IXY = np.zeros(N)
    H_X_given_SY = np.zeros(N)
    ISY = np.zeros(N)

    for i, gamma in enumerate(gammas):

        # Train the network 
        variational_PF_network = variational_privacy_fairness.VPAF(
            input_type=input_type, representation_type=representation_type, output_type=output_type, problem=problem, \
                gamma=gamma, input_dim=input_dim, representation_dim=representation_dim, output_dim=output_dim, s_dim=s_dim, \
                s_type=s_type, prior_type=prior_type
        ).to(device)
        variational_PF_network.apply(utils.weight_init)
        variational_PF_network.train()
        variational_PF_network.fit(trainset,testset,batch_size=batch_size,epochs=epochs,eval_rate=eval_rate, verbose=verbose)

        # Evaluate the network performance 
        variational_PF_network.eval()
        IXY[i], H_X_given_SY[i] = variational_PF_network.evaluate(testset,True,'')

        # Evaluate the network performance against an adversary
        variational_PF_network.to('cpu')
        X, S = trainset.data, trainset.hidden
        Y, Y_mean = variational_PF_network.encoder(torch.FloatTensor(X))
        print('MINE calculations...')
        mine_network = mine.MINE(s_dim,representation_dim, hidden_size=100, moving_average_rate=0.1).to(device)
        ISY[i] = mine_network.train(S.float(), Y.detach(), batch_size = 2*batch_size, n_iterations=int(5e4), n_verbose=-1, n_window=100, save_progress=-1)
        print(f'ISY: {ISY[i]}')
    
    np.save(logsdir+'gammas',gammas)
    np.save(logsdir+'IXY',IXY)
    np.save(logsdir+'H_X_given_SY',H_X_given_SY)
    np.save(logsdir+'ISY',ISY)

# Experiment 6: Fairness on the COMPAS dataset 

elif experiment == 6: 

    figsdir = '../results/images/compas/fairness/'
    logsdir = '../results/logs/compas/fairness/'
    os.makedirs(figsdir, exist_ok=True)
    os.makedirs(logsdir, exist_ok=True)

    input_type = 'vector'
    representation_type = 'vector'
    output_type = ['binary']
    output_dim = [1]
    problem = 'fairness'
    input_dim = 10
    eval_rate = 1000
    epochs = 150
    batch_size = 64
    learning_rate = 1e-4
    representation_dim = 2
    verbose=True

    trainset, testset = utils.get_compas()
    # Get information of the data 
    X, T, S = trainset.data, trainset.targets, trainset.hidden  
    random_chance_T = max(np.sum(T == 1) / len(T), 1.0 - np.sum(T == 1) / len(T))  
    random_chance_S = max(np.sum(S == 1) / len(S), 1.0 - np.sum(S == 1) / len(S))
    original_data_metrics_lin = evaluations.evaluate_fair_representations(None, trainset, testset, device, verbose=True, predictor_type='Linear')
    original_data_metrics_rf = evaluations.evaluate_fair_representations(None, trainset, testset, device, verbose=True, predictor_type='RandomForest')
    original_data_metrics_dummy = evaluations.evaluate_fair_representations(None, trainset, testset, device, verbose=True, predictor_type='Dummy')
    np.save(logsdir+'random_chance_t',random_chance_T)
    np.save(logsdir+'random_chance_s',random_chance_S)
    np.save(logsdir+'original_data_metrics_lin',original_data_metrics_lin)
    np.save(logsdir+'original_data_metrics_rf',original_data_metrics_rf)
    np.save(logsdir+'original_data_metrics_dummy',original_data_metrics_dummy)


    N = 30
    betas = np.linspace(1,50,N)
    IXY = np.zeros(N)
    IYT_given_S = np.zeros(N)
    metrics_lin = np.zeros((N, 5))
    metrics_rf = np.zeros((N, 5))
    ISY = np.zeros(N)

    for i, beta in enumerate(betas):

        # Train the network 
        network = variational_privacy_fairness.VPAF(
            input_type=input_type, representation_type=representation_type, output_type=output_type, 
            problem=problem, beta=beta, input_dim=input_dim, representation_dim=representation_dim, 
            output_dim=output_dim
        ).to(device)
        network.apply(utils.weight_init)
        network.train()
        network.fit(trainset, testset, batch_size=batch_size,
            epochs=epochs, eval_rate=eval_rate, verbose=verbose, learning_rate=learning_rate)

        # Evaluate the representations performance
        network.eval()
        IXY[i], IYT_given_S[i] = network.evaluate(testset,True,'')
        metrics_lin[i] = evaluations.evaluate_fair_representations(network.encoder, trainset, testset, device, verbose=True, predictor_type='Linear')
        metrics_rf[i] = evaluations.evaluate_fair_representations(network.encoder, trainset, testset, device, verbose=True, predictor_type='RandomForest')
        X, T, S = testset.data, testset.targets, testset.hidden  
        Y, Y_mean = network.encoder(torch.FloatTensor(X).to(device)) 
        mine_network = mine.MINE(1,representation_dim, hidden_size=100, moving_average_rate=0.1).to(device)
        print('MINE calculations...')
        ISY[i] = mine_network.train(S, Y.detach().cpu().numpy(), batch_size = int(len(S)/4), n_iterations=int(5e4), n_verbose=-1, n_window=100, save_progress=-1)
        print(f'ISY: {ISY[i]}')
    
    np.save(logsdir+'betas',betas)
    np.save(logsdir+'IXY',IXY)
    np.save(logsdir+'ITY_given_S',IYT_given_S)
    np.save(logsdir+'ISY',ISY)
    np.save(logsdir+'metrics_lin',metrics_lin)
    np.save(logsdir+'metrics_rf',metrics_rf)

# Experiment 7: Privacy on the Compas Dataset 

elif experiment == 7:

    figsdir = '../results/images/compas/privacy/'
    logsdir = '../results/logs/compas/privacy/'
    os.makedirs(figsdir, exist_ok=True)
    os.makedirs(logsdir, exist_ok=True)

    input_type = 'vector'
    representation_type = 'vector'
    output_type = ['regression','classes','classes','classes','classes','classes','classes','classes','classes','classes']
    output_dim = [1,2,2,2,2,2,2,2,2,2]
    problem = 'privacy'
    input_dim = 10
    s_dim = 1
    eval_rate = 1000
    epochs = 250
    batch_size = 64
    learning_rate = 1e-4
    representation_dim = 2
    prior_type = 'Gaussian'

    verbose=False

    N = 30
    gammas = np.logspace(0,np.log10(500),N)
    IXY = np.zeros(N)
    H_X_given_SY = np.zeros(N)
    metrics_lin = np.zeros((N, 5))
    metrics_rf = np.zeros((N, 5))
    ISY = np.zeros(N)

    trainset, testset = utils.get_compas('privacy')

    for i, gamma in enumerate(gammas):

        # Train the network 
        network = variational_privacy_fairness.VPAF(
            input_type=input_type, representation_type=representation_type, output_type=output_type, 
            problem=problem, gamma=gamma, input_dim=input_dim, representation_dim=representation_dim, 
            output_dim=output_dim, prior_type=prior_type
        ).to(device)
        network.apply(utils.weight_init)
        network.train()
        network.fit(trainset, testset, batch_size=batch_size,
            epochs=epochs, eval_rate=eval_rate, verbose=verbose, learning_rate=learning_rate)

        # Evaluate the representations performance
        network.eval()
        IXY[i], H_X_given_SY[i] = network.evaluate(testset,True,'')
        X, T, S = testset.data, testset.targets, testset.hidden  
        Y, Y_mean = network.encoder(torch.FloatTensor(X).to(device)) 
        mine_network = mine.MINE(1,representation_dim, hidden_size=100, moving_average_rate=0.1).to(device)
        print('MINE calculations...')
        ISY[i] = mine_network.train(S, Y.detach().cpu().numpy(), batch_size = int(len(S)/4), n_iterations=int(5e4), n_verbose=-1, n_window=100, save_progress=-1)
        print(f'ISY: {ISY[i]}')
    
    np.save(logsdir+'gammas',gammas)
    np.save(logsdir+'IXY',IXY)
    np.save(logsdir+'H_X_given_SY',H_X_given_SY)
    np.save(logsdir+'ISY',ISY)