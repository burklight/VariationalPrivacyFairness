import ffvae 
import utils
import torch 
import numpy as np 
import evaluations

torch.manual_seed(2020)
np.random.seed(2020)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

experiment = 2

if experiment == 1: 

    name = 'adult'
    
    # Dataset in the privacy terms for the auto-encoding
    trainset, testset = utils.get_adult('privacy') 
    # Dataset in the fairness terms for the evaluation
    trainset_, testset_ = utils.get_adult('fairness') 

    output_type = ['regression','classes','regression','classes','classes','classes','classes','classes','classes','regression','regression','regression','classes']
    output_dim = [1,9,1,16,16,7,15,6,5,1,1,1,42]
    input_dim = 13

elif experiment == 2: 

    name = 'compas'

    # Dataset in the privacy terms for the auto-encoding
    trainset, testset = utils.get_compas('privacy') 
    # Dataset in the fairness terms for the evaluation
    trainset_, testset_ = utils.get_compas('fairness')

    output_type = ['regression','classes','classes','classes','classes','classes','classes','classes','classes','classes']
    output_dim = [1,2,2,2,2,2,2,2,2,2]
    input_dim = 10

s_dim = 1
eval_rate = 1000
epochs = 150
batch_size = 1024
representation_dim = 2
verbose=True

logsdir = '../results/logs/FFVAE/' + name + '_'

alphas = np.array([1, 100, 200, 300, 400])
N = len(alphas)
metrics_lin = np.zeros((N, 5))
metrics_rf = np.zeros((N, 5))

for i, alpha in enumerate(alphas):

    # Train the network 
    network = ffvae.FFVAE(input_dim=input_dim, representation_dim=representation_dim, 
        sensitive_dim=s_dim, output_dim=output_dim, output_type=output_type, alpha=alpha).to(device)
    network.apply(utils.weight_init)
    network.train()
    network.fit(trainset, testset, batch_size=batch_size,
        epochs=epochs, eval_rate=eval_rate, verbose=verbose)

    # Evaluate the representations performance
    network.eval()
    metrics_lin[i] = evaluations.evaluate_fair_representations(network.encoder, trainset_, testset_, device, verbose=True, predictor_type='Linear')
    metrics_rf[i] = evaluations.evaluate_fair_representations(network.encoder, trainset_, testset_, device, verbose=True, predictor_type='RandomForest')

np.save(logsdir+'alphas',alphas)
np.save(logsdir+'metrics_lin',metrics_lin)
np.save(logsdir+'metrics_rf',metrics_rf)

