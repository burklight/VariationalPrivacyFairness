import ppvae 
import utils
import torch 
import numpy as np 
import evaluations
import mine

torch.manual_seed(2020)
np.random.seed(2020)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

experiment = 1

if experiment == 1: 
    name = 'adult' 
    trainset, testset = utils.get_adult('privacy')

    output_type = ['regression','classes','regression','classes','classes','classes','classes','classes','classes','regression','regression','regression','classes']
    output_dim = [1,9,1,16,16,7,15,6,5,1,1,1,42]
    input_dim = 13

    epochs = 150
    batch_size = 1024
    learning_rate = 1e-3

    batch_size_mine = 2 * batch_size

else: 
    name = 'compas'
    trainset, testset = utils.get_compas('privacy')

    output_type = ['regression','classes','classes','classes','classes','classes','classes','classes','classes','classes']
    output_dim = [1,2,2,2,2,2,2,2,2,2]
    input_dim = 10

    epochs = 250
    batch_size = 64
    learning_rate = 1e-4

    batch_size_mine = 463

logsdir = '../results/logs/PPVAE/' + name + '_'

s_dim = 1
representation_dim = 2

N = 30
betas = np.logspace(0,np.log10(50),N)
IYZ = np.zeros(N)
accuracy_s_lin = np.zeros(N)
accuracy_s_rf = np.zeros(N)
accuracy_s_prior = evaluations.evaluate_private_representations(None, trainset, testset, device, verbose=True, predictor_type='Dummy')

for i, beta in enumerate(betas):

    # Train the network 
    network = ppvae.PPVAE(
        input_dim=input_dim, representation_dim=representation_dim, output_dim=output_dim, output_type=output_type, 
        sensitive_dim=s_dim, beta=beta).to(device)
    network.apply(utils.weight_init)
    network.train()
    network.fit(trainset, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, verbose=True)

    # Evaluate the representations performance
    network.eval()
    accuracy_s_lin[i] = evaluations.evaluate_private_representations(network.encoder, trainset, testset, device, 
        verbose=True, predictor_type='Linear', respects_MC=False)
    accuracy_s_rf[i] = evaluations.evaluate_private_representations(network.encoder, trainset, testset, device, 
        verbose=True, predictor_type='RandomForest', respects_MC=False)
    X, Y = testset.data, testset.hidden  
    Z, Z_mean = network.encoder(torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device)) 
    mine_network = mine.MINE(1,representation_dim, hidden_size=100, moving_average_rate=0.1).to(device)
    print('MINE calculations...')
    IYZ[i] = mine_network.train(Y, Z.detach().cpu().numpy(), batch_size = batch_size_mine, n_iterations=int(5e4), 
        n_verbose=-1, n_window=100, save_progress=-1)
    print(f'I(Y;Z): {IYZ[i]}')

np.save(logsdir+'betas',betas)
np.save(logsdir+'IYZ',IYZ)
np.save(logsdir+'accuracy_s_lin',accuracy_s_lin)
np.save(logsdir+'accuracy_s_rf',accuracy_s_rf)
np.save(logsdir+'accuracy_s_prior',accuracy_s_prior)