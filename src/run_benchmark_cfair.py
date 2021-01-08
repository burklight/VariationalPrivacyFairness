import cfair 
import utils
import torch 
import numpy as np 
import evaluations

torch.manual_seed(2020)
np.random.seed(2020)

def get_probabilities(dataset): 

    Y, A = dataset.targets, dataset.hidden 
    py0 = np.sum(Y == 0) / len(Y) 
    py1 = 1.0 - py0 
    pa0_y0 = np.sum(A[Y==0] == 0) / len(Y[Y==0]) 
    pa1_y0 = 1.0 - pa0_y0 
    pa0_y1 = np.sum(A[Y==1] == 0) / len(Y[Y==1])
    pa1_y1 = 1.0 - pa0_y1 

    return py0, py1, pa0_y0, pa1_y0, pa0_y1, pa1_y1 


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

experiment = 2

if experiment == 1: 

    name = 'adult'
    trainset, testset = utils.get_adult('fairness') 
    input_dim = 13

elif experiment == 2: 

    name = 'compas'
    trainset, testset = utils.get_compas('fairness')
    input_dim = 10

s_dim = 1
target_dim = 1
eval_rate = 1000
epochs = 150
batch_size = 1024
representation_dim = 2
verbose=True

py0, py1, pa0_y0, pa1_y0, pa0_y1, pa1_y1 = get_probabilities(trainset)

logsdir = '../results/logs/CFAIR/' + name + '_'

lambdas_ = np.array([1/10000]) #np.array([0.1, 1, 10, 100, 1000])
N = len(lambdas_)
metrics_lin = np.zeros((N, 5))
metrics_rf = np.zeros((N, 5))

for i, lambda_ in enumerate(lambdas_):

    # Train the network 
    network = cfair.CFAIR(input_dim=input_dim, representation_dim=representation_dim, 
        sensitive_dim=s_dim, target_dim=target_dim, lambda_ = lambda_,
        py0=py0, py1=py1, pa0_y0=pa0_y0, pa1_y0=pa1_y0, pa0_y1=pa0_y1, pa1_y1=pa1_y1).to(device)
    network.apply(utils.weight_init)
    network.train()
    network.fit(trainset, testset, batch_size=batch_size,
        epochs=epochs, eval_rate=eval_rate, verbose=verbose)

    # Evaluate the representations performance
    network.eval()
    metrics_lin[i] = evaluations.evaluate_fair_representations(network.encoder, trainset, testset, device, verbose=True, predictor_type='Linear')
    metrics_rf[i] = evaluations.evaluate_fair_representations(network.encoder, trainset, testset, device, verbose=True, predictor_type='RandomForest')

#np.save(logsdir+'lambdas',lambdas_)
#np.save(logsdir+'metrics_lin',metrics_lin)
#np.save(logsdir+'metrics_rf',metrics_rf)


'''
network.to('cpu')
X, Y, A = testset.data, testset.targets, testset.hidden
X = torch.FloatTensor(X) 
Y = torch.FloatTensor(Y) 
A = torch.FloatTensor(A)
Z = network.encoder.encode(X) 
Y_hat = network.decoder(Z)
A_hat_0 = network.adv_decoder_0(Z[Y==0])
A_hat_1 = network.adv_decoder_1(Z[Y==1])
A_hat = torch.cat((A_hat_0, A_hat_1))
import metrics 
Y_hat = torch.cat((Y_hat[Y==0], Y_hat[Y==1]))
A = torch.cat((A[Y==0], A[Y==1]))
Y = torch.cat((Y[Y==0], Y[Y==1]))
accuracy = metrics.get_accuracy(Y_hat, Y, 2) 
accuracy_s = metrics.get_accuracy(A_hat, A, 2)  
discrimination = metrics.get_discrimination(Y_hat, Y, A) 
error_gap = metrics.get_error_gap(Y_hat, Y, A)
equalized_odds = metrics.get_eq_odds_gap(Y_hat, Y, A)
predictor_type = 'self'
print(f'Accuracy ({predictor_type}): {accuracy}') if verbose else 0 
print(f'Accuracy on S ({predictor_type}): {accuracy_s}') if verbose else 0 
print(f'Discrimination ({predictor_type}): {discrimination}') if verbose else 0 
print(f'Error gap ({predictor_type}): {error_gap}') if verbose else 0 
print(f'Equalized odds gap ({predictor_type}): {equalized_odds}') if verbose else 0 
'''