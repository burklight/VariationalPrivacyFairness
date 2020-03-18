from utils import get_mnist, get_adult
from variational_privacy_fairness import VPAF
import torch

torch.manual_seed(2020)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Experiment 1: Privacy on modified MNIST 

input_type = 'image' 
representation_type = 'image'
output_type = 'image'
problem = 'privacy'
input_dim = (3,28,28)
eval_rate = 5
beta = 1.0

trainset, testset = get_mnist()

variational_PF_network = VPAF(
    input_type=input_type, representation_type=representation_type, output_type=output_type, problem=problem, beta=beta, input_dim=input_dim, 
).to(device)
variational_PF_network.fit(trainset,batch_size=2048,eval_rate=eval_rate)

