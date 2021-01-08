import torch
from progressbar import progressbar
import math

class Encoder(torch.nn.Module):

    def __init__(self, input_dim=104, representation_dim=8, sensitive_dim=1):
        super(Encoder, self).__init__()

        self.representation_dim = representation_dim

        self.zb_given_x = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,2*representation_dim + sensitive_dim)
        )
    
    def encode(self, x):

        z_mu_logvar_b = self.zb_given_x(x)
        z_mu_logvar, b = z_mu_logvar_b[:,:-1], z_mu_logvar_b[:,-1]
        z_mu = z_mu_logvar[:,self.representation_dim:]
        z_logvar = z_mu_logvar[:,:self.representation_dim]
        return z_mu, z_logvar, b
    
    def reparametrize(self, z_mu, z_logvar):
        z = torch.randn_like(z_mu)*torch.exp(0.5*z_logvar) + z_mu 
        return z 
    
    def forward(self, x): 
        z_mu, z_logvar, b = self.encode(x) 
        z = self.reparametrize(z_mu, z_logvar) 
        return z, z_mu


class Decoder(torch.nn.Module): 

    def __init__(self, output_dim=104, representation_dim=8, sensitive_dim=1, n_regression=1):
        super(Decoder, self).__init__()

        self.sensitive_dim = sensitive_dim 
        self.logvar = torch.nn.Parameter(torch.Tensor(-1.0 * torch.ones(n_regression)))
        
        self.a_given_b = torch.nn.Sequential(
            torch.nn.Linear(sensitive_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, sensitive_dim)
        )

        self.x_given_zb = torch.nn.Sequential(
            torch.nn.Linear(representation_dim + sensitive_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, output_dim)
        )
    
    def forward(self, z, b): 

        x_hat = self.x_given_zb(torch.cat((z,b.view(-1,self.sensitive_dim)),1))
        a_hat = self.a_given_b(b.view(-1,self.sensitive_dim))
        return x_hat, a_hat

class AdversaryDiscriminator(torch.nn.Module): 

    def __init__(self, representation_dim=8, sensitive_dim=1): 
        super(AdversaryDiscriminator, self).__init__()

        self.sensitive_dim = sensitive_dim 

        self.real_given_zb = torch.nn.Sequential(
            torch.nn.Linear(representation_dim + sensitive_dim, 100), 
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        )

    def forward(self, z, b): 

        return torch.sigmoid(self.real_given_zb(torch.cat((z,b.view(-1,self.sensitive_dim)),1)))

class FFVAE(torch.nn.Module):

    def __init__(self, input_dim=104, representation_dim=8, sensitive_dim=1, output_dim=[1], output_type=['binary'],
        alpha=1, gamma=1): 
        super(FFVAE, self).__init__()

        self.alpha = alpha 
        self.gamma = gamma

        self.output_type = output_type 
        self.output_dim = output_dim
        self.output_len = sum(output_dim)
        self.n_regression = sum([1 for v in self.output_type if v == 'b'])
        self.sensitive_dim = sensitive_dim

        self.encoder = Encoder(input_dim, representation_dim, self.sensitive_dim)
        self.decoder = Decoder(self.output_len, representation_dim, self.sensitive_dim, self.n_regression) 
        self.adversary_discriminator = AdversaryDiscriminator(representation_dim, self.sensitive_dim)
    
    def get_reconstruction(self, x, x_hat):

        reconstruction_error = 0
        dim_start_x = 0
        dim_start_x_hat = 0
        reg_start = 0

        for output_type, output_dim in zip(self.output_type ,self.output_dim):

            if output_type == 'classes':
                sx = dim_start_x
                ex = dim_start_x + 1
                sx_hat = dim_start_x_hat
                ex_hat = dim_start_x_hat + output_dim
                CE = torch.nn.functional.cross_entropy(x_hat[:,sx_hat:ex_hat], x[:,sx:ex].long().view(-1), reduction='sum')
            elif output_type == 'binary':
                sx = dim_start_x
                ex = dim_start_x + 1
                sx_hat = dim_start_x_hat
                ex_hat = dim_start_x_hat + 1
                CE = torch.nn.functional.binary_cross_entropy_with_logits(x_hat[:,sx_hat:ex_hat].view(-1), x[:,sx:ex].view(-1), reduction='sum')
            else: # regression 
                sx = dim_start_x
                ex = dim_start_x + output_dim
                sx_hat = dim_start_x_hat
                ex_hat = dim_start_x_hat + output_dim
                sr = reg_start 
                er = reg_start + output_dim
                reg_start = er
                CE = 0.5 * torch.sum(
                    math.log(2*math.pi) + self.decoder.logvar[sr:er] + \
                        torch.pow(x_hat[:,sx_hat:ex_hat] - x[:,sx:ex], 2) / (torch.exp(self.decoder.logvar[sr:er]) + 1e-10)
                )
            
            reconstruction_error += CE

            dim_start_x = ex
            dim_start_x_hat = ex_hat
        
        return - reconstruction_error
    
    def get_predictivness(self, a, a_hat):

        return - torch.nn.functional.binary_cross_entropy_with_logits(a_hat.view(-1), a.view(-1), reduction = 'sum')
    
    def get_disentanglement(self, z, b):

        return 0 
    
    def get_prior(self, z_mu, z_logvar, b):

        if self.sensitive_dim == 1: 
            n_a = 2 
        
        KL_sens = 0 # it is deterministic
        KL_no_sens = 0.5 * torch.sum( 
            - z_logvar - 1.0 + torch.exp(z_logvar) + torch.pow(z_mu, 2)
        )

        return KL_sens + KL_no_sens

    def train_step(self, batch_size, dataloader, optimizer, verbose):

        device = 'cuda' if next(self.encoder.parameters()).is_cuda else 'cpu' 

        for x_in, x_out, a in progressbar(dataloader):

            x_in = x_in.to(device).float() 
            x_out = x_out.to(device).float()
            a = a.to(device).float() 

            optimizer.zero_grad() 
            z_mu, z_logvar, b = self.encoder.encode(x_in)
            z = self.encoder.reparametrize(z_mu, z_logvar)
            x_hat, a_hat = self.decoder(z,b)

            reconstruction = self.get_reconstruction(x_out, x_hat) 
            predictivness = self.get_predictivness(a, a_hat) 
            disentanglement = self.get_disentanglement(z, b) 
            prior = self.get_prior(z_mu, z_logvar, b)
            loss = - reconstruction - self.alpha * predictivness + self.gamma * disentanglement + prior
            
            loss.backward()
            optimizer.step()


    def fit(self, dataset_train, dataset_val, epochs=1000, learning_rate=1e-3, batch_size=1024, eval_rate=15, verbose=True):

        dataloader = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        #optimizer_adv = torch.optim.Adam(self.adversary_discriminator.parameters(), lr=learning_rate) in our case dim(a) = 1

        for epoch in range(epochs): 
            print(f'Epoch # {epoch+1}')
            self.train_step(batch_size, dataloader, optimizer, verbose)

            if epoch % eval_rate == eval_rate - 1:
                print(f'Evaluating TRAIN') if verbose else 0 