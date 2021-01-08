import torch
import math
from progressbar import progressbar

class Encoder(torch.nn.Module): 

    def __init__(self, input_dim = 104, representation_dim = 8, sensitive_dim = 1): 
        super(Encoder, self).__init__() 
    
        self.representation_dim = representation_dim
        self.sensitive_dim = sensitive_dim
        
        self.f = torch.nn.Sequential(
            torch.nn.Linear(input_dim+sensitive_dim, 100),
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 2 * representation_dim)
        )
    
    def forward(self, x, y): 

        z_mu_logvar = self.f(torch.cat((x,y.view(-1,self.sensitive_dim)),1))
        mu = z_mu_logvar[:,:self.representation_dim]
        logvar = z_mu_logvar[:,self.representation_dim:]
        z = torch.randn_like(mu) * logvar + mu  
        return z, (mu, logvar) 
    

class Decoder(torch.nn.Module): 

    def __init__(self, output_dim = 104, representation_dim = 8, sensitive_dim = 1): 
        super(Decoder, self).__init__() 

        self.sensitive_dim = sensitive_dim

        self.f = torch.nn.Sequential( 
            torch.nn.Linear(representation_dim+sensitive_dim, 100), 
            torch.nn.ReLU(),
            torch.nn.Linear(100,output_dim)
        )
    
    def forward(self, z, y): 

        return self.f(torch.cat((z,y.view(-1,self.sensitive_dim)),1)) 


class PPVAE(torch.nn.Module): 

    def __init__(self, input_dim=104, representation_dim=2, output_dim=[1], output_type=['classes'], 
        sensitive_dim=1, beta=1.0):
        super(PPVAE, self).__init__()

        self.beta = beta 

        self.output_type = output_type 
        self.output_dim = output_dim
        self.output_len = sum(output_dim)
        self.sensitive_dim = sensitive_dim

        self.encoder = Encoder(input_dim, representation_dim, self.sensitive_dim)
        self.decoder = Decoder(self.output_len, representation_dim, self.sensitive_dim) 


    def get_reconstruction(self, x_hat, x_out): 

        H_output_given_ZY_ub = 0
        dim_start_out = 0
        dim_start_x = 0
        reg_start = 0

        for output_type_, output_dim_ in zip(self.output_type, self.output_dim):

            if output_type_ == 'classes':
                so = dim_start_out
                eo = dim_start_out + output_dim_
                sx = dim_start_x
                ex = dim_start_x + 1
                CE = torch.nn.functional.cross_entropy(x_hat[:,so:eo], x_out[:,sx:ex].long().view(-1), reduction='sum')
            elif output_type_ == 'binary':
                so = dim_start_out
                eo = dim_start_out + 1
                sx = dim_start_x
                ex = dim_start_x + 1
                CE = torch.nn.functional.binary_cross_entropy_with_logits(x_hat[:,so:eo].view(-1), x_out[:,sx:ex].view(-1), reduction='sum')
            elif output_type_ == 'image':
                ex = ex = 0
                CE = torch.nn.functional.binary_cross_entropy(x_hat, x_out, reduction='sum')
            else: # regression 
                so = dim_start_out
                eo = dim_start_out + output_dim_
                sx = dim_start_x
                ex = dim_start_x + output_dim_
                CE = 0.5 * torch.sum(math.log(2*math.pi) + 1 + torch.pow(x_hat[:,so:eo] - x_out[:,sx:ex], 2))

            H_output_given_ZY_ub += CE / math.log(2) # in bits 

            dim_start_out = eo
            dim_start_x = ex

        return - H_output_given_ZY_ub

    
    def get_kl(self, z_mu, z_logvar):

        Dkl = -0.5 * torch.sum(1.0 + z_logvar - torch.pow(z_mu, 2) - torch.exp(z_logvar)) 
        return Dkl / math.log(2) # in bits


    def train_step(self, batch_size, dataloader, optimizer, verbose=True):

        device = 'cuda' if next(self.encoder.parameters()).is_cuda else 'cpu' 

        for x_in, x_out, y in progressbar(dataloader):

            x_in = x_in.to(device).float() 
            x_out = x_out.to(device).float()
            y = y.to(device).float() 

            optimizer.zero_grad() 
            z, (z_mu, z_logvar) = self.encoder(x_in, y)
            x_hat = self.decoder(z, y)

            reconstruction = self.get_reconstruction(x_hat, x_out) 
            kl_div = self.get_kl(z_mu, z_logvar)
            loss = kl_div - self.beta * reconstruction
            
            loss.backward()
            optimizer.step()
    
    def fit(self, dataset_train, epochs=1000, learning_rate=1e-3, batch_size=1024, verbose=True):

        dataloader = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        for epoch in range(epochs): 
            print(f'Epoch # {epoch+1}')
            self.train_step(batch_size, dataloader, optimizer, verbose)

    

