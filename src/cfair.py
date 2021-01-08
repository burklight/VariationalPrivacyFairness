import torch
from progressbar import progressbar
import math

class Encoder(torch.nn.Module):

    def __init__(self, input_dim=104, representation_dim=8):
        super(Encoder, self).__init__() 

        self.g = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 100), 
            torch.nn.ReLU(),
            torch.nn.Linear(100, representation_dim)
        )
    
    def forward(self, x):

        return self.g(x), 0
    
    def encode(self, x):

        return self.g(x) 

class Decoder(torch.nn.Module): 

    def __init__(self, representation_dim=8, target_dim=1):
        super(Decoder, self).__init__() 

        self.h = torch.nn.Sequential( 
            torch.nn.Linear(representation_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, target_dim)
        )
    
    def forward(self, z):

        return self.h(z)

class GradientReversalLayer(torch.nn.Module):

    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x): 

        return x 
    
    def backward(self, grad_output):

        return -grad_output

class AdversaryDecoder(torch.nn.Module):

    def __init__(self, representation_dim=8, sensitive_dim=1):
        super(AdversaryDecoder, self).__init__()

        self.h_prime = torch.nn.Sequential(
            torch.nn.Linear(representation_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, sensitive_dim),
            GradientReversalLayer()
        )
    
    def forward(self, z): 

        return self.h_prime(z)

class CFAIR(torch.nn.Module):

    def __init__(self, input_dim=104, representation_dim=8, sensitive_dim=1, target_dim=1, lambda_ = 1,
        py0=0.5, py1=0.5, pa0_y0=0.5, pa1_y0=0.5, pa0_y1=0.5, pa1_y1=0.5): 
        super(CFAIR, self).__init__()

        self.lambda_ = lambda_ 
        self.py0 = py0 
        self.py1 = py1 
        self.pa0_y0 = pa0_y0 
        self.pa1_y0 = pa1_y0 
        self.pa0_y1 = pa0_y1 
        self.pa1_y1 = pa1_y1

        self.encoder = Encoder(input_dim, representation_dim) 
        self.decoder = Decoder(representation_dim, target_dim) 
        self.adv_decoder_0 = AdversaryDecoder(representation_dim, sensitive_dim)
        self.adv_decoder_1 = AdversaryDecoder(representation_dim, sensitive_dim)
    
    def upper_bound_ber(self, h_hat, h, ph0, ph1): 

        CE_0 = torch.nn.functional.binary_cross_entropy_with_logits(h_hat[h==0].view(-1), h[h==0].view(-1), reduction = 'mean')
        CE_1 = torch.nn.functional.binary_cross_entropy_with_logits(h_hat[h==1].view(-1), h[h==1].view(-1), reduction = 'mean')
        # loss = 0.5 (CE_0 1/p(h=0) + CE_1 1/p(h=1)) 
        return 0.5 * (CE_0 / ph0 + CE_1 / ph1)
    
    def train_step(self, batch_size, dataloader, optimizer, verbose):

        device = 'cuda' if next(self.encoder.parameters()).is_cuda else 'cpu' 

        for x, y, a in progressbar(dataloader):

            x = x.to(device).float() 
            y = y.to(device).float()
            a = a.to(device).float() 

            optimizer.zero_grad() 

            z = self.encoder.encode(x)
            y_hat = self.decoder(z) 
            a_hat_0 = self.adv_decoder_0(z[y == 0]) 
            a_hat_1 = self.adv_decoder_1(z[y == 1])
            BER_y = self.upper_bound_ber(y_hat, y, self.py0, self.py1)
            BER_a_0 = self.upper_bound_ber(a_hat_0, a[y == 0], self.pa0_y0, self.pa1_y0)
            BER_a_1 = self.upper_bound_ber(a_hat_1, a[y == 1], self.pa0_y1, self.pa1_y1)

            loss = BER_y - self.lambda_ * (BER_a_0 + BER_a_1)
            
            loss.backward()
            optimizer.step()
    
    def fit(self, dataset_train, dataset_val, epochs=1000, learning_rate=1e-3, batch_size=1024, eval_rate=15, verbose=True):

        dataloader = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
            list(self.adv_decoder_0.parameters()) + list(self.adv_decoder_1.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        for epoch in range(epochs): 
            print(f'Epoch # {epoch+1}')
            self.train_step(batch_size, dataloader, optimizer, verbose)