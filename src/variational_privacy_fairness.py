import torch, torchvision 
from progressbar import progressbar
from networks import Encoder, Decoder
import math, os 
import umap 
import warnings 
from numba.errors import NumbaPerformanceWarning 

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

torch.manual_seed(2020)

class VPAF(torch.nn.Module):

    def __init__(self, input_type='image', representation_type='image', output_type='image', input_dim=104, representation_dim=8, output_dim=1, problem='privacy', beta=1.0, gamma=1.0):
        super(VPAF,self).__init__() 

        self.problem = problem   
        self.beta = beta
        self.gamma = gamma
        self.input_type = input_type
        self.representation_type = representation_type
        self.output_type = output_type  

        self.encoder = Encoder(input_type, representation_type, input_dim, representation_dim) 
        self.decoder = Decoder(representation_type, output_type, representation_dim, output_dim)
    
    def get_IXY_ub(self, y_mean):

        Dkl = -0.5 * torch.sum(
            1.0 + self.encoder.y_logvar_theta - torch.pow(y_mean, 2) - torch.exp(self.encoder.y_logvar_theta)
        )
        IXY_ub = Dkl / len(y_mean) / math.log(2) # in bits 

        return IXY_ub

    def get_IYT_given_S_lb(self, decoder_output, t):

        if self.output_type == 'classes':
            CE = torch.nn.functional.cross_entropy(decoder_output, t, reduction='sum')
        else: # regression 
            CE = 0.5 * torch.sum(
                math.log(2*math.pi) + self.decoder.t_logvar_phi - torch.pow(decoder_output - t, 2) / (torch.exp(self.decoder.t_logvar_phi) + 1e-10)
            )
        IYT_given_S_lb = CE / len(t) / math.log(2) # in bits 

        return IYT_given_S_lb
    
    def get_IXY_given_S_lb(self, decoder_output, x):

        if self.output_type == 'image':
            CE = torch.nn.functional.binary_cross_entropy(decoder_output, x, reduction='sum')
        elif self.output_type == 'classes':
            CE = torch.nn.functional.cross_entropy(decoder_output, x, reduction='sum')
        else: # regression 
            CE = - 0.5 * torch.sum(
                math.log(2*math.pi) + self.decoder.t_logvar_phi - torch.pow(decoder_output - x, 2) / (torch.exp(self.decoder.t_logvar_phi) + 1e-10)
            )
        IXY_given_S_lb = CE / len(x) / math.log(2) # in bits 

        return IXY_given_S_lb

    def fit(self, dataset, epochs=1000, learning_rate=1e-3, batch_size=1024, eval_rate=15, logs_dir='../results/logs/', figs_dir='../results/images/'):

        device = 'cuda' if next(self.encoder.parameters()).is_cuda else 'cpu' 
        
        N = len(dataset)
        N_tr = int(N * 0.85)
        N_val = N - N_tr
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [N_tr, N_val]) 
        del dataset
        
        dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val,batch_size=len(dataset_val),shuffle=False)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        for epoch in range(epochs): 

            running_loss = 0
            running_IXY_ub = 0 
            running_H_output_given_SY_ub = 0 

            for x, t, s in progressbar(dataloader_train):

                x = x.to(device).float() 
                t = t.to(device).float()
                s = s.to(device).float() 

                optimizer.zero_grad() 

                y, y_mean = self.encoder(x)
                output = self.decoder(y,s)

                IXY_ub = self.get_IXY_ub(y_mean)
                if self.problem == 'privacy':
                    H_output_given_SY_ub = self.get_IXY_given_S_lb(output, x)
                    loss = IXY_ub + self.beta * H_output_given_SY_ub
                else: # fairness
                    H_output_given_SY_ub = self.get_IYT_given_S_lb(output, t)
                    loss = IXY_ub + self.gamma * H_output_given_SY_ub
                
                loss.backward()
                optimizer.step()
                
                running_IXY_ub += IXY_ub.item() 
                running_H_output_given_SY_ub  += H_output_given_SY_ub.item() 
                running_loss += loss.item()
            
            print(f'Loss: {running_loss / len(dataloader_train)}')
            print(f'IXY_ub: {running_IXY_ub / len(dataloader_train)}')
            print(f'H_output_given_SY_ub: {running_H_output_given_SY_ub / len(dataloader_train)}')

            with torch.no_grad():

                if epoch % eval_rate == eval_rate - 1:
                                    
                    print(f'Evaluating...')

                    for x, t, s in progressbar(dataloader_val):

                        x = x.to(device).float()
                        t = t.to(device).float() 
                        s = s.to(device).float()

                        y, y_mean = self.encoder(x)
                        output = self.decoder(y,s)

                        if self.input_type == 'image' and self.representation_type == 'image' and self.output_type == 'image':
                            torchvision.utils.save_image(x[:12*8],os.path.join(figs_dir,'x.png'),nrow=12)
                            torchvision.utils.save_image(y_mean[:12*8],os.path.join(figs_dir,'y.png'),nrow=12)
                            torchvision.utils.save_image(output[:12*8],os.path.join(figs_dir,'x_hat.png'),nrow=12)

                        IXY_ub = self.get_IXY_ub(y_mean)
                        if self.problem == 'privacy':
                            H_output_given_SY_ub = self.get_IXY_given_S_lb(output, x)
                            loss = IXY_ub + self.beta * H_output_given_SY_ub
                        else: # fairness
                            H_output_given_SY_ub = self.get_IYT_given_S_lb(output, t)
                            loss = IXY_ub + self.gamma * H_output_given_SY_ub
                    
                    print(f'Loss: {loss.item()}')
                    print(f'IXY_ub: {IXY_ub.item()}')
                    print(f'H_output_given_SY_ub: {H_output_given_SY_ub.item()}')