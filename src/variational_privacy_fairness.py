import torch, torchvision 
from progressbar import progressbar
from networks import Encoder, Decoder
from metrics import get_accuracy, get_discrimination, get_error_gap, get_eq_odds_gap
from mutual_information import get_conditional_entropy, get_entropy
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
        IXY_ub = Dkl / y_mean.shape[0] / math.log(2) # in bits 

        return IXY_ub

    def get_IYT_given_S_lb(self, decoder_output, t):

        if self.output_type == 'classes':
            CE = torch.nn.functional.cross_entropy(decoder_output, t, reduction='sum')
        elif self.output_type == 'single_class':
            CE = torch.nn.functional.binary_cross_entropy_with_logits(decoder_output.view(-1), t, reduction='sum')
        else: # regression 
            CE = 0.5 * torch.sum(
                math.log(2*math.pi) + self.decoder.out_logvar_phi + torch.pow(decoder_output - t, 2) / (torch.exp(self.decoder.out_logvar_phi) + 1e-10)
            )
        IYT_given_S_lb = CE / t.shape[0] / math.log(2) # in bits 

        return IYT_given_S_lb
    
    def get_IXY_given_S_lb(self, decoder_output, x):

        if self.output_type == 'binary':
            CE = torch.nn.functional.binary_cross_entropy(decoder_output, x, reduction='sum')
        elif self.output_type == 'classes':
            CE = torch.nn.functional.cross_entropy(decoder_output.view(-1), x, reduction='sum')
        else: # regression 
            CE = 0.5 * torch.sum(
                math.log(2*math.pi) + self.decoder.out_logvar_phi + torch.pow(decoder_output - x, 2) / (torch.exp(self.decoder.out_logvar_phi) + 1e-10)
            )

        return IXY_given_S_lb
    
    def evaluate(self, dataset, verbose, figs_dir):

        device = 'cuda' if next(self.encoder.parameters()).is_cuda else 'cpu' 
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=len(dataset),shuffle=False)
        
        with torch.no_grad():
            for x, t, s in dataloader:
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
                    H_X_given_SY_ub = self.get_IXY_given_S_lb(output, t)
                    print(f'IXY: {IXY_ub.item()}') if verbose else 0 
                    print(f'HX_given_SY: {H_X_given_SY_ub.item()}') if verbose else 0 
                    return IXY_ub, H_X_given_SY_ub
                else: # fairness
                    H_T_given_S = get_conditional_entropy(dataset.targets, dataset.hidden)
                    H_T_given_SY_ub = self.get_IYT_given_S_lb(output, t)
                    accuracy = get_accuracy(output, t)
                    discrimination = get_discrimination(output, t, s)
                    error_gap = get_error_gap(output, t, s)
                    eq_odds_gap = get_eq_odds_gap(output, t, s)
                    IYT_given_S = H_T_given_S - H_T_given_SY_ub
                    print(f'IXY: {IXY_ub.item()}') if verbose else 0
                    print(f'ITY_given_S_approx: {IYT_given_S.item()}') if verbose else 0
                    print(f'Accuracy: {accuracy}') if verbose else 0 
                    print(f'Discrimination: {discrimination}') if verbose else 0 
                    print(f'Error gap: {error_gap}') if verbose else 0 
                    print(f'Equalized odds gap {eq_odds_gap}') if verbose else 0 
                    return IXY_ub, IYT_given_S, accuracy, discrimination, error_gap, eq_odds_gap

    def train_step(self, batch_size, learning_rate, dataloader, optimizer, verbose):

        device = 'cuda' if next(self.encoder.parameters()).is_cuda else 'cpu' 

        for x, t, s in progressbar(dataloader):

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


    def fit(self, dataset_train, dataset_val, epochs=1000, learning_rate=1e-3, batch_size=1024, eval_rate=15, \
        verbose=True, logs_dir='../results/logs/', figs_dir='../results/images/'):

        dataloader = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        for epoch in range(epochs): 
            print(f'Epoch # {epoch+1}')
            self.train_step(batch_size, learning_rate, dataloader, optimizer, verbose)

            if epoch % eval_rate == eval_rate - 1:
                print(f'Evaluating TRAIN') if verbose else 0 
                if self.problem == 'privacy':
                    IXY, HX_given_SY = self.evaluate(dataset_train, verbose, figs_dir)
                else: # fairness
                    IXY, ITY_given_S, accuracy, discrimination = self.evaluate(dataset_train, verbose, figs_dir)
                print(f'Evaluating VALIDATION/TEST') if verbose else 0 
                if self.problem == 'privacy':
                    IXY, HX_given_SY = self.evaluate(dataset_val)
                else: # fairness
                    IXY, ITY_given_S, accuracy, discrimination = self.evaluate(dataset_val, verbose, figs_dir)
