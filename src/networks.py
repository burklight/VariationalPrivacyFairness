import torch

torch.manual_seed(2020)

class ImageEncoder(torch.nn.Module):

    def __init__(self, input_dim):
        super(ImageEncoder,self).__init__() 

        C = input_dim[0]

        self.func = torch.nn.Sequential(
            torch.nn.Conv2d(C,5,5,padding=2),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU6(inplace=True),
            torch.nn.Conv2d(5,50,5,padding=2),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU6(inplace=True),
            torch.nn.Conv2d(50,3,5,padding=2),
        )
    
    def forward(self, x):

        return self.func(x)
    

class ImageEncoderVec(torch.nn.Module):

    def __init__(self, input_dim, representation_dim):
        super(ImageEncoderVec,self).__init__() 

        C = input_dim[1]

        self.func = torch.nn.Sequential(
            torch.nn.Conv2d(C,5,5,padding=1,stride=2),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(5,50,5,2),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU6(),
            torch.nn.Flatten(),
            torch.nn.Linear(50*5*5,100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU6(),
            torch.nn.Linear(100,representation_dim)
        )
    
    def forward(self, x):

        return self.func(x)
    

class VectorEncoder(torch.nn.Module):

    def __init__(self, input_dim, representation_dim):
        super(VectorEncoder,self).__init__()

        self.func = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU6(),
            torch.nn.Linear(128,32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU6(),
            torch.nn.Linear(32,representation_dim)
        )

        self.func = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,representation_dim)
        )
    
    def forward(self, x):

        return self.func(x)

class DecoderImage(torch.nn.Module):

    def __init__(self):
        super(DecoderImage,self).__init__() 
    
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,5,5,padding=2),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU6(inplace=True),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(5,50,5,padding=2),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU6(inplace=True),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(50,50,5,padding=2),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU6(inplace=True),
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(50,3,5,padding=2),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, y, s):

        s = s.view(-1,1)
        z = self.conv1(y + s.view((-1,) + (1,)*(len(y.shape)-1)))
        z = self.conv2(z + s.view((-1,) + (1,)*(len(z.shape)-1)))
        z = self.conv3(z + s.view((-1,) + (1,)*(len(z.shape)-1)))
        output = self.conv4(z + s.view((-1,) + (1,)*(len(z.shape)-1))) 

        return output

class VecDecoderImage(torch.nn.Module):

    def __init__(self, representation_dim):
        super(VecDecoderImage,self).__init__()

        self.lin_1 = torch.nn.Sequential(
            torch.nn.Linear(representation_dim+1,100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU6(),
        )

        self.lin_2 = torch.nn.Sequential(
            torch.nn.Linear(101,50*5*5),
            torch.nn.Fold(output_size=(5,5),kernel_size=(5,5)),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU6(),
        )

        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(50,5,5,2),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU6(),
            torch.nn.ConvTranspose2d(5,3,5,2),
        )
    
    def forward(self, y, s):

        s.view(-1,1)
        z = self.lin_1(torch.cat((y,s)))
        z = self.lin_2(torch.cat((z,s)))
        output = self.conv(z + s.view((-1) + (1,)*(len(z.shape)+1)))

        return output

class DecoderVector(torch.nn.Module):

    def __init__(self, representation_dim, output_dim):
        super(DecoderVector,self).__init__() 

        self.func = torch.nn.Sequential(
            torch.nn.Linear(representation_dim+1,32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU6(),
            torch.nn.Linear(32,16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU6(),
            torch.nn.Linear(16,output_dim)
        )

        self.func = torch.nn.Sequential(
            torch.nn.Linear(representation_dim+1, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,output_dim)
        )
    
    def forward(self, y, s):

        return self.func(torch.cat((y,s.view(-1,1)),1))


class Encoder(torch.nn.Module):

    def __init__(self, input_type='image', representation_type='image', input_dim=104, representation_dim=8):
        super(Encoder,self).__init__() 

        if input_type == 'image':
            if representation_type == 'image':
                self.f_theta = ImageEncoder(input_dim)
            else:
                self.f_theta = ImageEncoderVec(input_dim, representation_dim)
        else: 
            self.f_theta = VectorEncoder(input_dim, representation_dim)
        
        if representation_type == 'image':
            self.y_logvar_theta = torch.nn.Parameter(torch.Tensor([-1.0]))# * torch.ones(input_dim)))
        else:
            self.y_logvar_theta = torch.nn.Parameter(torch.Tensor(-1.0 * torch.ones(representation_dim)))

    def encode(self, x):

        y_mean = self.f_theta(x)
        return y_mean 

    def reparametrize(self, y_mean): 

        noise = torch.randn_like(y_mean) + self.y_logvar_theta
        y = y_mean + noise 
        return y 
    
    def forward(self, x, noise=True):

        y_mean = self.encode(x) 
        y = self.reparametrize(y_mean) if noise else y_mean 
        return y, y_mean 

class Decoder(torch.nn.Module):

    def __init__(self, representation_type='image', output_type='image', representation_dim=8, output_dim=1):
        super(Decoder,self).__init__() 

        if output_type == 'image':
            if representation_type == 'image':
                self.f_phi = DecoderImage()
            else:
                self.f_phi = VecDecoderImage(representation_dim) 
        else:
            self.f_phi = DecoderVector(representation_dim, output_dim)
            if output_type == 'regression':
                output_dim_reg = 1
                self.out_logvar_phi = torch.nn.Parameter(torch.Tensor(-1.0 * torch.ones(output_dim_reg)))
    

    def forward(self, y, s):

        output = self.f_phi(y, s)
        return output # either x or t