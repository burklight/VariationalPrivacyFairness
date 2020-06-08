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

        C = input_dim[0]

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
        )

        self.conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU6(),
            torch.nn.ConvTranspose2d(50,5,5,2),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU6(),
            torch.nn.ConvTranspose2d(5,3,5,2,padding=1,output_padding=1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, y, s):

        s = s.view(-1,1)
        z = self.lin_1(torch.cat((y,s),1))
        z = self.lin_2(torch.cat((z,s),1))
        z = z.view(-1,50,5,5)
        output = self.conv(z + s.view((-1,) + (1,)*(len(z.shape)-1)))

        return output

class DecoderVector(torch.nn.Module):

    def __init__(self, representation_dim, output_dim, s_dim):
        super(DecoderVector,self).__init__() 

        self.s_dim = s_dim

        self.func = torch.nn.Sequential(
            torch.nn.Linear(representation_dim+self.s_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,output_dim)
        )
    
    def forward(self, y, s):

        return self.func(torch.cat((y,s.view(-1,self.s_dim)),1))


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
        
        self.y_logvar_theta = torch.nn.Parameter(torch.Tensor([-1.0]))

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

    def __init__(self, representation_type='image', output_type=['image'], representation_dim=8, output_dim=[1], regression_dims=-1, s_dim=1):
        super(Decoder,self).__init__() 

        if 'image' in output_type:
            if representation_type == 'image':
                self.f_phi = DecoderImage()
            else:
                self.f_phi = VecDecoderImage(representation_dim) 
        else:
            self.f_phi = DecoderVector(representation_dim, sum(output_dim), s_dim)
            if 'regression' in output_type:
                self.out_logvar_phi = torch.nn.Parameter(torch.Tensor(-1.0 * torch.ones(output_dim[output_type.index('regression')])))
    

    def forward(self, y, s):

        output = self.f_phi(y, s)
        return output # either x or t