import torch 
import torchvision 
import random 
import numpy as np 
import PIL
import pandas as pd
from sklearn import preprocessing

torch.manual_seed(0)

def weight_init(m):
    '''
    This function is used to initialize the netwok weights
    '''

    if isinstance(m,torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m,torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)

class MNIST_dataset(torch.utils.data.Dataset):

    def __init__(self,data,targets,data_hidden,transform=None):
        self.data = data
        self.targets = targets
        self.hidden = data_hidden
        self.transform = transform
    
    def __getitem__(self,index):
        image, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        image, target, hidden = torchvision.transforms.functional.to_pil_image(datum), int(target), int(hidden)
        if self.transform is not None:
            datum = self.transform(datum)
        return image, target, hidden
    
    def __len__(self):
        return len(self.targets)

def get_mnist():

    # Load normal MNIST dataset 
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))
    
    # Add the color and normalize to 0..1
    N_tr = len(trainset)
    data_n = torch.zeros(N_tr,3,28,28)
    data_hidden = torch.arange(len(trainset.targets)) % 3
    for n in range(N_tr):
        data_n[n,data_hidden[n]] = trainset.data[n]
    data_n /= 255.0 
    trainset = MNIST_dataset(data_n,trainset.targets,data_hidden,trainset.transform)
    N_tst = len(testset)
    data_n = torch.zeros(N_tst,3,28,28)
    data_hidden = torch.arange(len(testset.targets)) % 3
    for n in range(N_tst):
        data_n[n,data_hidden[n]] = testset.data[n]
    data_n /= 255.0 
    testset = MNIST_dataset(data_n,testset.targets,data_hidden,testset.transform)

    return trainset, testset

class Adult_dataset(torch.utils.data.Dataset):

    def __init__(self,data,targets,data_hidden,transform=None):
        self.data = data
        self.targets = targets
        self.hidden = data_hidden
        self.transform = transform
    
    def __getitem__(self,index):
        datum, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        datum, target, hidden = float(datum), int(target), int(hidden)
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, target, hidden
    
    def __len__(self):
        return len(self.targets)

def get_adult():

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country','salary']
    dummy_variables = {
        'workclass': ['Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'],
        'education': ['Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, \
            12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'],
        'marital-status': ['Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent,\
            Married-AF-spouse'],
        'occupation': ['Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, \
            Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, \
            Protective-serv, Armed-Forces'],
        'relationship': ['Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'],
        'race': ['White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'],
        'sex': ['Female, Male'],
        'native-country' : ['United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), \
            India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, \
            Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, \
            Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands']
    }
    for k in dummy_variables:
        dummy_variables[k] = [v.strip() for v in dummy_variables[k][0].split(',')]
    
    # Load Adult dataset
    data_train = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        names=column_names,header=None
    )
    data_test = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        names=column_names,skiprows=1,header=None
    )
    data_train = data_train.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
    data_test = data_test.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)

    '''
    def get_variables(data):

        dummy_columns = list(dummy_variables.keys())
        dummy_columns.remove('sex')
        for k in dummy_variables:
            data[k] = data[k].astype('category').cat.set_categories(dummy_variables[k])
        X = pd.get_dummies(data.drop('sex', axis=1).drop('salary',axis=1), columns=dummy_columns).to_numpy()
        S = data['sex'].to_numpy()
        T = data['salary'].to_numpy()
        T = np.where(np.logical_or(T=='<=50K',T=='<=50K.'),0,1)
        S = np.where(S=='Male',0,1)

        return X.astype(float), S, T
        
    X_train, S_train, T_train = get_variables(data_train)
    X_test, S_test, T_test = get_variables(data_test)
    X_mean, X_std = X_train[:,:6].mean(), X_train[:,:6].std()
    X_train[:,:6] = (X_train[:,:6]-X_mean) / (X_std) 
    X_test[:,:6] = (X_test[:,:6]-X_mean) / (X_std)
    '''

    def get_variables(data):
    
        le = preprocessing.LabelEncoder()
        dummy_columns = list(dummy_variables.keys())
        dummy_columns.remove('sex')
        data[dummy_columns] = data[dummy_columns].apply(lambda col: le.fit_transform(col))
        X = data.drop('sex',axis=1).drop('salary',axis=1).to_numpy()
        S = data['sex'].to_numpy()
        T = data['salary'].to_numpy()
        T = np.where(np.logical_or(T=='<=50K',T=='<=50K.'),0,1)
        S = np.where(S=='Male',0,1)

        return X.astype(float), S, T 
    
    X_train, S_train, T_train = get_variables(data_train)
    X_test, S_test, T_test = get_variables(data_test)
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    X_train = (X_train-X_mean) / (X_std) 
    X_test = (X_test-X_mean) / (X_std)

    trainset = Adult_dataset(X_train, T_train, S_train)
    testset = Adult_dataset(X_test, T_test, S_test)

    return trainset, testset


class US_dataset(torch.utils.data.Dataset):

    def __init__(self,data,targets,data_hidden,transform=None):
        self.data = data
        self.targets = targets
        self.hidden = data_hidden
        self.transform = transform
    
    def __getitem__(self,index):
        datum, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        datum, target, hidden = float(datum), int(target), int(hidden)
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, target, hidden
    
    def __len__(self):
        return len(self.targets)


def get_us_cens():

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country','salary']
    dummy_variables = {
        'workclass': ['Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'],
        'education': ['Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, \
            12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'],
        'marital-status': ['Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent,\
            Married-AF-spouse'],
        'occupation': ['Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, \
            Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, \
            Protective-serv, Armed-Forces'],
        'relationship': ['Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'],
        'race': ['White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'],
        'sex': ['Female, Male'],
        'native-country' : ['United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), \
            India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, \
            Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, \
            Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands']
    }
    for k in dummy_variables:
        dummy_variables[k] = [v.strip() for v in dummy_variables[k][0].split(',')]
    
    # Load Adult dataset
    data_train = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        names=column_names,header=None
    )
    data_test = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        names=column_names,skiprows=1,header=None
    )
    data_train = data_train.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
    data_test = data_test.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)

    def get_variables(data):
    
        le = preprocessing.LabelEncoder()
        dummy_columns = list(dummy_variables.keys())
        data[dummy_columns] = data[dummy_columns].apply(lambda col: le.fit_transform(col))
        X = data[['age','sex','education']].to_numpy()
        S = data[['age','salary']].to_numpy()
        T = data[['age','sex','education']].to_numpy()

        return X.astype(float), S, T.astype(float)
    
    X_train, S_train, T_train = get_variables(data_train)
    X_test, S_test, T_test = get_variables(data_test)
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    X_train = (X_train-X_mean) / (X_std) 
    X_test = (X_test-X_mean) / (X_std)   
    T_train[:,0] = X_train[:,0]
    T_test[:,0] = X_test[:,0]

    trainset = US_dataset(X_train, T_train, S_train)
    testset = US_dataset(X_test, T_test, S_test)

    return trainset, testset