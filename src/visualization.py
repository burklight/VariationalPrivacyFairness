import numpy as np
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor

def plot_vs_figure(x, y, xname, yname, figsdir, figname, maxval=None):

    fig, ax = plt.subplots()
    model = GaussianProcessRegressor()
    model.fit(x.reshape(-1,1), y)
    x_plot = np.linspace(x.min(), x.max(), 100)
    y_plot = model.predict(x_plot.reshape(-1,1))
    plt.scatter(x,y,marker='.',color='black')
    plt.plot(x_plot, y_plot, linestyle = ':', color='black')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(figsdir + figname + '.eps', format='eps')

def plot_figure(x, y, xname, yname, figsdir, figname):

    fig, ax = plt.subplots()
    plt.plot(x, y, marker='.',linestyle = ':', color='black')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(figsdir + figname + '.eps', format='eps')