"""
Created on Thu Sep 15 01:54:17 2022

@author: abhil
"""
import sys
import datetime
import scipy.stats
import statsmodels.formula.api as smf
from torch.autograd import Variable
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import yfinance as yf
from csv import writer
from torchmin import minimize
import torch.nn as nn
from torch.distributions import Exponential, Uniform
from torch.nn import Softmax
import random
from cvxpylayers.torch import CvxpyLayer
import torch
import cvxpy as cp
import copy
# import statsmodels.api as sm
from scipy.stats import binned_statistic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
import os
import csv
import math
from scipy.stats import multivariate_normal
from torch import nn
import torch.distributions as D
from scipy.stats import f
from torch.distributions.transforms import CorrCholeskyTransform
import time
import pickle
from sqwash import SuperquantileReducer, SuperquantileSmoothReducer

import torch.autograd as autograd
from itertools import accumulate
from torchmetrics import R2Score
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_pinball_loss
from sklearn.svm import SVC, SVR # import SVC model
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import multivariate_normal
warnings.filterwarnings('ignore')

from torchmin import minimize_constr
torch.set_default_tensor_type(torch.DoubleTensor)
old_stdout = sys.stdout
log_file = open(r'D:\projects\spdlayers\prod_run\logs\simulated\output_log.log', "w")
sns.set(palette='colorblind', font_scale=1.3)
palette = sns.color_palette()
seed = 456
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)
# caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.append('D:/projects/spdlayers/')

torch.set_default_dtype(torch.double)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensor.to(device)


mod = 'fixed_target'  # ['find_target', 'fixed_target', 'no_reg']


def generate_mixture_data(mean_list,cov_list, prob_list, num_samples):
    mixK = len(prob_list)

    mix_rand_u = np.random.rand(num_samples)
    data = np.random.multivariate_normal(mean_list[0], cov_list[0], 0)

    sumPs = 0
        
    for mix_index in range(mixK):
        data_a = np.random.multivariate_normal(mean_list[mix_index], cov_list[mix_index], num_samples)
    
        sumPs = sumPs+prob_list[mix_index]
        mix_N = (mix_rand_u<=sumPs).sum()-len(data)
        data = np.concatenate((data, data_a[:mix_N,:]), axis=0)

    np.random.shuffle(data)
    np.savetxt("D:/projects/spdlayers/prod_run/data/simulated/sim_data.txt", data)
   
    return data


def plot_sim_data(mean_list,cov_list, prob_list, alpha_values):
    # Generate and plot the data for each alpha value
    fig = plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
    
    
    for i, alpha in enumerate(alpha_values):
        
        data = generate_mixture_data(mean_list,cov_list, prob_list, num_samples = 50000)
    
        # Separate the dimensions
        x = data[:, 0]
        y = data[:, 1]
        x1 = data[:, 2]
        y1 = data[:, 3]
    
        # Plot the data
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.scatter(x, y, x1, c='blue', label='Dependent Variables')
        ax.scatter(x, y, y1, c='red', label='Independent Variables')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Alpha = {alpha}')
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect values as needed
    plt.suptitle('Biased Mixture Data for Varying Alpha', fontsize=16)
    plt.show() 
    
def conditional_data_generation(data, mean_list, cov_list, prob_list,num_points):
    num_observations = data.shape[0]

    conditional_data = []

    for obs_index in range(num_observations):
        obs_data = data[obs_index]
        conditional_mean_list = []
        conditional_cov_list = []
        likelihood = np.zeros(len(mean_list))

        for i in range(len(mean_list)):
            mean = mean_list[i]
            cov = cov_list[i]
            prob = prob_list[i]
               

            cyy = cov[0:2, 0:2]  # Covariance matrix of the dependent variables
            cyx = cov[0:2, 2:4]  # Custom array only containing covariances, not variances
            cxy = cov[2:4, 0:2]  # Same as above
            cxx = cov[2:4, 2:4]  # Covariance matrix of independent variables
            

            my = mean[0:2]  # Mu of dependent variables
            mx = mean[2:4]  # Mu of independent variables
            
           
            conditional_mu = my + np.dot(cyx, np.linalg.inv(cxx)).dot((obs_data - mx))
            conditional_cov = cyy - np.dot(cyx, np.dot(np.linalg.inv(cxx), cxy))       #np.linalg.inv(np.linalg.inv(cov)[0:2, 0:2])
            
            

            conditional_mean_list.append(conditional_mu)
            conditional_cov_list.append(conditional_cov)

            likelihood[i] = multivariate_normal.pdf(obs_data, mx, cxx)*prob

        conditional_prob_list = (likelihood/sum(likelihood)).tolist()
       
        dependent_data = generate_mixture_data(conditional_mean_list,conditional_cov_list, conditional_prob_list, num_points)

        conditional_data.append(dependent_data)

    return np.array(conditional_data)
  

def plot_conditional_data(obs_data, conditional_data, radius, cov):
    num_observations = len(obs_data)
    num_rows = (num_observations + 1) // 2  # Number of subplot rows
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))

    colors = ['red', 'red', 'red', 'red', 'green', 'green', 'green', 'green']

    for i, ax in enumerate(axs.flatten()):
        if i < num_observations:
            ax.scatter(conditional_data[i][:, 0], conditional_data[i][:, 1], alpha=0.2, color=colors[i],
                       label='Conditional Data')
            ax.scatter(obs_data[i][0], obs_data[i][1], color='blue', marker='x', s=100,
                       label='Observed Data')

            eigvals, eigvecs = np.linalg.eig(cov[i])
            theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

            ellipse = Ellipse(xy=obs_data[i], width=2 * radius * np.sqrt(eigvals[0]),
                              height=2 * radius * np.sqrt(eigvals[1]), angle=theta,
                              edgecolor='black', facecolor='none', linestyle='dashed')
            ax.add_patch(ellipse)

            ax.set_xlabel('Variable 1')
            ax.set_ylabel('Variable 2')
            ax.set_title('Observation {}'.format(i + 1))
            ax.legend()
            ax.axis('equal')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def generate_covariance_matrix(variance_1, variance_2, correlation, spread_factor):
    # Create the 2x2 covariance matrix for the independent features
    cov_independent = np.array([[variance_1, 0],
                               [0, variance_2]])

    # Create the 2x2 submatrix with desired positive semi-definite properties
    submatrix = np.array([[variance_1, correlation * np.sqrt(variance_1 * variance_2)],
                          [correlation * np.sqrt(variance_1 * variance_2), variance_2]])

    # Create the 4x4 covariance matrix
    cov_matrix = np.zeros((4, 4))
    cov_matrix[:2, :2] = cov_independent
    cov_matrix[2:4, 2:4] = submatrix * spread_factor
    cov_matrix[0, 2] = correlation * np.sqrt(variance_1 * variance_2)
    cov_matrix[2, 0] = correlation * np.sqrt(variance_1 * variance_2)

    return cov_matrix

def generate_covariance_matrix_scaled(mean_a, spread, correlation, distance):
    cov_a = np.diag(spread[:2])  # Covariance matrix for A (dimensions 1 and 2)
    cov_b = np.diag(spread[2:])  # Covariance matrix for B (dimensions 3 and 4)

    # Calculate the covariance between A and B based on correlation
    cov_ab = np.sqrt(cov_a) * correlation * np.sqrt(cov_b)

    # Calculate the mean difference vector between A and B based on distance
    mean_diff = np.array([distance, distance])
    
    # Adjust the mean of B based on the mean difference vector
    mean_b = mean_a + mean_diff
    
    # Adjust the covariance of B based on the distance
    cov_b_adjusted = cov_b * (distance ** 2)
    
    # Assemble the 4x4 covariance matrix
    covariance_matrix = np.block([[cov_a, cov_ab], [cov_ab.T, cov_b_adjusted]])

    # Ensure the covariance matrix is positive semidefinite
    eigenvalues, eigenvectors = scipy.linalg.eigh(covariance_matrix)
    epsilon = 1e-6  # Small constant to ensure positive semidefiniteness

    if np.any(eigenvalues < -epsilon):
        eigenvalues[eigenvalues < -epsilon] = epsilon

    covariance_matrix = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))

    return covariance_matrix

def is_psd(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0)


def sim_data_gen_new_method(sim_data_params):
    
    def conditional_data_generation(mean, cov):
        c11 = cov[0:2, 0:2] # Covariance matrix of the dependent variables
        c12 = cov[0:2, 2:4] # Custom array only containing covariances, not variances
        c21 = cov[2:4, 0:2] # Same as above
        c22 = cov[2:4, 2:4] # Covariance matrix of independent variables
        
        m1 = mean[0:2].T # Mu of dependent variables
        m2 = mean[2:4].T # Mu of independent variables
        conditional_data = np.random.multivariate_normal(m2, c22, 1000)
        conditional_mu = m1 + c12.dot(np.linalg.inv(c22)).dot((conditional_data - m2).T).T
        conditional_cov = np.linalg.inv(np.linalg.inv(cov)[0:2, 0:2])
        dependent_data = np.array([np.random.multivariate_normal(c_mu, conditional_cov, 1)[0] for c_mu in conditional_mu])
        
        
    var_centers = sim_data_params[0]
    side_centers = sim_data_params[1]
    mean_1 = [0, -var_centers, - side_centers, 0]
    f = 10.0
    f1 = 0.9
    cov_1 = [[1,   0.0, 0.9, 0.9],
              [0.0,   1, 0.9, 0.9],
              [0.9, 0.9, f, 0.95*f],
              [0.9, 0.9, 0.95*f, f]]
    
    
    mean_2 = [0, -var_centers, -side_centers, 0]
    cov_2 = [[1,   0.0, 0.9, 0.9],
              [0.0,   1, 0.9, 0.9],
              [0.9, 0.9, 1, -0.5],
              [0.9, 0.9, -0.5, 1]]
    
    x, y, x1, y1 = np.random.multivariate_normal(mean_1, cov_1, 10000).T
    a, b, a1, b1 = np.random.multivariate_normal(mean_2, cov_2, 40000).T
    data1 = np.array([x, y, x1, y1]).T
    data2 = np.array([a, b, a1, b1]).T
    
    data1 = np.c_[ data1, np.ones(len(data1)) ]
    data2 = np.c_[ data2, 2*np.ones(len(data2)) ]
      
    data = np.concatenate((data1, data2), axis=0)
    data_pd = pd.DataFrame(data)
    data_final1 = data_pd.sample(frac=0.1)
    
   
    
    mean_3 = [0, var_centers, side_centers, 0]
    cov_3 = [[1,   0.0, -0.9, -0.9],
              [0.0,   1, -0.9, -0.9],
              [-0.9, -0.9, 1, -0.0],
              [-0.9, -0.9, -0.0, 1]]
    a, b, a1, b1 = np.random.multivariate_normal(mean_3, cov_3, 5000).T
    data3 = np.array([a, b, a1, b1]).T
    
    data3 = np.c_[ data3, 3*np.ones(len(data3)) ]
    
    data_3dist = np.concatenate((data_final1, data3), axis=0)
    data_3dist_pd = pd.DataFrame(data_3dist)
    data_final = data_3dist_pd.sample(frac=0.2)
    data_final = data_final.sample(frac=1).reset_index(drop=True)
    
    data_tosave = data_final.iloc[:, :4]

    data_final.to_csv('D:/projects/spdlayers/prod_run/data/simulated/sim_data_with_assignment.txt',
                      header=None, sep=',', index=False)
    data_tosave.to_csv('D:/projects/spdlayers/prod_run/data/simulated/sim_data.txt',
                      header=None, sep=',', index=False)
    
   
    
def sim_data_gen_conformal(sim_data_params):
    var_centers = 0
    side_centers1 = sim_data_params[0]
    side_centers2 = sim_data_params[1]
    sf = sim_data_params[2]
    mean_1 = [0, -var_centers, - side_centers1, 0]
    cov_1 = [[1,   0.0, 0.9, 0.9],
             [0.0,   1, 0.9, 0.9],
             [0.9, 0.9, 1, 0.0],
             [0.9, 0.9, 0.0, 1]]
    mean_2 = [0, var_centers, side_centers2, 0]
    cov_2 = [[1,   0.0, -0.9, -0.9],
             [0.0,   1, -0.9, -0.9],
             [-0.9, -0.9, 1, -0.0],
             [-0.9, -0.9, -0.0, 1]]
    mean_2_ = [0, var_centers, side_centers2, 0]
    cov_2_  = [[1,   0.0, -0.9, -0.9],
             [0.0,   1, -0.9, -0.9],
             [-0.9, -0.9, 1, sf],
             [-0.9, -0.9, sf, 1]]
    x, y, x1, y1 = np.random.multivariate_normal(mean_1, cov_1, 50000).T
    a, b, a1, b1 = np.random.multivariate_normal(mean_2, cov_2, 50000).T
    a_, b_, a1_, b1_ = np.random.multivariate_normal(mean_2_, cov_2_, 50000).T
    data1 = np.array([x, y, x1, y1]).T
    data2 = np.array([a, b, a1, b1]).T
    data3 = np.array([a_, b_, a1_, b1_]).T
    data2_ = np.concatenate((data2, data3), axis=0)
    data2_pd = np.array(pd.DataFrame(data2_).sample(frac=0.5))
    data = np.concatenate((data1, data2_pd), axis=0)
    data_pd = pd.DataFrame(data)
    data_final = data_pd.sample(frac=0.2)
    data_final.to_csv('D:/projects/spdlayers/prod_run/data/simulated/sim_data.txt',
                      header=None, sep=',', index=False)


def load_data(exp_type=None, params = None ):
    if exp_type == 'portfolio':
        # , 'VOD',  'ADBE', 'NVDA', 'CRM' ]
        symbols = ['GOOGL', 'TSLA', 'AMZN', 'AAPL', 'MSFT']
        all_stocks = pd.DataFrame()
        for symbol in symbols:
            tmp_close = yf.download(symbol,
                                    start='2015-01-01',
                                    end='2018-01-01',
                                    progress=False)['Adj Close']
            all_stocks = pd.concat([all_stocks, tmp_close], axis=1)
        all_stocks.columns = symbols
        returns = np.log(all_stocks/all_stocks.shift(1)).dropna(how="any")
        returns.reset_index(inplace=True)
        aux_data = pd.DataFrame()
        for symbol in symbols:
            tmp_close = yf.download(symbol,
                                    start='2015-01-01',
                                    end='2018-01-01',
                                    progress=False)['Volume']
            aux_data = pd.concat([aux_data, tmp_close], axis=1)

        aux_data.columns = symbols
        aux_data = aux_data.pct_change().dropna(how="any")
        aux_data.reset_index(inplace=True)
        print(returns.shape, aux_data.shape)
        return returns, aux_data
    if exp_type == 'simulated':
      
        sim_data = pd.read_csv(
            'D:/projects/spdlayers/prod_run/data/simulated/sim_data.txt', header=None, sep=',')
        train_returns = sim_data.iloc[:, :2]
        train_aux = sim_data.iloc[:, 2:]
        print(train_returns.shape, train_aux.shape)
        return train_returns, train_aux

def sample_batch_indices(x, y, batch_size, rs=None):
    if rs is None:
        rs = np.random.RandomState()
    train_ix = np.arange(len(x))
    rs.shuffle(train_ix)
    n_batches = int(np.ceil(len(x) / batch_size))
    batch_indices = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_indices.append(
            train_ix[start:end].tolist())

    return batch_indices

def train_test_scaler(train, val, test):
    scaler = StandardScaler()
    normalized_x_train = pd.DataFrame(scaler.fit_transform(train))
    normalized_x_val = pd.DataFrame( scaler.transform(val))
    normalized_x_test = pd.DataFrame(scaler.transform(test))
    return normalized_x_train, normalized_x_val, normalized_x_test
    
def compute_train_test_split( df, aux1, exp_type):
    if exp_type == "portfolio":
        # , 'VOD',  'ADBE', 'NVDA', 'CRM' ]
        symbols = ['GOOGL', 'TSLA', 'AMZN', 'AAPL', 'MSFT']
        x_train = np.array(aux1[aux1['index'].dt.year != 2017][symbols])
        x_test = np.array(aux1[aux1['index'].dt.year == 2017][symbols])
        y_train = np.array(df[df['index'].dt.year != 2017][symbols])
        y_test = np.array(df[df['index'].dt.year == 2017][symbols])
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        return (torch.from_numpy(x_train), torch.from_numpy(x_test), torch.from_numpy(y_train),            
                torch.from_numpy(y_test),)
    if exp_type == "simulated":
        x_train = np.array(aux1.iloc[:600, :])
        x_val   = np.array(aux1.iloc[600:1000, :])
        x_test  = np.array(aux1.iloc[1000:2000, :])
        y_train = np.array(df.iloc[:600, :])
        y_val   = np.array(df.iloc[600:1000,:])
        y_test  = np.array(df.iloc[1000:2000, :])
        print(x_train.shape,x_val.shape, x_test.shape, y_train.shape,y_val.shape, y_test.shape)
        return (torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(x_test),
                torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test),)


class VectorToTriangularMatrix(nn.Module):
    def __init__(self, vector_size):
        super(VectorToTriangularMatrix, self).__init__()
        self.vector_size = vector_size
        self.n = 2#int((2 * vector_size)**0.5 + 1)

    def forward(self, vec):
        # Initialize a matrix of zeros
        
        L = torch.zeros( vec.size(0), self.n, self.n, device=vec.device)

        # Fill the lower triangular part of the matrix using the vector
        idx_tril = torch.tril_indices(self.n, self.n, offset=-1)
        
        vec_reshaped = vec.reshape(shape=(vec.size(0),1))
        L[:,idx_tril[0], idx_tril[1]] = vec_reshaped
    
        # L[:,idx_tril[0], idx_tril[1]] = vec.squeeze()
        
        # Set the diagonal elements to 1
        L[:, torch.arange(self.n), torch.arange(self.n)] = 1
     
        return L
    
class DeepNormalModel(torch.nn.Module):
    def __init__(
        self,
        n_inputs,
        n_hidden,
        x_scaler,
        y_scaler, mod
    ):
        super().__init__()
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.jitter = 1e-6
        
        self.shared = torch.nn.Linear(n_inputs, n_hidden)
        self.mean_hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.mean_linear = torch.nn.Linear(n_hidden, n_hidden)
        
        self.cho_hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.cho_elements_module = nn.Linear(
            n_hidden, n_hidden * (n_hidden - 1) // 2)
        
        self.cho_diag_hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.cho_diag = torch.nn.Linear(n_hidden, n_hidden)
      
              
        if mod == 'find_target':
            self.target = nn.Parameter(torch.randn(1), requires_grad=True)
            self.input_to_output = nn.Linear(n_inputs, 1)
        elif mod == 'fixed_target':
           
            t_radius = torch.tensor(2.)  # torch.tensor(torch.randint(50, 100, (1,)).item())*0.01  #torch.tensor(1.8) 
            self.radius = nn.Parameter(t_radius, requires_grad=True)
            self.input_to_output = nn.Linear(n_inputs, 1, bias=True)
    def forward(self, x):

        shared_1 = self.shared(x)
        shared_2 = F.tanh(shared_1)      #leaky_relu(shared, negative_slope=0.01)
        
        mean_hidden_1 = self.mean_hidden(shared_2)
        mean_hidden_2 = F.tanh(mean_hidden_1)     #leaky_relu(mean_hidden, negative_slope=0.01)
        mean = self.mean_linear(mean_hidden_2)
        
      
        cho_hidden= self.cho_hidden(shared_2) 
        cho_hidden1= F.tanh(cho_hidden)
        cho_elements = self.cho_elements_module(cho_hidden1)       
        cho_elements = F.leaky_relu(cho_elements, negative_slope=0.01)
        cho = CorrCholeskyTransform()(cho_elements)
        
        
        cho_dh = self.cho_diag_hidden(shared_2)
        cho_dh1= F.tanh(cho_dh)
        cho_d = self.cho_diag(cho_dh1)
        cho_d1 = F.softplus(cho_d) 
        diagonal_indices = torch.arange(cho.shape[2])
        
        cho[:, diagonal_indices, diagonal_indices] = cho_d1
      
        if mod == 'find_target':
            target = self.target
            x = self.input_to_output(x)
            return mean, cho, target, torch.sigmoid(x)
        elif mod == 'fixed_target':
            
            # r = torch.exp(self.radius)

            r = torch.abs(self.radius)
         
            cho_scaled =  r*cho  # torch.matmul(cho, y)
            x = self.input_to_output(x)
            #torch.clamp(radius, min = 1e-2)

            return mean, cho_scaled,r, torch.sigmoid(x)
        else:
            return mean, cho
        
        
class DeepNormalModel_NLL(torch.nn.Module):
    def __init__(
        self,
        n_inputs,
        n_hidden,
        x_scaler,
        y_scaler, mod
    ):
        super().__init__()
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.jitter = 1e-6
        self.shared = torch.nn.Linear(n_inputs, n_hidden)
        self.mean_hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.mean_linear = torch.nn.Linear(n_hidden, n_hidden)
        
        self.cho_hidden = torch.nn.Linear(n_inputs, n_hidden)
        self.cho_elements_module = nn.Linear(
            n_hidden, n_hidden * (n_hidden - 1) // 2)
        
        self.cho_diag_hidden = torch.nn.Linear(n_inputs, n_hidden)
        self.cho_diag = torch.nn.Linear(n_hidden, n_hidden)
      
    
    def forward(self, x):

        shared_1 = self.shared(x)
        shared_2 = F.tanh(shared_1)      #leaky_relu(shared, negative_slope=0.01)
        # shared = self.dropout(shared)
        # Parametrization of the mean
        mean_hidden_1 = self.mean_hidden(shared_2)
        mean_hidden_2 = F.tanh(mean_hidden_1)     #leaky_relu(mean_hidden, negative_slope=0.01)
        mean = self.mean_linear(mean_hidden_2)
        
       
        cho_hidden= self.cho_hidden(shared_2)
        cho_hidden1= F.tanh(cho_hidden)
        cho_elements = self.cho_elements_module(cho_hidden1)        
        cho_elements = F.leaky_relu(cho_elements, negative_slope=0.01)
        
        cho = CorrCholeskyTransform()(cho_elements)
        
        
        cho_dh = self.cho_diag_hidden(x)
        cho_dh1= F.tanh(cho_dh)
        cho_d = self.cho_diag(cho_dh1)
        cho_d1 = F.softplus(cho_d) 
        diagonal_indices = torch.arange(cho.shape[2])
        
        cho[:, diagonal_indices, diagonal_indices] = cho_d1
                    
        return mean, cho
      

def compute_loss(model, x, y, kl_reg=0.1):
    y_scaled = model.y_scaler(y)
    mean, cho = model(x)
    y_hat = D.MultivariateNormal(mean, scale_tril=cho)
    m = y_hat.sample()
    neg_log_likelihood = -y_hat.log_prob(m)
    # neg_log_likelihood = -y_hat.log_prob(y_scaled)
    return torch.mean(neg_log_likelihood)

def multivariateNLL(model, x, y, mod):
    
    mean, cho = model(x)
   
    y_scaled = model.y_scaler(y)
   
    cov_inv = torch.matmul(torch.linalg.inv(torch.transpose(cho, 1, 2)), torch.linalg.inv(cho))
    cov = torch.matmul(cho, torch.transpose(cho, 1, 2))
    
    
    distance = y_scaled - mean
    
    temp_mat = torch.einsum('ij,ijk,ik->i', distance, cov_inv, distance)
    
    loss = 0.5*torch.log(torch.linalg.det(cov_inv)).view(y_scaled.shape[0], 1, 1)  + 0.5*temp_mat
    
    final_loss = loss.nansum()
    final_loss.retain_grad()
    return final_loss



# Logistic regression model defined externally for reuse and gradient flow
class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def dist_loss(model, x, y,cgd_train_batch, mod,gamma, target, test_mode=None):
    convergence_threshold=1e-4
    max_epochs=500
    lr=1e-2
    mean, cho, radius, _ = model(x)
    cov_inv = torch.matmul(torch.linalg.inv(torch.transpose(cho, 1, 2)), torch.linalg.inv(cho))

   
    def compute_distance(y_train, mean, cov_inv, radius):
        def dist(u, v, c):
            diff = u - v
            m = torch.matmul(torch.matmul(diff.T, c), diff)
            return torch.sqrt(m)
        dist_list = torch.empty(len(y_train))
        for i in range(len(y_train)):
            x = y_train[i]
            m = mean[i]
            c = cov_inv[i]  #*(radius**2)
            r = radius  #[i]
            dis = dist(x,m,c)
            # dis_div_ratio = dis/ r

            # temp = (1/(radius*radius))*torch.matmul(c[i].T, x - m)
            # dis = torch.linalg.norm(temp,  ord=2)
            dist_list[i] = dis   #dis_div_ratio
           
        return dist_list

    mahalanobis_dist = compute_distance(y, mean, cov_inv, radius)
    assign_list = (mahalanobis_dist <= radius).float().unsqueeze(-1)

    log_regr = LogisticRegression(x.shape[1])
    criterion = nn.BCELoss()
    optimizer_conf = optim.Adam(log_regr.parameters(), lr=lr)

    prev_loss = float('inf')
    for epoch in range(int(max_epochs)):
        optimizer_conf.zero_grad()
        predictions = log_regr(x)
        loss_conf = criterion(predictions.float(), assign_list.float())
        loss_conf.backward(create_graph=True)
        optimizer_conf.step()

        if abs(prev_loss - loss_conf.item()) < convergence_threshold:
            break
        prev_loss = loss_conf.item()

    predicted_prob = log_regr(x)
    
    
    conditional_loss = torch.mean((predicted_prob - (1 - target)) ** 2)

    if gamma != 0:
        penalty = nn.LeakyReLU()(target - assign_list.mean())
        loss_final = conditional_loss + 10 * penalty
    else:
        loss_final = conditional_loss
        
  
    return loss_final, assign_list.mean().item(), []



def compute_coverage_loss_for_other_models(model, x, y, mod,gamma, target, test_mode=None):
    y_scaled = y #model.y_scaler(y)
    if mod == 'find_target':
        mean, cho, target, y_hat = model(x)
    elif mod == 'fixed_target':
        mean, cho, radius, y_hat = model(x)
        radius =  torch.abs(radius)
    else:
        mean, cho = model(x)
    cov_inv = torch.matmul(torch.linalg.inv(torch.transpose(cho, 1, 2)), torch.linalg.inv(cho))
    # cov = torch.matmul(cho, torch.transpose(cho, 1, 2))
    theoritical_radius = scipy.stats.chi2.ppf(target, df= mean.shape[1])
    
    def compute_distance(y_train, mean, cov_inv, radius):
        def dist(u, v, c):
            diff = u - v
            m = torch.matmul(torch.matmul(diff.T, c), diff)
            return torch.sqrt(m)
        dist_list = torch.empty(len(y_train))
        for i in range(len(y_train)):
            x = y_train[i]
            m = mean[i]
            c = cov_inv[i]  #*(radius**2)
            r = radius  #[i]
            dis = dist(x,m,c)
            # dis_div_ratio = dis/ r

            # temp = (1/(radius*radius))*torch.matmul(c[i].T, x - m)
            # dis = torch.linalg.norm(temp,  ord=2)
            dist_list[i] = dis   #dis_div_ratio
           
        return dist_list
    
    
    def assignment_loss(loss_list, sig_level, type):
        if type == 'assign':
            assign = torch.Tensor([1 if i <= 1 else 0 for i in loss_list])
            assign = assign.view(y.shape[0], 1)
        elif type == 'distance':
            dist = torch.Tensor(loss_list) 
            m = nn.Sigmoid()
            # assign = m(-5*F.leaky_relu(dist.view(500, 1), negative_slope=5))
            assign = 1 - m(dist.view(y_scaled.shape[0],1))
        elif type == 'adjusted':
            assign = torch.Tensor([1 if i <= 1 else 0 for i in loss_list])
            dist = torch.Tensor(loss_list)
            m = nn.Sigmoid()
            dist = 1 -  m(dist.view(y_scaled.shape[0],1))
            # dist = m(-5*F.leaky_relu(dist.view(500, 1), negative_slope=5))
            assign = assign.view(y_scaled.shape[0], 1) - dist.detach() + dist
        return assign

    loss_list = compute_distance(y_scaled, mean, cov_inv, radius)
    # print(loss_list)
    assign_list = assignment_loss(loss_list, target, type='assign')

    return sum(assign_list).item()/len(assign_list)



def cvx_layer( init_w, mean_mat, cholesky_mat, radius_list, steps, target):
    z1 = init_w
   
    task_loss = []
    z_updated = []
    J = []
    
    for i in range(len(mean_mat)):
        
        z = z1[i]
        mean = mean_mat[i] #.view(-1)
        cholesky = cholesky_mat[i]
        radius = radius_list  #[i]
        cholesky_inv = torch.inverse(cholesky)
      
            
        Objective = lambda z,x, Psqrt, z1: z.T@mean + radius*torch.sqrt(radius)*cp.norm( cholesky_inv@z, 2)  if isinstance(z, cp.Variable) else z.T@mean + radius*torch.sqrt(radius)*torch.linalg.norm(cholesky_inv @ z)
        SumConstr =  lambda z,x, Psqrt, z1: cp.sum(z) - 1 if isinstance(z, cp.Variable) else z.sum() - 1
        LowBound  =  lambda z,x, Psqrt, z1: -z
        UpBound   = lambda z,x, Psqrt, z1: z - 1
        
        Obj = lambda z: z.T@mean + radius*torch.sqrt(radius)*cp.norm( cholesky_inv@z, 2) if isinstance(z, cp.Variable) else z.T@mean + radius*torch.sqrt(radius)*torch.linalg.norm(cholesky_inv @ z)
            
        equalities = [SumConstr]
        inequalities = [LowBound, UpBound]
        params = [mean, cholesky_inv, z]       
        
        res = minimize_constr(
            Obj, z,
            max_iter= steps,
            constr= [np.array([list(np.ones(len(mean))) ]),np.array([1]), np.array([1])],
            bounds= dict(lb= np.zeros(len(mean)), ub= np.ones(len(mean)), keep_feasible=True),
            disp=0 )
      
        z = [res.x]
        lam = [torch.tensor(res.v[1])] 
        nu = [torch.tensor(res.v[0])]
  
        
        def vec(z, lam, nu):
            return torch.cat([a.view(-1) for b in [z,lam,nu] for a in b])

        def mat(x):
            sz = [0] + list(accumulate([a.numel() for b in [z,lam,nu] for a in b]))
            val = [x[a:b] for a,b in zip(sz, sz[1:])]
            return ([val[i].view_as(z[i]) for i in range(len(z))],
                    [val[i+len(z)].view_as(lam[i]) for i in range(len(lam))],
                    [val[i+len(z)+len(lam)].view_as(nu[i]) for i in range(len(nu))])

        # computes the KKT residual
        def kkt(z, lam, nu, *params):
            g = [ineq(*z, *params) for ineq in inequalities]
            dnu = [eq(*z, *params) for eq in equalities]
            L = (Objective(*z, *params) + 
                 sum((u*v).sum() for u,v in zip(lam,g)) + sum((u*v).sum() for u,v in zip(nu,dnu)))
            dz = autograd.grad(L, z, create_graph=True)
            dlam = [lam[i]*g[i] for i in range(len(lam))]
            return dz, dlam, dnu
        
        
        # compute residuals and re-engage autograd tape
        y = vec(z, lam, nu)
        y = y - vec(*kkt([z_.clone().detach().requires_grad_() for z_ in z], lam, nu, *params))
        # compute jacobian and backward hook
        J.append(autograd.functional.jacobian(lambda x: vec(*kkt(*mat(x), *params)), y))
        y.register_hook(lambda grad,b=i : torch.linalg.solve( J[b].transpose(0,1), grad[:,None])[:,0])
        
        z_updated.append([torch.clamp(mat(y)[0][0], min = 0, max = 1)])
        loss = res.fun
        task_loss.append(loss)
    z_updated = [torch.stack(o, dim=0) for o in zip(*z_updated)]
    
 
    return z_updated[0], task_loss, res.success

def cvx_layer_onestep( init_w, alpha, S, steps, target):
    z1 = init_w
      
    def obj(z1):
        z_w = z1  # /torch.sum(z1)
        obj = - ( mean @ z_w - torch.norm(cov @ z_w, p=2))
        return obj
    task_loss = []
    z_updated = []

    for i in range(len(alpha)):
        z = z1[i]
        mean = alpha[i].view(-1)
        cov = S[i]
        res = minimize_constr(
            obj, z,
            max_iter=steps,
            constr=dict( fun=lambda x: x.sum().square(), lb=1, ub=1, keep_feasible=True),
            bounds= dict(lb=0, ub=1, keep_feasible=True),
            disp=0 )
        
        output = res.x  # /torch.sum(res.x)
        # slack_var = res.v 
        # z = [torch.tensor(output)]
        # lam = [torch.tensor(slack_var[1])]
        # nu = [torch.tensor(slack_var[0])]
        loss = res.fun
        output.retain_graph = True
        output.requires_grad = True
        loss.requires_grad = True
        z_updated.append(output)
        task_loss.append(res.fun)
    z_updated = [torch.stack(o, dim=0) for o in zip(*z_updated)]
    return z_updated, task_loss, res.success


def compute_nll_loss(epoch, steps, gamma,target, z_prev, model, x_batch, y_batch, mod=None):
    loss_nll = multivariateNLL(model, x_batch, y_batch, mod)
    return loss_nll

def compute_coverage_loss(epoch, steps, gamma,target, z_prev, model, x_batch, y_batch,cgd_train_batch, mod=None):
    loss, coverage, [] = dist_loss(model, x_batch, y_batch,cgd_train_batch, mod,gamma, target )
    return loss, coverage

def compute_e2e_loss_os(z_prev, model, x_batch, y_batch, mod, epoch, steps, gamma, target):
    mean, cholesky_mat, radius, _ = model(x_batch)
    
    
   
    w_tmp, task_loss, status = cvx_layer( z_prev, mean, cholesky_mat, radius, steps, target)
  
   
    prod = w_tmp.mul(y_batch)
    port_values = torch.sum(prod, dim=1)

    # print(torch.mean(port_values))
    method = 'cvar'

    if method == 'cvar':
       
        
        quant = int(target*len(port_values)) + 1
        port_sorted = torch.sort(port_values, descending=True)[0]
        quant = port_sorted[quant]

        port_le_quant = port_values.le(quant).float()
        port_le_quant.requires_grad = True
        cvar_loss =  port_values.mul(port_le_quant).sum() / port_le_quant.sum()
        loss = -cvar_loss
        
    elif method == 'mean':
        loss = - torch.mean(port_values)
        
    elif method == 'exponential':
        param = 1
        port_inter = (1 - torch.exp(-param*port_values))*(1/param)
        loss = torch.mean(port_values)
        
    final_loss = loss
   
    
    coverage = compute_coverage_loss_for_other_models(model, x_batch, y_batch, mod,gamma, target)

    return final_loss, _, coverage, cvar_loss, w_tmp, _

def compute_e2e_nll(z_prev, model, x_batch, y_batch, mod, epoch, steps, gamma, target):
    b, A, radius, _ = model(x_batch)
    w_tmp, task_loss, status = cvx_layer( z_prev, b, A, steps, target)

    prod = w_tmp.mul(y_batch)
    port_values = torch.sum(prod, dim=1)
    method = 'cvar'

    if method == 'cvar':
       
        
        quant = int(target*len(port_values)) + 1
        port_sorted = torch.sort(port_values, descending=True)[0]
        quant = port_sorted[quant]
        port_le_quant = port_values.le(quant).float()
        port_le_quant.requires_grad = True
        loss =  port_values.mul(port_le_quant).sum() / port_le_quant.sum()
       
        
    elif method == 'mean':
        loss = - torch.mean(port_values)
        # loss.requires_grad=True
    elif method == 'exponential':
        param = 1
        port_inter = (1 - torch.exp(-param*port_values))*(1/param)
        loss = torch.mean(port_values)
        # loss.requires_grad=True
        
    pred_loss = compute_nll_loss(
        epoch, steps, gamma,target, z_prev, model, x_batch, y_batch, mod)
    to_save = []
    
    final_loss = (1 - gamma) * loss + gamma*pred_loss
    
    coverage = compute_coverage_loss_for_other_models(model, x_batch, y_batch, mod,gamma, target)
    
    return final_loss, pred_loss,coverage, loss, w_tmp, to_save

def compute_nll_opt(z_prev, model, x_batch, y_batch, mod, epoch, steps, gamma, target):
    y = y_batch
    pred_loss = compute_nll_loss(
        epoch, steps, gamma,target, z_prev, model, x_batch, y_batch, mod)
    to_save = []
    w_tmp = []
    
    final_loss =  pred_loss
    loss = torch.tensor([0])
   
    radius = scipy.stats.chi2.ppf(target, df= x_batch.shape[1])  # np.sqrt
    
    mean, cho = model(x_batch)
    cov_inv = torch.matmul(torch.linalg.inv(torch.transpose(cho, 1, 2)), torch.linalg.inv(cho))

    def compute_distance(y_train, mean, cov_inv, radius):
        def dist(u, v, c):
            diff = u - v
            m = torch.matmul(torch.matmul(diff.T, c), diff)
            return torch.sqrt(m)
        dist_list = torch.empty(len(y_train))
        for i in range(len(y_train)):
            x = y_train[i]
            m = mean[i]
            c = cov_inv[i]  #*(radius**2)
            r = radius  #[i]
            dis = dist(x,m,c)
            dis_div_ratio = dis/ r
   
           
            dist_list[i] = dis_div_ratio
           
        return dist_list
  
    def assignment_loss(loss_list, sig_level, type):
        if type == 'assign':
            assign = torch.Tensor([1 if i <= 1 else 0 for i in loss_list])
            assign = assign.view(y.shape[0], 1)
       
        return assign

    loss_list = compute_distance(y, mean, cov_inv, radius)
     # print(loss_list)
    assign_list = assignment_loss(loss_list, target, type='assign')
    
    coverage = sum(assign_list).item()/len(assign_list)

    return final_loss, pred_loss,coverage, loss, w_tmp, to_save

def compute_e2e_coverage(z_prev, model, x_batch, y_batch,cgd_train_batch,x_val, y_val, mod, epoch, steps, gamma, target):
    
    b, A, radius, _ = model(x_batch)
    
         
    w_tmp, task_loss, status = cvx_layer( z_prev, b, A, radius, steps, target)
    
    # w_tmp = torch.stack(w_tmp)
    # w_tmp.retain_grad()
    prod = w_tmp.mul(y_batch)
    port_values = torch.sum(prod, dim=1)
    cov_prev = 0
    pred_loss_prev = torch.tensor([0])
    method = 'cvar'
    # print(port_values)
    if method == 'cvar':
      
        
        quant = int(target*len(port_values)) + 1
        port_sorted = torch.sort(port_values, descending=True)[0]
        quant = port_sorted[quant]
        
        port_le_quant = port_values.le(quant).float()
        
        port_le_quant.requires_grad = True
        cvar_loss =  torch.div(port_values.mul(port_le_quant).sum() , port_le_quant.sum())
        loss = -cvar_loss
        # loss.requires_grad=True
        
    elif method == 'mean':
        loss = - torch.mean(port_values)
        # loss.requires_grad=True
    elif method == 'exponential':
        param = 1
        port_inter = (1 - torch.exp(-param*port_values))*(1/param)
        loss = torch.mean(port_values)
        # loss.requires_grad=True
    
    pred_loss,coverage = compute_coverage_loss(
    epoch, steps, gamma,target, z_prev, model, x_batch, y_batch, cgd_train_batch, mod)
    to_save = []
    
  
    
    final_loss = (1 - gamma) * loss + gamma*pred_loss
    
    return final_loss, pred_loss, coverage, cvar_loss, w_tmp, to_save

def CVaR_optimization(mu, Y, target):
    # Defining initial inputs
    # mu = Y.mean().to_numpy().reshape(1,-1)
    mu_np = mu.detach().numpy()
    returns = Y #.detach().numpy()
    rearranged_returns = np.swapaxes(returns, 0, 1)
    
    
    alpha = target
    m = Y.shape[1]
    n = Y.shape[0]
    k = Y.shape[2]
    
   
    w = cp.Variable((n,k))
    t = cp.Variable((1, 1))
    s = cp.Variable((n,m))
   
    #

    returns_mult = cp.hstack([cp.sum(cp.multiply(w, rearranged_returns[r]), axis = 1) for r in range(m) ])
    returns_stacked = cp.reshape( returns_mult, (returns_mult.shape[0],1) )
    s_reshaped = cp.reshape(s,(returns_mult.shape[0], 1))
    
  

    constraints = [s >=0, w >=0, cp.sum(w, axis = 1) == 1, s_reshaped + returns_stacked + t >= 0]
   
    # Defining risk objective
    risk = t + (1/(m*n*(1 - alpha))) * cp.sum(s)
    objective = cp.Minimize(risk)

    # Solving problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
   
       
    
    def cvar(port_values):
        quant = int(target*len(port_values)) + 1
        # Sort port_values in descending order
        port_sorted = np.sort(port_values)[::-1]
                                
        # Find the quantile
        quant = port_sorted[quant]
        
        # Find the elements less than or equal to quant and convert to float
        port_le_quant = (port_values <= quant).astype(float)
        
        port_cvar = np.sum(port_values * port_le_quant) / np.sum(port_le_quant)
        return port_cvar
    
    optimal_port = w.value   
    port_values = np.sum(np.multiply(optimal_port, mu_np), axis = 1)
    
    port_sampled = np.sum(np.multiply(optimal_port, rearranged_returns[0]), axis = 1)
    
    
    optimal_port_repeat = np.repeat(w.value,m,axis = 0)
    returns_new = returns.reshape((m*n,k))
   
    port_values_all_samples = np.sum(np.multiply(optimal_port_repeat, returns_new), axis = 1)
    
    cvar_oos = cvar(port_values)
    cvar_sampled = cvar(port_sampled)
    cvar_all = cvar(port_values_all_samples)
    
    print(cvar_oos, cvar_sampled, cvar_all, prob.objective.value )
    
    return cvar_oos, prob.objective.value


def calculate_cvar_radius(mean, covariance_matrix, alpha, p):
    # Calculate portfolio standard deviation
    portfolio_std_dev = np.sqrt(np.dot(np.dot(mean, covariance_matrix), mean))

    # Calculate VaR at the specified confidence level
    var_alpha = np.dot(mean, mean) + norm.ppf(alpha) * portfolio_std_dev

    # Calculate CVaR
    cvar = -norm.ppf(p) * var_alpha

    # Calculate the radius of the CVaR uncertainty set
    cvar_radius = np.abs(norm.ppf(p)) * portfolio_std_dev

    return cvar_radius

def mahalanobis_distance_squared(x, mean, covariance):
    """
    Calculate the squared Mahalanobis distance.
    """
    diff = x - mean
    precision_matrix = np.linalg.inv(covariance)
    return np.dot(diff, np.dot(precision_matrix, diff))

def ellipsoid_overlap(mean1, cov1, radius1, mean2, cov2, radius2, num_samples=100000):
    """
    Calculate the Jaccard Index for the overlap of two ellipsoids.
    """
    # Generate random samples from multivariate normal distributions
    samples1 = np.random.multivariate_normal(mean1, cov1, num_samples)
    samples2 = np.random.multivariate_normal(mean2, cov2, num_samples)

    # Check if each sample lies within the ellipsoids
    inside_ellipsoid1 = np.array([mahalanobis_distance_squared(sample, mean1, cov1) <= chi2.ppf(0.99, len(mean1)) * radius1**2 for sample in samples1])
    inside_ellipsoid2 = np.array([mahalanobis_distance_squared(sample, mean2, cov2) <= chi2.ppf(0.99, len(mean2)) * radius2**2 for sample in samples2])

    # Calculate the Jaccard Index
    intersection = np.sum(inside_ellipsoid1 & inside_ellipsoid2)
    union = np.sum(inside_ellipsoid1 | inside_ellipsoid2)
    jaccard_index = intersection / union

    return jaccard_index, samples1, samples2

def plot_ellipsoid(mean, covariance, radius, ax, color):
    """
    Plot the ellipsoid.
    """
    num_points = 100000
    phi = np.linspace(0, 2 * np.pi, num_points)
    x = mean[0] + radius * np.cos(phi)
    y = mean[1] + radius * np.sin(phi)

    ax.plot(x, y, color)



def get_train_loss_and_w(algorithm, epoch, steps, gamma,target, z_prev, model, optimizer, x_batch, y_batch,cgd_train_batch,x_val, y_val, mod):
    
    if algorithm == 'e2e_k_step':
        final_loss, pred_loss, coverage,loss, w_tmp, to_save = compute_e2e_loss_os(
         z_prev, model, x_batch, y_batch, mod, epoch, steps, gamma, target)
    elif algorithm == 'e2e_nll':
        final_loss, pred_loss,coverage, loss, w_tmp, to_save = compute_e2e_nll(
         z_prev, model, x_batch, y_batch, mod, epoch, steps, gamma, target)
    elif algorithm == 'e2e_coverage':
        final_loss, pred_loss,coverage, loss, w_tmp, to_save = compute_e2e_coverage(
         z_prev, model, x_batch, y_batch,cgd_train_batch,x_val, y_val, mod, epoch, steps, gamma, target)        
    elif algorithm == 'nll+opt':
        final_loss, pred_loss,coverage, loss, w_tmp, to_save = compute_nll_opt(
         z_prev, model, x_batch, y_batch, mod, epoch, steps, gamma, target)


    return final_loss, pred_loss, coverage,loss, w_tmp, to_save


def test(model, x_test, y_test, steps, target):
    alpha, S, radius, _ = model(x_test)
    
    z_prev = torch.abs(torch.randn_like(y_test))
    z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
    
    w_tmp, _,_ = cvx_layer(z_prev, alpha, S, steps, target)
    
    
    prod = w_tmp.mul(y_test)
    port_values = torch.sum(prod, dim=1)
    
    quant = int(target*len(port_values)) + 1
    port_sorted = torch.sort(port_values, descending=True)[0]
    quant = port_sorted[quant]
    port_le_quant = port_values.le(quant).float()
    port_le_quant.requires_grad = True
    port_cvar =  port_values.mul(port_le_quant).sum() / port_le_quant.sum()
    
    return port_cvar.item(), port_values



def test_and_save(model, x_test, y_test, steps, target, pickle_filename):
    # Existing functionality
    if pickle_filename == 'nll+opt':
        alpha, S = model(x_test)
        radius =  torch.tensor(scipy.stats.chi2.ppf(target, df= x_test.shape[1]))
        
    else:
        alpha, S, radius, _ = model(x_test)
        
  
    
    z_prev = torch.abs(torch.randn_like(y_test))
    z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
    
    w_tmp, _, _ = cvx_layer(z_prev, alpha, S, radius, steps, target)
    
    prod = w_tmp.mul(y_test)
    port_values = torch.sum(prod, dim=1)
    
    quant = int(target*len(port_values)) + 1
    port_sorted = torch.sort(port_values, descending=True)[0]
    quant = port_sorted[quant]
    port_le_quant = port_values.le(quant).float()
    port_le_quant.requires_grad = True
    port_cvar =  port_values.mul(port_le_quant).sum() / port_le_quant.sum()

    # Create a dictionary to store PyTorch arrays
    torch_data = {
        'y_test': y_test,
        'alpha': alpha,
        'S': S,
        'radius': radius
    }
    print("cvar", "var")
    print(port_cvar.item(), quant)
   
    return port_cvar.item(), port_values, torch_data

def test_and_save_conditional_samples(model, x_test, y_test, cgd_test, steps, target, gamma, pickle_filename):
    # Existing functionality
    if pickle_filename == 'nll+opt':
        alpha, S = model(x_test)
        radius =  torch.tensor(scipy.stats.chi2.ppf(target, df= x_test.shape[1]))
        
    else:
        alpha, S, radius, _ = model(x_test)
        
        
    z_prev = torch.abs(torch.randn_like(y_test))
    z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
    
    w_tmp, _, _ = cvx_layer(z_prev, alpha, S, radius, steps, target)
    
    
    
    
    def cvar(port_values):
        quant = int(target*len(port_values)) + 1
        # Sort port_values in descending order
        port_sorted = np.sort(port_values)[::-1]
                                
        # Find the quantile
        quant = port_sorted[quant]
        
        # Find the elements less than or equal to quant and convert to float
        port_le_quant = (port_values <= quant).astype(float)
        
        port_cvar = np.sum(port_values * port_le_quant) / np.sum(port_le_quant)
        return port_cvar
    
    optimal_port = w_tmp.detach().numpy()
    port_values = np.sum(np.multiply(optimal_port, y_test.detach().numpy()), axis = 1)
    
    rearranged_returns = np.swapaxes(cgd_test, 0, 1)
    port_sampled = np.sum(np.multiply(optimal_port, rearranged_returns[0]), axis = 1)
    
    m = rearranged_returns.shape[0]
    n = rearranged_returns.shape[1]
    k = rearranged_returns.shape[2]
    
    optimal_port_repeat = np.repeat(optimal_port,m,axis = 0)
    returns_new = cgd_test.reshape((m*n,k))   
    port_values_all_samples = np.sum(np.multiply(optimal_port_repeat, returns_new), axis = 1)
    
    cvar_oos = cvar(port_values)
    cvar_sampled = cvar(port_sampled)
    cvar_all = cvar(port_values_all_samples)
    
  
    
    with open(f"output.txt", "a") as file:
        # Write the values into the file
        file.write(f"{pickle_filename},{cvar_oos},{cvar_sampled}, {cvar_all}\n")
        
  
    
    
 


def ellipsoid_comparision(model, x_test, y_test, steps, target, pickle_filename):
    
    # Existing functionality
    if pickle_filename == 'nll+opt':
        mean, cho = model(x_test)
        radius =  torch.tensor(scipy.stats.chi2.ppf(target, df= x_test.shape[1]))
        
    else:
        mean, cho, radius, _ = model(x_test)
        
    cov = torch.matmul(cho, torch.transpose(cho, 1, 2))
    
    print("here")
    print(y_test[:2])
    print(mean[:2])
    print(cov[:2])
    import sys
    sys.exit()
    
    to_save  = {
    'data': y_test,
    'mean': mean,
    'cov': cov,
    'radius': radius}
    
    import os
    path = './ellipsoid_data'
                                    
    # Create the folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Define the pickle file path
    pickle_file_path = os.path.join(path, str(pickle_filename)+".pkl")
    
       
    # Save the dictionary to a pickle file
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(to_save, pickle_file)
        
  
    
    


def test_nll_pred_opt(model, x_test, y_test, steps, target):
    alpha, S, radius, _ = model(x_test)
    
    z_prev = torch.abs(torch.randn_like(y_test))
    z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
    
    w_tmp, _,_ = cvx_layer(z_prev, alpha, S, steps, target)
    
    
    prod = w_tmp.mul(y_test)
    port_values = torch.sum(prod, dim=1)
    
    quant = int(target*len(port_values)) + 1
    port_sorted = torch.sort(port_values, descending=True)[0]
    quant = port_sorted[quant]
    port_le_quant = port_values.le(quant).float()
    port_le_quant.requires_grad = True
    port_cvar =  port_values.mul(port_le_quant).sum() / port_le_quant.sum()
    
    return port_cvar.item(), port_values

def optimization_conformal(mean, cholesky, radius,  y_test, cgd_test, target):
    
    n = mean.shape[1]
    z = cp.Variable(n)
    x = cp.Parameter(n)
    Psqrt = cp.Parameter((n,n))
    # rad = cp.Parameter(1)
    
    z_prev = torch.abs(torch.randn_like(y_test))
    z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
    steps = 1000
    radius = torch.tensor(radius)
    w_tmp, _, _ = cvx_layer(z_prev, mean, cholesky, radius, steps, target)
    
  

    prod = w_tmp.mul(y_test)
    port_values = torch.sum(prod, dim=1)
    
    quant = int(target*len(port_values)) + 1
    port_sorted = torch.sort(port_values, descending=True)[0]
    quant = port_sorted[quant]
    port_le_quant = port_values.le(quant).float()
    port_le_quant.requires_grad = True
    port_cvar =  port_values.mul(port_le_quant).sum() / port_le_quant.sum()
    
    print("cvar","var")
    print(port_cvar, quant)
    
    return port_cvar, port_values


def optimization_conformal_conditional_samples(mean, cholesky, radius,  y_test, cgd_test, target):
    
    n = mean.shape[1]
    z = cp.Variable(n)
    x = cp.Parameter(n)
    Psqrt = cp.Parameter((n,n))
    # rad = cp.Parameter(1)
    
    z_prev = torch.abs(torch.randn_like(y_test))
    z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
    steps = 1000
    radius = torch.tensor(radius)
    w_tmp, _, _ = cvx_layer(z_prev, mean, cholesky, radius, steps, target)
    
    def cvar(port_values):
        quant = int(target*len(port_values)) + 1
        # Sort port_values in descending order
        port_sorted = np.sort(port_values)[::-1]
                                
        # Find the quantile
        quant = port_sorted[quant]
        
        # Find the elements less than or equal to quant and convert to float
        port_le_quant = (port_values <= quant).astype(float)
        
        port_cvar = np.sum(port_values * port_le_quant) / np.sum(port_le_quant)
        return port_cvar
    
    optimal_port = w_tmp.detach().numpy()
    port_values = np.sum(np.multiply(optimal_port, y_test.detach().numpy()), axis = 1)
    
    rearranged_returns = np.swapaxes(cgd_test, 0, 1)
    port_sampled = np.sum(np.multiply(optimal_port, rearranged_returns[0]), axis = 1)
    
    m = rearranged_returns.shape[0]
    n = rearranged_returns.shape[1]
    k = rearranged_returns.shape[2]
    
    optimal_port_repeat = np.repeat(optimal_port,m,axis = 0)
    returns_new = cgd_test.reshape((m*n,k))   
    port_values_all_samples = np.sum(np.multiply(optimal_port_repeat, returns_new), axis = 1)
    
    cvar_oos = cvar(port_values)
    cvar_sampled = cvar(port_sampled)
    cvar_all = cvar(port_values_all_samples)
    
    
    with open(f"output.txt", "a") as file:
        # Write the values into the file
        file.write(f"{cvar_oos},{cvar_sampled}, {cvar_all}\n")
    
  



def get_coverage(alg, model, x, y, mod, target, factor = 1):
  
    y_scaled = y #model.y_scaler(y)
    if alg == 'nll+opt':
        mean, cho = model(x)
        radius =  scipy.stats.chi2.ppf(target, df= x.shape[1])
    
    else:
        mean, cho, radius, y_hat = model(x)
    
   
    radius =  radius
    cov_inv = torch.matmul(torch.linalg.inv(torch.transpose(cho, 1, 2)), torch.linalg.inv(cho))
    # cov = torch.matmul(cho, torch.transpose(cho, 1, 2))
       
    def compute_distance(y_train, mean, cov_inv, radius):
        def dist(u, v, c):
            diff = u - v
            m = torch.matmul(torch.matmul(diff.T, c), diff)
            return torch.sqrt(m)
        dist_list = torch.empty(len(y_train))
        for i in range(len(y_train)):
            x = y_train[i]
            m = mean[i]
            c = cov_inv[i]  #*(radius**2)
            r = radius  #[i]
            dis = dist(x,m,c)
            # dis_div_ratio = dis/ r
            
            # temp = (1/(radius*radius))*torch.matmul(c[i].T, x - m)
            # dis = torch.linalg.norm(temp,  ord=2)
            if alg == 'nll+opt':
                dist_list[i] = dis/r
            else:
                dist_list[i] = dis   #dis_div_ratio
           
        return dist_list
    
    
    def assignment_loss(loss_list, sig_level, type):
        if type == 'assign':
            assign = torch.Tensor([1 if i <= 1 else 0 for i in loss_list])
            assign = assign.view(y.shape[0], 1)
        return assign
 
    loss_list = compute_distance(y_scaled, mean, cov_inv, radius)
    # print(loss_list[:5])
    assign_list = assignment_loss(loss_list, target, type='assign')
    coverage_perc = 100*(sum(assign_list)/len(assign_list))
    
    return coverage_perc.item(), assign_list



def get_conditional_coverage(alg, model, x, y, mod, target, conditional_data, factor = 1):
    y_scaled = y #model.y_scaler(y)
    if alg == 'nll+opt':
        mean1, cho = model(x)
        radius =  scipy.stats.chi2.ppf(target, df= x.shape[1])
    
    else:
        mean1, cho, radius, y_hat = model(x)
        radius = radius.detach().numpy()

    conditional_mu = np.array(mean1.detach().numpy())
   
    radius =  radius
    cov_inv = torch.matmul(torch.linalg.inv(torch.transpose(cho, 1, 2)), torch.linalg.inv(cho))
   
    def dist(u, v, c):
            diff = u - v
            m = torch.matmul(torch.matmul(diff.T, c), diff)
            return torch.sqrt(m)

    result = []
    for i in range(len(conditional_data)):
        m = mean1[i]
        c = cov_inv[i]*(1/factor)

        num_sim = 100
        dependent_data = conditional_data[i]
        # dist_list = torch.empty(len(dependent_data))
        

        mean = m.detach().numpy()
        cov = c  #.detach().numpy()
        rad = radius
        
        
      
        
        dist_list = []
        for j in range(len(dependent_data) ):
            
            x = torch.tensor(dependent_data[j])
           
            
            dis = dist(x,mean,cov)
           
            dist_list.append(dis.item())
            
        if alg == 'nll+opt':
            assign = torch.Tensor([1 if i <= rad else 0 for i in dist_list])
        else:
            assign = torch.Tensor([1 if i <= 1 else 0 for i in dist_list])

        
        coverage = 100*(sum(assign)/len(assign))
        
       
        
        result.append([ coverage.item()])

    result_np = np.array(result)

    # np.savetxt("e2e_coverage.csv", result_np, delimiter=",")
    return np.round([np.mean(result),np.std(result_np), np.min(result), np.max(result), np.quantile(result, 0.1),  np.quantile(result, 0.9)], 2), np.sort(np.array(result).T)[0]

  

def predict_then_optimize(alg, model, x, y, steps, target, mod, cgd_test):
    
    overall_cov, assign_list = get_coverage(alg, model, x, y, mod, target, factor = 1)
   
        
    if cgd_test is None:
        test_coverage, test_cov_list = None
    else:
        test_coverage, test_cov_list  = get_conditional_coverage(alg, model, x, y, mod, target, cgd_test, factor = 1)
    
    # test_cvar, test_returns = test(model, x, y,steps, target)
    
    return overall_cov, assign_list, test_coverage, test_cov_list
            
    
    
    
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_distribution_cov_frequency(freq_list, title_list):
 
    plt.figure(figsize=(10, 6))
    
    colors = ['green', 'lime', 'red', 'pink', 'skyblue']
    alg_list =  ['conditional_conformal', 'conformal', 'e2e_coverage', 'e2e_k_step', 'nll+opt']
    
    colors = ['green', 'red', 'blue']
    alg_list =  ['conditional_conformal',  'e2e_coverage', 'e2e_k_step']
    # for i, cov_freq_dist in enumerate(freq_list):
    #     print()
    i = 0
    for key in alg_list:
        
        cov_freq_dist = freq_list[key]
        # cov_freq_dist = np.array(cov_freq_dist)
        title = key
        
        # Compute mean and bounds
        if np.sum(cov_freq_dist) == 0.0:
           
            continue
        else:           
          
            mean_cumulative_freq = np.mean(cov_freq_dist, axis=0)
            x = mean_cumulative_freq/100
            y = np.arange(len(mean_cumulative_freq)) / len(mean_cumulative_freq)
           
            data_transposed = np.transpose(cov_freq_dist)
            
            percentiles_5 = np.percentile(data_transposed, q=10, axis=1)/100
            percentiles_95 = np.percentile(data_transposed, q=90, axis=1)/100          
                    
            # if key in ['conformal', 'nll+opt']:
            #     continue
            # else:
           
            plt.plot( x,y, label=title, color = colors[i], linewidth = 2)
            
            # # # # Plot shaded region representing bounds
            plt.fill_betweenx(y, x, percentiles_5, color=colors[i], alpha=0.3)
            # # # # Plot shaded region representing bounds
            plt.fill_betweenx(y, x, percentiles_95, color=colors[i], alpha=0.3)
            
            
        print(key, i)
        i = i+1
        
        
    # Set the x-axis formatter to display values as percentages
    percent_formatter = mticker.PercentFormatter(xmax=1.0, decimals=0)
    plt.gca().xaxis.set_major_formatter(percent_formatter)

    # plt.title("Distribution of Coverage Frequencies")
    plt.axvline(x=0.9, linestyle = '--', color = 'black', linewidth = 2)
    plt.xlabel("Coverage Frequency", fontsize = 20)
    plt.ylabel("Cumulative Probability", fontsize = 20)
    plt.grid(None)
    # plt.legend(loc=2, prop={'size': 10})
    
    
    custom_ticks = [ 0.,   0.2,  0.4,  0.6,  0.8, 0.9, 1. ]
    plt.xticks(custom_ticks)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    # Add white background and axis
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    ax.tick_params(colors='black')
    ax.grid(None)
    plt.tight_layout()
    plt.savefig(r'D:\projects\spdlayers\E2E-CRO\models\outputs\final_plots\cumdistplot.pdf')
    # plt.show()




def train_and_save_best(model, optimizer, mod, x_train, x_val, y_train, y_val, cgd_train, n_epochs, steps, gamma, target, algorithm, batch_size, scheduler=None, print_every=10, save_path="best_model.pth"):

    best_model = None
    best_cvar_loss = float("inf")
    best_epoch = None
    
    
    train_losses, val_losses, pred_losses, port_losses = [], [], [], []
    z_prev = torch.abs(torch.randn_like(y_train))
    z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
    radius_prev = 0
    total_coverage = 0
    
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):
       
        batch_indices = sample_batch_indices(x_train, y_train, batch_size)
        
        batch_losses_t, port_losses_t, b_coverage_t, pred_losses_t = [], [], [], []
        to_save_tot = []

        for batch_ix in batch_indices:
                        
            z_prev_subset = z_prev[batch_ix]
            model.train()
            
          
            
            optimizer.zero_grad()
            
            # print("done printing params")
            
            if cgd_train is None:
                b_train_loss, b_pred_loss, b_coverage, b_port_loss, w_tmp, to_save = get_train_loss_and_w(
                algorithm, epoch, steps, gamma, target, z_prev_subset, model, optimizer, x_train[batch_ix], y_train[batch_ix], None,x_val, y_val, mod)
            
            else:
                b_train_loss, b_pred_loss, b_coverage, b_port_loss, w_tmp, to_save = get_train_loss_and_w(
                algorithm, epoch, steps, gamma, target, z_prev_subset, model, optimizer, x_train[batch_ix], y_train[batch_ix], cgd_train[batch_ix],x_val, y_val, mod)

            for i in range(len(w_tmp)):
                z_prev[batch_ix[i]] = w_tmp[i]
            # model.radius.retain_grad()
            
            if algorithm != 'nll+opt':
                radius_prev = model.radius.item()
                
          
  

            to_save_tot.extend(to_save)
            batch_losses_t.append(b_train_loss)
            b_coverage_t.append(b_coverage)
            pred_losses_t.append(b_pred_loss.detach().numpy())
            port_losses_t.append(b_port_loss.detach().numpy())

        if scheduler is not None:
            scheduler.step()
  
      

        train_loss_fin = torch.stack(batch_losses_t, dim=0).sum(dim=0).sum(dim=0)
        pred_loss_fin = np.mean([np.mean(data) for data in pred_losses_t])
        port_loss_fin = np.mean(port_losses_t)
        train_losses.append(train_loss_fin.detach().numpy())
        pred_losses.append(pred_loss_fin)
        port_losses.append(port_loss_fin)
        total_coverage = np.mean(b_coverage_t)
        
        train_loss_fin.backward(retain_graph=True)
        optimizer.step()
        
        avg_loss = 0
        avg_perc = 0
        port_perc = 0
        pred_perc = 0

        avg_length = 10
        if len(train_losses) > avg_length:
            avg_loss = sum(train_losses[-avg_length:]) / avg_length
            avg_perc = 100 * (train_loss_fin - avg_loss) / avg_loss

            avg_port = sum(port_losses[-avg_length:]) / avg_length
            port_perc = (port_loss_fin - avg_port) / abs(avg_port)

            avg_pred = sum(pred_losses[-avg_length:]) / avg_length
            pred_perc = 100 * (pred_loss_fin - avg_pred) / avg_pred

          
            
        if epoch == 0 or (epoch + 1) % print_every == 0:
            print(
                f'Epoch {epoch+1} | Train = {train_loss_fin:.2f}, {abs(avg_perc):.2f}  | Port = {port_loss_fin:.2f}, {abs(port_perc):.2f}  | Pred = {pred_loss_fin:.2f}, {abs(pred_perc):.2f} | coverage = {total_coverage:.2f} | Radius = {radius_prev:.2f}')
        
        dir_id = str(datetime.datetime.now().strftime('%Y%m%d'))
        # Specify the folder to save the models
        save_folder = str(algorithm)+"_saved_models/"+str(dir_id)
        os.makedirs(save_folder, exist_ok=True)
        print(epoch, n_epochs)
        
        if algorithm == 'e2e_coverage':
        
            if (total_coverage >= target) or (epoch == n_epochs-1):
                print("Entered model saving")
                # Save the model
                model_path = os.path.join(save_folder, f"model_epoch_{epoch}.pt")
                torch.save(model, model_path)
        
                # Save epoch number and loss information to CSV
                csv_path = os.path.join(save_folder, "training_info.csv")
           
                header = ["epoch", "total_loss", "port_loss", "coverage_loss"]
                
                # Check if the CSV file already exists
                write_header = not os.path.exists(csv_path)
                import csv
                with open(csv_path, mode="a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    if write_header:
                        writer.writerow(header)
                    
                    # Write epoch number and loss to CSV
                    writer.writerow([epoch, train_loss_fin.item(), port_loss_fin.item(), pred_loss_fin.item() ])
            
                # Stop saving if the difference between current loss and threshold is less than 1%
               
                if (abs(train_loss_fin - avg_loss) / avg_loss < 0.01) or (epoch == n_epochs-1):
                     
                    print("Training complete.")
                    
            
                    # Post-training step: Load the best model based on the epoch with the lowest loss
                    csv_path = os.path.join(save_folder, "training_info.csv")
                    with open(csv_path, mode="r") as csv_file:
                        reader = csv.DictReader(csv_file)
                        best_epoch_info = min(reader, key=lambda x: float(x["total_loss"]))
                    
                    best_epoch = int(best_epoch_info["epoch"])
                   
                    print(best_epoch)
                    best_model_path = os.path.join(save_folder, f"model_epoch_{best_epoch}.pt").replace("\\","/")
                    best_model = torch.load(best_model_path)
                    
                    break
                
        elif (algorithm == 'nll+opt') or (algorithm == 'e2e_k_step'):
            
            # Stop saving if the difference between current loss and threshold is less than 1%
           
            if (abs(train_loss_fin - avg_loss) / avg_loss < 0.01) or (epoch == n_epochs-1):
                 
                print("Training complete.")
                
                # Save the model
                model_path = os.path.join(save_folder, f"model_epoch_{epoch}.pt")
                torch.save(model, model_path)
        
                best_model = model
                
                break
            
                
               
    return best_model, train_losses, best_cvar_loss, radius_prev

def uniqueid_generator(alg,exp_type, steps, gamma, target):
    dir_id = str(datetime.datetime.now().strftime('%Y%m%d'))
    if not os.path.exists(r"D:\projects\spdlayers\prod_run\results\simulated\%s" %(dir_id)):
        os.mkdir(r"D:\projects\spdlayers\prod_run\results\simulated\%s" %(dir_id))
    if not os.path.exists(r"D:\projects\spdlayers\prod_run\saved_models\simulated\%s" %(dir_id)):
        os.mkdir(r"D:\projects\spdlayers\prod_run\saved_models\simulated\%s" %(dir_id))
    # +str(datetime.datetime.now().strftime('%H%M%S'))
    file_id = str(exp_type)+"_"+str(alg)+"_"+str(steps)+"_"+str(int(gamma*100))+"_"+str(int(target*100))
    return dir_id, file_id


def saving_files(folder_path, file_name):
    # Define the file path
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the file exists
    if os.path.exists(file_path):
        print(f"File '{file_name}' already exists in '{folder_path}'")
    else:
        print(f"File '{file_name}' doesn't exist in '{folder_path}'. Creating...")

        header = ["date","time","algorithm","target","steps", "gamma","alpha", "phi", "test_conditional_cov", "test_cvar", "test_coverage"]
        # Write data to CSV file
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write header row
            # for index, value in enumerate(random_list):
            #     writer.writerow([index + 1, value])  # Write data row

        print(f"Results saved to '{file_path}'")
        
def write_list_to_csv(file_path, data):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    print(f"List written to '{file_path}' as a row in CSV")
        

    
def run():
    return None
    

if __name__ == '__main__': 
    run()
