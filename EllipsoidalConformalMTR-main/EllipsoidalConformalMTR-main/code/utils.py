# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 01:54:17 2022

@author: abhil
"""
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
# import minimize_constr as mc
from torch import nn
import torch.distributions as D
from torch.distributions.transforms import CorrCholeskyTransform
import time
import pickle
from sqwash import SuperquantileReducer, SuperquantileSmoothReducer

import sys
# sys.path.insert(1, r'D:\projects\spdlayers\pytorch-minimize\torchmin_')
from torchmin_.minimize_constr import minimize_constr

# from torchmin import minimize_constr
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


exp_type = 'simulated'
learning_rate = 1e-5*5000
momentum = 0.9
weight_decay = 1e-3
n_epochs = 2000
batch_size = 200
print_every = 1
step_list =  [3]  #[3, 10, 50]  # [2,10, 50]
gamma_list = [0.9]   #[0.9, 0.1, 0]  #gamma cant be one
target_list = [0.99]  #[0.99, 0.9, 0.8, 0.7]
# sim_data_params = [2, 1]
mod = 'fixed_target'  # ['find_target', 'fixed_target', 'no_reg']



def sim_data_gen_new_method(sim_data_params):

    var_centers = sim_data_params[0]
    side_centers = sim_data_params[1]

    mean_1 = [0, -var_centers, - side_centers, 0]
    cov_1 = [[1,   0.0, 0.9, 0.9],
             [0.0,   1, 0.9, 0.9],
             [0.9, 0.9, 1, 0.0],
             [0.9, 0.9, 0.0, 1]]

    mean_2 = [0, var_centers, side_centers, 0]
    cov_2 = [[1,   0.0, -0.9, -0.9],
             [0.0,   1, -0.9, -0.9],
             [-0.9, -0.9, 1, -0.0],
             [-0.9, -0.9, -0.0, 1]]

    x, y, x1, y1 = np.random.multivariate_normal(mean_1, cov_1, 50000).T
    a, b, a1, b1 = np.random.multivariate_normal(mean_2, cov_2, 50000).T

    data1 = np.array([x, y, x1, y1]).T
    data2 = np.array([a, b, a1, b1]).T

    data = np.concatenate((data1, data2), axis=0)
    data_pd = pd.DataFrame(data)
    data_final = data_pd.sample(frac=0.2)

    data_final.to_csv('D:/projects/spdlayers/prod_run/data/simulated/sim_data.txt',
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
    
    

def load_data(exp_type=None, sim_data_params = None):

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

        sim_data_gen_conformal(sim_data_params)

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
            train_ix[start:end].tolist()
        )

    return batch_indices


# class StandardScaler(object):
#     """
#     Standardize data by removing the mean and scaling to unit variance.
#     """

#     def __init__(self):
#         self.mean = None
#         self.scale = None

#     def fit(self, sample):
#         self.mean = sample.mean(0, keepdim=True)
#         self.scale = sample.std(0, unbiased=False, keepdim=True)
#         return self

#     def __call__(self, sample):
#         return self.transform(sample)

#     def transform(self, sample):
#         return (sample - self.mean) / self.scale

#     def inverse_transform(self, sample):
#         return sample * self.scale + self.mean
    
def train_test_scaler(train, val, test):
    
    scaler = StandardScaler()

    normalized_x_train = pd.DataFrame(
        scaler.fit_transform(train)
        # ,
        # columns = train.columns
    )
    
    
    normalized_x_val = pd.DataFrame(
        scaler.transform(val)
        # ,
        # columns = val.columns
    )
    
    normalized_x_test = pd.DataFrame(
        scaler.transform(test)
        # ,
        # columns = test.columns
    )
    
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
        return (
            torch.from_numpy(x_train),
            torch.from_numpy(x_test),
            torch.from_numpy(y_train),
            torch.from_numpy(y_test),
        )

    if exp_type == "simulated":

        x_train = np.array(aux1.iloc[:600, :])
        x_val   = np.array(aux1.iloc[600:1000, :])
        x_test  = np.array(aux1.iloc[1000:2000, :])

        y_train = np.array(df.iloc[:600, :])
        y_val   = np.array(df.iloc[600:1000,:])
        y_test  = np.array(df.iloc[1000:2000, :])

        print(x_train.shape,x_val.shape, x_test.shape, y_train.shape,y_val.shape, y_test.shape)
        return (
            torch.from_numpy(x_train),
            torch.from_numpy(x_val),
            torch.from_numpy(x_test),
            torch.from_numpy(y_train),
            torch.from_numpy(y_val),
            torch.from_numpy(y_test),
        )



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

       

        self.cho_elements_module = nn.Linear(
            n_inputs, n_hidden * (n_hidden - 1) // 2)
        self.cho_elements_module1 = nn.Linear(
            n_hidden * (n_hidden - 1) // 2, n_hidden * (n_hidden - 1) // 2)
       

        if mod == 'find_target':
            self.target = nn.Parameter(torch.randn(1), requires_grad=True)
            self.input_to_output = nn.Linear(n_inputs, 1)
        elif mod == 'fixed_target':
            # self.i_o_hidden1 = nn.Linear(n_inputs, 10, bias=True)
            # self.i_o_hidden2 = nn.Linear(10, 10, bias=True)
            # self.i_o_hidden3 = nn.Linear(10, n_inputs, bias=True)
            self.radius = nn.Parameter(torch.randn(1), requires_grad=True)
            self.input_to_output = nn.Linear(n_inputs, 1, bias=True)

        # self.dropout = torch.nn.Dropout()

    def forward(self, x):
        # Normalization

     
        # Shared layer
        shared_1 = self.shared(x)
        shared_2 = F.tanh(shared_1)      #leaky_relu(shared, negative_slope=0.01)
        # shared = self.dropout(shared)
        # Parametrization of the mean
        mean_hidden_1 = self.mean_hidden(shared_2)
        mean_hidden_2 = F.tanh(mean_hidden_1)     #leaky_relu(mean_hidden, negative_slope=0.01)

        mean = self.mean_linear(mean_hidden_2)

        # Parametrization fo the standard deviation
        # std_hidden = self.std_hidden(shared)
        # std_hidden = F.relu(std_hidden)
        # std_hidden = self.dropout(std_hidden)
        # std = F.softplus(self.std_linear(std_hidden)) + self.jitter

        cho_elements = self.cho_elements_module(x)
        cho_elements = F.leaky_relu(cho_elements, negative_slope=0.01)
        cho_elements1 = self.cho_elements_module1(cho_elements)
        cho_elements1 = F.leaky_relu(cho_elements1, negative_slope=0.01)
        # cho_elements2 = self.cho_elements_module2(cho_elements1)
        # cho_elements2 = F.leaky_relu(cho_elements2, negative_slope=0.01)
        cho = CorrCholeskyTransform()(cho_elements1)


        if mod == 'find_target':

            target = self.target
            x = self.input_to_output(x)
            return mean, cho, target, torch.sigmoid(x)
        elif mod == 'fixed_target':
            # x = self.i_o_hidden1(x)
            # x = F.leaky_relu(x, negative_slope=0.01)
            # x = self.i_o_hidden2(x)
            # x = F.leaky_relu(x, negative_slope=0.01)
            # x = self.i_o_hidden3(x)
            # x = F.leaky_relu(x, negative_slope=0.01)
           
         
            r = torch.sqrt(torch.abs(self.radius))
            temp = torch.eye(cho.shape[1])
            temp = temp.reshape((1, cho.shape[1], cho.shape[1]))
            y = r*temp.repeat(cho.shape[0], 1, 1)
            
            cho_scaled = torch.matmul(cho, y)
            
            x = self.input_to_output(x)
            #torch.clamp(radius, min = 1e-2)
            return mean, cho_scaled,r, torch.sigmoid(x)

        else:
            return mean, cho

def compute_loss(model, x, y, kl_reg=0.1):

    y_scaled = model.y_scaler(y)

    # mean, std = model(x)
    # y_hat = torch.distributions.Normal(mean, std)

    mean, cho = model(x)
    y_hat = D.MultivariateNormal(mean, scale_tril=cho)

    m = y_hat.sample()
    neg_log_likelihood = -y_hat.log_prob(m)

    # neg_log_likelihood = -y_hat.log_prob(y_scaled)

    return torch.mean(neg_log_likelihood)


def multivariateNLL(model, x, y, mod):

    if mod == 'find_target':
        mean, cho, target, y_hat = model(x)
    elif mod == 'fixed_target':
        mean, cho,  y_hat = model(x)
    else:
        mean, cho = model(x)

    # model.train()
    # optimizer.zero_grad()
    y_scaled = model.y_scaler(y)
    # mean, cho = model(x)
    cov_inv = torch.matmul(torch.linalg.inv(
        cho), torch.transpose(torch.linalg.inv(cho), 1, 2))
    cov = torch.matmul(torch.transpose(cho, 1, 2), cho)
    distance = y_scaled - mean

    temp_mat = torch.matmul(torch.matmul(torch.transpose(
        distance.view(500, 2, 1), 1, 2), cov_inv), distance.view(500, 2, 1))

    loss = 0.5*torch.log(torch.linalg.det(cov_inv)
                         ).view(500, 1, 1) - 0.5*temp_mat

    final_loss = -loss.sum()
    final_loss.retain_grad()

    # final_loss.backward()
    # optimizer.step()

    return final_loss


def dist_loss(model, x, y, mod, target, test_mode=None):
    y_scaled = y #model.y_scaler(y)

    if mod == 'find_target':
        mean, cho, target, y_hat = model(x)
    elif mod == 'fixed_target':
        mean, cho, radius, y_hat = model(x)
        radius =  torch.abs(radius)
    else:
        mean, cho = model(x)

    cov_inv = torch.matmul(torch.linalg.inv(
        cho), torch.transpose(torch.linalg.inv(cho), 1, 2))
    cov = torch.matmul(torch.transpose(cho, 1, 2), cho)
    
    theoritical_radius =  np.sqrt(scipy.stats.chi2.ppf(target, df= mean.shape[1]))
    
    print(radius.item(), theoritical_radius)
    

    # class LinearRegression(nn.Module):
    #   def __init__(self, input_dim: int, output_dim: int) -> None:
    #     super(LinearRegression, self).__init__()
    #     self.input_to_output = nn.Linear(input_dim, output_dim)
    #   def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.input_to_output(x)
    #     return torch.sigmoid(x)

    def compute_distance(y_train, mean, cov_inv):
        def dist(u, v, c):
            diff = u - v

            m = (1/radius*radius)*torch.matmul(torch.matmul(diff.T, c), diff)
            return torch.sqrt(m)
        dist_list = torch.empty(len(y_train))
        for i in range(len(y_train)):
            x = y_train[i]
            m = mean[i]
            c = cov_inv[i]
            # dis = dist(x,m,c)

            temp = (1/radius*radius)*torch.matmul(cho[i].T, x - m)
            dis = torch.linalg.norm(temp,  ord=2)

            dist_list[i] = dis

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

    loss_list = compute_distance(y_scaled, mean, cov_inv)
    assign_list = assignment_loss(loss_list, target, type='distance')
    
    if test_mode == 'on':
        print(loss_list.T)
        print(assign_list.T)
        
        sys.exit()
        output = torch.sum(assign_list) / len(assign_list)
        return output.item()

    
    # temp = assignment_loss(loss_list, target, type='adjusted')
    

    # print(assign_list[:10].T)
    # print(loss_list[:10].T)
    # print(temp[:10].T)
    # import sys
    # sys.exit()

    if mod == 'find_target':

        x_tr = pd.DataFrame(x.detach().numpy())
        x_tr.columns = ['x1', 'x2']
        # x_tr['intercept'] = 1
        x_tr['target'] = assign_list.detach().numpy()

        mod = smf.logit("target ~ x1 + x2", data=x_tr)
        res = mod.fit()
        # print(res.summary())

        net_out = torch.Tensor(res.predict(x_tr))

        out_target = (target - torch.Tensor(net_out))
        relu_out = F.relu(out_target)
        loss_final = torch.sum(torch.square(out_target))

        temp_list = assignment_loss(loss_list, target, type='assign')
        print(target, torch.sum(temp_list) / len(temp_list))

    elif mod == 'no_reg':

        loss_final = torch.sum(torch.square(1 - assign_list))
        temp_list = assignment_loss(loss_list, target, type='adjusted')

        print(torch.sum(temp_list) / len(temp_list))
        
    # elif mod == 'hyp_test':
    #     print(y_hat)
    #     print(assign_list)
    #     import sys
    #     sys.exit()
        
    #     params = list(model.input_to_output.parameters()) #+\
    #         # list(model.i_o_hidden1.parameters()) +\
    #         # list(model.i_o_hidden2.parameters()) +\
    #         # list(model.i_o_hidden3.parameters())

    #     optimizer_conf = torch.optim.Adam(
    #         params,
    #         lr=1e-2,
    #         weight_decay=weight_decay,
    #     )
        
    else:
        
       
        criterion = nn.BCELoss(reduction='sum')
        # criterion = nn.BCEWithLogitsLoss()
 
        params = list(model.input_to_output.parameters()) #+\
            # list(model.i_o_hidden1.parameters()) +\
            # list(model.i_o_hidden2.parameters()) +\
            # list(model.i_o_hidden3.parameters())

        optimizer_conf = torch.optim.Adam(
            params,
            lr=1e-2,
            weight_decay=weight_decay,
        )

        for epoch in range(10):
           
            model.train()
            optimizer_conf.zero_grad()
            loss_conf = criterion(y_hat, assign_list)

            loss_conf.backward(retain_graph=True)
            optimizer_conf.step()

        model.eval()
        _, _, _,y_hat = model(x)

        from torchmetrics import R2Score
        r2score = R2Score()
        r2 = r2score(y_hat, assign_list)
                
        loss_n = (r2*(assign_list.shape[0] - len(params)))/((1 - r2)*len(params))
        
        from scipy.stats import f
        p_value = f.cdf(loss_n.detach().numpy(), 199, 1)
        print(p_value)
      
        out_target = target - torch.Tensor(assign_list) 
        
        sys.stdout = log_file

        # print(torch.Tensor(y_hat)[:10].T)
        # print(assign_list[:10].T)
        # print(loss_list[:10].T)
        loss_final = torch.sum(torch.square(out_target))
        # loss_final = torch.sum(relu_out)

        temp_list = assignment_loss(loss_list, target, type='adjusted')

        print(target, torch.sum(temp_list) / len(temp_list))

        sys.stdout = old_stdout

        # print(target, torch.sum(temp_list) / len(temp_list))

    # loss_final = Variable(loss_final, requires_grad = True)
    # loss_final.retain_grad()
    loss_n.retain_grad()

    return -loss_n#loss_final


def compute_rmse(model, x_test, y_test):
    model.eval()
    # mean, std = model(x_test)
    # y_hat = torch.distributions.Normal(mean, std)

    mean, cho = model(x_test)
    y_hat = D.MultivariateNormal(mean, scale_tril=cho)
    # y_hat = model(x_test)
    pred = model.y_scaler.inverse_transform(y_hat.mean)
    return torch.sqrt(torch.mean((pred - y_test)**2))




def cvx_layer_new(n):
    z1 = cp.Variable(n)
    z = cp.Variable(n)

    # w = cp.Parameter(n)
    alpha = cp.Parameter(n)
    S = cp.Parameter((n, n))

    obj = alpha @ z1 + cp.pnorm(z, p=2)

    constraints = [cp.sum(z1) == 1, z1 >= 0, z == S @ z1]

    prob = cp.Problem(
        cp.Minimize(obj),
        constraints
    )

    layer = CvxpyLayer(prob, parameters=[alpha, S], variables=[z1, z])
    return layer


def cvx_layer_onestep( init_w, alpha, S, steps, target):

    z1 = init_w
    
    theoritical_radius =  np.sqrt(scipy.stats.chi2.ppf(target, df= z1.shape[1]))

    def obj(z1):
        z_w = z1  # /torch.sum(z1)
        obj = - ( mean @ z_w - torch.norm(cov @ z_w, p=2))
        # print(z_w)
        # print(mean)
        # print(cov)
        # print(mean @ z_w)
        # print(torch.norm(cov @ z_w, p=2))
       

        return obj

    task_loss = []
    z_updated = []

    # print(torch.round(alpha[100:1], decimals=3))
    for i in range(len(alpha)):

        z = z1[i]

        mean = alpha[i].view(-1)
        cov = S[i]

        res = minimize_constr(
            obj, z,
            max_iter=steps,
            constr=dict(
                fun=lambda x: x.sum().square(),
                lb=1, ub=1, keep_feasible=True
            ),
            bounds=dict(lb=0, ub=1, keep_feasible=True),
            disp=0
        )
        # print(res)
        # import sys
        # sys.exit()
        output = res.x  # /torch.sum(res.x)
        loss = res.fun
        
        output.retain_graph = True
        output.requires_grad = True
        loss.requires_grad = True
        # print(output)

        # if (i == 145):
        #     print(mean)
        #     print(cov)
        #     print(output)
        z_updated.append(output)
        task_loss.append(res.fun)

    return z_updated, task_loss, res.success



def cvx_layer_onestep_for_conformal( init_w, alpha, S,radius, steps, target):
    
    z1 = init_w
        
    theoritical_radius =  np.sqrt(scipy.stats.chi2.ppf(target, df= z1.shape[1]))

    def obj(z1):
        z_w = z1  # /torch.sum(z1)
        obj = - ( mean @ z_w - radius*torch.norm(cov @ z_w @ cov.T, p=2))
        # print(z_w)
        # print(mean)
        # print(cov)
        # print(mean @ z_w)
        # print(torch.norm(cov @ z_w, p=2))
       

        return obj

    task_loss = []
    z_updated = []

    # print(torch.round(alpha[100:1], decimals=3))
    for i in range(len(alpha)):

        z = z1[i]

        mean = alpha[i].view(-1)
        
        if len(S) == 2:
            cov = S
        else:        
            cov = S[i]

        res = minimize_constr(
            obj, z,
            max_iter=steps,
            constr=dict(
                fun=lambda x: x.sum().square(),
                lb=1, ub=1, keep_feasible=True
            ),
            bounds=dict(lb=0, ub=1, keep_feasible=True),
            disp=0
        )
        # print(res)
        # import sys
        # sys.exit()
        output = res.x  # /torch.sum(res.x)
        loss = res.fun
        
        output.retain_graph = True
        output.requires_grad = True
        loss.requires_grad = True
        # print(output)

        # if (i == 145):
        #     print(mean)
        #     print(cov)
        #     print(output)
        z_updated.append(output)
        task_loss.append(res.fun)

    return z_updated, task_loss, res.success


# layer1 = cvx_layer_new(n)


# def compute_e2e_loss(model, x_batch, y_batch):
#     b, A = model(x_batch)

#     overall_loss = []
#     returns = torch.zeros(batch_size, 0)
#     for i in range(len(b)):

#         mean = b[i].view(-1)
#         cov = A[i]

#         w_tmp = layer1(mean, cov, solver_args={'solve_method': 'ECOS'})

#         # z = layer(A, b, solver_args={'solve_method':'ECOS'})
#         cov_np = cov.detach().numpy()
#         S = np.array(np.linalg.inv(cov_np @ cov_np.T))
#         r = []
#         # print(mean)
#         # print(cov)
#         r += [np.exp(np.random.multivariate_normal(mean.detach().numpy(), S)) - 1]
#         r = torch.from_numpy(np.array(r))

#         total_return = 1 + (w_tmp[0] * y_batch[i].view(1, 5)).sum(1)
#         # returns = torch.cat([returns, total_return.unsqueeze(-1)], axis=1)

#         loss = -total_return  # .unsqueeze(-1)
#         # loss =  ((w_tmp[i]-y_batch[i])**2).sum()
#         overall_loss.append(loss)
#     final_loss = sum(overall_loss)

#     return final_loss


def compute_po_loss(epoch, steps, gamma,target, z_prev, model, x_batch, y_batch, mod=None):
    # loss = compute_loss(model, x_batch, y_batch)
    loss = dist_loss(model, x_batch, y_batch, mod, target )
    # loss_nll = multivariateNLL(model, x_batch, y_batch, mod)

    return loss  #+ loss_nll


def compute_e2e_loss_os(z_prev, model, x_batch, y_batch, mod, epoch, steps, gamma, target):
    b, A, radius, _ = model(x_batch)
    
   
    w_tmp, task_loss, status = cvx_layer_onestep( z_prev, b, A, steps, target)
    w_tmp = torch.stack(w_tmp)
    w_tmp.retain_grad()
    prod = w_tmp.mul(y_batch)
    port_values = torch.sum(prod, dim=1)
    
    
    # print(w_tmp[:5])
    # print(prod[:5])
    # print(torch.sort(port_values, descending=True)[0])
    # print(cvar_loss)
    # print("------------------")
    

    method = 'cvar'

    if method == 'cvar':
        reducer = SuperquantileReducer(superquantile_tail_fraction= target)
        cvar_loss = reducer(port_values)
        
        loss = cvar_loss
        
       
        # reducer2 = SuperquantileSmoothReducer(superquantile_tail_fraction= target, smoothing_coefficient=1.0)
        # cvar_smooth = reducer2(port_values)
        
        # loss = cvar_smooth
        
        
        
        # quant = int(target*len(port_values)) + 1
        
        # port_sorted = torch.sort(port_values, descending=True)[0]
        # quant = port_sorted[quant]

        # port_le_quant = port_values.le(quant).float()
        # port_le_quant.requires_grad = True
        # loss =   port_values.mul(port_le_quant).sum() / port_le_quant.sum()
        # loss.requires_grad=True
        


    elif method == 'mean':
        loss = - torch.mean(port_values)
        # loss.requires_grad=True
    elif method == 'exponential':
        param = 1
        port_inter = (1 - torch.exp(-param*port_values))*(1/param)

        loss = torch.mean(port_values)
        # loss.requires_grad=True
    
    loss = loss 
    pred_loss = compute_po_loss(
        epoch, steps, gamma,target, z_prev, model, x_batch, y_batch, mod)

    # pred_loss.requires_grad = True
    final_loss = (1 - gamma) * loss + gamma*pred_loss
    # final_loss.requires_grad=True
        

    return final_loss, pred_loss, loss, w_tmp


# def train_one_step_old(model, optimizer, x_batch, y_batch):
#     model.train()
#     optimizer.zero_grad()
#     # loss = compute_loss(model, x_batch, y_batch)
#     loss = compute_e2e_loss(model, x_batch, y_batch)
#     loss.backward()
#     optimizer.step()
#     return loss


def train_one_step(epoch, steps, gamma,target, z_prev, model, optimizer, x_batch, y_batch, mod):
    # model.train()
    # optimizer.zero_grad()
    # loss = compute_loss(model, x_batch, y_batch)
    final_loss, pred_loss, loss, w_tmp = compute_e2e_loss_os(
         z_prev, model, x_batch, y_batch, mod, epoch, steps, gamma, target)

    # final_loss.backward()
    # optimizer.step()

    return final_loss, pred_loss, loss, w_tmp

def test(model, x_test, y_test, target):
    z_prev = torch.abs(torch.randn_like(y_test))
    z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
    
    epoch = None
    steps = 1000
    gamma = 0
   
    final_loss, pred_loss, loss, w_tmp = compute_e2e_loss_os(
         z_prev, model, x_test, y_test, mod, epoch , steps , gamma, target)

    return loss.item()


def train(model, optimizer, mod, x_train, x_val, y_train, y_val, n_epochs, steps, gamma,target, batch_size, scheduler=None, print_every=10):
    train_losses, val_losses, pred_losses, port_losses = [], [], [], []

    z_prev = torch.abs(torch.randn_like(y_train))
    z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
    
    radius_prev = 0

    # 0.2*torch.ones_like(y_train)  # [:,:y_train.shape[1]-1])


    for epoch in range(n_epochs):
        
        # print(z_prev[:50])
        
        batch_indices = sample_batch_indices(x_train, y_train, batch_size)
        batch_losses_t, port_losses_t, pred_losses_t = [], [], []
        

        for batch_ix in batch_indices:
            
            z_prev_subset = z_prev[batch_ix]

            model.train()
            optimizer.zero_grad()

            b_train_loss, b_pred_loss, b_port_loss, w_tmp = train_one_step(
                epoch, steps, gamma,target, z_prev_subset, model, optimizer, x_train[batch_ix], y_train[batch_ix], mod)
            
            for i in range(len(w_tmp)):

                z_prev[batch_ix[i]] = w_tmp[i]
                
            model.radius.retain_grad()
            radius_prev =  torch.sqrt(torch.abs(model.radius)).item()
                             
            b_train_loss.backward(retain_graph = True)
               
            optimizer.step()
            
            # params = []
            # grads = []
            # for param in model.parameters():
            #     params.append(param)
            #     grads.append(param.grad)
            # print(params)
            # print(grads)
            # print(model.cho_elements_module.weight.grad)
            # from torchviz import make_dot

            # make_dot(model(x_train), params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
            
          
            batch_losses_t.append(b_train_loss)  #.detach().numpy())
            pred_losses_t.append(b_pred_loss.detach().numpy())
            port_losses_t.append(b_port_loss.detach().numpy())

        if scheduler is not None:
            scheduler.step()

        train_loss_fin = torch.stack(batch_losses_t, dim=0).sum(dim=0).sum(dim=0)
        pred_loss_fin = np.mean(pred_losses_t)
        port_loss_fin = np.mean(port_losses_t)
     

        train_losses.append(train_loss_fin.detach().numpy())
        pred_losses.append(pred_loss_fin)
        port_losses.append(port_loss_fin)
        
        avg_perc  = 0
        port_perc = 0
        pred_perc = 0
        
        avg_length = 10
        if len(train_losses) > avg_length:
            avg_loss = sum(train_losses[-avg_length:])/avg_length
            avg_perc = 100*(train_loss_fin - avg_loss)/avg_loss
            
            avg_port = sum(port_losses[-avg_length:])/avg_length
            port_perc = 100*(port_loss_fin - avg_port)/avg_port
            
            avg_pred = sum(pred_losses[-avg_length:])/avg_length
            pred_perc = 100*(pred_loss_fin - avg_pred)/avg_pred
            
            
            if (abs(avg_perc) <= 0.05):
                if  ((port_perc) > 0):
                    continue
                else:
                    break

        if epoch == 0 or (epoch + 1) % print_every == 0:
            print(
                f'Epoch {epoch+1} | Train = {train_loss_fin:.4f}, {abs(avg_perc):.4f}  | Port = {port_loss_fin:.4f}, {abs(port_perc):.4f}  | Pred = {pred_loss_fin:.4f}, {abs(pred_perc):.4f} ')

    return train_losses, port_loss_fin.item(), radius_prev # , val_losses




# def train_pred_model(model, optimizer, mod, x_train, x_val, y_train, y_val, n_epochs, steps, gamma, batch_size, scheduler=None, print_every=10):
#     train_losses, val_losses = [], []
#     z_prev = 0.2*torch.ones_like(y_train)  # [:,:y_train.shape[1]-1])

#     batch_indices = sample_batch_indices(x_train, y_train, batch_size)

#     for epoch in range(n_epochs):

#         batch_losses_t, port_losses, pred_losses = [], [], []

#         for batch_ix in batch_indices:

#             model.train()
#             optimizer.zero_grad()

#             pred_loss = compute_po_loss(
#                 epoch, steps, gamma, z_prev, model, x_train[batch_ix], y_train[batch_ix], mod)

#             # rmse_loss =  -compute_loss(model,x_train[batch_ix], y_train[batch_ix])
#             # pred_loss = rmse_loss  #pred_loss + rmse_loss
#             pred_losses.append(pred_loss)

#             pred_loss.backward()
#             optimizer.step()
#             # if model.cho_elements_module.weight.grad is not None:
#             #     print(model.mean_linear.weight.grad[0][0].item())

#         if scheduler is not None:
#             scheduler.step()

#         train_loss = torch.stack(pred_losses, dim=0).sum(dim=0).sum(dim=0)

#         # train_loss.requires_grad = True

#         # train_loss.backward()
#         # optimizer.step()

#         train_losses.append(train_loss.detach().numpy())

#         # if len(train_losses) > 5:
#         #     avg_loss = sum(train_losses[-5:])/5
#         #     perc = 100*(train_loss - avg_loss)/avg_loss
#         #     # print(perc)
#         #     if abs(perc) <= 0.05:
#         #         break

#         if epoch == 0 or (epoch + 1) % print_every == 0:
#             print(f'Epoch {epoch+1} | Train loss = {train_loss:.4f}')

#     return train_losses  # , val_losses


def uniqueid_generator(exp_type, steps, gamma, target):
    dir_id = str(datetime.datetime.now().strftime('%Y%m%d'))
    
    if not os.path.exists(r"D:\projects\spdlayers\prod_run\results\simulated\%s" %(dir_id)):
        os.mkdir(r"D:\projects\spdlayers\prod_run\results\simulated\%s" %(dir_id))
        
    if not os.path.exists(r"D:\projects\spdlayers\prod_run\saved_models\simulated\%s" %(dir_id)):
        os.mkdir(r"D:\projects\spdlayers\prod_run\saved_models\simulated\%s" %(dir_id))
    # +str(datetime.datetime.now().strftime('%H%M%S'))
    file_id = str(exp_type)+"_"+str(steps)+"_"+str(int(gamma*100))+"_"+str(int(target*100))
    
    return dir_id, file_id


def run():
    df, aux1 = load_data(exp_type=exp_type)


    for steps in step_list:  # [2, 10, 50]:
        for gamma in gamma_list:
            for target in target_list:
                st = time.time()
        
                x_train, x_val, x_test, y_train, y_val, y_test = compute_train_test_split(
                    aux1, df, exp_type=exp_type)
        
                x_scaler = StandardScaler().fit(x_train)
                y_scaler = StandardScaler().fit(y_train)
                
                x_val_scal = StandardScaler().fit(x_val)
                y_val_scal = StandardScaler().fit(y_val)
        
                x_test_scal = StandardScaler().fit(x_test)
                y_test_scal = StandardScaler().fit(y_test)
        
                n_hidden = y_train.shape[1]
        
                model = DeepNormalModel(
                    n_inputs=x_train.shape[1],
                    n_hidden=n_hidden,
                    x_scaler=x_scaler,
                    y_scaler=y_scaler,
                    mod=mod
                )
                
                
        
                pytorch_total_params = sum(p.numel()
                                           for p in model.parameters() if p.requires_grad)
                print(f'{pytorch_total_params:,} trainable parameters')
        
        
                if mod == 'find_target':
                    # target = nn.Parameter(torch.randn(1), requires_grad=True)
                    params = list(model.parameters())  # + list(target)
                if mod in ('fixed_target', 'no_reg'):
                    # target = None
                    params = list(model.parameters())
               
                
                optimizer = torch.optim.Adam(
                    params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
        
                scheduler = None
        
                train_losses, train_loss_fin, radius = train(
                    model,
                    optimizer, mod,
                    x_train,
                    x_val,
                    y_train,
                    y_val,
                    n_epochs=n_epochs,
                    steps=steps,
                    gamma=gamma,
                    target= target,
                    batch_size=batch_size,
                    scheduler=scheduler,
                    print_every=print_every,
                )
                et = time.time()
                elapsed_time = np.round(et - st, 3)
                
                del et, st
                
                val_var = test(model, x_val, y_val, target)
                test_var = test(model, x_test, y_test, target)
                
                train_coverage = dist_loss(model, x_train, y_train, mod, target, test_mode= 'on')
                val_coverage   = dist_loss(model, x_val,   y_val,   mod, target, test_mode= 'on')
                test_coverage  = dist_loss(model, x_test,  y_test,  mod, target, test_mode= 'on')
                
        
                print([target,steps, gamma, elapsed_time,train_loss_fin, val_var, test_var, train_coverage, val_coverage, test_coverage])
                results = [target,steps, gamma, elapsed_time,train_loss_fin, val_var, test_var,train_coverage, val_coverage, test_coverage]
                dir_id, file_id = uniqueid_generator(exp_type, steps, gamma, target)
                pickle.dump(model, open(r"D:\projects\spdlayers\prod_run\saved_models\simulated\%s\%s.pkl" %(dir_id, file_id), 'wb'))
                with open(r"D:\projects\spdlayers\prod_run\results\simulated\%s\%s_results.csv" %(dir_id, file_id), 'a') as f_object:
        
                    writer_object = writer(f_object)
                    writer_object.writerow(results)
                    f_object.close()
        
                with open(r'D:\projects\spdlayers\prod_run\results\simulated\%s\%s_epoch_results.csv' %(dir_id, file_id), 'a') as f_object_:
        
                    writer_object = writer(f_object_)
                    writer_object.writerow(train_losses)
                    f_object_.close()
    
    
    log_file.close()
    







if __name__ == '__main__': 
    run()
