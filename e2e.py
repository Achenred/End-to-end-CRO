# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 08:46:51 2023

@author: abhil
"""

import torch
import seaborn as sns
import numpy as np
import pandas as pd
import time as gettime
import pickle
import sys
import datetime 
from torch.optim import lr_scheduler
import os
from utils import load_data, compute_train_test_split, train_test_scaler, DeepNormalModel, DeepNormalModel_NLL, train,train_and_save_best, test, dist_loss,get_coverage,get_p_value, uniqueid_generator,get_conditional_coverage, generate_mixture_data, plot_sim_data, conditional_data_generation, generate_covariance_matrix, is_psd, generate_covariance_matrix_scaled, plot_distribution_cov_frequency, predict_then_optimize, test_and_save, ellipsoid_comparision,saving_files,write_list_to_csv

from conformal_synthetic_data import conformal_train

torch.set_default_tensor_type(torch.DoubleTensor)
old_stdout = sys.stdout
log_file = open(r'.\simulated\output_log.log', "w")

sns.set(palette='colorblind', font_scale=1.3)
palette = sns.color_palette()

seed = 456
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)


torch.set_default_dtype(torch.double)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensor.to(device)


exp_type = 'simulated'
learning_rate = [1e-1, 1e-2, 1e-3]
momentum = 1e-08
weight_decay = 1e-8
n_epochs = [100,200,500,1000]
batch_size = [200,400,600]
print_every = 1
step_list =  [3,5, 10, 50] 
gamma_list = [0.1, 0.3, 0.5, 0.7,0.9]
target_list = [0.7, 0.8, 0.9]  
sim_data_params = [2, 1]
mod = 'fixed_target' 
algorithm = [ 'conformal', 'conditional_conformal', 'nll+opt', 'e2e_k_step', 'e2e_coverage']  

def data_gen(alpha, phi):
  
    # Parameters
    mean_a = np.array([0, 0, 0, 0])
    mean_b = np.array([0, 5, 5, 0])
    mean_c = mean_b
    
    np.random.seed(123)
    # Example usage
    variance_1 = 1.0
    variance_2 = 1.5
    correlation = 0.3
    spread_factor = 2.0
        
    cov_a = generate_covariance_matrix(variance_1, variance_2, correlation, spread_factor)
    
    num_samples = 2000
     # Varying values of alpha
    alpha = alpha  
    phi = phi
    p_a = phi
    p_b = (1 - p_a) / (alpha + 1)
    p_c = (1 - p_a) * alpha / (alpha + 1)
    
    cov_b = cov_a * alpha
    cov_c = cov_a / alpha
    
    mean_list = [mean_a, mean_b, mean_c]
    cov_list = [cov_a, cov_b, cov_c]
    prob_list = [p_a, p_b, p_c]
    
    print(alpha, phi)
    data = generate_mixture_data(mean_list,cov_list, prob_list, num_samples)
    
    df = pd.DataFrame(data[:, :2])
    aux1 = pd.DataFrame(data[:, 2:])
 
    
    x_train, x_val, x_test, y_train, y_val, y_test = compute_train_test_split(df, aux1, exp_type=exp_type)
    x_scaler, x_val_scaler, x_test_scaler = x_train, x_val, x_test #train_test_scaler(x_train, x_val, x_test)
    n_hidden = y_train.shape[1]
    
    
    # Example usage
    mean_list = [mean_a, mean_b, mean_c]
    cov_list = [cov_a, cov_b, cov_c]
    prob_list = [p_a, p_b, p_c]

    num_conditional_points = 100
    cgd_train = conditional_data_generation(np.array(x_train), mean_list, cov_list, prob_list,num_conditional_points)
    num_conditional_points = 100
    cgd_test = conditional_data_generation(np.array(x_test), mean_list, cov_list, prob_list,num_conditional_points)

    return x_scaler, x_val_scaler, x_test_scaler, y_train, y_val, y_test, cgd_train, cgd_test
 

def portfolio_data_gen(year, seed):
    year_directory = f"data/{year}_samples"
    returns_file = f"{year}_returns_{0}.txt"
    side_file = f"{year}_data_side_{0}.txt"
    cols_list = []
         
    returns_file_path = os.path.join(year_directory, returns_file)
    side_file_path = os.path.join(year_directory, side_file)
    
    with open(returns_file_path, 'r') as f:
        lines = f.readlines()
        returns_lists = [list( line.strip().split(',')) for line in lines]
        
    with open(side_file_path, 'r') as f:
        lines = f.readlines()
        side_info_lists = [list( line.strip().split(',')) for line in lines]
        
    returns_cols = returns_lists[seed] + ['DATE']
    side_info_cols = side_info_lists[seed] + ['DATE']
    
    returns = pd.read_csv("./data/expected_return.csv")
    data_side = pd.read_csv("./data/side_info.csv") 
    
    def create_train_val_test(df, year, col_list):        
    
        df_sub = df[col_list]
        df_sub['DATE'] = pd.to_datetime(df_sub['DATE'])
        df_sub['year'] = df_sub['DATE'].dt.year
        df_sub['year'] = df_sub['year'].astype(int)
        
        # Parse start year to integer
        start_year = int(year)
        train_end_year = start_year + 2
        val_start_year = train_end_year+1
        test_start_year= val_start_year+1 
    
        df_train = df_sub[(df_sub.year >= start_year) & (df_sub.year <= train_end_year)]
        df_val = df_sub[df_sub.year == val_start_year]
        df_test = df_sub[df_sub.year == test_start_year]
        

        df_train.drop(['DATE', 'year'], axis = 1, inplace = True)  
        df_val.drop(['DATE', 'year'],axis = 1, inplace = True)
        df_test.drop(['DATE', 'year'],axis = 1, inplace = True)
        
        df_train[np.isnan(df_train)] = 0
        df_val[np.isnan(df_val)] = 0
        df_test[np.isnan(df_test)] = 0
     
        
        
        return df_train, df_val , df_test    
   
    returns_train, returns_val, returns_test = create_train_val_test( returns, year, returns_cols)
    returns_train_s, returns_val_s, returns_test_s = train_test_scaler(returns_train, returns_val, returns_test )
    
    
    returns_train_s =  torch.from_numpy(100*returns_train.values)
    returns_val_s   =  torch.from_numpy(100*returns_val.values)
    returns_test_s  =  torch.from_numpy(100*returns_test.values)
    

    side_train, side_val, side_test = create_train_val_test( data_side, year, side_info_cols)
    side_train_s, side_val_s, side_test_s = train_test_scaler(side_train, side_val, side_test)
    side_train_s, side_val_s, side_test_s = torch.from_numpy(side_train_s.values), torch.from_numpy(side_val_s.values), torch.from_numpy(side_test_s.values)
    
      
    
    return side_train_s, side_val_s, side_test_s, returns_train_s, returns_val_s, returns_test_s
    
        
plots_data_path = r'./outputs/plots_data'
if not os.path.exists(plots_data_path):
    os.makedirs(plots_data_path)
results_file_path = r'./outputs/'
file_name = str(exp_type)+'_output_summary.csv'
 # Create the folder if it doesn't exist
saving_files(results_file_path, file_name)
date = datetime.datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
time = datetime.datetime.now().strftime("%H:%M:%S")  # Format: HH:MM:SS

if exp_type == 'simulated':
    list_1 =  [0.01, 0.05, 0.1, 0.5, 1,5, 10, 20, 50, 100]
    list_2 =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9]
else:
    list_1 = [2012, 2013, 2014, 2015, 2016, 2017]
    list_2 = list(range(0,10))

freq_list = []
for alg in algorithm:
    for target in target_list:
        for alpha in list_1:
            for phi in list_2:
                
                if exp_type == 'simulated':
                    x_scaler, x_val_scaler, x_test_scaler, y_train, y_val, y_test, cgd_train, cgd_test = data_gen(alpha, phi)
                    
                   
                else:
                    x_scaler, x_val_scaler, x_test_scaler, y_train, y_val, y_test = portfolio_data_gen(alpha, phi)
                    
                    x_scaler = torch.nan_to_num(x_scaler)        
              
                    cgd_train = None
                    cgd_test = None
                                       
                
                if alg in ['e2e_coverage','e2e_k_step', 'e2e_nll', 'nll+opt']:
                
                    for steps in step_list:
                        for gamma in gamma_list:
                            print(str([alg, target, steps, gamma]))
                            
                            st = gettime.time()
                            n_hidden = y_train.shape[1]
                            
                                                    
                            if alg == 'nll+opt':
                                nn_model = DeepNormalModel_NLL(
                                    n_inputs=x_scaler.shape[1],
                                    n_hidden=n_hidden,
                                    x_scaler= x_scaler,
                                    y_scaler=y_train,
                                    mod=mod
                                )
                            else:
                            
                                nn_model = DeepNormalModel(
                                    n_inputs=x_scaler.shape[1],
                                    n_hidden=n_hidden,
                                    x_scaler= x_scaler,
                                    y_scaler=y_train,
                                    mod=mod
                                )
                            
                            pytorch_total_params = sum(p.numel()
                                                       for p in nn_model.parameters() if p.requires_grad)
                            print(f'{pytorch_total_params:,} trainable parameters')
                            if mod == 'find_target':
                                # target = nn.Parameter(torch.randn(1), requires_grad=True)
                                params = list(nn_model.parameters())  # + list(target)
                            if mod in ('fixed_target', 'no_reg'):
                                # target = None
                                params = list(nn_model.parameters())
                                
                            all_params = []
                            selected_params = []
                            for name, param in nn_model.named_parameters():
                                if 'radius' in name:
                                    selected_params.append(param)
                                else:
                                    all_params.append(param)
                                    
                                  
                            optimizer = torch.optim.Adam([
     {'params': all_params, 'lr': learning_rate, 'weight_decay':weight_decay},
     {'params': selected_params, 'lr': learning_rate*0.5, 'weight_decay':weight_decay}
 ])
                            
                            if alg == 'nll+opt':
                                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100, 200], gamma=1.1)
                            
                            else:                                
                                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.9)
                           
                            model, train_losses, train_loss_fin, radius = train_and_save_best(
                                nn_model,
                                optimizer, mod,
                                x_scaler,
                                x_val_scaler,
                                y_train,
                                y_val,
                                cgd_train,
                                n_epochs=n_epochs,
                                steps=steps,
                                gamma=gamma,
                                target= target,
                                algorithm = alg,
                                batch_size=batch_size,
                                scheduler=scheduler,
                                print_every=print_every,
                            )
                            et = gettime.time()
                            elapsed_time = np.round(et - st, 3)
                            
                            del et, st
                            
                            if alg == 'nll+opt':
                               
                                test_cvar, test_returns, torch_data = test_and_save(model, x_test_scaler, y_test,steps, target, pickle_filename = alg)
                                test_coverage, test_cov_list, test_conditional_cov, cov_freq_dist_e2e = predict_then_optimize(alg, model, x_test_scaler, y_test, steps, target, mod, cgd_test)
                                
                               
                                # Define the pickle file path
                                cov_path = os.path.join(plots_data_path, str(alg)+"_"+str(target)+"_"+str(steps)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(phi)+"_cov.pkl").replace("\\","/")
                                data_path = os.path.join(plots_data_path, str(alg)+"_"+str(target)+"_"+str(steps)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(phi)+"_data.pkl").replace("\\","/")
                                # Save the list as a pickle file
                                
                                
                                with open(cov_path, 'wb') as pickle_file:
                                    print(cov_path)
                                    pickle.dump(cov_freq_dist_e2e, pickle_file)
                                    
                                with open(data_path, 'wb') as pickle_file:                                    
                                    pickle.dump(torch_data, pickle_file)
                                    
                                
                                
                                print(test_cvar, test_coverage, test_conditional_cov)
                                print(cov_freq_dist_e2e)
                                                            
                            else :
                               
                                test_cvar, test_returns, torch_data = test_and_save(model, x_test_scaler, y_test,steps, target, pickle_filename = alg)
                            
                                if cgd_test is not None:
                                    
                                    test_conditional_cov, cov_freq_dist_e2e = get_conditional_coverage(algorithm,model, x_test_scaler, y_test, mod, target, cgd_test)
                                else:
                                    test_conditional_cov = None
                                    cov_freq_dist_e2e = None
                                
                                test_coverage, test_cov_list  = get_coverage(algorithm,model, x_test_scaler, y_test, mod, target)
                             
                                # Define the pickle file path
                               
                                print(test_cvar, test_coverage, test_conditional_cov)
                                print(cov_freq_dist_e2e)
                                
                                # Define the pickle file path
                                cov_path = os.path.join(plots_data_path, str(alg)+"_"+str(target)+"_"+str(steps)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(phi)+"_cov.pkl").replace("\\","/")
                                data_path = os.path.join(plots_data_path, str(alg)+"_"+str(target)+"_"+str(steps)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(phi)+"_data.pkl").replace("\\","/")
                                # Save the list as a pickle file
                                
                               
                                with open(cov_path, 'wb') as pickle_file:
                                    pickle.dump(cov_freq_dist_e2e, pickle_file)
                                    
                                with open(data_path, 'wb') as pickle_file:                                    
                                    pickle.dump(torch_data, pickle_file)
                                
                            results = [date,time,alg,target,steps, gamma,alpha, phi, test_conditional_cov, test_cvar, test_coverage]
                            
                            csv_file_path = os.path.join(results_file_path, file_name)
                            
                            write_list_to_csv(csv_file_path, results)
                                                                            
                            del model
        
                            
                elif alg in ['conformal', 'conditional_conformal']:
                   
                    
                    steps = gamma = None
                    
                   
                    train_conditional_cov, test_conditional_cov,freq_list_conf, test_cvar,port_values, test_pvalue, test_coverage, torch_data = conformal_train(alg,
                            x_scaler,
                            x_val_scaler, x_test_scaler,
                            y_train,
                            y_val, y_test,
                            cgd_train, cgd_test,
                            target= target,
                            algorithm = alg,
                            batch_size=batch_size,
                            print_every=print_every,)
                                  
                    print(test_cvar)
                    print(test_conditional_cov)
                    print(freq_list_conf)
                    
                                       
                                       
                    cov_path = os.path.join(plots_data_path, str(alg)+"_"+str(target)+"_"+str(steps)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(phi)+"_cov.pkl").replace("\\","/")
                    data_path = os.path.join(plots_data_path, str(alg)+"_"+str(target)+"_"+str(steps)+"_"+str(gamma)+"_"+str(alpha)+"_"+str(phi)+"_data.pkl").replace("\\","/")
                    # Save the list as a pickle file
                    
                    
                    # Save the list as a pickle file
                    with open(cov_path, 'wb') as pickle_file:
                        pickle.dump(freq_list_conf, pickle_file)
                        
                    with open(data_path, 'wb') as pickle_file:                                    
                        pickle.dump(torch_data, pickle_file)
                         
                    results = [date, time, alg,target,steps, gamma,alpha, phi,  test_conditional_cov, test_cvar,  test_coverage]
                    
                    print(results)
                    csv_file_path = os.path.join(results_file_path, file_name)
                    write_list_to_csv(csv_file_path, results)
                    
                 
                   