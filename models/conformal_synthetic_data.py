import sys
import math
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import torch 
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from sklearn.svm import SVC # import SVC model
import torch.nn.functional as F
from torch import nn,optim
from utils import optimization_conformal, optimization_conformal_conditional_samples

sys.path.append("../")
sys.path.append("../../")


from utilities.ellipsoidal_conformal_utilities import (
    ellipsoidal_non_conformity_measure,
    ellipse_global_alpha_s,
    ellipse_local_alpha_s,
    local_ellipse_validity_efficiency,
    ellipse_volume,
    plot_ellipse_global,
    plot_ellipse_local,
)

from utilities.copula_conformal_utilities import (
    prepare_norm_data,
    simple_mlp,
    std_emp_conf_predict,
    norm_emp_conf_predict,
    std_emp_conf_all_targets_alpha_s,
    norm_emp_conf_all_targets_alpha_s,
    empirical_conf_validity,
    empirical_conf_efficiency,
    plot_standard_rectangle,
    plot_normalized_rectangle,
)

n_out = 2
lam = 0.9


standard_empirical_validity_all = []
standard_empirical_efficiency_all = []
normalized_empirical_copula_validity_all = []
normalized_empirical_copula_efficiency_all = []
standard_ellipse_validity_all = []
standard_ellipse_efficiency_all = []
normalized_ellipse_validity_all = []
normalized_ellipse_efficiency_all = []


    
class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, hidden_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear  = torch.nn.Linear(input_dim, input_dim, bias = False)
         self.linear1 = torch.nn.Linear(input_dim, input_dim, bias = False)
         self.linear2 = torch.nn.Linear(input_dim, input_dim, bias = False)
         self.linear3 = torch.nn.Linear(input_dim, input_dim, bias = False)
         self.linear4 = torch.nn.Linear(input_dim, output_dim, bias = False)
     def forward(self, x):
         hidden1 = self.linear(x)
         hidden2 = self.linear1(hidden1)
         hidden3 = self.linear2(hidden2)
         hidden4 = self.linear3(hidden3)
         hidden5 = self.linear4(hidden4)
         outputs = torch.sigmoid(hidden5)
         return outputs
        


def get_conditional_cov(y,y_pred, conditional_data, cov_train, alpha_s):
    
    if conditional_data is not None:

        print(y.shape, y_pred.shape, conditional_data.shape)
    
        conditional_mu = np.array(y_pred)
        mu_pred = np.array(y_pred)
        # conditional_cov = np.linalg.inv(np.linalg.inv(cov)[2:4, 2:4])
    
        result = []
        for i in range(len(conditional_mu)):
            mu = conditional_mu[i]
    
            dependent_data = conditional_data[i]
    
            inv_cov_train = np.linalg.inv(cov_train)
            error_test = (dependent_data - mu )  #.detach().numpy()
           
    
            non_conf_multi_test = ellipsoidal_non_conformity_measure(error_test, inv_cov_train)
            coverage = np.mean(non_conf_multi_test < alpha_s) * 100
           
            result.append( [coverage])   #[conditional_mu[0],conditional_mu[1], coverage])
    
    
        result_np = np.array(result)
        np.savetxt("coverage.csv", result_np, delimiter=",")
    
        return np.round([np.mean(result), np.std(result), np.min(result), np.max(result), np.quantile(result, 0.1),  np.quantile(result, 0.9)], 2), np.sort(np.array(result).T)[0]
    else:
        return None, None

def get_local_conditional_cov(
    knn,
    x,
    y_pred_train,
    y_true_train,
    y_pred_test,
    y_true_test,
    local_alpha_s,
    lam,
    cov_train, conditional_data, x_train, epsilon
):
    """
    Calculates conformal validity and efficiency performance results for the normalized local ellipsoidal non-conformity measure
    :param knn: knn
    :param y_pred: y_pred_test
    :param y_true: y_true_test
    :param local_alpha_s: Local $\alpha_s$ value
    :param lam: $\lambda$ value
    :param cov_train: Covariance matrix estimated from proper training instances
    :param cgd_test: conditionally generated data

    XXX:param local_neighbors_test: Obtained kNN neighbors for each instance in test data
    XXX:param y_true_test: Test data's ground truth
    XXX:param y_pred_test: Test data's predictions
    XXX:param y_true_train: Proper Training data's ground truth
    XXX:param y_pred_train: Proper Training data's predictions
    XXX:param local_alpha_s: Local $\alpha_s$ value
    XXX:param dim: Output dimension number k
    XXX:param lam: $\lambda$ value
    XXX:param cov_train: Covariance matrix estimated from proper training instances
    XXX:return: validity, efficiency
    """

    conditional_mu = np.array(y_pred_test)
    mu_pred = np.array(y_pred_test)
    # conditional_cov = np.linalg.inv(np.linalg.inv(cov)[2:4, 2:4])

    conditional_mu = np.array(y_pred_test)
    local_neighbors = knn.kneighbors(x, return_distance=False)
    
    if conditional_data is not None:

        transposed_conditional_data = conditional_data.transpose((1, 0, 2))
    
        result = []
        
        
    
        for j in  range(len(conditional_mu)):
    
            mu = conditional_mu[j]
            dependent_data = conditional_data[j]
    
            #OLD wrong
            #local_y_minus_y_true = (y_true_train - y_pred_train)[local_neighbors[j, :], :  ]
            #old
            y_pred_test_j = y_pred_test[j]
            #Erick: new
            #y_pred_test_j = np.mean(y_true_train[local_neighbors[j, :],:],axis=0)
            local_y_minus_y_true = y_true_train[local_neighbors[j, :],:] - y_pred_test_j
    
            #local_cov_test = np.cov(local_y_minus_y_true.T)
            local_cov_test = (1/(local_y_minus_y_true.shape[0]-1))*local_y_minus_y_true.T@local_y_minus_y_true
            local_cov_test_regularized = lam * local_cov_test + (1 - lam) * cov_train
            local_inv_cov_test = np.linalg.inv(local_cov_test_regularized)
            local_error_test = dependent_data - y_pred_test_j
            local_non_conf_multi_test_all = ellipsoidal_non_conformity_measure(local_error_test, local_inv_cov_test)
            
            # print(local_cov_test)
            coverage = np.mean(local_non_conf_multi_test_all < local_alpha_s) * 100
           
            # local_cov.append(coverage)
            result.append( [coverage])
    
        result_np = np.array(result)
    
        np.savetxt("conditional_coverage.csv", result_np, delimiter=",")
        return np.round([np.mean(result_np),np.std(result_np), np.min(result_np), np.max(result_np), np.quantile(result_np, 0.1),  np.quantile(result_np, 0.9)], 2), np.sort(result_np.T)[0]
    else:
        return None, None

import pickle
def get_overall_cov(y_true, y_pred, x, cov_train, alpha_s, target):
        y_true_test = y_true
        y_pred_test = y_pred
        x_test = x
        

        inv_cov_train = np.linalg.inv(cov_train)
        error_test = (y_true_test - y_pred_test)
        non_conf_multi_test = ellipsoidal_non_conformity_measure(error_test, inv_cov_train)

        overall_cov = np.mean(non_conf_multi_test < alpha_s) * 100
        # print("alpha_a")
        # print(alpha_s)
        
        # Create a dictionary to store PyTorch arrays
        torch_data = {
            'y_test': y_true,
            'alpha': y_pred,
            'S': cov_train,
            'radius': alpha_s
        }
        pickle_filename = 'conformal'
        # Save PyTorch arrays as a single pickle file
        with open(pickle_filename, 'wb') as pickle_filename:
            pickle.dump(torch_data, pickle_filename)
        return overall_cov

def get_overall_cov_conditional(
    local_neighbors_test,
    y_true_test,
    y_pred_test,
    y_true_train,
    y_pred_train,
    local_alpha_s,
    lam,
    cov_train, epsilon
):
    """
    Calculates conformal validity and efficiency performance results for the normalized local ellipsoidal non-conformity measure
    :param local_neighbors_test: Obtained kNN neighbors for each instance in test data
    :param y_true_test: Test data's ground truth
    :param y_pred_test: Test data's predictions
    :param y_true_train: Proper Training data's ground truth
    :param y_pred_train: Proper Training data's predictions
    :param local_alpha_s: Local $\alpha_s$ value
    :param dim: Output dimension number k
    :param lam: $\lambda$ value
    :param cov_train: Covariance matrix estimated from proper training instances
    :return: validity, efficiency
    """
    local_non_conf_multi_test_all = []
    
    local_cov = []

    for i in range(local_neighbors_test.shape[0]):
        y_pred_i = y_pred_test[i,:]
        #y_pred_i = np.mean(y_true_train[local_neighbors_test[i, :],:],axis=0)
        #local_y_minus_y_true = (y_true_train - y_pred_train)[local_neighbors_test[i, :], :  ]
        local_y_minus_y_true = y_true_train[local_neighbors_test[i, :],:] - y_pred_i
        #local_cov_test = np.cov(local_y_minus_y_true.T)
        local_cov_test = (1/(local_y_minus_y_true.shape[0]-1))*local_y_minus_y_true.T@local_y_minus_y_true
        local_cov_test_regularized = lam * local_cov_test + (1 - lam) * cov_train
        local_cov.append(local_cov_test_regularized)
        local_inv_cov_test = np.linalg.inv(local_cov_test_regularized)
        local_error_test = y_true_test[i, :] - y_pred_i
        local_non_conf_multi_test_all.append(
            ellipsoidal_non_conformity_measure(local_error_test, local_inv_cov_test)
        )
        
    
    overall_coverage = (
        np.mean(local_non_conf_multi_test_all < local_alpha_s) * 100
    )
    
    # Create a dictionary to store PyTorch arrays
    torch_data = {
        'y_test': y_true_test,
        'alpha': y_pred_test,
        'S': local_cov,
        'radius': local_alpha_s
    }
    pickle_filename = 'conditional_conformal'
    # Save PyTorch arrays as a single pickle file
    with open(pickle_filename, 'wb') as pickle_filename:
        pickle.dump(torch_data, pickle_filename)

    return  overall_coverage


def get_cvar_stat_cov(y_true, y_pred, x, cov_train, cgd_test, alpha_s, target):
    y_true_test = y_true
    y_pred_test = y_pred
    x_test = x
    
    inv_cov_train = np.linalg.inv(cov_train)
    error_test = (y_true_test - y_pred_test)  #.detach().numpy()
    non_conf_multi_test = ellipsoidal_non_conformity_measure(error_test, inv_cov_train)
               
    radius = alpha_s
    # cov_mat = torch.tensor(np.tile(np.sqrt(radius)*cov_train, (len(y_pred_test),1,1)))
    cov_mat = torch.tensor(np.tile(cov_train, (len(y_pred_test),1,1)))
    cholesky = torch.cholesky(cov_mat)
    mean = torch.tensor(y_pred_test)

   
    torch_data = {
        'y_test': y_true,
        'alpha': y_pred,
        'S': cov_mat,
        'radius': radius
    }
    
    optimization_conformal_conditional_samples(mean, cholesky, radius, torch.tensor(y_true_test),cgd_test, target)
    cvar, port_values = optimization_conformal(mean, cholesky, radius, torch.tensor(y_true_test),cgd_test, target)    
    assign_flag = [1 if i <= alpha_s else 0 for i in non_conf_multi_test]    
    test_stat =  3198  # get_p_value_conformal(assign_flag, x_test, target)    
    
    return cvar,port_values, test_stat, torch_data




def predict_knn(knn,x_train,y_train, x_predict):
  y_predict = [];
  neighbors = knn.kneighbors(x_predict, return_distance=False)
  for i in range(len(x_predict)):
    y_predict.append(np.mean(y_train[neighbors[i,:],:],axis=0))
  y_predict = np.array(y_predict)
  print(y_predict.shape)
  return y_predict

class MultiOutputRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(MultiOutputRegression,self).__init__()
        hidden_size = input_size
        self.input =nn.Linear(input_size,hidden_size)
        self.hidden=nn.Linear(hidden_size,hidden_size)
        self.output = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        yhat_hidden=self.input(x)
        yhat_hidden1 = F.tanh(yhat_hidden)
        yhat_hidden_in = self.hidden(yhat_hidden1)
        yhat_hidden_out = F.tanh(yhat_hidden_in)
        yhat = self.output(yhat_hidden_out)
        return yhat
    

def conformal_train(alg,x_train,x_val,x_test, y_train,y_val,y_test,cgd_train, cgd_test, target,algorithm,batch_size,
print_every):

    
    x_train_np, y_true_train = np.array(x_train), y_train.detach().numpy()
    x_cal, y_true_cal = np.array(x_val), y_val.detach().numpy()
    x_test, y_true_test = np.array(x_test), y_test.detach().numpy()


    # get number of neighbors
    n_neighbors = math.ceil(np.sqrt(x_train_np.shape[0]))

    lam = 0.75 #1 #ERICK
    predictor = "nn"
    #predictor = "knn"
    if predictor == "rf":
        # train regressor
        clf = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=1337)
        ).fit(x_train_np, y_true_train)
        # get regressor's predictions
        y_pred_train = clf.predict(x_train_np)
        y_pred_cal = clf.predict(x_cal)
        y_pred_test = clf.predict(x_test)
      
    elif predictor == 'nn':
        nn_model = MultiOutputRegression(x_train.shape[1], y_train.shape[1])
        optimizer = optim.SGD(nn_model.parameters(), lr = 0.1)
        criterion = nn.MSELoss()
        epochs=1000   
        for epoch in range(epochs):            
            
            yhat=nn_model(torch.tensor(x_train_np))
            loss=criterion(yhat,torch.tensor(y_true_train) )          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        y_pred_train = nn_model(torch.tensor(x_train_np)).detach().numpy()
        y_pred_cal = nn_model(torch.tensor(x_cal)).detach().numpy()
        y_pred_test = nn_model(torch.tensor(x_test)).detach().numpy()
        
     
    else:
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(x_train_np)
        # get regressor's predictions
        y_pred_train = predict_knn(knn,x_train,y_train, x_train_np)
        y_pred_cal = predict_knn(knn,x_train,y_train, x_cal)
        y_pred_test = predict_knn(knn,x_train,y_train, x_test)
      

    # get regressor's predictions with bias correction (ERICK)()
    bias = np.mean(y_true_train-y_pred_train,axis=0)
    
    print("bias",bias)
#    y_pred_train = clf.predict(x_train_np)+bias
#    y_pred_cal = clf.predict(x_cal)+bias
#    y_pred_test = clf.predict(x_test)+bias
    
    cov_train, alpha_s = ellipse_global_alpha_s(
        y_true_train, y_pred_train, y_true_cal, y_pred_cal, 1 - target)
    
    print(alpha_s)
  

    if alg == 'conformal':

        test_conditional_cov,cov_freq_dist = get_conditional_cov(y_true_test, y_pred_test, cgd_test, cov_train, alpha_s)
        
        # print(y_test[:2])
        # print(y_pred_test[:2])
        # print(cov_train)
        # print(alpha_s)
        # import sys
        # sys.exit()

        test_overall_cov = get_overall_cov(y_true_test, y_pred_test, x_test, cov_train, alpha_s, target)
#        print(test_conditional_cov)
#        print(test_overall_cov)
        test_cvar,port_values, test_stat, torch_data = get_cvar_stat_cov(y_true_test, y_pred_test, x_test, cov_train, cgd_test, alpha_s, target)


    elif alg == 'conditional_conformal':

        knn, local_alpha_s = ellipse_local_alpha_s(
            x_train,
            x_cal,
            y_true_train,
            y_pred_train,
            y_true_cal,
            y_pred_cal,
            1 - target,
            n_neighbors,
            lam,
            cov_train,
        )
        

   
        test_conditional_cov,cov_freq_dist = get_local_conditional_cov(
                knn,
                x_test,
                y_pred_train,
                y_true_train,
                y_pred_test,
                y_true_test,
                local_alpha_s,
                lam,
                cov_train, cgd_test, x_train, 1 - target
            )

        local_neighbors_test = knn.kneighbors(x_test, return_distance=False)


        test_overall_cov = get_overall_cov_conditional(  local_neighbors_test,
        y_true_test,
        y_pred_test,
        y_true_train,
        y_pred_train,
        local_alpha_s,
        lam,
        cov_train,1 - target    )
        
        def get_cvar_stat_conditional_conf(y_true, y_pred, x, cov_train, alpha_s, target):
            y_true_test = y_true
            y_pred_test = y_pred
            x_test = x
 
            train_conditional_cov = None

           
            local_neighbors_test = knn.kneighbors(x_test, return_distance=False)
            
    
            (coverage, normalized_ellipse_efficiency, binary_flag, cov_list) = local_ellipse_validity_efficiency(  local_neighbors_test,
            y_true_test,
            y_pred_test,
            y_true_train,
            y_pred_train,
            local_alpha_s,
            n_out,
            lam,
            cov_train,    )
                        
  
            radius = local_alpha_s
            cov_mat = torch.tensor(np.array(cov_list))  #np.sqrt(local_alpha_s)*
            mean = torch.tensor(y_pred_test)
            cholesky = torch.cholesky(cov_mat)
            
            # print(radius)
            # print(cov_mat[:5])
            # import sys
            # sys.exit()
            
            torch_data = {
        'y_test': y_test,
        'alpha': y_pred,
        'S': cov_mat,
        'radius': radius
    }
            optimization_conformal_conditional_samples(mean, cholesky, radius, torch.tensor(y_true_test),cgd_test, target)
            cvar, port_values = optimization_conformal(mean, cov_mat, radius, y_true_test, cgd_test, target)
            assign_flag = binary_flag
            
            # print(y_true[:5])
            # print(mean[:5])
            # print(cov_mat[:5])
            # print(radius)
            # import sys
            # sys.exit()
            
            test_stat = 3198   # get_p_value_conformal(assign_flag, x_test, target)
            
                       
            return  cvar, port_values, test_stat, torch_data
        
        test_cvar,port_values, test_stat, torch_data = get_cvar_stat_conditional_conf(y_test, y_pred_test, x_test, cov_train, alpha_s, target)
    
   
    # print(test_conditional_cov)
    # print(test_overall_cov)
    train_conditional_cov = 999
    return train_conditional_cov, test_conditional_cov,cov_freq_dist, test_cvar.item(),port_values, test_stat, test_overall_cov, torch_data

























'''
    
def get_conditional_cov(y,y_pred, clf, conditional_data, cov_train, alpha_s):

              
    conditional_mu = np.array(y_pred)
    mu_pred = np.array(y_pred)
    # conditional_cov = np.linalg.inv(np.linalg.inv(cov)[2:4, 2:4])
        
    result = []
    for i in range(len(conditional_mu)):
        mu = conditional_mu[i]
                        
        dependent_data = conditional_data[i]
     
        inv_cov_train = np.linalg.inv(cov_train)
        error_test = ( mu - dependent_data )  #.detach().numpy()
        
        
        non_conf_multi_test = ellipsoidal_non_conformity_measure(error_test, inv_cov_train)

        coverage = np.mean(non_conf_multi_test < alpha_s) * 100
        # print(mu, sum(dependent_data)/1000, coverage, alpha_s)
        
        # print([conditional_mu[0],conditional_mu[1], coverage])
        result.append( [coverage])   #[conditional_mu[0],conditional_mu[1], coverage])

        
    result_np = np.array(result)
    np.savetxt("coverage.csv", result_np, delimiter=",")
    
    return np.round([np.mean(result), np.std(result), np.min(result), np.max(result), np.quantile(result, 0.1),  np.quantile(result, 0.9)], 2)


    # dependent_data = np.array([np.random.multivariate_normal(c_mu, conditional_cov, 1)[0] for c_mu in conditional_mu])
    
    
    
def get_local_conditional_cov(
    knn,
    x,
    y_pred,
    y_true,
    local_alpha_s,
    lam,
    cov_train, conditional_data
):
    """
    Calculates conformal validity and efficiency performance results for the normalized local ellipsoidal non-conformity measure
    :param local_neighbors_test: Obtained kNN neighbors for each instance in test data
    :param y_true_test: Test data's ground truth
    :param y_pred_test: Test data's predictions
    :param y_true_train: Proper Training data's ground truth
    :param y_pred_train: Proper Training data's predictions
    :param local_alpha_s: Local $\alpha_s$ value
    :param dim: Output dimension number k
    :param lam: $\lambda$ value
    :param cov_train: Covariance matrix estimated from proper training instances
    :return: validity, efficiency
    """
         
    conditional_mu = np.array(y_pred)
    mu_pred = np.array(y_pred)
    # conditional_cov = np.linalg.inv(np.linalg.inv(cov)[2:4, 2:4])
    
    conditional_mu = np.array(y_pred)
    local_neighbors = knn.kneighbors(x, return_distance=False)
    
    result = []
    for j in range(len(conditional_mu)):
        mu = conditional_mu[j]
        dependent_data = conditional_data[j]            
        local_y_minus_y_true = (y_true - y_pred)[ local_neighbors[j, :], :]      
        local_cov_test = np.cov(local_y_minus_y_true.T)
        local_cov_test_regularized = lam * local_cov_test + (1 - lam) * cov_train
        local_inv_cov_test = np.linalg.inv(local_cov_test_regularized)
        
        local_error_test =  mu -  dependent_data  #y_true_test[i, :] - y_pred_test[i, :]
        non_conf_multi_test = ellipsoidal_non_conformity_measure(local_error_test, np.linalg.inv(cov_train)) 

        coverage = np.mean(non_conf_multi_test < local_alpha_s) * 100
        result.append( [coverage])  #[mu[0],mu[1], coverage])
        
    result_np = np.array(result)

    np.savetxt("conditional_coverage.csv", result_np, delimiter=",")
        
    return np.round([np.mean(result),np.std(result), np.min(result), np.max(result), np.quantile(result, 0.1),  np.quantile(result, 0.9)], 2)



def conformal_train(alg,x_train,x_val,x_test, y_train,y_val,y_test,cgd_train, cgd_test, n_epochs,target,algorithm,batch_size,
print_every):
    
    x_train_np, y_true_train = np.array(x_train), y_train.detach().numpy()
    x_cal, y_true_cal = np.array(x_val), y_val.detach().numpy()
    x_test, y_true_test = np.array(x_test), y_test.detach().numpy()
    

    # get number of neighbors
    n_neighbors = math.ceil(x_train_np.shape[0] * 0.15)
    # train regressor
    clf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=1337)
    ).fit(x_train_np, y_true_train)
    # get regressor's predictions
    y_pred_train = clf.predict(x_train_np)
    y_pred_cal = clf.predict(x_cal)
    y_pred_test = clf.predict(x_test)
    
         
    cov_train, alpha_s = ellipse_global_alpha_s(
        y_true_train, y_pred_train, y_true_cal, y_pred_cal, 1 - target
    )
    

    # 3 - Standard global ellipsoidal NCM (ellipsoid)
    if alg == 'conformal':
        
        train_conditional_cov = 3198 #  get_conditional_cov(y_true_train, y_pred_train, clf, cgd_train, cov_train, alpha_s)
        test_conditional_cov, freq_list_conf = get_conditional_cov(y_true_test, y_pred_test, clf, cgd_test, cov_train, alpha_s)
        
        # print(train_conditional_cov, test_conditional_cov)
        
                
        def get_cvar_stat_cov(y_true, y_pred, x, cov_train, alpha_s, target):
            y_true_test = y_true
            y_pred_test = y_pred
            x_test = x
            
            inv_cov_train = np.linalg.inv(cov_train)
            error_test = (y_true_test - y_pred_test).detach().numpy()
            non_conf_multi_test = ellipsoidal_non_conformity_measure(error_test, inv_cov_train)
            
            
            coverage = np.mean(non_conf_multi_test < alpha_s) * 100
            # standard_ellipse_validity_all.append(np.mean(non_conf_multi_test < alpha_s) * 100)
                   
            radius = alpha_s
            cov_mat = torch.tensor(np.tile(np.sqrt(radius)*cov_train, (len(y_pred_test),1,1)))
            mean = torch.tensor(y_pred_test)
            
            cvar = optimization_conformal(mean, cov_mat, radius, y_true_test, target)
            
            assign_flag = [1 if i <= alpha_s else 0 for i in non_conf_multi_test]
  
            
            test_stat =  3198  # get_p_value_conformal(assign_flag, x_test, target)
            
            
            return cvar, test_stat, coverage
        
        # train_cvar, train_test_stat, train_cov = get_cvar_stat_cov(y_train, y_pred_train, x_train, cov_train, alpha_s, target)
        # val_cvar, val_test_stat, val_cov = get_cvar_stat_cov(y_val, y_pred_cal, x_val, cov_train, alpha_s, target)
        test_cvar, test_test_stat, test_cov = get_cvar_stat_cov(y_test, y_pred_test, x_test, cov_train, alpha_s, target)
        
        return train_conditional_cov, test_conditional_cov, test_cvar.item(), test_test_stat, test_cov
        
    
    
    # 4 - Normalized local ellipsoidal NCM (ellipsoid)
    elif alg == 'conditional_conformal':
        knn, local_alpha_s = ellipse_local_alpha_s(
            x_train,
            x_cal,
            y_true_train,
            y_pred_train,
            y_true_cal,
            y_pred_cal,
            1 - target,
            n_neighbors,
            lam,
            cov_train,
        )
        
        
        def get_cvar_stat_cov(y_true, y_pred, x, cov_train, alpha_s, target):
            y_true_test = y_true
            y_pred_test = y_pred
            x_test = x
            
            
            train_conditional_cov = None
            # get_local_conditional_cov(
            #     knn,
            #     x_train,
            #     y_pred_train,
            #     y_true_train,                
            #     local_alpha_s,
            #     lam,
            #     cov_train, cgd_train
            # )
           
            test_conditional_cov, freq_list_conf = get_local_conditional_cov(
                knn,
                x_test,
                y_pred,
                y_true,                
                local_alpha_s,
                lam,
                cov_train, cgd_test
            )
            
            local_neighbors_test = knn.kneighbors(x_test, return_distance=False)
            
    
            (coverage, normalized_ellipse_efficiency, binary_flag, cov_list) = local_ellipse_validity_efficiency(  local_neighbors_test,
            y_true_test,
            y_pred_test,
            y_true_train,
            y_pred_train,
            local_alpha_s,
            n_out,
            lam,
            cov_train,    )
                        
  
            radius = local_alpha_s
            cov_mat = torch.tensor(np.sqrt(local_alpha_s)*np.array(cov_list))
            mean = torch.tensor(y_pred_test)
            cvar = optimization_conformal(mean, cov_mat, radius, y_true_test, target)
            assign_flag = binary_flag
            
            test_stat = 3198   # get_p_value_conformal(assign_flag, x_test, target)
            
                       
            return  train_conditional_cov, test_conditional_cov, cvar, test_stat, coverage
        
        
        # train_cvar, train_test_stat, train_cov = get_cvar_stat_cov(y_train, y_pred_train, x_train, cov_train, alpha_s, target)
        # val_cvar, val_test_stat, val_cov = get_cvar_stat_cov(y_val, y_pred_cal, x_val, cov_train, alpha_s, target)
        train_conditional_cov, test_conditional_cov, test_cvar, test_test_stat, test_cov = get_cvar_stat_cov(y_test, y_pred_test, x_test, cov_train, alpha_s, target)
        
        return  train_conditional_cov, test_conditional_cov, freq_list_conf, test_cvar.item(), test_test_stat, test_cov
    
'''    
    

    # print(binary_flag)
    
    # target = np.array(binary_flag)
    # x = np.array(x_test)
    # import statsmodels.api as sm
    # import scipy

    # x = sm.add_constant(x)
    # full_model = sm.OLS(target, x).fit()

    # #calculate log-likelihood of model
    # full_ll = full_model.llf
    # print(full_ll)
    # print(x)
    # x1 = x.T[0]
    
    # reduced_model = sm.OLS(target, x1).fit()

    # #calculate log-likelihood of model
    # reduced_ll = reduced_model.llf
    # print(reduced_ll)
    # #calculate likelihood ratio Chi-Squared test statistic
    # LR_statistic = -2*(reduced_ll-full_ll)
    
    # print(LR_statistic)
        
    # #calculate p-value of test statistic using 2 degrees of freedom
    # p_val = scipy.stats.chi2.sf(LR_statistic, 2)
    
    # print(p_val)
        
    


#     normalized_ellipse_validity_all.append(normalized_ellipse_validity)
#     normalized_ellipse_efficiency_all.append(normalized_ellipse_efficiency)
    


#     title = "normalized_ellipse_synthetic.jpg"
#     plot_ellipse_local(
#         title,
#         max_points,
#         y_true_test,
#         y_pred_test,
#         alpha_s,
#         y_true_train,
#         y_pred_train,
#         local_neighbors_test,
#     )
    
    # print("=================================")
    # print("RESULTS")
    # print("=================================")
    # print("epsilon", epsilon)
    # print("====================")
    
    # print(
    #     "Standard empirical validity : $"
    #     + str(np.mean(standard_empirical_validity_all))
    #     + " \pm "
    #     + str(np.std(standard_empirical_validity_all))
    #     + "$"
    # )
    # print(
    #     "Standard empirical efficiency : $"
    #     + str(np.mean(standard_empirical_efficiency_all))
    #     + " \pm "
    #     + str(np.std(standard_empirical_efficiency_all))
    #     + "$"
    # )
    # print("====================")
    
    # print(
    #     "Normalized empirical validity : $"
    #     + str(np.mean(normalized_empirical_copula_validity_all))
    #     + " \pm "
    #     + str(np.std(normalized_empirical_copula_validity_all))
    #     + "$"
    # )
    # print(
    #     "Normalized empirical efficiency : $"
    #     + str(np.mean(normalized_empirical_copula_efficiency_all))
    #     + " \pm "
    #     + str(np.std(normalized_empirical_copula_efficiency_all))
    #     + "$"
    # )
    # print("====================")
    
    # print(
    #     "Standard ellipse validity : $"
    #     + str(np.mean(standard_ellipse_validity_all))
    #     + " \pm "
    #     + str(np.std(standard_ellipse_validity_all))
    #     + "$"
    # )
    # print(
    #     "Standard ellipse efficiency : $"
    #     + str(np.mean(standard_ellipse_efficiency_all))
    #     + " \pm "
    #     + str(np.std(standard_ellipse_efficiency_all))
    #     + "$"
    # )
    # print("====================")
    
    # print(
    #     "Normalized ellipse validity : $"
    #     + str(np.mean(normalized_ellipse_validity_all))
    #     + " \pm "
    #     + str(np.std(normalized_ellipse_validity_all))
    #     + "$"
    # )
    # print(
    #     "Normalized ellipse efficiency : $"
    #     + str(np.mean(normalized_ellipse_efficiency_all))
    #     + " \pm "
    #     + str(np.std(normalized_ellipse_efficiency_all))
    #     + "$"
    # )
    # print("====================")
    # print("====================")



if __name__ == '__main__': 
    conformal_train()
