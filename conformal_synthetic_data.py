import sys
import math
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import torch 
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from sklearn.svm import SVC # import SVC model
import torch.nn.functional as F
from torch import nn,optim
from utils import optimization_conformal

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
        

def get_p_value_conformal(assign_flag, x_scaler, target):
    
    if torch.is_tensor(assign_flag):
        y_hat_original = torch.tensor(assign_flag).double()
    else:
        y_hat_original = torch.tensor(assign_flag).double().unsqueeze(1)
        
    x = torch.tensor(np.array(x_scaler))
    
  
    test_stat_bootstrap = []
    predicted_target = y_hat_original
    
    model = SVC(kernel = 'rbf',probability=True, random_state = 0)
   
    model.fit(x, predicted_target)
    
    # Get the best model and its corresponding parameters
    best_model = model  #grid_search.best_estimator_
    
   
    # Evaluate the best model on the test set and predict probabilities
    y_new_flag = best_model.predict(x)
    y_new = best_model.predict_proba(x)[:,1]

    
    del model
    RSS_0 =  (predicted_target -  target)**2  #(predicted_target -  math.log(target/(1-target)))**2
    RSS_0_sum = sum(RSS_0)
    
    RSS_A =  (np.subtract(predicted_target.flatten() , y_new))**2   #(np.subtract(predicted_target.reshape(-1,1) , y_new_flag))**2 
    RSS_A_sum = sum(RSS_A)
    
    
    test_stat_original = (x.shape[0]/2)*math.log(RSS_0_sum/RSS_A_sum)
    
    RSS_A_new = sum((np.subtract(target , y_new))**2 )

    test_stat_original_new = RSS_A_new    #(x.shape[0]/2)*math.log(RSS_0_sum/RSS_A_new)
   
    prob_np = np.tile(target, (len(y_new),1))  
    # prob_np = np.array(y_new.detach().numpy())
                
    test_stat_bootstrap = []
    test_stat_new_bootstramp = []
    

    
    for i in tqdm(range(100)):
        values = np.random.binomial(1, prob_np)
       
        pred = np.array(values) #.reshape(-1)
        
        model_bs = SVC(kernel = 'rbf',probability=True, random_state = 0)
        
        model_bs.fit(x, pred)
        
               
        # Get the best model and its corresponding parameters
       
        # Evaluate the best model on the test set and predict probabilities
        y_new = model_bs.predict_proba(x)[:,1]
    
        del model_bs
        
        RSS_0 = (pred -  target)**2
        RSS_0_sum = sum(RSS_0)
        
        RSS_A = (np.subtract(pred.flatten() , y_new))**2 #(np.subtract(pred.reshape(-1,1) , y_new_flag))**2 
        RSS_A_sum = sum(RSS_A)                                                                                            
        
        test_stat = (x.shape[0]/2)*math.log(RSS_0_sum/RSS_A_sum)
        test_stat_bootstrap.append(test_stat)
        
        RSS_A_new = sum((np.subtract(target , y_new))**2 )
        test_stat_new =  RSS_A_new  #(x.shape[0]/2)*math.log(RSS_0_sum/RSS_A_new)
        
        test_stat_new_bootstramp.append(test_stat_new)
    
    
    p_value = (sum(test_stat_original <= i  for i in test_stat_bootstrap))/len(test_stat_bootstrap)
    p_val_new = (sum(test_stat_original_new <= i  for i in test_stat_new_bootstramp))/len(test_stat_new_bootstramp)
   
    print(p_value, p_val_new)

    return  [p_value,p_val_new]  # p_value



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
           
            result.append( [coverage])  
    
    
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
    
    conditional_mu = np.array(y_pred_test)
    local_neighbors = knn.kneighbors(x, return_distance=False)
    
    if conditional_data is not None:
    
        result = []
        for j in  range(len(conditional_mu)):

            dependent_data = conditional_data[j]
    
            y_pred_test_j = y_pred_test[j]
           
            local_y_minus_y_true = y_true_train[local_neighbors[j, :],:] - y_pred_test_j
    
            local_cov_test = (1/(local_y_minus_y_true.shape[0]-1))*local_y_minus_y_true.T@local_y_minus_y_true
            local_cov_test_regularized = lam * local_cov_test + (1 - lam) * cov_train
            local_inv_cov_test = np.linalg.inv(local_cov_test_regularized)
            local_error_test = dependent_data - y_pred_test_j
            local_non_conf_multi_test_all = ellipsoidal_non_conformity_measure(local_error_test, local_inv_cov_test)
           
            coverage = np.mean(local_non_conf_multi_test_all < local_alpha_s) * 100
        
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

        local_y_minus_y_true = y_true_train[local_neighbors_test[i, :],:] - y_pred_i

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


def get_cvar_stat_cov(y_true, y_pred, x, cov_train, alpha_s, target):
    y_true_test = y_true
    y_pred_test = y_pred
    x_test = x
    
    inv_cov_train = np.linalg.inv(cov_train)
    error_test = (y_true_test - y_pred_test)  #.detach().numpy()
    non_conf_multi_test = ellipsoidal_non_conformity_measure(error_test, inv_cov_train)
               
    radius = alpha_s

    cov_mat = torch.tensor(np.tile(cov_train, (len(y_pred_test),1,1)))
    cholesky = torch.cholesky(cov_mat)
    mean = torch.tensor(y_pred_test)

    torch_data = {
        'y_test': y_true,
        'alpha': y_pred,
        'S': cov_mat,
        'radius': radius
    }
    
    
    cvar, port_values = optimization_conformal(mean, cholesky, radius, torch.tensor(y_true_test), target)    
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

    cov_train, alpha_s = ellipse_global_alpha_s(
        y_true_train, y_pred_train, y_true_cal, y_pred_cal, 1 - target)
    
    print(alpha_s)
  

    if alg == 'conformal':

        test_conditional_cov,cov_freq_dist = get_conditional_cov(y_true_test, y_pred_test, cgd_test, cov_train, alpha_s)

        test_overall_cov = get_overall_cov(y_true_test, y_pred_test, x_test, cov_train, alpha_s, target)

        test_cvar,port_values, test_stat, torch_data = get_cvar_stat_cov(y_true_test, y_pred_test, x_test, cov_train, alpha_s, target)


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
            cov_mat = torch.tensor(np.array(cov_list))
            mean = torch.tensor(y_pred_test)
            cholesky = torch.cholesky(cov_mat)

            torch_data = {
        'y_test': y_test,
        'alpha': y_pred,
        'S': cov_mat,
        'radius': radius
    }
            
            print("here")
            
            cvar, port_values = optimization_conformal(mean, cov_mat, radius, y_true_test, target)
            assign_flag = binary_flag

            test_stat = 3198 
            
                       
            return  cvar, port_values, test_stat, torch_data
        
        test_cvar,port_values, test_stat, torch_data = get_cvar_stat_conditional_conf(y_test, y_pred_test, x_test, cov_train, alpha_s, target)

    train_conditional_cov = 999
    return train_conditional_cov, test_conditional_cov,cov_freq_dist, test_cvar.item(),port_values, test_stat, test_overall_cov, torch_data



if __name__ == '__main__': 
    conformal_train()
