import sys
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import torch
from sqwash import SuperquantileReducer, SuperquantileSmoothReducer


from utils import load_data, compute_train_test_split, train_test_scaler, DeepNormalModel, train, test, dist_loss, uniqueid_generator, cvx_layer_onestep_for_conformal

sys.path.append("../")
sys.path.append("../../")

from utilities.simulate import generate_synthetic_data

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
lam = 0.95
n_sample = 50000
n_kfold = 1
epsilon = 0.05
max_points = 30

standard_empirical_validity_all = []
standard_empirical_efficiency_all = []
normalized_empirical_copula_validity_all = []
normalized_empirical_copula_efficiency_all = []
standard_ellipse_validity_all = []
standard_ellipse_efficiency_all = []
normalized_ellipse_validity_all = []
normalized_ellipse_efficiency_all = []

def ellipsoidal_non_conformity_measure(error, inv_cov):
    """
    Calculates the ellipsoidal non-conformity score
    :param error: Vector $(y_i - \hat{y_i})$
    :param inv_cov: Inverse-covariance matrix
    :return: Ellipsoidal non-conformity score
    """
    return np.sqrt(np.sum(error.T * (inv_cov @ error.T), axis=0))

exp_type = 'simulated'

result = []
for k in np.arange(0, 100, 1):
    # prepare train, cal and test data
    df, aux1 = load_data(exp_type= 'simulated', sim_data_params = [0, 4, k])
    
    x_train, x_val, x_test, y_train, y_val, y_test = compute_train_test_split(df, aux1, exp_type=exp_type)
    x_scaler, x_val_scaler, x_test_scaler = train_test_scaler(x_train, x_val, x_test)
    
    x_train, y_true_train =  np.array(x_scaler), np.array(y_train)
    x_cal, y_true_cal = np.array(x_val_scaler), np.array(y_val)
    x_test, y_true_test = np.array(x_test_scaler), np.array(y_test)


    # data = generate_synthetic_data(n_sample)
    # x_train, y_true_train = data["train"]
    # x_cal, y_true_cal = data["cal"]
    # x_test, y_true_test = data["test"]
    
    # get number of neighbors
    n_neighbors = math.ceil(x_train.shape[0] * 0.05)
    # train regressor
    clf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=1337)
    ).fit(x_train, y_true_train)
    # get regressor's predictions
    y_pred_train = clf.predict(x_train)
    y_pred_cal = clf.predict(x_cal)
    y_pred_test = clf.predict(x_test)

    '''
    # 1 - Standard empirical Copula (hyper-rectangle)
    alphas = std_emp_conf_all_targets_alpha_s(y_true_cal, y_pred_cal, epsilon=epsilon)
    conf_test_preds = std_emp_conf_predict(y_pred_test, alphas)
    standard_empirical_validity_all.append(
        empirical_conf_validity(conf_test_preds, y_true_test)
    )
    standard_empirical_efficiency_all.append(empirical_conf_efficiency(conf_test_preds))


    # title = "standard_empirical_synthetic.eps"
    # plot_standard_rectangle(title, max_points, y_true_test, y_pred_test, alphas)

    # 2 - Normalized empirical Copula (hyper-rectangle)
    x_train_norm, x_val_norm, y_true_train_norm, y_true_val_norm = train_test_split(
        x_train, y_true_train, test_size=0.1, random_state=1337
    )

    y_pred_train_norm = clf.predict(x_train_norm)
    y_pred_val_norm = clf.predict(x_val_norm)

    y_true_train_norm, y_true_val_norm = prepare_norm_data(
        y_true_train_norm, y_true_val_norm, y_pred_train_norm, y_pred_val_norm
    )

    checkpoint = ModelCheckpoint(
        "multi.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=1)
    redonplat = ReduceLROnPlateau(
        monitor="val_loss", mode="min", patience=10, verbose=2
    )
    callbacks_list = [checkpoint, early, redonplat]

    norm_model = simple_mlp(n_continuous=2, n_outputs=n_out)

    norm_model.fit(
        x_train_norm,
        y_true_train_norm,
        validation_data=(x_val_norm, y_true_val_norm),
        epochs=400,
        verbose=2,
        callbacks=callbacks_list,
        batch_size=256,
    )

    try:
        norm_model.load_weights("multi.h5")
    except:
        pass

    mu_cal = norm_model.predict(x_cal)
    mu_test = norm_model.predict(x_test)

    alphas = norm_emp_conf_all_targets_alpha_s(
        y_true_cal, y_pred_cal, epsilon=epsilon, mu=mu_cal, beta=0.1
    )
    norm_conf_test_preds, norm_alphas = norm_emp_conf_predict(
        y_pred_test, mu_test, alphas, beta=0.1
    )
    normalized_empirical_copula_validity_all.append(
        empirical_conf_validity(norm_conf_test_preds, y_true_test)
    )
    normalized_empirical_copula_efficiency_all.append(
        empirical_conf_efficiency(norm_conf_test_preds)
    )

    title = "normalized_empirical_synthetic.eps"
    plot_normalized_rectangle(title, max_points, y_true_test, y_pred_test, norm_alphas)

    '''
    # 3 - Standard global ellipsoidal NCM (ellipsoid)
    cov_train, alpha_s = ellipse_global_alpha_s(
        y_true_train, y_pred_train, y_true_cal, y_pred_cal, epsilon
    )
    inv_cov_train = np.linalg.inv(cov_train)
    
    global_cov = torch.tensor(cov_train)
    

    error_test = y_true_test - y_pred_test
    non_conf_multi_test = ellipsoidal_non_conformity_measure(error_test, inv_cov_train)
    

    standard_ellipse_validity_all.append(np.mean(non_conf_multi_test < alpha_s) * 100)
    standard_ellipse_efficiency_all.append(
        ellipse_volume(inv_cov_train, alpha_s, n_out)
    )
    
    standard_target_flag = [1 if i < alpha_s else 0 for i in non_conf_multi_test]
    
    
 
    # title = "standard_ellipse_synthetic.eps"
    # plot_ellipse_global(title, max_points, y_true_test, y_pred_test, alpha_s, cov_train)

    # 4 - Normalized local ellipsoidal NCM (ellipsoid)
    knn, local_alpha_s = ellipse_local_alpha_s(
        x_train,
        x_cal,
        y_true_train,
        y_pred_train,
        y_true_cal,
        y_pred_cal,
        epsilon,
        n_neighbors,
        lam,
        cov_train,
    )

    local_neighbors_test = knn.kneighbors(x_test, return_distance=False)

    (
        normalized_ellipse_validity,
        normalized_ellipse_efficiency,
    ) = local_ellipse_validity_efficiency(
        local_neighbors_test,
        y_true_test,
        y_pred_test,
        y_true_train,
        y_pred_train,
        local_alpha_s,
        n_out,
        lam,
        cov_train,
    )

    normalized_ellipse_validity_all.append(normalized_ellipse_validity)
    normalized_ellipse_efficiency_all.append(normalized_ellipse_efficiency)
    
    local_non_conf_multi_test_all = []
    for i in range(local_neighbors_test.shape[0]):
        local_y_minus_y_true = (y_true_train - y_pred_train)[
            local_neighbors_test[i, :], :
        ]
            

        local_cov_test = np.cov(local_y_minus_y_true.T)
        
        local_cov_test_regularized = lam * local_cov_test + (1 - lam) * cov_train
        local_inv_cov_test = np.linalg.inv(local_cov_test_regularized)

        local_error_test = y_true_test[i, :] - y_pred_test[i, :]
        
        
        
        local_non_conf_multi_test_all.append(
            ellipsoidal_non_conformity_measure(local_error_test, local_inv_cov_test)
        )

    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    import scipy

    x = sm.add_constant(x_test_scaler)
    
    
    # print(local_non_conf_multi_test_all)
    target_flag = [1 if i < local_alpha_s else 0 for i in local_non_conf_multi_test_all]
    
    # print(sum(target_flag))
    # print(sum(standard_target_flag))
    
    #fit regression model
    full_model = sm.OLS(target_flag, x).fit()
    #calculate log-likelihood of model
    full_ll = full_model.llf
    
    #fit regression model
    reduced_model = sm.OLS(target_flag, math.log(9)*x['const']).fit()
    #calculate log-likelihood of model
    reduced_ll = reduced_model.llf
    
    #calculate likelihood ratio Chi-Squared test statistic
    LR_statistic = -2*(reduced_ll-full_ll)
    #calculate p-value of test statistic using 2 degrees of freedom
    p_val = scipy.stats.chi2.sf(LR_statistic, 3)
    
    
    
    #fit regression model
    full_model_st = sm.OLS(standard_target_flag, x).fit()
    #calculate log-likelihood of model
    full_ll_st = full_model_st.llf
    
    #fit regression model
    reduced_model_st = sm.OLS(standard_target_flag, math.log(9)*x['const']).fit()
    #calculate log-likelihood of model
    reduced_ll_st = reduced_model_st.llf
    
    #calculate likelihood ratio Chi-Squared test statistic
    LR_statistic_st = -2*(reduced_ll_st-full_ll_st)
    #calculate p-value of test statistic using 2 degrees of freedom
    p_val_st = scipy.stats.chi2.sf(LR_statistic_st, 3)
    
    print(k,LR_statistic, p_val, LR_statistic_st, p_val_st )
    result.append(np.round([k,LR_statistic, p_val, LR_statistic_st, p_val_st], 3))
    

print(result)
    
import sys
sys.exit()
    
















    
#     z_prev = torch.abs(torch.randn_like(torch.tensor(y_true_test)))
#     z_prev = z_prev / z_prev.sum(dim=-1).unsqueeze(-1)
#     steps = 100
#     target = 1 - epsilon
#     b = torch.tensor(y_pred_test)
#     cov_mat = []
#     global_cov_mat = []
#     for i in range(local_neighbors_test.shape[0]):
#         local_y_minus_y_true = (y_true_train - y_pred_train)[
#             local_neighbors_test[i, :], :
#         ]
            

#         local_cov_test = np.cov(local_y_minus_y_true.T)
        
#         cov_mat.append(local_cov_test)
    

#     # A = torch.tensor(cov_mat)
#     A = torch.tensor(global_cov)
    
#     rad = alpha_s
#     w_tmp, task_loss, status = cvx_layer_onestep_for_conformal( z_prev, b, A, rad, steps, target)
    
#     w_tmp = torch.stack(w_tmp)
#     w_tmp.retain_grad()
#     prod = w_tmp.mul(torch.tensor(y_true_test))
#     port_values = torch.sum(prod, dim=1)
    
#     reducer = SuperquantileReducer(superquantile_tail_fraction= target)
#     cvar_loss = reducer(port_values)
    
#     loss1 = cvar_loss
    
    
    
#     quant = int(target*len(port_values)) + 1
    
#     port_sorted = torch.sort(port_values, descending=True)[0]
#     quant = port_sorted[quant]

#     port_le_quant = port_values.le(quant).float()
#     port_le_quant.requires_grad = True
#     loss2 =   port_values.mul(port_le_quant).sum() / port_le_quant.sum()
    
   
    
    
    

#     # title = "normalized_ellipse_synthetic.eps"
#     # plot_ellipse_local(
#     #     title,
#     #     max_points,
#     #     y_true_test,
#     #     y_pred_test,
#     #     alpha_s,
#     #     y_true_train,
#     #     y_pred_train,
#     #     local_neighbors_test,
#     # )

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
