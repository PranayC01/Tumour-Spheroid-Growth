import numpy as np
from scipy.optimize import minimize_scalar

def profile_likelihood(data, params, func):
    """
    Computes the profile likelihood of a given set of data and parameters.
    
    Parameters:
        data (numpy array): The set of data to be analyzed.
        params (list or numpy array): The parameters of the model being analyzed.
        func (function): The function used to model the data.
        
    Returns:
        negative log-likelihood (float): The negative log-likelihood of the model given the data and parameters.
    """
    return -np.sum(np.log(func(data, params)))

def find_confidence_interval(data, params, func, param_index, interval=0.95):
    """
    Finds the profile likelihood confidence interval for a given parameter.
    
    Parameters:
        data (numpy array): The set of data to be analyzed.
        params (list or numpy array): The parameters of the model being analyzed.
        func (function): The function used to model the data.
        param_index (int): The index of the parameter for which the confidence interval is being calculated.
        interval (float): The desired confidence interval (default is 0.95).
        
    Returns:
        lower_bound (float): The lower bound of the confidence interval.
        upper_bound (float): The upper bound of the confidence interval.
    """
    def neg_log_likelihood(param):
        params_copy = params.copy()
        params_copy[param_index] = param
        return profile_likelihood(data, params_copy, func)
    
    res = minimize_scalar(neg_log_likelihood)
    best_fit = res.x
    best_likelihood = res.fun
    
    confidence_level = 1 - (1 - interval) / 2
    threshold = -np.log(confidence_level)
    
    lower_bound = best_fit
    upper_bound = best_fit
    step = best_fit / 10
    
    while profile_likelihood(data, params[param_index] = lower_bound, func) > best_likelihood + threshold:
        lower_bound -= step
        
    while profile_likelihood(data, params[param_index] = upper_bound, func) > best_likelihood + threshold:
        upper_bound += step
        
    return lower_bound, upper_bound
