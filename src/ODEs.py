import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.stats import chi2
import timeit

#------------------------------------------------------------------------------------------------------------------------------------------#
'''
def Mendelsohn(t, v, a, b):
    return a*(v**b)

# Special case of Mendelsohn (Power Law) Model with b=1.
def Exponential(t, v, a):
    return a*v

def Logistic(t, v, r, k):
    return r*v*(1-v/k)

def Gompertz(t, v, r, k):
    return r*np.log(k/v)*v

def Bertalanffy(t, v, b, d):
    return b*(v**(2/3))-d*v

#------------------------------------------------------------------------------------------------------------------------------------------#

# Analytical solution of Mendelsohn ODE (theta = V_0, a, b)
def mend_sol(theta, t):
    return ((1 - theta[2])*(theta[1]*t + (theta[0]**(1-theta[2]))/(1-theta[2])))**(1/(1-theta[2]))

# Analytical solution of Exponential ODE (theta = V_0, a)
def exp_sol(theta, t):
    return theta[0]*np.exp(theta[1]*t)

# Analytical solution of Logistic ODE (theta = V_0, r, K)
def log_sol(theta, t):
    return (theta[0]/((theta[0]/theta[2])+(1 - theta[0]/theta[2])*np.exp(-theta[1]*t)))

# Analytical solution of Gompertz ODE (theta = V_0, r, K)
def gomp_sol(theta, t):
    return theta[2]*(theta[0]/theta[2])**np.exp(-theta[1]*t)

# Analytical solution of Bertalanffy ODE (theta = V_0, b, d)
def bert_sol(theta, t):
    return ((theta[1]/theta[2])*(1 - np.exp(-theta[2]*t/3)) + (theta[0]**(1/3))*np.exp(-theta[2]*t/3))**3

#------------------------------------------------------------------------------------------------------------------------------------------#


# Numerical solution to Exponential ODE, theta = [V0, a]
def exp_num_sol(theta):
    return solve_ivp(Exponential, [0, 10], [theta[0]], args = (theta[1],), t_eval = np.linspace(0, 10, 101))

# Numerical solution to Mendelsohn ODE, theta = [V0, a, b]
def mend_num_sol(theta): 
    return solve_ivp(Mendelsohn, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))

# Numerical solution to Logistic ODE, theta = [V0, r, K]
def log_num_sol(theta): 
    return solve_ivp(Logistic, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))

# Numerical solution to Gompertz ODE, theta = [V0, r, K]
def gomp_num_sol(theta): 
    return solve_ivp(Gompertz, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))

# Numerical solution to Bertalanffy ODE, theta = [V0, b, d]
def bert_num_sol(theta):
    return solve_ivp(Bertalanffy, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))
#------------------------------------------------------------------------------------------------------------------------------------------#

t_eval = np.linspace(0, 10, 101)
def exp_sol_noise(noise):
    return exp_sol([0.01, 1], t_eval) * (1 + noise * np.random.normal(0,1,101))
def mend_sol_noise(noise):
    return mend_sol([0.01, 1, 2], t_eval) * (1 + noise * np.random.normal(0,1,101))
def log_sol_noise(noise):
    return log_sol([0.01, 1, 10], t_eval) * (1 + noise * np.random.normal(0,1,101))
def gomp_sol_noise(noise):
    return gomp_sol([0.01, 1, 10], t_eval) * (1 + noise * np.random.normal(0,1,101))
def bert_sol_noise(noise):
    return bert_sol([0.01, 1, 1], t_eval) * (1 + noise * np.random.normal(0,1,101))
def bert_growth_sol_noise(noise):
    return bert_sol([0.01, 2, 1], t_eval) * (1 + noise * np.random.normal(0,1,101))
def bert_decay_sol_noise(noise):
    return bert_sol([0.01, 1, 2], t_eval) * (1 + noise * np.random.normal(0,1,101))

#------------------------------------------------------------------------------------------------------------------------------------------#


def exp_res(theta, noise):
    return exp_sol(theta, t_eval) - exp_sol_noise(noise)

def mend_res(theta, noise):
    return mend_sol(theta, t_eval) - mend_sol_noise(noise)

def log_res(theta, noise):
    return log_sol(theta, t_eval) - log_sol_noise(noise)

def gomp_res(theta, noise):
    return gomp_sol(theta, t_eval) - gomp_sol_noise(noise)

def bert_res(theta, noise):
    return bert_sol(theta, t_eval) - bert_sol_noise(noise)

def bert_growth_res(theta, noise):
    return bert_sol(theta, t_eval) - bert_growth_sol_noise(noise)
    
def bert_decay_res(theta, noise):
    return bert_sol(theta, t_eval) - bert_decay_sol_noise(noise)

#------------------------------------------------------------------------------------------------------------------------------------------#
    
def exp_consts(noise):
    theta0 = [0.01, 1]
    return least_squares(exp_res, theta0, args = (0.05))

def mend_consts(noise):
    theta0 = [0.01, 1, 2]
    return least_squares(mend_res, theta0, args = (0.05))

def log_consts(noise):
    theta0 = [0.01, 1, 10]
    return least_squares(log_res, theta0, args = (0.05))

def gomp_consts(noise):
    theta0 = [0.01, 1, 10]
    return least_squares(gomp_res, theta0, args = (0.05))

def bert_consts(noise):
    theta0 = [0.01, 1, 1]
    return least_squares(bert_res, theta0, args = (0.05))

def bert_growth_consts(noise):
    theta0 = [0.01, 2, 1]
    return least_squares(bert_growth_res, theta0, args = (0.05))

def bert_decay_consts(noise):
    theta0 = [0.01, 1, 2]
    return least_squares(bert_decay_res, theta0, args = (0.05))

#------------------------------------------------------------------------------------------------------------------------------------------#
# Least Squares

print("Least Squares estimates:")
print("Exponential Model: ", "[V0, a] = ", exp_consts(0.05).x)
print("Mendelsohn Model: ", "[V0, a, b] = ", mend_consts(0.05).x)
print("Logistic Model: ", "[V0, r, K] = ", log_consts(0.05).x)
print("Gompertz Model: ", "[V0, r, K] = ", gomp_consts(0.05).x)
print("Bertalanffy Model: ", "[V0, b, d] = ", bert_consts(0.05).x)
print("Bertalanffy Growth Model: ", "[V0, b, d] = ", bert_growth_consts(0.05).x)
print("Bertalanffy Decay Model: ", "[V0, b, d] = ", bert_decay_consts(0.05).x)



#------------------------------------------------------------------------------------------------------------------------------------------#

# Profile Likelihood
'''
n=101
# g is analytical solution for an ODE, v = g*(1 + noise*N(0,1)).

def log_l(theta, g, v, noise):
    log_l = -(n/2)*np.log(noise**2*(2*np.pi)) -n/(2*noise**2) - (1/(2*noise**2))*np.sum(np.square(v)/np.square(g(theta).y[0]) - 2*v/g(theta).y[0]) - np.sum(np.log(g(theta).y[0]))
    return log_l

def neg_log_l(theta, g, v, noise):
    return -log_l(theta, g, v, noise)
'''
#print(exp_num_sol.y[0])
#print(exp_sol_noise)

log_l([0.01, 1], exp_num_sol, exp_sol_noise, 0.05)
log_l([0.01, 1, 2], mend_num_sol, mend_sol_noise, 0.05)
log_l([0.01, 1, 10], log_num_sol, log_sol_noise, 0.05)
log_l([0.01, 1, 10], gomp_num_sol, gomp_sol_noise, 0.05)
log_l([0.01, 1, 1], bert_num_sol, bert_sol_noise, 0.05)

def exp_mle(noise): 
    return minimize(neg_log_l, [0.01, 1], method = 'Nelder-Mead', args=(exp_num_sol, exp_sol_noise(noise), noise))
def mend_mle(noise): 
    return minimize(neg_log_l, [0.01, 1, 2], method = 'Nelder-Mead', args=(mend_num_sol, mend_sol_noise(noise), noise))
#Note: No. of max iterations exceeded without convergence. Can use options={'maxiter':1000}
def log_mle(noise): 
    return minimize(neg_log_l, [0.01, 1, 10], method = 'Nelder-Mead', args=(log_num_sol, log_sol_noise(noise), noise))
def gomp_mle(noise):
    return minimize(neg_log_l, [0.01, 1, 10], method = 'Nelder-Mead', args=(gomp_num_sol, gomp_sol_noise(noise), noise))
def bert_mle(noise):
    return minimize(neg_log_l, [0.01, 1, 1], method = 'Nelder-Mead', args=(bert_num_sol, bert_sol_noise(noise), noise))

print("Maximum likelihood estimates:")
print("Exponential Model: ", "[V0, a] = ", exp_mle(noise).x)
print("Mendelsohn Model: ", "[V0, a, b] = ", mend_mle(noise).x)
print("Logistic Model: ", "[V0, r, K] = ", log_mle(noise).x)
print("Gompertz Model: ", "[V0, r, K] = ", gomp_mle(noise).x)
print("Bertalanffy Model: ", "[V0, b, d] = ", bert_mle(noise).x)
print(log_l(exp_mle(noise).x, exp_num_sol, exp_sol_noise, noise))
print(log_l(exp_mle(noise).x, exp_num_sol, exp_sol_noise, noise) + exp_mle(noise).fun)


#------------------------------------------------------------------------------------------------------------------------------------------#

# Profile Likelihood confidence intervals

def exp_CI(confidence, noise, param):
    df = 1
    if param == "a":
        def test(a):
            return -exp_mle(noise).fun - log_l([exp_mle(noise).x[0], a], exp_num_sol, exp_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*exp_mle(noise).x[1], x1 = 0.9*exp_mle(noise).x[1])
        root2 = root_scalar(test, x0 = 1.5*exp_mle(noise).x[1], x1 = 1.1*exp_mle(noise).x[1])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    elif param == "V0":
        def test(V0):
            return -exp_mle(noise).fun - log_l([V0, exp_mle(noise).x[1]], exp_num_sol, exp_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*exp_mle(noise).x[0], x1 = 0.9*exp_mle(noise).x[0])
        root2 = root_scalar(test, x0 = 1.5*exp_mle(noise).x[0], x1 = 1.1*exp_mle(noise).x[0])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    else:
        print("Check param value")

def log_CI(confidence, noise, param):
    df = 1
    if param == "r":
        def test(r):
            return -log_mle(noise).fun - log_l([log_mle(noise).x[0], r, log_mle(noise).x[2]], log_num_sol, log_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*log_mle(noise).x[1], x1 = 0.9*log_mle(noise).x[1])
        root2 = root_scalar(test, x0 = 1.5*log_mle(noise).x[1], x1 = 1.1*log_mle(noise).x[1])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    elif param == "K":
        def test(K):
            return -log_mle(noise).fun - log_l([log_mle(noise).x[0], log_mle(noise).x[1], K], log_num_sol, log_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*log_mle(noise).x[2], x1 = 0.9*log_mle(noise).x[2])
        root2 = root_scalar(test, x0 = 1.5*log_mle(noise).x[2], x1 = 1.1*log_mle(noise).x[2])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    elif param == "V0":
        def test(V0):
            return -log_mle(noise).fun - log_l([V0, log_mle(noise).x[1], log_mle(noise).x[2]], log_num_sol, log_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*log_mle(noise).x[0], x1 = 0.9*log_mle(noise).x[0])
        root2 = root_scalar(test, x0 = 1.5*log_mle(noise).x[0], x1 = 1.1*log_mle(noise).x[0])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    else:
        print("Check param value")

def gomp_CI(confidence, noise, param):
    df = 1
    if param == "r":
        def test(r):
            return -gomp_mle(noise).fun - log_l([gomp_mle(noise).x[0], r, gomp_mle(noise).x[2]], gomp_num_sol, gomp_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*gomp_mle(noise).x[1], x1 = 0.9*gomp_mle(noise).x[1])
        root2 = root_scalar(test, x0 = 1.5*gomp_mle(noise).x[1], x1 = 1.1*gomp_mle(noise).x[1])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    elif param == "K":
        def test(K):
            return -gomp_mle(noise).fun - log_l([gomp_mle(noise).x[0], gomp_mle(noise).x[1], K], gomp_num_sol, gomp_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*gomp_mle(noise).x[2], x1 = 0.9*gomp_mle(noise).x[2])
        root2 = root_scalar(test, x0 = 1.5*gomp_mle(noise).x[2], x1 = 1.1*gomp_mle(noise).x[2])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    elif param == "V0":
        def test(V0):
            return -gomp_mle(noise).fun - log_l([V0, gomp_mle(noise).x[1], gomp_mle(noise).x[2]], gomp_num_sol, gomp_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*gomp_mle(noise).x[0], x1 = 0.9*gomp_mle(noise).x[0])
        root2 = root_scalar(test, x0 = 1.5*gomp_mle(noise).x[0], x1 = 1.1*gomp_mle(noise).x[0])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    else:
        print("Check param value")

def bert_CI(confidence, noise, param):
    df = 1
    if param == "b":
        def test(b):
            return -bert_mle(noise).fun - log_l([bert_mle(noise).x[0], b, bert_mle(noise).x[2]], bert_num_sol, bert_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*bert_mle(noise).x[1], x1 = 0.9*bert_mle(noise).x[1])
        root2 = root_scalar(test, x0 = 1.5*bert_mle(noise).x[1], x1 = 1.1*bert_mle(noise).x[1])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    elif param == "d":
        def test(d):
            return -bert_mle(noise).fun - log_l([bert_mle(noise).x[0], bert_mle(noise).x[1], d], bert_num_sol, bert_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*bert_mle(noise).x[2], x1 = 0.9*bert_mle(noise).x[2])
        root2 = root_scalar(test, x0 = 1.5*bert_mle(noise).x[2], x1 = 1.1*bert_mle(noise).x[2])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    elif param == "V0":
        def test(V0):
            return -bert_mle(noise).fun - log_l([V0, bert_mle(noise).x[1], bert_mle(noise).x[2]], bert_num_sol, bert_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*bert_mle(noise).x[0], x1 = 0.9*bert_mle(noise).x[0])
        root2 = root_scalar(test, x0 = 1.5*bert_mle(noise).x[0], x1 = 1.1*bert_mle(noise).x[0])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    else:
        print("Check param value")

def mend_CI(confidence, noise, param):
    df = 1
    if param == "a":
        def test(a):
            return -mend_mle(noise).fun - log_l([mend_mle(noise).x[0], a, mend_mle(noise).x[2]], mend_num_sol, mend_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*mend_mle(noise).x[1], x1 = 0.9*mend_mle(noise).x[1])
        root2 = root_scalar(test, x0 = 1.5*mend_mle(noise).x[1], x1 = 1.1*mend_mle(noise).x[1])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    elif param == "b":
        def test(b):
            return -mend_mle(noise).fun - log_l([mend_mle(noise).x[0], mend_mle(noise).x[1], b], mend_num_sol, mend_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*mend_mle(noise).x[2], x1 = 0.9*mend_mle(noise).x[2])
        root2 = root_scalar(test, x0 = 1.5*mend_mle(noise).x[2], x1 = 1.1*mend_mle(noise).x[2])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    elif param == "V0":
        def test(V0):
            return -mend_mle(noise).fun - log_l([V0, mend_mle(noise).x[1], mend_mle(noise).x[2]], mend_num_sol, mend_sol_noise(noise), noise) - chi2.ppf(confidence, df)/2
        root1 = root_scalar(test, x0 = 0.5*mend_mle(noise).x[0], x1 = 0.9*mend_mle(noise).x[0])
        root2 = root_scalar(test, x0 = 1.5*mend_mle(noise).x[0], x1 = 1.1*mend_mle(noise).x[0])
        CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
        return CI
    else:
        print("Check param value")

start = timeit.default_timer()


print("Profile Likelihood confidence interval (95%) for Exponential model (a), with noise 5%:", exp_CI(0.95, 0.05, "a"))
print("Profile Likelihood confidence interval (99%) for Exponential model (a), with noise 5%:", exp_CI(0.99, 0.05, "a"))
print("Profile Likelihood confidence interval (95%) for Exponential model (a), with noise 10%:", exp_CI(0.95, 0.1, "a"))
print("Profile Likelihood confidence interval (99%) for Exponential model (a), with noise 10%:", exp_CI(0.99, 0.1, "a"))
print("Profile Likelihood confidence interval (95%) for Exponential model (V0), with noise 5%:", exp_CI(0.95, 0.05, "V0"))
#Runtime ~ 4 mins

print("Profile Likelihood confidence interval (95%) for Logistic model (r), with noise 5%:", log_CI(0.95, 0.05, "r"))
print("Profile Likelihood confidence interval (95%) for Logistic model (K), with noise 5%:", log_CI(0.95, 0.05, "K"))
print("Profile Likelihood confidence interval (95%) for Logistic model (V0), with noise 5%:", log_CI(0.95, 0.05, "V0"))
#Runtime ~ 7 mins

print("Profile Likelihood confidence interval (95%) for Gompertz model (r), with noise 5%:", gomp_CI(0.95, 0.05, "r"))
print("Profile Likelihood confidence interval (95%) for Gompertz model (K), with noise 5%:", gomp_CI(0.95, 0.05, "K"))
print("Profile Likelihood confidence interval (95%) for Gompertz model (V0), with noise 5%:", gomp_CI(0.95, 0.05, "V0"))
#Runtime ~ 8 mins

print("Profile Likelihood confidence interval (95%) for Bertalanffy model (b), with noise 5%:", bert_CI(0.95, 0.05, "b"))
print("Profile Likelihood confidence interval (95%) for Bertalanffy model (d), with noise 5%:", bert_CI(0.95, 0.05, "d"))
print("Profile Likelihood confidence interval (95%) for Bertalanffy model (V0), with noise 5%:", bert_CI(0.95, 0.05, "V0"))
#Runtime ~ 5 mins

print("Profile Likelihood confidence interval (95%) for Mendelsohn model (a), with noise 5%:", mend_CI(0.95, 0.05, "a"))
print("Profile Likelihood confidence interval (95%) for Mendelsohn model (b), with noise 5%:", mend_CI(0.95, 0.05, "b"))
print("Profile Likelihood confidence interval (95%) for Mendelsohn model (V0), with noise 5%:", mend_CI(0.95, 0.05, "V0"))
#Error - no values in range tried (Mendelsohn estimates inaccurate.)


stop = timeit.default_timer()

print('Time: ', stop - start)
'''
#################################################################################################################################################################

# Dosage Optimisation

# Objective function with costant D
def obj_function(D, theta, v, time, c):
    return v(time, theta, D)[-1]**2 + c*(D**2)*time[1]











#################################################################################################################################################################

#Classes

t_eval = np.linspace(0, 10, 101)

class Exponential:
    def __init__(self, noise):
        self.noise = noise
        self.data = self.exp_sol_noise()

    # Exponential ODE
    def ode(self, t, v, a):
        return a*v
    # Analytical solution of Exponential ODE theta = [V_0, a]
    def exp_sol(self, theta, t):
        return theta[0]*np.exp(theta[1]*t)
    # Numerical solution to Exponential ODE, theta = [V0, a]
    def exp_num_sol(self, theta):
        return solve_ivp(self.ode, [0, 10], [theta[0]], args = (theta[1],), t_eval = np.linspace(0, 10, 101))
    # Solution with noise
    def exp_sol_noise(self):
        return self.exp_sol([0.01, 1], t_eval) * (1 + self.noise * np.random.normal(0,1,101))
    
    # Residual
    def exp_res(self, theta):
        return self.exp_sol(theta, t_eval) - self.data
    # Set Least Squares estimates
    def l_squares(self):
        theta0 = [0.01, 1]
        return least_squares(self.exp_res, theta0)
    # Get Least Squares estimates
    def get_l_s(self):
        print(self.l_squares().x)
    
    # Set MLE estimates
    def exp_mle(self): 
        return minimize(neg_log_l, [0.01, 1], method = 'Nelder-Mead', args=(self.exp_num_sol, self.data, self.noise))
    # Get MLE estimates
    def get_mle(self):
        print(self.exp_mle().x)
    
    # Set PL CI
    def exp_CI(self, confidence, param):
        df = 1
        if param == "a":
            def test(a):
                return -self.exp_mle().fun - log_l([self.exp_mle().x[0], a], self.exp_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.exp_mle().x[1], x1 = 0.9*self.exp_mle().x[1])
            root2 = root_scalar(test, x0 = 1.5*self.exp_mle().x[1], x1 = 1.1*self.exp_mle().x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -self.exp_mle().fun - log_l([V0, self.exp_mle().x[1]], self.exp_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2            
            root1 = root_scalar(test, x0 = 0.5*self.exp_mle().x[0], x1 = 0.9*self.exp_mle().x[0])
            root2 = root_scalar(test, x0 = 1.5*self.exp_mle().x[0], x1 = 1.1*self.exp_mle().x[0])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        else:
            print("Check param value")
    # Get PL CI
    def get_CI(self, confidence, param):
        print(self.exp_CI(confidence, param))

    # Plot general solution
    def plot(self, theta):
        V0 = str(theta[0])
        a = str(theta[1])
        plt.plot(self.exp_num_sol(theta).t, self.exp_num_sol(theta).y[0])
        plt.title("Exponential model with V0 = "  + V0 + " and a = " + a)
        plt.xlabel("Time")
        plt.ylabel("Volume")
        plt.show()

    
#################################################################################################################################################################


class Mendelsohn:
    def __init__(self, noise):
        self.noise = noise
        self.data = self.mend_sol_noise()

    # Mendelsohn ODE
    def ode(self, t, v, a, b):
        return a*(v**b)
    # Analytical solution of Mendelsohn ODE theta = [V_0, a, b]
    def mend_sol(self, theta, t):
        return ((1 - theta[2])*(theta[1]*t + (theta[0]**(1-theta[2]))/(1-theta[2])))**(1/(1-theta[2]))
    # Numerical solution to Mendelsohn ODE, theta = [V0, a, b]
    def mend_num_sol(self, theta): 
        return solve_ivp(self.ode, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))
    # Solution with noise
    def mend_sol_noise(self):
        return self.mend_sol([0.01, 1, 2], t_eval) * (1 + self.noise * np.random.normal(0,1,101))
    
    # Residuals
    def mend_res(self, theta):
        return self.mend_sol(theta, t_eval) - self.data
    # Set Least Squares estimates
    def l_squares(self):
        theta0 = [0.01, 1, 2]
        return least_squares(self.mend_res, theta0)
    # Get Least Squares estimates
    def get_l_s(self):
        print(self.l_squares().x)
    
    # Set MLE estimates
    def mend_mle(self): 
        return minimize(neg_log_l, [0.01, 1, 2], method = 'Nelder-Mead', args=(self.mend_num_sol, self.data, self.noise))
    # Get MLE estimates
    def get_mle(self):
        print(self.mend_mle().x)

    # Set PL CI
    def mend_CI(self, confidence, param):
        df = 1
        if param == "a":
            def test(a):
                return -self.mend_mle().fun - log_l([self.mend_mle().x[0], a, self.mend_mle().x[2]], self.mend_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.mend_mle().x[1], x1 = 0.9*self.mend_mle().x[1])
            root2 = root_scalar(test, x0 = 1.5*self.mend_mle().x[1], x1 = 1.1*self.mend_mle().x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "b":
            def test(b):
                return -self.mend_mle().fun - log_l([self.mend_mle().x[0], self.mend_mle().x[1], b], self.mend_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.mend_mle().x[2], x1 = 0.9*self.mend_mle().x[2])
            root2 = root_scalar(test, x0 = 1.5*self.mend_mle().x[2], x1 = 1.1*self.mend_mle().x[2])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -self.mend_mle().fun - log_l([V0, self.mend_mle().x[1], self.mend_mle().x[2]], self.mend_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.mend_mle().x[0], x1 = 0.9*self.mend_mle().x[0])
            root2 = root_scalar(test, x0 = 1.5*self.mend_mle().x[0], x1 = 1.1*self.mend_mle().x[0])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        else:
            print("Check param value")
    # Get PL CI
    def get_CI(self, confidence, param):
        print(self.mend_CI(confidence, param))

    # Plot general solution
    def plot(self, theta):
        V0 = str(theta[0])
        a = str(theta[1])
        b = str(theta[2])
        plt.plot(self.mend_num_sol(theta).t, self.mend_num_sol(theta).y[0])
        plt.title("Mendelsohn model with V0 = "  + V0 + ", a = " + a + " and b = " + b)
        plt.xlabel("Time")
        plt.ylabel("Volume")
        plt.show()    


#################################################################################################################################################################



class Logistic:
    def __init__(self, noise):
        self.noise = noise
        self.data = self.log_sol_noise()
    
    # ODE
    def ode(self, t, v, r, k):
        return r*v*(1-v/k)
    # Analytical solution of Logistic ODE theta = [V_0, r, K]
    def log_sol(self, theta, t):
        return (theta[0]/((theta[0]/theta[2])+(1 - theta[0]/theta[2])*np.exp(-theta[1]*t)))
    # Numerical solution to Logistic ODE, theta = [V0, r, K]
    def log_num_sol(self, theta): 
        return solve_ivp(self.ode, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))
    # Solution with noise
    def log_sol_noise(self):
        return self.log_sol([0.01, 1, 10], t_eval) * (1 + self.noise * np.random.normal(0,1,101))

    # Residuals
    def log_res(self, theta):
        return self.log_sol(theta, t_eval) - self.data
    # Set Least Squares estimates
    def l_squares(self):
        theta0 = [0.01, 1, 10]
        return least_squares(self.log_res, theta0)
    # Get Least Squares estimates
    def get_l_s(self):
        print(self.l_squares().x)
    
    # Set MLE estimates
    def log_mle(self): 
        return minimize(neg_log_l, [0.01, 1, 10], method = 'Nelder-Mead', args=(self.log_num_sol, self.data, self.noise))
    # Get MLE estimates
    def get_mle(self):
        print(self.log_mle().x)

    # Set PL CI
    def log_CI(self, confidence, param):
        df = 1
        if param == "r":
            def test(r):
                return -self.log_mle().fun - log_l([self.log_mle().x[0], r, self.log_mle().x[2]], self.log_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.log_mle().x[1], x1 = 0.9*self.log_mle().x[1])
            root2 = root_scalar(test, x0 = 1.5*self.log_mle().x[1], x1 = 1.1*self.log_mle().x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "K":
            def test(K):
                return -self.log_mle().fun - log_l([self.log_mle().x[0], self.log_mle().x[1], K], self.log_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.log_mle().x[2], x1 = 0.9*self.log_mle().x[2])
            root2 = root_scalar(test, x0 = 1.5*self.log_mle().x[2], x1 = 1.1*self.log_mle().x[2])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -self.log_mle().fun - log_l([V0, self.log_mle().x[1], self.log_mle().x[2]], self.log_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.log_mle().x[0], x1 = 0.9*self.log_mle().x[0])
            root2 = root_scalar(test, x0 = 1.5*self.log_mle().x[0], x1 = 1.1*self.log_mle().x[0])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        else:
            print("Check param value")
    # Get PL CI
    def get_CI(self, confidence, param):
        print(self.log_CI(confidence, param))


    # Plot general solution
    def plot(self, theta):
        V0 = str(theta[0])
        r = str(theta[1])
        K = str(theta[2])
        plt.plot(self.log_num_sol(theta).t, self.log_num_sol(theta).y[0])
        plt.title("Logistic model with V0 = "  + V0 + ", r = " + r + " and K = " + K)
        plt.xlabel("Time")
        plt.ylabel("Volume")
        plt.show()
    

    # Dosage ODE
    def dos_ode(self, t, v, r, k, D):
        return r*v*(1-v/k) - D*v
    # Dosage Analytical Solution with theta = [V0, r, k]
    def dos_sol(self, time, theta, D):
        return theta[0]/((theta[0]*theta[1])/(theta[2]*(theta[1]-D)) + (1 - (theta[0]*theta[1])/(theta[2]*(theta[1]-D)))*np.exp((D-theta[1])*time))
    # Dosage Numerical Solution with theta = [V0, r, k]
    def num_dos(self, time, theta):
        return solve_ivp(self.dos_ode, time, [theta[0]], args=(theta[1], theta[2]), t_eval = np.linspace(time[0], time[1], 101))
    # Solution with noise
    def dos_noise(self, time, theta, D):
        return self.dos_sol(time, theta, D) * (1 + self.noise * np.random.normal(time[0],time[1],101))
    # Find Estimate for D
    def D_est(self, time, theta, c):
        return minimize(obj_function, 1, args=(theta, self.dos_noise, time, c))
    # Get Estimate for D
    def get_D(self, time, theta, c):
        print(self.D_est(time, theta, c).x)


#################################################################################################################################################################



class Gompertz:
    def __init__(self, noise):
        self.noise = noise
        self.data = self.gomp_sol_noise()
    
    # ODE
    def ode(self, t, v, r, k):
        return r*np.log(k/v)*v
    # Analytical solution of Gompertz ODE theta = [V_0, r, K]
    def gomp_sol(self, theta, t):
        return theta[2]*(theta[0]/theta[2])**np.exp(-theta[1]*t)
    # Numerical solution to Gompertz ODE, theta = [V0, r, K]
    def gomp_num_sol(self, theta): 
        return solve_ivp(self.ode, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))
    # Solution with noise
    def gomp_sol_noise(self):
        return self.gomp_sol([0.01, 1, 10], t_eval) * (1 + self.noise * np.random.normal(0,1,101))

    # Residuals
    def gomp_res(self, theta):
        return self.gomp_sol(theta, t_eval) - self.data
    # Set Least Squares estimates
    def l_squares(self):
        theta0 = [0.01, 1, 10]
        return least_squares(self.gomp_res, theta0)
    # Get Least Squares estimates
    def get_l_s(self):
        print(self.l_squares().x)
    
    # Set MLE estimates
    def gomp_mle(self):
        return minimize(neg_log_l, [0.01, 1, 10], method = 'Nelder-Mead', args=(self.gomp_num_sol, self.data, self.noise))
    # Get MLE estimates
    def get_mle(self):
        print(self.gomp_mle().x)

    # Set PL CI
    def gomp_CI(self, confidence, param):
        df = 1
        if param == "r":
            def test(r):
                return -self.gomp_mle().fun - log_l([self.gomp_mle().x[0], r, self.gomp_mle().x[2]], self.gomp_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.gomp_mle().x[1], x1 = 0.9*self.gomp_mle().x[1])
            root2 = root_scalar(test, x0 = 1.5*self.gomp_mle().x[1], x1 = 1.1*self.gomp_mle().x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "K":
            def test(K):
                return -self.gomp_mle().fun - log_l([self.gomp_mle().x[0], self.gomp_mle().x[1], K], self.gomp_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.gomp_mle().x[2], x1 = 0.9*self.gomp_mle().x[2])
            root2 = root_scalar(test, x0 = 1.5*self.gomp_mle().x[2], x1 = 1.1*self.gomp_mle().x[2])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -self.gomp_mle().fun - log_l([V0, self.gomp_mle().x[1], self.gomp_mle().x[2]], self.gomp_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.gomp_mle().x[0], x1 = 0.9*self.gomp_mle().x[0])
            root2 = root_scalar(test, x0 = 1.5*self.gomp_mle().x[0], x1 = 1.1*self.gomp_mle().x[0])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        else:
            print("Check param value")
    # Get PL CI
    def get_CI(self, confidence, param):
        print(self.gomp_CI(confidence, param))

    # Plot general solution
    def plot(self, theta):
        V0 = str(theta[0])
        r = str(theta[1])
        K = str(theta[2])
        plt.plot(self.gomp_num_sol(theta).t, self.gomp_num_sol(theta).y[0])
        plt.title("Gompertz model with V0 = "  + V0 + ", r = " + r + " and K = " + K)
        plt.xlabel("Time")
        plt.ylabel("Volume")
        plt.show()


#################################################################################################################################################################



class Bertalanffy:
    def __init__(self, noise):
        self.noise = noise
        self.data = self.bert_sol_noise()
    
    # ODE
    def ode(self, t, v, b, d):
        return b*(v**(2/3))-d*v
    # Analytical solution of Bertalanffy ODE theta = [V_0, b, d]
    def bert_sol(self, theta, t):
        return ((theta[1]/theta[2])*(1 - np.exp(-theta[2]*t/3)) + (theta[0]**(1/3))*np.exp(-theta[2]*t/3))**3
    # Numerical solution to Bertalanffy ODE, theta = [V0, b, d]
    def bert_num_sol(self, theta):
        return solve_ivp(self.ode, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))
    # Solution with noise
    def bert_sol_noise(self):
        return self.bert_sol([0.01, 1, 1], t_eval) * (1 + self.noise * np.random.normal(0,1,101))

    # Residuals
    def bert_res(self, theta):
        return self.bert_sol(theta, t_eval) - self.data
    # Set Least Squares estimates
    def l_squares(self):
        theta0 = [0.01, 1, 1]
        return least_squares(self.bert_res, theta0)
    # Get Least Squares estimates
    def get_l_s(self):
        print(self.l_squares().x)
    
    # Set MLE estimates
    def bert_mle(self):
        return minimize(neg_log_l, [0.01, 1, 1], method = 'Nelder-Mead', args=(self.bert_num_sol, self.data, self.noise))
    # Get MLE estimates
    def get_mle(self):
        print(self.bert_mle().x)

    # Set PL CI
    def bert_CI(self, confidence, param):
        df = 1
        if param == "b":
            def test(b):
                return -self.bert_mle().fun - log_l([self.bert_mle().x[0], b, self.bert_mle().x[2]], self.bert_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.bert_mle().x[1], x1 = 0.9*self.bert_mle().x[1])
            root2 = root_scalar(test, x0 = 1.5*self.bert_mle().x[1], x1 = 1.1*self.bert_mle().x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "d":
            def test(d):
                return -self.bert_mle().fun - log_l([self.bert_mle().x[0], self.bert_mle().x[1], d], self.bert_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.bert_mle().x[2], x1 = 0.9*self.bert_mle().x[2])
            root2 = root_scalar(test, x0 = 1.5*self.bert_mle().x[2], x1 = 1.1*self.bert_mle().x[2])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -self.bert_mle().fun - log_l([V0, self.bert_mle().x[1], self.bert_mle().x[2]], self.bert_num_sol, self.data, self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*self.bert_mle().x[0], x1 = 0.9*self.bert_mle().x[0])
            root2 = root_scalar(test, x0 = 1.5*self.bert_mle().x[0], x1 = 1.1*self.bert_mle().x[0])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        else:
            print("Check param value")
    # Get PL CI
    def get_CI(self, confidence, param):
        print(self.bert_CI(confidence, param))

    # Plot general solution
    def plot(self, theta):
        V0 = str(theta[0])
        b = str(theta[1])
        d = str(theta[2])
        plt.plot(self.bert_num_sol(theta).t, self.bert_num_sol(theta).y[0])
        plt.title("Bertalanffy model with V0 = "  + V0 + ", b = " + b + " and d = " + d)
        plt.xlabel("Time")
        plt.ylabel("Volume")
        plt.show()    


#################################################################################################################################################################


obj = Logistic(0.05)
print(obj.dos_sol([0,10], [1, 5, 100], 1))
#obj.get_D([0,10], [0.01, 5, 100], 10000)
# How to restrict D positive?

#################################################################################################################################################################


