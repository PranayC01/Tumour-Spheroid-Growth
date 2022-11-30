import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.stats import chi2
import timeit

#------------------------------------------------------------------------------------------------------------------------------------------#

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


def plot(sol):
    print(sol.y[0])
    plt.plot(sol.t, sol.y[0], label = 'v(0) = 0.01')
    plt.plot(sol.t, sol.y[1], label = 'v(0) = 0.03')
    plt.plot(sol.t, sol.y[2], label = 'v(0) = 0.05')
    plt.plot(sol.t, sol.y[3], label = 'v(0) = 0.09')
    plt.plot(sol.t, sol.y[4], label = 'v(0) = 0.099')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    if (sol.y[0] == exp_num_sol.y[0]).all():
        plt.title('Exponential with a=1')
    elif (sol.y[0] == mend_num_sol.y[0]).all():
        plt.title('Mendelsohn with a=1 and b=2')
    elif (sol.y[0] == log_num_sol.y[0]).all():
        plt.title('Logistic with r=1 and k=10')
    elif (sol.y[0] == gomp_num_sol.y[0]).all():
        plt.title('Gompertz with r=1 and k=10')
    elif (sol.y[0] == bert_num_sol.y[0]).all():
        plt.title('Bertalanffy with b = d = 1')
    plt.show()

def plot_with_noise(sol, noise):
    plt.plot(sol.t, sol.y[0], label = 'v(0) = 0.01')
    plt.plot(sol.t, sol.y[0] * (1 + np.random.normal(0,1,101) * noise), label = 'v(0) = 0.01', marker="x")
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    if (sol.y[0] == exp_num_sol.y[0]).all():
        plt.title('Exponential with noise, a=1')
    elif (sol.y[0] == mend_num_sol.y[0]).all():
        plt.title('Mendelsohn with noise, a=1 and b=2')
    elif (sol.y[0] == log_num_sol.y[0]).all():
        plt.title('Logistic with noise, r=1 and k=10')
    elif (sol.y[0] == gomp_num_sol.y[0]).all():
        plt.title('Gompertz with noise, r=1 and k=10')
    elif (sol.y[0] == bert_num_sol.y[0]).all():
        plt.title('Bertalanffy with noise, b = d = 1')
    plt.show()

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
    return least_squares(exp_res, theta0, noise)

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
'''
print("Least Squares estimates:")
print("Exponential Model: ", "[V0, a] = ", exp_consts(0.05).x)
print("Mendelsohn Model: ", "[V0, a, b] = ", mend_consts(0.05).x)
print("Logistic Model: ", "[V0, r, K] = ", log_consts(0.05).x)
print("Gompertz Model: ", "[V0, r, K] = ", gomp_consts(0.05).x)
print("Bertalanffy Model: ", "[V0, b, d] = ", bert_consts(0.05).x)
print("Bertalanffy Growth Model: ", "[V0, b, d] = ", bert_growth_consts(0.05).x)
print("Bertalanffy Decay Model: ", "[V0, b, d] = ", bert_decay_consts(0.05).x)
'''


#------------------------------------------------------------------------------------------------------------------------------------------#

# Profile Likelihood

n=101
# g is analytical solution for an ODE, v = g*(1 + noise*N(0,1)).

def log_l(theta, g, v, noise):
    log_l = -(n/2)*np.log(noise**2*(2*np.pi)) -n/(2*noise**2) - (1/(2*noise**2))*np.sum(np.square(v)/np.square(g(theta).y[0]) - 2*v/g(theta).y[0]) - np.sum(np.log(g(theta).y[0]))
    return log_l

def neg_log_l(theta, g, v, noise):
    return -log_l(theta, g, v, noise)

#print(exp_num_sol.y[0])
#print(exp_sol_noise)
'''
log_l([0.01, 1], exp_num_sol, exp_sol_noise, 0.05)
log_l([0.01, 1, 2], mend_num_sol, mend_sol_noise, 0.05)
log_l([0.01, 1, 10], log_num_sol, log_sol_noise, 0.05)
log_l([0.01, 1, 10], gomp_num_sol, gomp_sol_noise, 0.05)
log_l([0.01, 1, 1], bert_num_sol, bert_sol_noise, 0.05)
'''
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
'''
print("Maximum likelihood estimates:")
print("Exponential Model: ", "[V0, a] = ", exp_mle(noise).x)
print("Mendelsohn Model: ", "[V0, a, b] = ", mend_mle(noise).x)
print("Logistic Model: ", "[V0, r, K] = ", log_mle(noise).x)
print("Gompertz Model: ", "[V0, r, K] = ", gomp_mle(noise).x)
print("Bertalanffy Model: ", "[V0, b, d] = ", bert_mle(noise).x)
print(log_l(exp_mle(noise).x, exp_num_sol, exp_sol_noise, noise))
print(log_l(exp_mle(noise).x, exp_num_sol, exp_sol_noise, noise) + exp_mle(noise).fun)
'''

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

'''
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
'''

stop = timeit.default_timer()

print('Time: ', stop - start) 