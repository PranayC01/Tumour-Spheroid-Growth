from pickletools import optimize
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy import optimize as op
import numdifftools as nd
from ci_rvm import find_CI


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

# Plot of solution to Exponential ODE.
exp_num_sol = solve_ivp(Exponential, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1,), t_eval = np.linspace(0, 10, 101))

# Plot of solution to Mendelsohn ODE.
mend_num_sol = solve_ivp(Mendelsohn, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 2), t_eval = np.linspace(0, 10, 101))

# Plot of solution to Logistic ODE.
log_num_sol = solve_ivp(Logistic, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 10), t_eval = np.linspace(0, 10, 101))

# Plot of solution to Gompertz ODE.
gomp_num_sol = solve_ivp(Gompertz, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 10), t_eval = np.linspace(0, 10, 101))

# Plot of solution to Bertalanffy ODE with equal 'birth and death' rates, b and d, respectively.
bert_num_sol = solve_ivp(Bertalanffy, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 1), t_eval = np.linspace(0, 10, 101))

# Plot of solution to Bertalanffy ODE with 'birth rate' > 'death rate' (i.e. b>d).
bert_growth_num_sol = solve_ivp(Bertalanffy, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (2, 1), t_eval = np.linspace(0, 10, 101))

# Plot of solution to Bertalanffy ODE with 'birth rate' < 'death rate' (i.e. b<d).
bert_decay_num_sol = solve_ivp(Bertalanffy, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 2), t_eval = np.linspace(0, 10, 101))

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
    elif (sol.y[0] == bert_growth_num_sol.y[0]).all():
        plt.title('Bertalanffy with b = 2, d = 1')
    elif (sol.y[0] == bert_decay_num_sol.y[0]).all():
        plt.title('Bertalanffy with b = 1, d = 2')
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
    elif (sol.y[0] == bert_growth_num_sol.y[0]).all():
        plt.title('Bertalanffy with noise, b = 2, d = 1')
    elif (sol.y[0] == bert_decay_num_sol.y[0]).all():
        plt.title('Bertalanffy with noise, b = 1, d = 2')
    plt.show()

t_eval = np.linspace(0, 10, 101)
noise = 0.05

exp_sol_noise = exp_sol([0.01, 1], t_eval) + (noise * np.random.normal(0,1,101))
mend_sol_noise = mend_sol([0.01, 1, 2], t_eval) + (noise * np.random.normal(0,1,101))
log_sol_noise = log_sol([0.01, 1, 10], t_eval) + (noise * np.random.normal(0,1,101))
gomp_sol_noise = gomp_sol([0.01, 1, 10], t_eval) + (noise * np.random.normal(0,1,101))
bert_sol_noise = bert_sol([0.01, 1, 1], t_eval) + (noise * np.random.normal(0,1,101))
bert_growth_sol_noise = bert_sol([0.01, 2, 1], t_eval) + (noise * np.random.normal(0,1,101))
bert_decay_sol_noise = bert_sol([0.01, 1, 2], t_eval) + (noise * np.random.normal(0,1,101))

def exp_res(theta):
    return exp_sol(theta, t_eval) - exp_sol_noise

def mend_res(theta):
    return mend_sol(theta, t_eval) - mend_sol_noise

def log_res(theta):
    return log_sol(theta, t_eval) - log_sol_noise

def gomp_res(theta):
    return gomp_sol(theta, t_eval) - gomp_sol_noise

def bert_res(theta):
    return bert_sol(theta, t_eval) - bert_sol_noise

def bert_growth_res(theta):
    return bert_sol(theta, t_eval) - bert_growth_sol_noise
    
def bert_decay_res(theta):
    return bert_sol(theta, t_eval) - bert_decay_sol_noise

theta0 = [0.01, 1]
exp_consts = least_squares(exp_res, theta0)
theta0 = [0.01, 1, 2]
mend_consts = least_squares(mend_res, theta0)
theta0 = [0.01, 1, 10]
log_consts = least_squares(log_res, theta0)
theta0 = [0.01, 1, 10]
gomp_consts = least_squares(gomp_res, theta0)
theta0 = [0.01, 1, 1]
bert_consts = least_squares(bert_res, theta0)
theta0 = [0.01, 2, 1]
bert_growth_consts = least_squares(bert_growth_res, theta0)
theta0 = [0.01, 1, 2]
bert_decay_consts = least_squares(bert_decay_res, theta0)

'''
print(exp_consts.x)
print(mend_consts.x)
print(log_consts.x)
print(gomp_consts.x)
print(bert_consts.x)
print(bert_growth_consts.x)
print(bert_decay_consts.x)
'''

#
#
#

# Profile Likelihood

n = 101

def logL_exp(params):
    V0, a = params
    print(n*np.log(V0) + a*sum(np.linspace(0,1,n))) 
    return n*np.log(V0) + a*sum(np.linspace(0,1,n))

logL_exp([0.01,1])

neglogL_exp = lambda params: -logL_exp(params)

x0 = [0, 0]

result = op.minimize(neglogL_exp, x0)

jac = nd.Gradient(logL_exp)
hess = nd.Hessian(logL_exp)

CIs = find_CI(result.x, logL_exp, jac, hess, disp=True)

print(CIs)