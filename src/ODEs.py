import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

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

# Plot of solution to Exponential ODE.
def exp_plot():
    exp_sol = solve_ivp(Exponential, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1,), t_eval = np.linspace(0, 10, 101))
    print(exp_sol)
    plt.plot(exp_sol.t, exp_sol.y[0], label = 'v(0) = 0.01')
    plt.plot(exp_sol.t, exp_sol.y[1], label = 'v(0) = 0.03')
    plt.plot(exp_sol.t, exp_sol.y[2], label = 'v(0) = 0.05')
    plt.plot(exp_sol.t, exp_sol.y[3], label = 'v(0) = 0.09')
    plt.plot(exp_sol.t, exp_sol.y[4], label = 'v(0) = 0.099')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.title('Exponential with a=1')
    plt.show()

# Plot of solution to Mendelsohn ODE.
def mend_plot():
    mend_sol = solve_ivp(Mendelsohn, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 2), t_eval = np.linspace(0, 10, 101))
    print(mend_sol)
    plt.plot(mend_sol.t, mend_sol.y[0], label = 'v(0) = 0.01')
    plt.plot(mend_sol.t, mend_sol.y[1], label = 'v(0) = 0.03')
    plt.plot(mend_sol.t, mend_sol.y[2], label = 'v(0) = 0.05')
    plt.plot(mend_sol.t, mend_sol.y[3], label = 'v(0) = 0.09')
    plt.plot(mend_sol.t, mend_sol.y[4], label = 'v(0) = 0.099')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.title('Mendelsohn with a=1 and b=2')
    plt.show()


# Plot of solution to Logistic ODE.
def log_plot():
    log_sol = solve_ivp(Logistic, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 10), t_eval = np.linspace(0, 10, 101))
    print(log_sol)
    plt.plot(log_sol.t, log_sol.y[0], label = 'v(0) = 0.01')
    plt.plot(log_sol.t, log_sol.y[1], label = 'v(0) = 0.03')
    plt.plot(log_sol.t, log_sol.y[2], label = 'v(0) = 0.05')
    plt.plot(log_sol.t, log_sol.y[3], label = 'v(0) = 0.09')
    plt.plot(log_sol.t, log_sol.y[4], label = 'v(0) = 0.099')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.title('Logistic with r=1 and k=10')
    plt.show()

# Plot of solution to Gompertz ODE.
def gomp_plot():
    gomp_sol = solve_ivp(Gompertz, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 10), t_eval = np.linspace(0, 10, 101))
    print(gomp_sol)
    plt.plot(gomp_sol.t, gomp_sol.y[0], label = 'v(0) = 0.01')
    plt.plot(gomp_sol.t, gomp_sol.y[1], label = 'v(0) = 0.03')
    plt.plot(gomp_sol.t, gomp_sol.y[2], label = 'v(0) = 0.05')
    plt.plot(gomp_sol.t, gomp_sol.y[3], label = 'v(0) = 0.09')
    plt.plot(gomp_sol.t, gomp_sol.y[4], label = 'v(0) = 0.099')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.title('Gompertz with r=1 and k=10')
    plt.show()

# Plot of solution to Bertalanffy ODE with equal 'birth and death' rates, b and d, respectively.
def bert_plot():
    bert_sol = solve_ivp(Bertalanffy, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 1), t_eval = np.linspace(0, 10, 101))
    print(bert_sol)
    plt.plot(bert_sol.t, bert_sol.y[0], label = 'v(0) = 0.01')
    plt.plot(bert_sol.t, bert_sol.y[1], label = 'v(0) = 0.03')
    plt.plot(bert_sol.t, bert_sol.y[2], label = 'v(0) = 0.05')
    plt.plot(bert_sol.t, bert_sol.y[3], label = 'v(0) = 0.09')
    plt.plot(bert_sol.t, bert_sol.y[4], label = 'v(0) = 0.099')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.title('Bertalanffy with b = d = 1')
    plt.show()

    # Plot of solution to Bertalanffy ODE with 'birth rate' > 'death rate' (i.e. b>d).
    bert_growth_sol = solve_ivp(Bertalanffy, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (2, 1), t_eval = np.linspace(0, 10, 101))
    print(bert_growth_sol)
    plt.plot(bert_growth_sol.t, bert_growth_sol.y[0], label = 'v(0) = 0.01')
    plt.plot(bert_growth_sol.t, bert_growth_sol.y[1], label = 'v(0) = 0.03')
    plt.plot(bert_growth_sol.t, bert_growth_sol.y[2], label = 'v(0) = 0.05')
    plt.plot(bert_growth_sol.t, bert_growth_sol.y[3], label = 'v(0) = 0.09')
    plt.plot(bert_growth_sol.t, bert_growth_sol.y[4], label = 'v(0) = 0.099')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.title('Bertalanffy with b = 2, d = 1')
    plt.show()

    # Plot of solution to Bertalanffy ODE with 'birth rate' < 'death rate' (i.e. b<d).
    bert_decay_sol = solve_ivp(Bertalanffy, [0, 10], [0.01, 0.03, 0.05, 0.09, 0.099], args = (1, 2), t_eval = np.linspace(0, 10, 101))
    print(bert_decay_sol)
    plt.plot(bert_decay_sol.t, bert_decay_sol.y[0], label = 'v(0) = 0.01')
    plt.plot(bert_decay_sol.t, bert_decay_sol.y[1], label = 'v(0) = 0.03')
    plt.plot(bert_decay_sol.t, bert_decay_sol.y[2], label = 'v(0) = 0.05')
    plt.plot(bert_decay_sol.t, bert_decay_sol.y[3], label = 'v(0) = 0.09')
    plt.plot(bert_decay_sol.t, bert_decay_sol.y[4], label = 'v(0) = 0.099')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.title('Bertalanffy with b = 1, d = 2')
    plt.show()
