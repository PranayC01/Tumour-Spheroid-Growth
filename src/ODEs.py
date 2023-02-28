import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.special import lambertw
from scipy.stats import chi2
from tabulate import tabulate
import timeit

#################################################################################################################################################################
# Profile Likelihood


# g is analytical solution for an ODE, v = g*(1 + noise*N(0,1)).

def log_l(theta, g, v, noise):
    n=101
    log_l = -(n/2)*np.log((noise**2)*(2*np.pi)) -n/(2*noise**2) - (1/(2*noise**2))*np.sum(np.square(v)/np.square(g(theta).y[0]) - 2*v/g(theta).y[0]) - np.sum(np.log(g(theta).y[0]))
    return log_l

def neg_log_l(theta, g, v, noise):
    return -log_l(theta, g, v, noise)

#################################################################################################################################################################

# Dosage Optimisation

# Objective function with costant D
def obj_function(D, theta, v, time, c):
    return v(time, theta, D)[-1]**2 + c*(D**2)*time[-1]

#################################################################################################################################################################

# Classes

t_eval = np.linspace(0, 10, 1001)
np.random.seed(1)
X = np.random.normal(0, 1, 1001)

class Exponential:
    def __init__(self, noise, V0, a):
        self.noise = noise
        self.a = a
        self.V0 = V0
        self.data = self.exp_sol_noise([V0, a])

    
    # Exponential ODE
    def ode(self, t, v, a):
        return a*v
    # Analytical solution of Exponential ODE theta = [V0, a]
    def exp_sol(self, theta, t):
        return theta[0]*np.exp(theta[1]*t)
    # Numerical solution to Exponential ODE, theta = [V0, a]
    def exp_num_sol(self, theta):
        return solve_ivp(self.ode, [0, 10], [theta[0]], args = (theta[1],), t_eval = np.linspace(0, 10, 101))
    # Solution with noise
    def exp_sol_noise(self, theta):
        np.random.seed(1)
        X = np.random.normal(0, 1, 1001)
        return self.exp_sol(theta, t_eval) * (1 + self.noise * X)
    
    
    # Log-likelihood for Exponential Model
    def log_l(self, theta, noise):
        n=1001
        g = self.exp_sol(theta, np.linspace(0, 10, n))
        log_l = -(n/2)*np.log((noise**2)*(2*np.pi)) -n/(2*noise**2) - (1/(2*noise**2))*np.sum(np.square(self.data)/np.square(g) - 2*self.data/g) - np.sum(np.log(g))
        return log_l
    
    # Negative Log-Likelihood
    def neg_log_l(self, theta, noise):
        return -self.log_l(theta, noise)
    
    # Set MLE estimates
    def exp_mle(self, guess): 
        return minimize(self.neg_log_l, guess, method = 'Nelder-Mead', args=(self.noise))
    # Get MLE estimates
    def get_mle(self, guess):
        return self.exp_mle(guess).x
    
    # Set PL CI
    def exp_CI(self, guess, confidence, param):
        df = 1
        mle = self.exp_mle(guess)
        if param == "a":
            def test(a):
                return -mle.fun - self.log_l([mle.x[0], a], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = mle.x[1], x1 = 0.8*mle.x[1])
            root2 = root_scalar(test, x0 = mle.x[1], x1 = 1.2*mle.x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -mle.fun - self.log_l([V0, mle.x[1]], self.noise) - chi2.ppf(confidence, df)/2            
            root1 = root_scalar(test, x0 = 0.99*mle.x[0], x1 = 0.8*mle.x[0])
            root2 = root_scalar(test, x0 = 1.01*mle.x[0], x1 = 1.2*mle.x[0])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        else:
            print("Check param value")
    # Get PL CI
    def get_CI(self, guess, confidence, param):
        print(self.exp_CI(guess, confidence, param))

    # Plot general solution
    def plot(self, theta):
        V0 = str(theta[0])
        a = str(theta[1])
        plt.plot(self.exp_num_sol(theta).t, self.exp_num_sol(theta).y[0])
        plt.title("Exponential model with V0 = "  + V0 + " and a = " + a)
        plt.xlabel("Time")
        plt.ylabel("Volume")
        plt.show()



    # Dosage ODE
    def dos_ode(self, t, v, a, D):
        return a*v - D*v
    # Dosage Analytical Solution with theta = [V0, a]
    def dos_sol(self, t, theta, D):
        return theta[0]*np.exp(theta[1]*t - D*t)
    # Dosage Numerical Solution with theta = [V0, a]
    def num_dos(self, t, theta, D):
        return solve_ivp(self.dos_ode, t, [theta[0]], args=(theta[1], D), t_eval = np.linspace(t[0], t[-1], 101))
    # Find Estimate for D
    def D_est(self, t, theta, c):
        return minimize(obj_function, 1, args=(theta, self.dos_sol, t, c))
    # Get Estimate for D
    def get_D(self, t, theta, c):
        return self.D_est(t, theta, c).x[0]
    # Objective vs D plot
    def obj_func_plot(self, t, theta, c):
        D = np.linspace(0,10,101)
        obj = [obj_function(i, theta, self.dos_sol, t, c) for i in D]
        plt.plot(D, obj)
        plt.show()

    # Optimal Control

    # Tumour Volume Trajectory
    def v(self, t, V0, a, T, c):
        return V0*np.exp(((-1/(2*T))*np.real(lambertw((2*(V0**2)*T*np.exp(2*a*T))/c)) + a)*t)
    # Co State
    def p(self, t, V0, a, T, c):
        return (-c/(V0*T))*np.real(lambertw((2*(V0**2)*T*np.exp(2*a*T))/c))*np.exp(((1/(2*T))*lambertw((2*(V0**2)*T*np.exp(2*a*T))/c) - a)*t)
    # Optimal Dosage Scheduling
    def D(self, t, V0, a, T, c):
        return (1/(2*T))*np.real(lambertw((2*(V0**2)*T*np.exp(2*a*T))/c))
    # Plot of Volume Trajectory
    def plot_v(self, V0, a, T, c):
        t = np.linspace(0, T, 101)
        v = [self.v(i, V0, a, T, c) for i in t]
        plt.plot(t, v)
        V0 = str(V0)
        a = str(a)
        plt.title("Exponential Volume Trajectory with V0 = " + V0 + ", a = " +  a)
        plt.ylabel("Volume")
        plt.xlabel("Time")
        plt.show()
    # Plot of Co-State over time
    def plot_p(self, V0, a, T, c):
        t = np.linspace(0, T, 101)
        p = [self.p(i, V0, a, T, c) for i in t]
        plt.plot(t, p)
        V0 = str(V0)
        a = str(a)
        plt.title("Exponential Co-state Trajectory with V0 = " + V0 + ", a = " +  a)
        plt.ylabel("Co-state")
        plt.xlabel("Time")
        plt.show()
    # Optimal Dosage Scheduling Plot
    def plot_D(self, V0, a, T, c):
        t = np.linspace(0, T, 101)
        D = [self.D(i, V0, a, T, c) for i in t]
        plt.plot(t, D)
        V0 = str(V0)
        a = str(a)
        plt.title("Exponential Dosage Schedule with V0 = " + V0 + ", a = " +  a)
        plt.ylabel("Dosage")
        plt.xlabel("Time")
        plt.show()
    # Plot Comparison of Volume and Dosage over Time
    def plot_v_D(self, V0, a, T, c):
        t = np.linspace(0, T, 101)
        v = [self.v(i, V0, a, T, c) for i in t]
        D = [self.D(i, V0, a, T, c) for i in t]
        plt.plot(t, v, label = "Volume")
        plt.plot(t, D, label = "Dosage")
        plt.legend()
        plt.title("Volume/Dosage Plot for Exponential Model")
        plt.ylabel("Volume/Dosage")
        plt.xlabel("Time")
        plt.show()
    

#################################################################################################################################################################

class Mendelsohn:
    def __init__(self, noise, V0, a, b, c):
        self.noise = noise
        self.a = a
        self.b = b
        self.c = c
        self.V0 = V0
        self.data = self.mend_sol_noise([V0, a, b])

    # Mendelsohn ODE
    def ode(self, t, v, a, b):
        return a*(v**b)
    # Analytical solution of Mendelsohn ODE, theta = [V_0, a, b]
    def mend_sol(self, theta, t):
        return ((1 - theta[2])*theta[1]*t + theta[0]**(1-theta[2]))**(1/(1-theta[2]))
    # Numerical solution to Mendelsohn ODE, theta = [V0, a, b]
    def mend_num_sol(self, theta): 
        return solve_ivp(self.ode, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))
    # Solution with noise
    def mend_sol_noise(self, theta):
        np.random.seed(1)
        X = np.random.normal(0, 1, 1001)
        return self.mend_sol(theta, t_eval) * (1 + self.noise * X)
    
    # Log-likelihood for Mendelsohn Model
    def log_l(self, theta, noise):
        n=1001
        g = self.mend_sol(theta, np.linspace(0, 10, n))
        log_l = -(n/2)*np.log((noise**2)*(2*np.pi)) -n/(2*noise**2) - (1/(2*noise**2))*np.sum(np.square(self.data)/np.square(g) - 2*self.data/g) - np.sum(np.log(g))
        return log_l
    
    # Negative Log-Likelihood
    def neg_log_l(self, theta, noise):
        return -self.log_l(theta, noise)
    
    # Set MLE estimates
    def mend_mle(self, guess): 
        return minimize(self.neg_log_l, guess, method = 'Nelder-Mead', args=(self.noise))
    # Get MLE estimates
    def get_mle(self, guess):
        return self.mend_mle(guess).x

    # Set PL CI
    def mend_CI(self, guess, confidence, param):
        df = 1
        mle = self.mend_mle(guess)
        if param == "a":
            def test(a):
                return -mle.fun - self.log_l([mle.x[0], a, mle.x[2]], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = mle.x[1], x1 = 0.8*mle.x[1])
            root2 = root_scalar(test, x0 = mle.x[1], x1 = 1.2*mle.x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "b":
            def test(b):
                return -mle.fun - self.log_l([mle.x[0], mle.x[1], b], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.8*mle.x[2], x1 = 0.9*mle.x[2])
            root2 = root_scalar(test, x0 = 1.2*mle.x[2], x1 = 1.1*mle.x[2])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -mle.fun - self.log_l([V0, mle.x[1], mle.x[2]], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = mle.x[0], x1 = 0.8*mle.x[0])
            root2 = root_scalar(test, x0 = mle.x[0], x1 = 1.2*mle.x[0])
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



    # Dosage ODE
    def dos_ode(self, t, v, a, b, D):
        return a*(v**b) - D*v
    # Dosage Analytical Solution with theta = [V0, a, b]
    def dos_sol(self, time, theta, D):
        return (theta[1]/D + (theta[0]**(1-theta[2]) - (theta[1]/D))*np.exp((theta[2] - 1)*D*time))**(1/(1-theta[2]))
    # Dosage Numerical Solution with theta = [V0, a, b]
    def num_dos(self, time, theta, D):
        return solve_ivp(self.dos_ode, time, [theta[0]], args=(theta[1], theta[2], D), t_eval = np.linspace(time[0], time[-1], 101))
    # Find Estimate for D
    def D_est(self, time, theta, c):
        return minimize(obj_function, 1, args=(theta, self.dos_sol, time, c))
    # Get Estimate for D
    def get_D(self, time, theta, c):
        return self.D_est(time, theta, c).x[0]
    # Objective vs D plot
    def obj_func_plot(self, time, theta, c):
        D = np.linspace(0,10,101)
        obj = [obj_function(i, theta, self.dos_sol, time, c) for i in D]
        plt.plot(D, obj)
        plt.show()

    # Optimal Control

    # For t > V0^(1-b)/a(b-1)
    # Tumour Volume Trajectory
    def v(self, t, V0, a, b, T, c):
        return ((1-b)*a*t + V0**(1-b))**(1/(1-b))
    # Co State
    def p(self, t, V0, a, b, T, c):
        return -2*(((1-b)*a*T + V0**(1-b))**(-1))*((1-b)*a*t + V0**(1-b))**(b/(b-1))
    # Plot of Volume Trajectory
    def plot_v_1(self, V0, a, b, T, c):
        t = np.linspace(0, (V0**(1-b))/(a*(b-1)), 1000, endpoint=False)
        v = [self.v(i, V0, a, b, T, c) for i in t]
        plt.plot(t, v)
        plt.show()
    # Plot of Co-State over time
    def plot_p(self, V0, a, b, T, c):
        t = np.linspace(0, (V0**(1-b))/(a*(b-1)), 1000, endpoint=False)
        p = [self.p(i, V0, a, b, T, c) for i in t]
        plt.plot(t, p)
        plt.show()

    
    # For t > V0^(1-b)/a(b-1)
    # ODE System
    def sys_ode(self, t, y):
        v, p = y
        return [self.a*(v**self.b) + p*v**2/(2*self.c), -self.a*self.b*p*(v**(self.b-1)) - v*p**2/(2*self.c)]
    # Numerical solution of System of ODEs with intial conditions [V0, P0]
    def num_sol_sys(self, P0, T, V0):
        t_star = (V0**(1-self.b))/(self.a*(self.b - 1))
        return solve_ivp(self.sys_ode, [t_star, T], [V0, P0], t_eval = np.linspace(t_star, T, 1000)[1:])
    # Shooting Method Residual
    def res_bc(self, P0, T, V0):
        return self.num_sol_sys(P0, T, V0).y[1][-1] + 2*(self.num_sol_sys(P0, T, V0).y[0][-1])
    # Not a scalar (Root Result)
    def find_p0(self, T, V0):
        return root_scalar(self.res_bc, x0 = -10, x1 = -10, args=(T, V0), maxiter=5000)
    def get_p0(self, T, V0):
        return self.find_p0(T, V0).root
    def plot_res(self, T, V0):
        p = np.linspace(-0.1, -40, 101)
        res = [self.res_bc(x, T, V0) for x in p]
        plt.plot(p, res)
        plt.show()
    def get_v(self, T, V0):
        p0 = self.get_p0(T, V0)
        return self.num_sol_sys(p0, T, V0).y[0]
    def get_p(self, T, V0):
        p0 = self.get_p0(T, V0)
        return self.num_sol_sys(p0, T, V0).y[1] 
    def opt_d(self, T, V0):
        return -((self.get_v(T, V0))*(self.get_p(T, V0)))/(2*self.c)
    def plot_v(self, T, V0):
        #t = np.linspace(0, T, 999)
        t = np.linspace((V0**(1-self.b))/(self.a*(self.b-1)), T, 1000)[1:]
        plt.plot(t, self.get_v(T, V0))
        plt.ylabel("Volume")
        plt.xlabel("Time")
        V0 = str(V0)
        plt.title("Mendelsohn Model of Tumour Volume with initial condition: V0 = " + V0)
        plt.show()
    def plot_d(self, T, V0):
        t = np.linspace((V0**(1-self.b))/(self.a*(self.b-1)), T, 1000)[1:]
        plt.plot(t, self.opt_d(T, V0))
        plt.ylabel("Dosage")
        plt.xlabel("Time")
        V0 = str(V0)
        plt.title("Mendelsohn Model of Optimal Dosage with initial condition: V0 = " + V0)
        plt.show()
    # Volume/Dosage Plot
    def plot_v_d(self, T, V0):
        t = np.linspace((V0**(1-self.b))/(self.a*(self.b-1)), T, 1000)[1:]
        plt.plot(t, self.get_v(T, V0), label = "Volume")
        plt.plot(t, self.opt_d(T, V0), label = "Dosage")
        plt.legend()
        plt.title("Volume/Dosage over time for Mendelsohn model")
        plt.ylabel("Volume/Dosage")
        plt.xlabel("Time")
        plt.show()


#################################################################################################################################################################

class Logistic:
    def __init__(self, noise, V0, r, k, c):
        self.noise = noise
        self.data = self.log_sol_noise([V0, r, k])
        self.V0 = V0
        self.r = r
        self.k = k
        self.c = c
    
    # ODE
    def ode(self, t, v, r, k):
        return r*v*(1-v/k)
    # Analytical solution of Logistic ODE theta = [V_0, r, K]
    def log_sol(self, theta, t):
        #return (theta[0]/((theta[0]/theta[2])+(1 - theta[0]/theta[2])*np.exp(-theta[1]*t)))
        return (theta[0]*theta[2])/(theta[0] + (theta[2] - theta[0])*np.exp(-theta[1]*t))
    # Numerical solution to Logistic ODE, theta = [V0, r, K]
    def log_num_sol(self, theta): 
        return solve_ivp(self.ode, [0, 10], [theta[0]], args = (theta[1], theta[2]), t_eval = np.linspace(0, 10, 101))
    # Solution with noise
    def log_sol_noise(self, theta):
        np.random.seed(1)
        X = np.random.normal(0, 1, 1001)
        return self.log_sol(theta, t_eval) * (1 + self.noise * X)

    # Log-likelihood for Logistic Model
    def log_l(self, theta, noise):
        n=1001
        g = self.log_sol(theta, np.linspace(0, 10, n))
        log_l = -(n/2)*np.log((noise**2)*(2*np.pi)) -n/(2*noise**2) - (1/(2*noise**2))*np.sum(np.square(self.data)/np.square(g) - 2*self.data/g) - np.sum(np.log(g))
        return log_l
    
    # Negative Log-Likelihood
    def neg_log_l(self, theta, noise):
        return -self.log_l(theta, noise)
    
    # Set MLE estimates
    def log_mle(self, guess): 
        return minimize(self.neg_log_l, guess, method = 'Nelder-Mead', args=(self.noise))
    # Get MLE estimates
    def get_mle(self, guess):
        return self.log_mle(guess).x

    # Set PL CI
    def log_CI(self, guess, confidence, param):
        df = 1
        mle = self.log_mle(guess)
        if param == "r":
            def test(r):
                return -mle.fun - self.log_l([mle.x[0], r, mle.x[2]], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.75*mle.x[1], x1 = 0.9*mle.x[1])
            root2 = root_scalar(test, x0 = 1.25*mle.x[1], x1 = 1.1*mle.x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "K" or "k":
            def test(K):
                return -mle.fun - self.log_l([mle.x[0], mle.x[1], K], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*mle.x[2], x1 = 0.9*mle.x[2])
            root2 = root_scalar(test, x0 = 1.5*mle.x[2], x1 = 1.1*mle.x[2])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -mle.fun - self.log_l([V0, mle.x[1], mle.x[2]], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*mle.x[0], x1 = 0.9*mle.x[0])
            root2 = root_scalar(test, x0 = 1.5*mle.x[0], x1 = 1.1*mle.x[0])
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
    def num_dos(self, time, theta, D):
        return solve_ivp(self.dos_ode, time, [theta[0]], args=(theta[1], theta[2], D), t_eval = np.linspace(time[0], time[-1], 101))
    # Find Estimate for D
    def D_est(self, time, theta, c):
        return minimize(obj_function, 1, args=(theta, self.dos_sol, time, c))
    # Get Estimate for D
    def get_D(self, time, theta, c):
        return self.D_est(time, theta, c).x[0]
    # Objective vs D plot
    def obj_func_plot(self, time, theta, c):
        D = np.linspace(0,10,101)
        obj = [obj_function(i, theta, self.dos_sol, time, c) for i in D]
        plt.plot(D, obj)
        plt.show()

    # Optimal Control

    # System of ODEs, y = [v, p]
    def sys_ode(self, t, y):
        v, p = y
        return [self.r*v*(1-v/self.k) + (p*v**2)/(2*self.c), -self.r*p*(1-(2*v/self.k)) - (v*p**2)/(2*self.c)]
    # Numerical solution of System of ODEs with intial conditions [V0, P0]
    def num_sol_sys(self, P0, T, V0):
        return solve_ivp(self.sys_ode, [0, T], [V0, P0], t_eval = np.linspace(0, T, 101))
    # Shooting Method
    def res_bc(self, P0, T, V0):
        return self.num_sol_sys(P0, T, V0).y[1][-1] + 2*(self.num_sol_sys(P0, T, V0).y[0][-1])
    # Not a scalar (Root Result)
    def find_p0(self, T, V0):
        return root_scalar(self.res_bc, x0 = -0.5, x1 = -1, args=(T, V0), maxiter=200)
    def get_p0(self, T, V0):
        return self.find_p0(T, V0).root
    def plot_res(self, T, V0):
        p = np.linspace(2, -2, 101)
        res = [self.res_bc(x, T, V0) for x in p]
        plt.plot(p, res)
        plt.show()
    def get_v(self, T, V0):
        p0 = self.get_p0(T, V0)
        return self.num_sol_sys(p0, T, V0).y[0]
    def get_p(self, T, V0):
        p0 = self.get_p0(T, V0)
        return self.num_sol_sys(p0, T, V0).y[1] 
    def opt_d(self, T, V0):
        return -((self.get_v(T, V0))*(self.get_p(T, V0)))/(2*self.c)
    def plot_v(self, T, V0):
        t = np.linspace(0, T, 101)
        plt.plot(t, self.get_v(T, V0))
        plt.ylabel("Volume")
        plt.xlabel("Time")
        V0 = str(V0)
        plt.title("Logistic Model of Tumour Volume with initial condition: V0 = " + V0)
        plt.show()
    def plot_d(self, T, V0):
        t = np.linspace(0, T, 101)
        plt.plot(t, self.opt_d(T, V0))
        plt.ylabel("Dosage")
        plt.xlabel("Time")
        V0 = str(V0)
        plt.title("Logistic Model of Optimal Dosage with initial condition: V0 = " + V0)
        plt.show()


#################################################################################################################################################################

class Gompertz:
    def __init__(self, noise, V0, r, k):
        self.noise = noise
        self.data = self.gomp_sol_noise([V0, r, k])
        self.V0 = V0
        self.r = r
        self.k = k
    
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
    def gomp_sol_noise(self, theta):
        np.random.seed(1)
        X = np.random.normal(0, 1, 1001)
        return self.gomp_sol(theta, t_eval) * (1 + self.noise * X)

    # Log-likelihood for Gompertz Model
    def log_l(self, theta, noise):
        n=1001
        g = self.gomp_sol(theta, np.linspace(0, 10, n))
        log_l = -(n/2)*np.log((noise**2)*(2*np.pi)) -n/(2*noise**2) - (1/(2*noise**2))*np.sum(np.square(self.data)/np.square(g) - 2*self.data/g) - np.sum(np.log(g))
        return log_l
    
    # Negative Log-Likelihood
    def neg_log_l(self, theta, noise):
        return -self.log_l(theta, noise)
    
    # Set MLE estimates
    def gomp_mle(self, guess):
        return minimize(self.neg_log_l, guess, method = 'Nelder-Mead', args=(self.noise))
    # Get MLE estimates
    def get_mle(self, guess):
        return self.gomp_mle(guess).x

    # Set PL CI
    def gomp_CI(self, guess, confidence, param):
        df = 1
        mle = self.gomp_mle(guess)
        if param == "r":
            def test(r):
                return -mle.fun - self.log_l([mle.x[0], r, mle.x[2]], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*mle.x[1], x1 = 0.9*mle.x[1])
            root2 = root_scalar(test, x0 = 1.5*mle.x[1], x1 = 1.1*mle.x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "K" or "k":
            def test(K):
                return -mle.fun - self.log_l([mle.x[0], mle.x[1], K], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*mle.x[2], x1 = 0.9*mle.x[2])
            root2 = root_scalar(test, x0 = 1.5*mle.x[2], x1 = 1.1*mle.x[2])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -mle.fun - self.log_l([V0, mle.x[1], mle.x[2]], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*mle.x[0], x1 = 0.9*mle.x[0])
            root2 = root_scalar(test, x0 = 1.5*mle.x[0], x1 = 1.1*mle.x[0])
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

    

    # Dosage ODE
    def dos_ode(self, t, v, r, k, D):
        return r*np.log(k/v)*v - D*v
    # Dosage Analytical Solution with theta = [V0, r, k]
    def dos_sol(self, time, theta, D):
        return theta[2] * np.exp(((D - theta[1]*np.log(theta[2]/theta[0]))*np.exp(-theta[1]*time) - D)/theta[1])
    # Dosage Numerical Solution with theta = [V0, r, k]
    def num_dos(self, time, theta, D):
        return solve_ivp(self.dos_ode, time, [theta[0]], args=(theta[1], theta[2], D), t_eval = np.linspace(time[0], time[-1], 101))
    # Find Estimate for D
    def D_est(self, time, theta, c):
        return minimize(obj_function, 1, args=(theta, self.dos_sol, time, c))
    # Get Estimate for D
    def get_D(self, time, theta, c):
        return self.D_est(time, theta, c).x[0]
    # Objective vs D plot
    def obj_func_plot(self, time, theta, c):
        D = np.linspace(0,10,101)
        obj = [obj_function(i, theta, self.dos_sol, time, c) for i in D]
        plt.plot(D, obj)
        plt.show()

    # Optimal Control, theta = [V0, r, k]

    # Tumour Volume Trajectory
    def v(self, t, V0, r, K, T, c):
        g = (1 - np.exp(-2*r*T))*((K**2)/(c*r))*((V0/K)**(2*np.exp(-r*T)))
        return K*np.exp(((np.real(lambertw(g)))*(np.exp(-r*t) - np.exp(r*t)))/(2*(np.exp(r*T) - np.exp(-r*T))))*((V0/K)**(np.exp(-r*t)))
    # Co State
    def p(self, t, V0, r, K, T, c):
        g = (1 - np.exp(-2*r*T))*((K**2)/(c*r))*((V0/K)**(2*np.exp(-r*T)))
        return -((2*c*r*np.real(lambertw(g))*np.exp(r*t))/(K*(np.exp(r*T) - np.exp(-r*T))))*(np.exp(((lambertw(g))*(np.exp(r*t) - np.exp(-r*t)))/(2*(np.exp(r*T) - np.exp(-r*T))))*((V0/K)**(-np.exp(-r*t))))
    # Optimal Dosage Scheduling
    def D(self, t, V0, r, K, T, c):
        g = (1 - np.exp(-2*r*T))*((K**2)/(c*r))*((V0/K)**(2*np.exp(-r*T)))
        return (r*np.real(lambertw(g))*np.exp(r*t))/(np.exp(r*T) - np.exp(-r*T))
    # Plot of Volume Trajectory
    def plot_v(self, V0, r, K, T, c):
        t = np.linspace(0, T, 101)
        v = [self.v(i, V0, r, K, T, c) for i in t]
        plt.plot(t, v)
        V0 = str(V0)
        r = str(r)
        K = str(K)
        plt.title("Gompertz Volume Trajectory with V0 = " + V0 + ", r = " + r + ", K = " + K)
        plt.ylabel("Volume")
        plt.xlabel("Time")
        plt.show()
    # Plot of Co-State over time
    def plot_p(self, V0, r, K, T, c):
        t = np.linspace(0, T, 101)
        p = [self.p(i, V0, r, K, T, c) for i in t]
        plt.plot(t, p)
        V0 = str(V0)
        r = str(r)
        K = str(K)
        plt.title("Gompertz Co-State Trajectory with V0 = " + V0 + ", r = " + r + ", K = " + K)
        plt.ylabel("Co-State")
        plt.xlabel("Time")
        plt.show()
    # Optimal Dosage Scheduling Plot
    def plot_D(self, V0, r, K, T, c):
        t = np.linspace(0, T, 101)
        D = [self.D(i, V0, r, K, T, c) for i in t]
        plt.plot(t, D)
        V0 = str(V0)
        r = str(r)
        K = str(K)
        plt.title("Gompertz Dosage Schedule with V0 = " + V0 + ", r = " + r + ", K = " + K)
        plt.ylabel("Dosage")
        plt.xlabel("Time")
        plt.show()


#################################################################################################################################################################

class Bertalanffy:
    def __init__(self, noise, V0, b, d, c):
        self.noise = noise
        self.data = self.bert_sol_noise([V0, b, d])
        self.V0 = V0
        self.b = b
        self.d = d
        self.c = c
    
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
    def bert_sol_noise(self, theta):
        np.random.seed(1)
        X = np.random.normal(0, 1, 1001)
        return self.bert_sol(theta, t_eval) * (1 + self.noise * X)

    # Log-likelihood for Bertalanffy Model
    def log_l(self, theta, noise):
        n=1001
        g = self.bert_sol(theta, np.linspace(0, 10, n))
        log_l = -(n/2)*np.log((noise**2)*(2*np.pi)) -n/(2*noise**2) - (1/(2*noise**2))*np.sum(np.square(self.data)/np.square(g) - 2*self.data/g) - np.sum(np.log(g))
        return log_l
    
    # Negative Log-Likelihood
    def neg_log_l(self, theta, noise):
        return -self.log_l(theta, noise)
    
    # Set MLE estimates
    def bert_mle(self, guess):
        return minimize(self.neg_log_l, guess, method = 'Nelder-Mead', args=(self.noise))
    # Get MLE estimates
    def get_mle(self, guess):
        return self.bert_mle(guess).x

    # Set PL CI
    def bert_CI(self, guess, confidence, param):
        df = 1
        mle = self.bert_mle(guess)
        if param == "b":
            def test(b):
                return -mle.fun - self.log_l([mle.x[0], b, mle.x[2]], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*mle.x[1], x1 = 0.9*mle.x[1])
            root2 = root_scalar(test, x0 = 1.5*mle.x[1], x1 = 1.1*mle.x[1])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "d":
            def test(d):
                return -mle.fun - self.log_l([mle.x[0], mle.x[1], d], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*mle.x[2], x1 = 0.9*mle.x[2])
            root2 = root_scalar(test, x0 = 1.5*mle.x[2], x1 = 1.1*mle.x[2])
            CI = [min(root1.root, root2.root), max(root1.root, root2.root)]
            return CI
        elif param == "V0":
            def test(V0):
                return -mle.fun - self.log_l([V0, mle.x[1], mle.x[2]], self.noise) - chi2.ppf(confidence, df)/2
            root1 = root_scalar(test, x0 = 0.5*mle.x[0], x1 = 0.9*mle.x[0])
            root2 = root_scalar(test, x0 = 1.5*mle.x[0], x1 = 1.1*mle.x[0])
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



    # Dosage ODE
    def dos_ode(self, t, v, b, d, D):
        return b*(v**(2/3))-(d+D)*v
    # Dosage Analytical Solution with theta = [V0, b, d]
    def dos_sol(self, time, theta, D):
        return ((theta[1]/(theta[2] + D))*(1 - np.exp(-(theta[2]+D)*time/3)) + (theta[0]**(1/3))*np.exp(-(theta[2]+D)*time/3))**3
    # Dosage Numerical Solution with theta = [V0, b, d]
    def num_dos(self, time, theta, D):
        return solve_ivp(self.dos_ode, time, [theta[0]], args=(theta[1], theta[2], D), t_eval = np.linspace(time[0], time[-1], 101))
    # Find Estimate for D
    def D_est(self, time, theta, c):
        return minimize(obj_function, 1, args=(theta, self.dos_sol, time, c))
    # Get Estimate for D
    def get_D(self, time, theta, c):
        return self.D_est(time, theta, c).x[0]
    # Objective vs D plot
    def obj_func_plot(self, time, theta, c):
        D = np.linspace(0,10,101)
        obj = [obj_function(i, theta, self.dos_sol, time, c) for i in D]
        plt.plot(D, obj)
        plt.show()

    # Optimal Control
    # System of ODEs, y = [v, p]
    def sys_ode(self, t, y):
        v, p = y
        return [self.b*(v**(2/3)) - self.d*v + (p*(v**2))/(2*self.c), self.d*p - (2/3)*self.b*p*(v**(-1/3)) - ((p**2)*v)/(2*self.c)]      
    # Numerical solution of System of ODEs with intial conditions [V0, P0]
    def num_sol_sys(self, P0, T, V0):
        return solve_ivp(self.sys_ode, [0, T], [V0, P0], t_eval = np.linspace(0, T, 101))
    # Shooting Method
    def res_bc(self, P0, T, V0):
        return self.num_sol_sys(P0, T, V0).y[1][-1] + 2*(self.num_sol_sys(P0, T, V0).y[0][-1])
    # Not a scalar (Root Result)
    def find_p0(self, T, V0):
        return root_scalar(self.res_bc, x0 = -1, x1 = -1.5, args=(T, V0), maxiter=1000)
    def get_p0(self, T, V0):
        return self.find_p0(T, V0).root
    def plot_res(self, T, V0):
        p = np.linspace(-0.1, 0.5, 101)
        res = [self.res_bc(x, T, V0) for x in p]
        plt.plot(p, res)
        plt.show()
    def get_v(self, T, V0):
        p0 = self.get_p0(T, V0)
        return self.num_sol_sys(p0, T, V0).y[0]
    def get_p(self, T, V0):
        p0 = self.get_p0(T, V0)
        return self.num_sol_sys(p0, T, V0).y[1] 
    def opt_d(self, T, V0):
        return -((self.get_v(T, V0))*(self.get_p(T, V0)))/(2*self.c)
    def plot_v(self, T, V0):
        t = np.linspace(0, T, 101)
        plt.plot(t, self.get_v(T, V0))
        plt.ylabel("Volume")
        plt.xlabel("Time")
        V0 = str(V0)
        plt.title("Bertalanffy Model of Tumour Volume with initial condition: V0 = " + V0)
        plt.show()
    def plot_d(self, T, V0):
        t = np.linspace(0, T, 101)
        plt.plot(t, self.opt_d(T, V0))
        plt.ylabel("Dosage")
        plt.xlabel("Time")
        V0 = str(V0)
        plt.title("Bertalanffy Model of Optimal Dosage with initial condition: V0 = " + V0)
        plt.show()  


#################################################################################################################################################################

### RESULTS ###

# Exponential Models with noise = 0.05, V0 = 0.01 and varying a
exp1 = Exponential(0.05, 0.01, 1)
exp2 = Exponential(0.05, 0.01, 2)
exp3 = Exponential(0.05, 0.01, 5)
exp4 = Exponential(0.05, 0.01, 10)
exp5 = Exponential(0.05, 0.01, 20)

# Exponential Models with noise = 0.05, V0 = 1 and varying a
exp6 = Exponential(0.05, 1, 1)
exp7 = Exponential(0.05, 1, 2)
exp8 = Exponential(0.05, 1, 5)
exp9 = Exponential(0.05, 1, 10)
exp10 = Exponential(0.05, 1, 20)

# Exponential Models with noise = 0.05, V0 = 10 and varying a
exp11 = Exponential(0.05, 10, 1)
exp12 = Exponential(0.05, 10, 2)
exp13 = Exponential(0.05, 10, 5)
exp14 = Exponential(0.05, 10, 10)
exp15 = Exponential(0.05, 10, 20)

# Table: noise = 0.05
# Create data
data1 = [[exp1.V0, exp1.a, exp1.exp_CI([0.01,1], 0.95, "V0"), exp1.exp_CI([0.01,1], 0.99, "V0")],
        [exp2.V0, exp2.a, exp2.exp_CI([0.01,2], 0.95, "V0"), exp2.exp_CI([0.01,2], 0.99, "V0")],
        [exp3.V0, exp3.a, exp3.exp_CI([0.01,5], 0.95, "V0"), exp3.exp_CI([0.01,5], 0.99, "V0")],
        [exp4.V0, exp4.a, exp4.exp_CI([0.01,10], 0.95, "V0"), exp4.exp_CI([0.01,10], 0.99, "V0")],
        [exp5.V0, exp5.a, exp5.exp_CI([0.01,20], 0.95, "V0"), exp5.exp_CI([0.01,20], 0.99, "V0")]
        ]
data2 = [[exp1.V0, exp1.a, exp1.exp_CI([0.01,1], 0.95, "a"), exp1.exp_CI([0.01,1], 0.99, "a")],
        [exp2.V0, exp2.a, exp2.exp_CI([0.01,2], 0.95, "a"), exp2.exp_CI([0.01,2], 0.99, "a")],
        [exp3.V0, exp3.a, exp3.exp_CI([0.01,5], 0.95, "a"), exp3.exp_CI([0.01,5], 0.99, "a")],
        [exp4.V0, exp4.a, exp4.exp_CI([0.01,10], 0.95, "a"), exp4.exp_CI([0.01,10], 0.99, "a")],
        [exp5.V0, exp5.a, exp5.exp_CI([0.01,20], 0.95, "a"), exp5.exp_CI([0.01,20], 0.99, "a")]
        ]
data3 = [[exp6.V0, exp6.a, exp6.exp_CI([1,1], 0.95, "V0"), exp6.exp_CI([1,1], 0.99, "V0")],
        [exp7.V0, exp7.a, exp7.exp_CI([1,2], 0.95, "V0"), exp7.exp_CI([1,2], 0.99, "V0")],
        [exp8.V0, exp8.a, exp8.exp_CI([1,5], 0.95, "V0"), exp8.exp_CI([1,5], 0.99, "V0")],
        [exp9.V0, exp9.a, exp9.exp_CI([1,10], 0.95, "V0"), exp9.exp_CI([1,10], 0.99, "V0")],
        [exp10.V0, exp10.a, exp10.exp_CI([1,20], 0.95, "V0"), exp10.exp_CI([1,20], 0.99, "V0")]
        ]
data4 = [[exp6.V0, exp6.a, exp6.exp_CI([1,1], 0.95, "a"), exp6.exp_CI([1,1], 0.99, "a")],
        [exp7.V0, exp7.a, exp7.exp_CI([1,2], 0.95, "a"), exp7.exp_CI([1,2], 0.99, "a")],
        [exp8.V0, exp8.a, exp8.exp_CI([1,5], 0.95, "a"), exp8.exp_CI([1,5], 0.99, "a")],
        [exp9.V0, exp9.a, exp9.exp_CI([1,10], 0.95, "a"), exp9.exp_CI([1,10], 0.99, "a")],
        [exp10.V0, exp10.a, exp10.exp_CI([1,20], 0.95, "a"), exp10.exp_CI([1,20], 0.99, "a")]
        ]
# Define header names
col_names1 = ["V0", "a", "95% Confidence Interval for V0", "99% Confidence Interval for V0"]
col_names2 = ["V0", "a", "95% Confidence Interval for a", "99% Confidence Interval for a"]

#print(tabulate(data1, headers=col_names1, tablefmt="latex"))
#print(tabulate(data2, headers=col_names2, tablefmt="latex"))
# print(tabulate(data3, headers=col_names1, tablefmt="latex"))
# print(tabulate(data4, headers=col_names2, tablefmt="latex"))


# Exponential Models with noise = 0.2, V0 = 0.01 and varying a
exp1 = Exponential(0.2, 0.01, 1)
exp2 = Exponential(0.2, 0.01, 2)
exp3 = Exponential(0.2, 0.01, 5)
exp4 = Exponential(0.2, 0.01, 10)
exp5 = Exponential(0.2, 0.01, 20)

# Exponential Models with noise = 0.2, V0 = 1 and varying a
exp6 = Exponential(0.2, 1, 1)
exp7 = Exponential(0.2, 1, 2)
exp8 = Exponential(0.2, 1, 5)
exp9 = Exponential(0.2, 1, 10)
exp10 = Exponential(0.2, 1, 20)

# Exponential Models with noise = 0.2, V0 = 10 and varying a
exp11 = Exponential(0.2, 10, 1)
exp12 = Exponential(0.2, 10, 2)
exp13 = Exponential(0.2, 10, 5)
exp14 = Exponential(0.2, 10, 10)
exp15 = Exponential(0.2, 10, 20)

# Table: noise = 0.2
# Create data
data1 = [[exp1.V0, exp1.a, exp1.exp_CI([0.01,1], 0.95, "V0"), exp1.exp_CI([0.01,1], 0.99, "V0")],
        [exp2.V0, exp2.a, exp2.exp_CI([0.01,2], 0.95, "V0"), exp2.exp_CI([0.01,2], 0.99, "V0")],
        [exp3.V0, exp3.a, exp3.exp_CI([0.01,5], 0.95, "V0"), exp3.exp_CI([0.01,5], 0.99, "V0")],
        [exp4.V0, exp4.a, exp4.exp_CI([0.01,10], 0.95, "V0"), exp4.exp_CI([0.01,10], 0.99, "V0")],
        [exp5.V0, exp5.a, exp5.exp_CI([0.01,20], 0.95, "V0"), exp5.exp_CI([0.01,20], 0.99, "V0")]
        ]
data2 = [[exp1.V0, exp1.a, exp1.exp_CI([0.01,1], 0.95, "a"), exp1.exp_CI([0.01,1], 0.99, "a")],
        [exp2.V0, exp2.a, exp2.exp_CI([0.01,2], 0.95, "a"), exp2.exp_CI([0.01,2], 0.99, "a")],
        [exp3.V0, exp3.a, exp3.exp_CI([0.01,5], 0.95, "a"), exp3.exp_CI([0.01,5], 0.99, "a")],
        [exp4.V0, exp4.a, exp4.exp_CI([0.01,10], 0.95, "a"), exp4.exp_CI([0.01,10], 0.99, "a")],
        [exp5.V0, exp5.a, exp5.exp_CI([0.01,20], 0.95, "a"), exp5.exp_CI([0.01,20], 0.99, "a")]
        ]
data3 = [[exp6.V0, exp6.a, exp6.exp_CI([1,1], 0.95, "V0"), exp6.exp_CI([1,1], 0.99, "V0")],
        [exp7.V0, exp7.a, exp7.exp_CI([1,2], 0.95, "V0"), exp7.exp_CI([1,2], 0.99, "V0")],
        [exp8.V0, exp8.a, exp8.exp_CI([1,5], 0.95, "V0"), exp8.exp_CI([1,5], 0.99, "V0")],
        [exp9.V0, exp9.a, exp9.exp_CI([1,10], 0.95, "V0"), exp9.exp_CI([1,10], 0.99, "V0")],
        [exp10.V0, exp10.a, exp10.exp_CI([1,20], 0.95, "V0"), exp10.exp_CI([1,20], 0.99, "V0")]
        ]
data4 = [[exp6.V0, exp6.a, exp6.exp_CI([1,1], 0.95, "a"), exp6.exp_CI([1,1], 0.99, "a")],
        [exp7.V0, exp7.a, exp7.exp_CI([1,2], 0.95, "a"), exp7.exp_CI([1,2], 0.99, "a")],
        [exp8.V0, exp8.a, exp8.exp_CI([1,5], 0.95, "a"), exp8.exp_CI([1,5], 0.99, "a")],
        [exp9.V0, exp9.a, exp9.exp_CI([1,10], 0.95, "a"), exp9.exp_CI([1,10], 0.99, "a")],
        [exp10.V0, exp10.a, exp10.exp_CI([1,20], 0.95, "a"), exp10.exp_CI([1,20], 0.99, "a")]
        ]
# Define header names
col_names1 = ["V0", "a", "95% Confidence Interval for V0", "99% Confidence Interval for V0"]
col_names2 = ["V0", "a", "95% Confidence Interval for a", "99% Confidence Interval for a"]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))
# print(tabulate(data2, headers=col_names2, tablefmt="latex"))
# print(tabulate(data3, headers=col_names1, tablefmt="latex"))
# print(tabulate(data4, headers=col_names2, tablefmt="latex"))

# Exponential Models with noise = 0.5, V0 = 0.01 and varying a
exp1 = Exponential(0.5, 0.01, 1)
exp2 = Exponential(0.5, 0.01, 2)
exp3 = Exponential(0.5, 0.01, 5)
exp4 = Exponential(0.5, 0.01, 10)
exp5 = Exponential(0.5, 0.01, 20)

# Exponential Models with noise = 0.5, V0 = 1 and varying a
exp6 = Exponential(0.5, 1, 1)
exp7 = Exponential(0.5, 1, 2)
exp8 = Exponential(0.5, 1, 5)
exp9 = Exponential(0.5, 1, 10)
exp10 = Exponential(0.5, 1, 20)

# Exponential Models with noise = 0.5, V0 = 10 and varying a
exp11 = Exponential(0.5, 10, 1)
exp12 = Exponential(0.5, 10, 2)
exp13 = Exponential(0.5, 10, 5)
exp14 = Exponential(0.5, 10, 10)
exp15 = Exponential(0.5, 10, 20)

# Table: noise = 0.5
# Create data
data1 = [[exp1.V0, exp1.a, exp1.exp_CI([0.01,1], 0.95, "V0"), exp1.exp_CI([0.01,1], 0.99, "V0")],
        [exp2.V0, exp2.a, exp2.exp_CI([0.01,2], 0.95, "V0"), exp2.exp_CI([0.01,2], 0.99, "V0")],
        [exp3.V0, exp3.a, exp3.exp_CI([0.01,5], 0.95, "V0"), exp3.exp_CI([0.01,5], 0.99, "V0")],
        [exp4.V0, exp4.a, exp4.exp_CI([0.01,10], 0.95, "V0"), exp4.exp_CI([0.01,10], 0.99, "V0")],
        [exp5.V0, exp5.a, exp5.exp_CI([0.01,20], 0.95, "V0"), exp5.exp_CI([0.01,20], 0.99, "V0")]
        ]
data2 = [[exp1.V0, exp1.a, exp1.exp_CI([0.01,1], 0.95, "a"), exp1.exp_CI([0.01,1], 0.99, "a")],
        [exp2.V0, exp2.a, exp2.exp_CI([0.01,2], 0.95, "a"), exp2.exp_CI([0.01,2], 0.99, "a")],
        [exp3.V0, exp3.a, exp3.exp_CI([0.01,5], 0.95, "a"), exp3.exp_CI([0.01,5], 0.99, "a")],
        [exp4.V0, exp4.a, exp4.exp_CI([0.01,10], 0.95, "a"), exp4.exp_CI([0.01,10], 0.99, "a")],
        [exp5.V0, exp5.a, exp5.exp_CI([0.01,20], 0.95, "a"), exp5.exp_CI([0.01,20], 0.99, "a")]
        ]
data3 = [[exp6.V0, exp6.a, exp6.exp_CI([1,1], 0.95, "V0"), exp6.exp_CI([1,1], 0.99, "V0")],
        [exp7.V0, exp7.a, exp7.exp_CI([1,2], 0.95, "V0"), exp7.exp_CI([1,2], 0.99, "V0")],
        [exp8.V0, exp8.a, exp8.exp_CI([1,5], 0.95, "V0"), exp8.exp_CI([1,5], 0.99, "V0")],
        [exp9.V0, exp9.a, exp9.exp_CI([1,10], 0.95, "V0"), exp9.exp_CI([1,10], 0.99, "V0")],
        [exp10.V0, exp10.a, exp10.exp_CI([1,20], 0.95, "V0"), exp10.exp_CI([1,20], 0.99, "V0")]
        ]
data4 = [[exp6.V0, exp6.a, exp6.exp_CI([1,1], 0.95, "a"), exp6.exp_CI([1,1], 0.99, "a")],
        [exp7.V0, exp7.a, exp7.exp_CI([1,2], 0.95, "a"), exp7.exp_CI([1,2], 0.99, "a")],
        [exp8.V0, exp8.a, exp8.exp_CI([1,5], 0.95, "a"), exp8.exp_CI([1,5], 0.99, "a")],
        [exp9.V0, exp9.a, exp9.exp_CI([1,10], 0.95, "a"), exp9.exp_CI([1,10], 0.99, "a")],
        [exp10.V0, exp10.a, exp10.exp_CI([1,20], 0.95, "a"), exp10.exp_CI([1,20], 0.99, "a")]
        ]
# Define header names
col_names1 = ["V0", "a", "95% Confidence Interval for V0", "99% Confidence Interval for V0"]
col_names2 = ["V0", "a", "95% Confidence Interval for a", "99% Confidence Interval for a"]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))
# print(tabulate(data2, headers=col_names2, tablefmt="latex"))
# print(tabulate(data3, headers=col_names1, tablefmt="latex"))
# print(tabulate(data4, headers=col_names2, tablefmt="latex"))

# Exponential model with varying noise
exp1 = Exponential(0.05, 1, 1)
exp2 = Exponential(0.2, 1, 1)
exp3 = Exponential(0.5, 1, 1)
exp4 = Exponential(1, 1, 1)

# Exponential MLE results
data1 = [[exp1.V0, exp1.a, exp1.noise, exp1.get_mle([1,1])[0], exp1.get_mle([1,1])[1]],
         [exp2.V0, exp2.a, exp2.noise, exp2.get_mle([1,1])[0], exp2.get_mle([1,1])[1]],
         [exp3.V0, exp3.a, exp3.noise, exp3.get_mle([1,1])[0], exp3.get_mle([1,1])[1]],
         [exp4.V0, exp4.a, exp4.noise, exp4.get_mle([1,1])[0], exp4.get_mle([1,1])[1]]
         ]

col_names1 = ["V0", "a", "\u03C3", "MLE for V0", "MLE for a"]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))




#######################################################################################################################################

# Mendelsohn models with noise = 0.05 and varying b
mend1 = Mendelsohn(0.05, 0.01, 1, 0.1, 1)
mend2 = Mendelsohn(0.05, 0.01, 1, 0.5, 1)
mend3 = Mendelsohn(0.05, 0.01, 1, 2, 1)
mend4 = Mendelsohn(0.05, 0.01, 1, 2.5, 1)

data1 = [[mend1.V0, mend1.a, mend1.b, mend1.mend_CI([0.01, 1, 0.1], 0.95, "b"), mend1.mend_CI([0.01, 1, 0.1], 0.99, "b")],
         [mend2.V0, mend2.a, mend2.b, mend2.mend_CI([0.01, 1, 0.5], 0.95, "b"), mend2.mend_CI([0.01, 1, 0.5], 0.99, "b")],
         [mend3.V0, mend3.a, mend3.b, mend3.mend_CI([0.01, 1, 2], 0.95, "b"), mend3.mend_CI([0.01, 1, 2], 0.99, "b")],
         [mend4.V0, mend4.a, mend4.b, mend4.mend_CI([0.01, 1, 2.5], 0.95, "b"), mend4.mend_CI([0.01, 1, 2.5], 0.99, "b")]
         ]

col_names1 = ["V0", "a", "b", "95% Confidence Interval for b", "99% Confidence Interval for b"]

#print(tabulate(data1, headers=col_names1, tablefmt="latex"))

# Mendelsohn models with noise = 0.2 and varying b
mend5 = Mendelsohn(0.2, 0.01, 1, 0.1, 1)
mend6 = Mendelsohn(0.2, 0.01, 1, 0.5, 1)
mend7 = Mendelsohn(0.2, 0.01, 1, 2, 1)
mend8 = Mendelsohn(0.2, 0.01, 1, 2.5, 1)

data1 = [[mend5.V0, mend5.a, mend5.b, mend5.mend_CI([0.01, 1, 0.1], 0.95, "b"), mend5.mend_CI([0.01, 1, 0.1], 0.99, "b")],
         [mend6.V0, mend6.a, mend6.b, mend6.mend_CI([0.01, 1, 0.5], 0.95, "b"), mend6.mend_CI([0.01, 1, 0.5], 0.99, "b")],
         [mend7.V0, mend7.a, mend7.b, mend7.mend_CI([0.01, 1, 2], 0.95, "b"), mend7.mend_CI([0.01, 1, 2], 0.99, "b")],
         [mend8.V0, mend8.a, mend8.b, mend8.mend_CI([0.01, 1, 2.5], 0.95, "b"), mend8.mend_CI([0.01, 1, 2.5], 0.99, "b")]
         ]

col_names1 = ["V0", "a", "b", "95% Confidence Interval for b", "99% Confidence Interval for b"]

#print(tabulate(data1, headers=col_names1, tablefmt="latex"))

# Mendelsohn models with noise = 0.2 and varying b
mend9 = Mendelsohn(0.5, 0.01, 1, 0.1, 1)
mend10 = Mendelsohn(0.5, 0.01, 1, 0.5, 1)
mend11 = Mendelsohn(0.5, 0.01, 1, 2, 1)
mend12 = Mendelsohn(0.5, 0.01, 1, 2.5, 1)

data1 = [[mend9.V0, mend9.a, mend9.b, mend9.mend_CI([0.01, 1, 0.1], 0.95, "b"), mend9.mend_CI([0.01, 1, 0.1], 0.99, "b")],
         [mend10.V0, mend10.a, mend10.b, mend10.mend_CI([0.01, 1, 0.5], 0.95, "b"), mend10.mend_CI([0.01, 1, 0.5], 0.99, "b")],
         [mend11.V0, mend11.a, mend11.b, mend11.mend_CI([0.01, 1, 2], 0.95, "b"), mend11.mend_CI([0.01, 1, 2], 0.99, "b")],
         [mend12.V0, mend12.a, mend12.b, mend12.mend_CI([0.01, 1, 2.5], 0.95, "b"), mend12.mend_CI([0.01, 1, 2.5], 0.99, "b")]
         ]

col_names1 = ["V0", "a", "b", "95% Confidence Interval for b", "99% Confidence Interval for b"]

#print(tabulate(data1, headers=col_names1, tablefmt="latex"))

# Mendelsohn model with varying noise
mend1 = Mendelsohn(0.05, 1, 1, 0.5, 1)
mend2 = Mendelsohn(0.2, 1, 1, 0.5, 1)
mend3 = Mendelsohn(0.5, 1, 1, 0.5, 1)
mend4 = Mendelsohn(1, 1, 1, 0.5, 1)

# Mendelsohn MLE results
data1 = [[mend1.V0, mend1.a, mend1.b, mend1.noise, mend1.get_mle([1,1,0.5])[0], mend1.get_mle([1,1,0.5])[1], mend1.get_mle([1,1,0.5])[2]],
         [mend2.V0, mend2.a, mend2.b, mend2.noise, mend2.get_mle([1,1,0.5])[0], mend2.get_mle([1,1,0.5])[1], mend2.get_mle([1,1,0.5])[2]],
         [mend3.V0, mend3.a, mend3.b, mend3.noise, mend3.get_mle([1,1,0.5])[0], mend3.get_mle([1,1,0.5])[1], mend3.get_mle([1,1,0.5])[2]],
         [mend4.V0, mend4.a, mend4.b, mend4.noise, mend4.get_mle([1,1,0.5])[0], mend4.get_mle([1,1,0.5])[1], mend4.get_mle([1,1,0.5])[2]]
         ]

col_names1 = ["V0", "a", "b", "\u03C3", "MLE for V0", "MLE for a", "MLE for b"]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))

#######################################################################################################################################

# Logistic models with noise = 0.05 and varying r (Low Initial Volume)
log1 = Logistic(0.05, 0.01, 0.1, 10, 1)
log2 = Logistic(0.05, 0.01, 0.5, 10, 1)
log3 = Logistic(0.05, 0.01, 1, 10, 1)
log4 = Logistic(0.05, 0.01, 5, 10, 1)
log5 = Logistic(0.05, 0.01, 25, 10, 1)
log6 = Logistic(0.05, 0.01, 100, 10, 1)

data1 = [[log1.V0, log1.r, log1.k, log1.log_CI([0.01, 0.1, 10], 0.95, "r"), log1.log_CI([0.01, 0.1, 10], 0.99, "r")],
         [log2.V0, log2.r, log2.k, log2.log_CI([0.01, 0.5, 10], 0.95, "r"), log2.log_CI([0.01, 0.5, 10], 0.99, "r")],
         [log3.V0, log3.r, log3.k, log3.log_CI([0.01, 1, 10], 0.95, "r"), log3.log_CI([0.01, 1, 10], 0.99, "r")],
         [log4.V0, log4.r, log4.k, log4.log_CI([0.01, 5, 10], 0.95, "r"), log4.log_CI([0.01, 5, 10], 0.99, "r")],
         [log5.V0, log5.r, log5.k, log5.log_CI([0.01, 25, 10], 0.95, "r"), log5.log_CI([0.01, 25, 10], 0.99, "r")],
         [log6.V0, log6.r, log6.k, log6.log_CI([0.01, 100, 10], 0.95, "r"), log6.log_CI([0.01, 100, 10], 0.99, "r")]
         ]
data2 = [[log1.V0, log1.r, log1.k, log1.log_CI([0.01, 0.1, 10], 0.95, "K"), log1.log_CI([0.01, 0.1, 10], 0.99, "K")],
         [log2.V0, log2.r, log2.k, log2.log_CI([0.01, 0.5, 10], 0.95, "K"), log2.log_CI([0.01, 0.5, 10], 0.99, "K")],
         [log3.V0, log3.r, log3.k, log3.log_CI([0.01, 1, 10], 0.95, "K"), log3.log_CI([0.01, 1, 10], 0.99, "K")],
         [log4.V0, log4.r, log4.k, log4.log_CI([0.01, 5, 10], 0.95, "K"), log4.log_CI([0.01, 5, 10], 0.99, "K")],
         [log5.V0, log5.r, log5.k, log5.log_CI([0.01, 25, 10], 0.95, "K"), log5.log_CI([0.01, 25, 10], 0.99, "K")],
         [log6.V0, log6.r, log6.k, log6.log_CI([0.01, 100, 10], 0.95, "K"), log6.log_CI([0.01, 100, 10], 0.99, "K")]
         ]

col_names1 = ["V0", "r", "k", "95% Confidence Interval for r", "99% Confidence Interval for r"]
col_names2 = ["V0", "r", "k", "95% Confidence Interval for k", "99% Confidence Interval for k"]

#print(tabulate(data1, headers=col_names1, tablefmt="latex"))
#print(tabulate(data2, headers=col_names2, tablefmt="latex"))


# Logistic models with noise = 0.05 and varying k (Low Initial Volume)
log1 = Logistic(0.05, 0.01, 1, 1, 1)
log2 = Logistic(0.05, 0.01, 1, 5, 1)
log3 = Logistic(0.05, 0.01, 1, 10, 1)
log4 = Logistic(0.05, 0.01, 1, 25, 1)
log5 = Logistic(0.05, 0.01, 1, 100, 1)
log6 = Logistic(0.05, 0.01, 1, 1000, 1)

data1 = [[log1.V0, log1.r, log1.k, log1.log_CI([0.01, 1, 1], 0.95, "r"), log1.log_CI([0.01, 1, 1], 0.99, "r")],
         [log2.V0, log2.r, log2.k, log2.log_CI([0.01, 1, 5], 0.95, "r"), log2.log_CI([0.01, 1, 5], 0.99, "r")],
         [log3.V0, log3.r, log3.k, log3.log_CI([0.01, 1, 10], 0.95, "r"), log3.log_CI([0.01, 1, 10], 0.99, "r")],
         [log4.V0, log4.r, log4.k, log4.log_CI([0.01, 1, 25], 0.95, "r"), log4.log_CI([0.01, 1, 25], 0.99, "r")],
         [log5.V0, log5.r, log5.k, log5.log_CI([0.01, 1, 100], 0.95, "r"), log5.log_CI([0.01, 1, 100], 0.99, "r")],
         [log6.V0, log6.r, log6.k, log6.log_CI([0.01, 1, 1000], 0.95, "r"), log6.log_CI([0.01, 1, 1000], 0.99, "r")]
         ]
data2 = [[log1.V0, log1.r, log1.k, log1.log_CI([0.01, 1, 1], 0.95, "k"), log1.log_CI([0.01, 1, 1], 0.99, "k")],
         [log2.V0, log2.r, log2.k, log2.log_CI([0.01, 1, 5], 0.95, "k"), log2.log_CI([0.01, 1, 5], 0.99, "k")],
         [log3.V0, log3.r, log3.k, log3.log_CI([0.01, 1, 10], 0.95, "k"), log3.log_CI([0.01, 1, 10], 0.99, "k")],
         [log4.V0, log4.r, log4.k, log4.log_CI([0.01, 1, 25], 0.95, "k"), log4.log_CI([0.01, 1, 25], 0.99, "k")],
         [log5.V0, log5.r, log5.k, log5.log_CI([0.01, 1, 100], 0.95, "k"), log5.log_CI([0.01, 1, 100], 0.99, "k")],
         [log6.V0, log6.r, log6.k, log6.log_CI([0.01, 1, 1000], 0.95, "k"), log6.log_CI([0.01, 1, 1000], 0.99, "k")]
         ]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))
# print(tabulate(data2, headers=col_names2, tablefmt="latex"))


# Logistic models with noise = 0.05 and varying k (High Initial Volume)
log1 = Logistic(0.05, 10, 1, 1, 1)
log2 = Logistic(0.05, 10, 1, 5, 1)
log3 = Logistic(0.05, 10, 1, 10, 1)
log4 = Logistic(0.05, 10, 1, 25, 1)
log5 = Logistic(0.05, 10, 1, 100, 1)
log6 = Logistic(0.05, 10, 1, 1000, 1)

data1 = [[log1.V0, log1.r, log1.k, log1.log_CI([10, 1, 1], 0.95, "r"), log1.log_CI([10, 1, 1], 0.99, "r")],
         [log2.V0, log2.r, log2.k, log2.log_CI([10, 1, 5], 0.95, "r"), log2.log_CI([10, 1, 5], 0.99, "r")],
         [log4.V0, log4.r, log4.k, log4.log_CI([10, 1, 25], 0.95, "r"), log4.log_CI([10, 1, 25], 0.99, "r")],
         [log5.V0, log5.r, log5.k, log5.log_CI([10, 1, 100], 0.95, "r"), log5.log_CI([10, 1, 100], 0.99, "r")],
         [log6.V0, log6.r, log6.k, log6.log_CI([10, 1, 1000], 0.95, "r"), log6.log_CI([10, 1, 1000], 0.99, "r")]
         ]
data2 = [[log1.V0, log1.r, log1.k, log1.log_CI([10, 1, 1], 0.95, "k"), log1.log_CI([10, 1, 1], 0.99, "k")],
         [log2.V0, log2.r, log2.k, log2.log_CI([10, 1, 5], 0.95, "k"), log2.log_CI([10, 1, 5], 0.99, "k")],
         [log4.V0, log4.r, log4.k, log4.log_CI([10, 1, 25], 0.95, "k"), log4.log_CI([10, 1, 25], 0.99, "k")],
         [log5.V0, log5.r, log5.k, log5.log_CI([10, 1, 100], 0.95, "k"), log5.log_CI([10, 1, 100], 0.99, "k")],
         [log6.V0, log6.r, log6.k, log6.log_CI([10, 1, 1000], 0.95, "k"), log6.log_CI([10, 1, 1000], 0.99, "k")]
         ]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))
# print(tabulate(data2, headers=col_names2, tablefmt="latex"))

# Logistic model with varying noise
log1 = Logistic(0.05, 1, 1, 10, 1)
log2 = Logistic(0.2, 1, 1, 10, 1)
log3 = Logistic(0.5, 1, 1, 10, 1)
log4 = Logistic(1, 1, 1, 10, 1)

# Logistic MLE results
data1 = [[log1.V0, log1.r, log1.k, log1.noise, log1.get_mle([1,1,10])[0], log1.get_mle([1,1,10])[1], log1.get_mle([1,1,10])[2]],
         [log2.V0, log2.r, log2.k, log2.noise, log2.get_mle([1,1,10])[0], log2.get_mle([1,1,10])[1], log2.get_mle([1,1,10])[2]],
         [log3.V0, log3.r, log3.k, log3.noise, log3.get_mle([1,1,10])[0], log3.get_mle([1,1,10])[1], log3.get_mle([1,1,10])[2]],
         [log4.V0, log4.r, log4.k, log4.noise, log4.get_mle([1,1,10])[0], log4.get_mle([1,1,10])[1], log4.get_mle([1,1,10])[2]]
         ]

col_names1 = ["V0", "r", "k", "\u03C3", "MLE for V0", "MLE for r", "MLE for k"]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))

#######################################################################################################################################

# Gompertz models with noise = 0.05 and varying r (Low Initial Volume)
gomp1 = Gompertz(0.05, 0.01, 0.1, 10)
gomp2 = Gompertz(0.05, 0.01, 0.5, 10)
gomp3 = Gompertz(0.05, 0.01, 1, 10)
gomp4 = Gompertz(0.05, 0.01, 5, 10)
gomp5 = Gompertz(0.05, 0.01, 25, 10)
gomp6 = Gompertz(0.05, 0.01, 100, 10)

data1 = [[gomp1.V0, gomp1.r, gomp1.k, gomp1.gomp_CI([0.01, 0.1, 10], 0.95, "r"), gomp1.gomp_CI([0.01, 0.1, 10], 0.99, "r")],
         [gomp2.V0, gomp2.r, gomp2.k, gomp2.gomp_CI([0.01, 0.5, 10], 0.95, "r"), gomp2.gomp_CI([0.01, 0.5, 10], 0.99, "r")],
         [gomp3.V0, gomp3.r, gomp3.k, gomp3.gomp_CI([0.01, 1, 10], 0.95, "r"), gomp3.gomp_CI([0.01, 1, 10], 0.99, "r")],
         [gomp4.V0, gomp4.r, gomp4.k, gomp4.gomp_CI([0.01, 5, 10], 0.95, "r"), gomp4.gomp_CI([0.01, 5, 10], 0.99, "r")],
         [gomp5.V0, gomp5.r, gomp5.k, gomp5.gomp_CI([0.01, 25, 10], 0.95, "r"), gomp5.gomp_CI([0.01, 25, 10], 0.99, "r")],
         [gomp6.V0, gomp6.r, gomp6.k, gomp6.gomp_CI([0.01, 100, 10], 0.95, "r"), gomp6.gomp_CI([0.01, 100, 10], 0.99, "r")]
         ]
data2 = [[gomp1.V0, gomp1.r, gomp1.k, gomp1.gomp_CI([0.01, 0.1, 10], 0.95, "k"), gomp1.gomp_CI([0.01, 0.1, 10], 0.99, "k")],
         [gomp2.V0, gomp2.r, gomp2.k, gomp2.gomp_CI([0.01, 0.5, 10], 0.95, "k"), gomp2.gomp_CI([0.01, 0.5, 10], 0.99, "k")],
         [gomp3.V0, gomp3.r, gomp3.k, gomp3.gomp_CI([0.01, 1, 10], 0.95, "k"), gomp3.gomp_CI([0.01, 1, 10], 0.99, "k")],
         [gomp4.V0, gomp4.r, gomp4.k, gomp4.gomp_CI([0.01, 5, 10], 0.95, "k"), gomp4.gomp_CI([0.01, 5, 10], 0.99, "k")],
         [gomp5.V0, gomp5.r, gomp5.k, gomp5.gomp_CI([0.01, 25, 10], 0.95, "k"), gomp5.gomp_CI([0.01, 25, 10], 0.99, "k")],
         [gomp6.V0, gomp6.r, gomp6.k, gomp6.gomp_CI([0.01, 100, 10], 0.95, "k"), gomp6.gomp_CI([0.01, 100, 10], 0.99, "k")]
         ]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))
# print(tabulate(data2, headers=col_names2, tablefmt="latex"))



# Gompertz models with noise = 0.05 and varying k (Low Initial Volume)
gomp1 = Gompertz(0.05, 0.01, 1, 1)
gomp2 = Gompertz(0.05, 0.01, 1, 5)
gomp3 = Gompertz(0.05, 0.01, 1, 10)
gomp4 = Gompertz(0.05, 0.01, 1, 25)
gomp5 = Gompertz(0.05, 0.01, 1, 100)
gomp6 = Gompertz(0.05, 0.01, 1, 1000)

data1 = [[gomp1.V0, gomp1.r, gomp1.k, gomp1.gomp_CI([0.01, 1, 1], 0.95, "r"), gomp1.gomp_CI([0.01, 1, 1], 0.99, "r")],
         [gomp2.V0, gomp2.r, gomp2.k, gomp2.gomp_CI([0.01, 1, 5], 0.95, "r"), gomp2.gomp_CI([0.01, 1, 5], 0.99, "r")],
         [gomp3.V0, gomp3.r, gomp3.k, gomp3.gomp_CI([0.01, 1, 10], 0.95, "r"), gomp3.gomp_CI([0.01, 1, 10], 0.99, "r")],
         [gomp4.V0, gomp4.r, gomp4.k, gomp4.gomp_CI([0.01, 1, 25], 0.95, "r"), gomp4.gomp_CI([0.01, 1, 25], 0.99, "r")],
         [gomp5.V0, gomp5.r, gomp5.k, gomp5.gomp_CI([0.01, 1, 100], 0.95, "r"), gomp5.gomp_CI([0.01, 1, 100], 0.99, "r")],
         [gomp6.V0, gomp6.r, gomp6.k, gomp6.gomp_CI([0.01, 1, 1000], 0.95, "r"), gomp6.gomp_CI([0.01, 1, 1000], 0.99, "r")]
         ]
data2 = [[gomp1.V0, gomp1.r, gomp1.k, gomp1.gomp_CI([0.01, 1, 1], 0.95, "k"), gomp1.gomp_CI([0.01, 1, 1], 0.99, "k")],
         [gomp2.V0, gomp2.r, gomp2.k, gomp2.gomp_CI([0.01, 1, 5], 0.95, "k"), gomp2.gomp_CI([0.01, 1, 5], 0.99, "k")],
         [gomp3.V0, gomp3.r, gomp3.k, gomp3.gomp_CI([0.01, 1, 10], 0.95, "k"), gomp3.gomp_CI([0.01, 1, 10], 0.99, "k")],
         [gomp4.V0, gomp4.r, gomp4.k, gomp4.gomp_CI([0.01, 1, 25], 0.95, "k"), gomp4.gomp_CI([0.01, 1, 25], 0.99, "k")],
         [gomp5.V0, gomp5.r, gomp5.k, gomp5.gomp_CI([0.01, 1, 100], 0.95, "k"), gomp5.gomp_CI([0.01, 1, 100], 0.99, "k")],
         [gomp6.V0, gomp6.r, gomp6.k, gomp6.gomp_CI([0.01, 1, 1000], 0.95, "k"), gomp6.gomp_CI([0.01, 1, 1000], 0.99, "k")]
         ]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))
# print(tabulate(data2, headers=col_names2, tablefmt="latex"))


# Gompertz models with noise = 0.05 and varying k (High Initial Volume)
gomp1 = Gompertz(0.05, 10, 1, 1)
gomp2 = Gompertz(0.05, 10, 1, 5)
gomp3 = Gompertz(0.05, 10, 1, 10)
gomp4 = Gompertz(0.05, 10, 1, 25)
gomp5 = Gompertz(0.05, 10, 1, 100)
gomp6 = Gompertz(0.05, 10, 1, 1000)

data1 = [[gomp1.V0, gomp1.r, gomp1.k, gomp1.gomp_CI([10, 1, 1], 0.95, "r"), gomp1.gomp_CI([10, 1, 1], 0.99, "r")],
         [gomp2.V0, gomp2.r, gomp2.k, gomp2.gomp_CI([10, 1, 5], 0.95, "r"), gomp2.gomp_CI([10, 1, 5], 0.99, "r")],
         [gomp4.V0, gomp4.r, gomp4.k, gomp4.gomp_CI([10, 1, 25], 0.95, "r"), gomp4.gomp_CI([10, 1, 25], 0.99, "r")],
         [gomp5.V0, gomp5.r, gomp5.k, gomp5.gomp_CI([10, 1, 100], 0.95, "r"), gomp5.gomp_CI([10, 1, 100], 0.99, "r")],
         [gomp6.V0, gomp6.r, gomp6.k, gomp6.gomp_CI([10, 1, 1000], 0.95, "r"), gomp6.gomp_CI([10, 1, 1000], 0.99, "r")]
         ]
data2 = [[gomp1.V0, gomp1.r, gomp1.k, gomp1.gomp_CI([10, 1, 1], 0.95, "k"), gomp1.gomp_CI([10, 1, 1], 0.99, "k")],
         [gomp2.V0, gomp2.r, gomp2.k, gomp2.gomp_CI([10, 1, 5], 0.95, "k"), gomp2.gomp_CI([10, 1, 5], 0.99, "k")],
         [gomp4.V0, gomp4.r, gomp4.k, gomp4.gomp_CI([10, 1, 25], 0.95, "k"), gomp4.gomp_CI([10, 1, 25], 0.99, "k")],
         [gomp5.V0, gomp5.r, gomp5.k, gomp5.gomp_CI([10, 1, 100], 0.95, "k"), gomp5.gomp_CI([10, 1, 100], 0.99, "k")],
         [gomp6.V0, gomp6.r, gomp6.k, gomp6.gomp_CI([10, 1, 1000], 0.95, "k"), gomp6.gomp_CI([10, 1, 1000], 0.99, "k")]
         ]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))
# print(tabulate(data2, headers=col_names2, tablefmt="latex"))


# Gompertz model with varying noise
gomp1 = Gompertz(0.05, 1, 1, 10)
gomp2 = Gompertz(0.2, 1, 1, 10)
gomp3 = Gompertz(0.5, 1, 1, 10)
gomp4 = Gompertz(1, 1, 1, 10)

# Logistic MLE results
data1 = [[gomp1.V0, gomp1.r, gomp1.k, gomp1.noise, gomp1.get_mle([1,1,10])[0], gomp1.get_mle([1,1,10])[1], gomp1.get_mle([1,1,10])[2]],
         [gomp2.V0, gomp2.r, gomp2.k, gomp2.noise, gomp2.get_mle([1,1,10])[0], gomp2.get_mle([1,1,10])[1], gomp2.get_mle([1,1,10])[2]],
         [gomp3.V0, gomp3.r, gomp3.k, gomp3.noise, gomp3.get_mle([1,1,10])[0], gomp3.get_mle([1,1,10])[1], gomp3.get_mle([1,1,10])[2]],
         [gomp4.V0, gomp4.r, gomp4.k, gomp4.noise, gomp4.get_mle([1,1,10])[0], gomp4.get_mle([1,1,10])[1], gomp4.get_mle([1,1,10])[2]]
         ]

col_names1 = ["V0", "r", "k", "\u03C3", "MLE for V0", "MLE for r", "MLE for k"]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))

#######################################################################################################################################


# Bertalanffy models with noise = 0.05 (Low Initial Volume)
bert1 = Bertalanffy(0.05, 0.01, 1, 1, 1)
bert2 = Bertalanffy(0.05, 0.01, 2, 1, 1)
bert3 = Bertalanffy(0.05, 0.01, 1, 2, 1)

data1 = [[bert1.V0, bert1.b, bert1.d, bert1.bert_CI([0.01, 1, 1], 0.95, "b"), bert1.bert_CI([0.01, 1, 1], 0.99, "b")],
         [bert2.V0, bert2.b, bert2.d, bert2.bert_CI([0.01, 2, 1], 0.95, "b"), bert2.bert_CI([0.01, 2, 1], 0.99, "b")],
         [bert3.V0, bert3.b, bert3.d, bert3.bert_CI([0.01, 1, 2], 0.95, "b"), bert3.bert_CI([0.01, 1, 2], 0.99, "b")]
         ]
data2 = [[bert1.V0, bert1.b, bert1.d, bert1.bert_CI([0.01, 1, 1], 0.95, "d"), bert1.bert_CI([0.01, 1, 1], 0.99, "d")],
         [bert2.V0, bert2.b, bert2.d, bert2.bert_CI([0.01, 2, 1], 0.95, "d"), bert2.bert_CI([0.01, 2, 1], 0.99, "d")],
         [bert3.V0, bert3.b, bert3.d, bert3.bert_CI([0.01, 1, 2], 0.95, "d"), bert3.bert_CI([0.01, 1, 2], 0.99, "d")]
         ]

col_names1 = ["V0", "b", "d", "95% Confidence Interval for b", "99% Confidence Interval for b"]
col_names2 = ["V0", "b", "d", "95% Confidence Interval for d", "99% Confidence Interval for d"]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))
# print(tabulate(data2, headers=col_names2, tablefmt="latex"))

# Bertalanffy models with noise = 0.05 (High Initial Volume)
bert1 = Bertalanffy(0.05, 10, 1, 1, 1)
bert2 = Bertalanffy(0.05, 10, 2, 1, 1)
bert3 = Bertalanffy(0.05, 10, 1, 2, 1)

data1 = [[bert1.V0, bert1.b, bert1.d, bert1.bert_CI([10, 1, 1], 0.95, "b"), bert1.bert_CI([10, 1, 1], 0.99, "b")],
         [bert2.V0, bert2.b, bert2.d, bert2.bert_CI([10, 2, 1], 0.95, "b"), bert2.bert_CI([10, 2, 1], 0.99, "b")],
         [bert3.V0, bert3.b, bert3.d, bert3.bert_CI([10, 1, 2], 0.95, "b"), bert3.bert_CI([10, 1, 2], 0.99, "b")]
         ]
data2 = [[bert1.V0, bert1.b, bert1.d, bert1.bert_CI([10, 1, 1], 0.95, "d"), bert1.bert_CI([10, 1, 1], 0.99, "d")],
         [bert2.V0, bert2.b, bert2.d, bert2.bert_CI([10, 2, 1], 0.95, "d"), bert2.bert_CI([10, 2, 1], 0.99, "d")],
         [bert3.V0, bert3.b, bert3.d, bert3.bert_CI([10, 1, 2], 0.95, "d"), bert3.bert_CI([10, 1, 2], 0.99, "d")]
         ]

col_names1 = ["V0", "b", "d", "95% Confidence Interval for b", "99% Confidence Interval for b"]
col_names2 = ["V0", "b", "d", "95% Confidence Interval for d", "99% Confidence Interval for d"]

# print(tabulate(data1, headers=col_names1, tablefmt="latex"))
# print(tabulate(data2, headers=col_names2, tablefmt="latex"))

# Bertalanffy model with varying noise
bert1 = Bertalanffy(0.05, 1, 1, 10, 1)
bert2 = Bertalanffy(0.2, 1, 1, 10, 1)
bert3 = Bertalanffy(0.5, 1, 1, 10, 1)
bert4 = Bertalanffy(1, 1, 1, 10, 1)

# Logistic MLE results
data1 = [[bert1.V0, bert1.b, bert1.d, bert1.noise, bert1.get_mle([1,1,10])[0], bert1.get_mle([1,1,10])[1], bert1.get_mle([1,1,10])[2]],
         [bert2.V0, bert2.b, bert2.d, bert2.noise, bert2.get_mle([1,1,10])[0], bert2.get_mle([1,1,10])[1], bert2.get_mle([1,1,10])[2]],
         [bert3.V0, bert3.b, bert3.d, bert3.noise, bert3.get_mle([1,1,10])[0], bert3.get_mle([1,1,10])[1], bert3.get_mle([1,1,10])[2]],
         [bert4.V0, bert4.b, bert4.d, bert4.noise, bert4.get_mle([1,1,10])[0], bert4.get_mle([1,1,10])[1], bert4.get_mle([1,1,10])[2]]
         ]

col_names1 = ["V0", "b", "d", "\u03C3", "MLE for V0", "MLE for b", "MLE for d"]

print(tabulate(data1, headers=col_names1, tablefmt="latex"))
















# log = Logistic(0.05, 5, 1, 10, 10)
# gomp = Gompertz(0.05)
# bert = Bertalanffy(0.05, 1, 2, 1, 1)


# log.obj_func_plot(t_eval, [1, 5, 10], 1)
# exp.obj_func_plot(t_eval, [1, 1], 1)
# mend.obj_func_plot(t_eval, [1, 1, 2], 1)
# gomp.obj_func_plot(t_eval, [1, 5, 10], 1)
# bert.obj_func_plot(t_eval, [1, 5, 5], 1)

# print(log.num_sol_opt(10))
# print(log.get_p(10))
# plt.plot(np.linspace(0, 10, len(log.get_v(10))), log.get_v(10))
# plt.show()
# print(log.num_sol_opt(10))
# print(log.get_v(10))
# print(log.get_p(10))


# print(gomp.find_coeff(1, 1, 10, 1, 5))
# print(gomp.find_other_coeff(1, 1, 10, 1, 5))
# #print(gomp.opt_d(1, 1, 10, 10000, 5))
# print(gomp.opt_sol(1, 1, 10, 1, 5))
# print(4*np.log(1/10) - 4*gomp.find_coeff(1, 1, 10, 1, 5))
# print(gomp.co_state(1, 1, 10, 1, 5))
# gomp.plot_sol(1, 1, 10, 10000, 5)



# print(log.num_sol_sys(10, 1, -1).y)
# print(log.num_sol_sys(10, 1, -5).y)
# log.plot_v(10, 1)
# print(log.find_p0(10, 1))
# print(log.num_sol_sys(log.find_p0(10, 1).root, 10, 1))
# # log.plot_res(10, 1)
# print(log.opt_d(10, 1))
# log.plot_d(2, 9.99)



# print(bert.get_v(10, 1))
# print(bert.get_p(10, 1))
# print(bert.opt_d(10, 1))
# bert.plot_v(10, 1)
# bert.plot_d(10, 1)


# exp.plot_v(2, 2, 10, 0.001)
# exp.plot_p(2, 2, 10, 0.001)
# exp.plot_D(2, 2, 10, 0.001)
# exp.plot_v_D(2, 1, 10, 1)


# mend.plot_v(10, 1)
# mend.plot_res(10, 1)
# mend.plot_v_1(1, 1, 2, 10, 1)
# mend.plot_p(1, 1, 2, 10, 1)


# gomp.plot_v(1, 5, 20, 10, 0.000001)
# gomp.plot_p(1, 5, 20, 10, 1)
# gomp.plot_D(1, 5, 20, 10, 0.000001)
#print([np.real(gomp.p(i, 1, 5, 20, 10, 0.001)) for i in np.linspace(0, 10, 101)])
# print(mend.res_bc(-0.001,10,2))
# mend.plot_res(10, 5)
# mend.plot_v_d(10, 5)
# print(mend.num_sol_sys(-40, 10, 5))



# y=[-mend8.mend_mle([0.01,1,2.5]).fun - mend8.log_l([mend8.mend_mle([0.01,1,2.5]).x[0], mend8.mend_mle([0.01,1,2.5]).x[1], b], mend8.noise) - chi2.ppf(0.95, 1)/2 for b in np.linspace(0, 3, 101)]
# plt.plot(np.linspace(0, 3, 101), y)
# plt.show()



#################################################################################################################################################################


