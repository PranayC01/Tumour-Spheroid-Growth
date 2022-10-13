import numpy as np
import math

def Mendelsohn(v, a, b):
    return a*(v**b)

def Exponential(v, a):
    return Mendelsohn(v, a, 1)

def Logistic(v, r, k):
    return r*v*(1-v/k)

def Gompertz(v, r, k):
    return r*math.log(k/v)*v

def Bertalanffy(v, r, d):
    return r*(v**(2/3))-d*v
