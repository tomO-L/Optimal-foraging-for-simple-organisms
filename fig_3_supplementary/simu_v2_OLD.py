import scipy as sp
import pickle as pk
import os
import numpy as np
import dill
import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds

def gamma_of_rho(rho,t_d0):

    res = rho/t_d0

    return res

def depletion(rho_r, pars):
    
    t_d0 = pars[0]
    t = pars[1]
    
    f = rho_r * (1 - np.exp(-t/t_d0))

    return f

def simu(rho_init, T, t_d0, n_r, r, v_lim, eta_target, noise=0, learning=False, trade_off=False, alpha=0, a=1):

    rho_0 = np.copy(rho_init)
    rho = np.copy(rho_init)
    rho = np.concatenate((rho,np.ones(n_r*2)*rho[-1]))
    rho = np.concatenate((np.ones(n_r * 2) * rho[0], rho))
    length = len(rho)
    v = 2.5
    v_list = []

    n = 0
    t = 0
		
    def J_minimize(v,pars):
		
        t_d = pars[0]
        r = pars[1]
        eta_star = pars[2]
        eta_m_star = pars[3]
		
        res = (eta_m_star * v*t_d/r * (np.exp(r/(t_d*v)) - 1) - eta_star)**2
		
        return res

    while n + n_r < length:
        ###################### Feeding ###
		
		### math rule 2 ###
		
        rho_depleted = depletion(rho[n:n + n_r], [t_d0, r / n_r / v, ])
        eta_test = np.sum(rho_depleted) * v * np.random.uniform(1 - noise, 1 + noise)
        
        if eta_test!=0:

            # minimize J
            
            v = minimize(J_minimize, v, [t_d0,r,eta_test,eta_target], bounds=Bounds(lb=0, ub=v_lim)).x[0]

        else:
            v = v_lim
        
        rho_depleted = depletion(rho[n:n + n_r], [t_d0, r / n_r / v, ])
        rho[n:n + n_r] = rho[n:n + n_r] - rho_depleted
        eta = np.sum(rho_depleted) * v * np.random.uniform(1 - noise, 1 + noise)
        
        v_list.append(v)
        n += 1
    schedule = r / n_r / np.array(v_list)

    return schedule



if __name__=='__main__':

    ### Model parameters ###
    t_d0 = 1 # depletion time scale
    n_r = 20 # size in bins of the depletion section
    r = 1*n_r # physical size of the depletion section
    v_lim = 200 # maximum speed
    
    eta_target = 4 * r/t_d0
    noise = 0.
    learning = False
    alpha = 1e-1 # learning coeficient. alpha = 0 when you care just about the past, and alpha = 1 when you care just about the present
    trade_off = False
    a = 250 # trade-off coeficient. The higher it is, the less trade-off you have

    T = 100

    with open("../environment_test.pkl", 'rb') as fileopen:
        rho = pk.load(fileopen) * 10


    ### Simulation ###
    inpt = [rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
    outpt = simu(rho, T, t_d0, n_r, r, v_lim, eta_target, noise=noise, learning=learning, trade_off=trade_off, alpha=alpha, a=a)
    ### Saving simulation ###

    res_sim = [inpt,outpt]

    name = "simulation_v2"

    os.makedirs(name, exist_ok=True )

    with open(f"{name}/{name}.pkl", 'wb') as file_to_write:
        dill.dump(res_sim, file_to_write)

