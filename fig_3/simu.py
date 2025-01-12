import scipy as sp
import pickle as pk
import os
import numpy as np
import dill
import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
    rho = np.concatenate((rho,rho[:2*n_r]))
    rho = np.concatenate((rho[-2*n_r:], rho))
    length = len(rho)
    v = v_lim
    v_list = []

    n = 0
    t = 0

    while n + n_r < length:

        ###################### Feeding ###
        
        rho_depleted = depletion(rho[n:n + n_r], [t_d0, r / n_r / v, ])
        rho[n:n + n_r] = rho[n:n + n_r] - rho_depleted
        eta = np.sum(rho_depleted) * v * np.random.uniform(1 - noise, 1 + noise)

        v_list.append(v)
		
		### math rule 1 ###
		
        if eta!=0:

            #est_rho_0 = rho_0[n+int(n_r/2)]
            est_rho_0 = eta/v * 1/(1-np.exp(-r/(v*t_d0))) #rho_0[n+int(n_r/2)]

            if est_rho_0>eta_target*t_d0/r:
                v = -r/n_r/t_d0 / np.log(eta_target*t_d0/r/est_rho_0)*n_r
                v = min(v,v_lim)
            else:
                v = v_lim

        else:
            v = v_lim
        n += 1

    schedule = r / n_r / np.array(v_list)

    return schedule



if __name__=='__main__':

    script_dir = os.path.dirname(__file__)
    script_dir_parent = os.path.abspath(os.path.join(script_dir, os.pardir))

    ### Model parameters ###
    t_d0 = 1 # depletion time scale
    n_r = 20 # size in bins of the depletion section
    r = 1*n_r # physical size of the depletion section
    v_lim = 200 # maximum speed
    
    aux = np.load(os.path.join(script_dir_parent, 'fig_2', "opt_0.npy"))
    opt_eta_target_list = aux[0]
    opt_eta_average_list = aux[1]

    opt_eta_target = opt_eta_target_list[np.argmax(opt_eta_average_list)]

    eta_target = opt_eta_target
    noise = 0.
    learning = False
    alpha = 1e-1 # learning coeficient. alpha = 0 when you care just about the past, and alpha = 1 when you care just about the present
    trade_off = False
    a = 250 # trade-off coeficient. The higher it is, the less trade-off you have

    T = 100

    # with open("../environment_test.pkl", 'rb') as fileopen:
    #     rho = pk.load(fileopen) * 10
    
    rho = np.load(os.path.join(script_dir_parent, 'rho_0.npy'))


    ### Simulation ###
    inpt = [rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
    outpt = simu(rho, T, t_d0, n_r, r, v_lim, eta_target, noise=noise, learning=learning, trade_off=trade_off, alpha=alpha, a=a)
    ### Saving simulation ###

    res_sim = [inpt,outpt]

    name = "simulation"

    os.makedirs(name, exist_ok=True )

    with open(os.path.join(script_dir, name, f"{name}.pkl"), 'wb') as file_to_write:
        dill.dump(res_sim, file_to_write)
