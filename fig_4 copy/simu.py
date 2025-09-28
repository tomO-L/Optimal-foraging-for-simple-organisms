import scipy as sp
import pickle as pk
import os
import numpy as np
import dill
import tqdm
import matplotlib.pyplot as plt

def gamma_of_rho(rho,t_d0):

    res = rho/t_d0

    return res

def depletion(rho_r, pars):
    
    t_d0 = pars[0]
    t = pars[1]
    
    f = rho_r * (1 - np.exp(-t/t_d0))

    return f

def simu(rho_init, T, t_d0, n_r, r, v_lim, eta_target, noise_std_ratio = 0.01, noise=0, learning=False, trade_off=False, alpha=0, a=1):

    rho = np.copy(rho_init)
    rho = np.concatenate((rho,rho[:2*n_r]))
    rho = np.concatenate((rho[-2*n_r:], rho))
    rho_0 = np.copy(rho)
    length = len(rho)
    v = 2.5
    v_list = []

    ### Noise and mouth parameters ###
    n_mouth = 0
    noise_max_memory = 5
    noise_memory = noise_max_memory
    noise_std = n_r * noise_std_ratio

    ###

    n_pos = 0
    t = 0

    n_stop = n_r
    jump_size = n_r


    while n_pos + n_r < length:

        ### Speed ###
            
        v = 100
            
        ### Mouth jitter ###

        if noise_memory==noise_max_memory:

            noise = np.random.normal(scale = noise_std)
            noise = int(np.round(noise))
            noise_memory = 0

            while n_pos+noise<0 or n_pos+noise>length:

                noise = np.random.normal(scale = noise_std)
                noise = int(np.round(noise))

        else: 
            
            noise_memory += 1

        ###
        n_mouth = n_pos + noise
        print(n_mouth)
        
        ### Feeding ###

        rho_depleted = depletion(rho[n_mouth:n_mouth + n_r], [t_d0, r / n_r / v, ])
        rho[n_mouth:n_mouth + n_r] = rho[n_mouth:n_mouth + n_r] - rho_depleted
        eta = np.sum(rho_depleted) * v 

        v_list.append(v)
        n_pos = n_pos + 1
        t = t + r / n_r / v

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
    
    # with open(f"../fig_2/opt_0.pkl", 'rb') as fileopen:
    #     opt_eta_target_list, opt_eta_average_list = dill.load(fileopen)
    aux = np.load(os.path.join(script_dir_parent, 'fig_2', "opt_0.npy"))
    opt_eta_target_list = aux[0]
    opt_eta_average_list = aux[1]

    opt_eta_target = opt_eta_target_list[np.argmax(opt_eta_average_list)]

    eta_target = opt_eta_target #4 * r/t_d0
    noise = 0.
    learning = False
    alpha = 1e-1 # learning coeficient. alpha = 0 when you care just about the past, and alpha = 1 when you care just about the present
    trade_off = False
    a = 250 # trade-off coeficient. The higher it is, the less trade-off you have

    T = 100

    # with open("../environment_test.pkl", 'rb') as fileopen:
    #     rho = pk.load(fileopen) * 10
        
    rho = np.load(os.path.join(script_dir_parent, 'rho_0.npy'))
    rho = np.ones(len(rho))

    ### Simulation ###
    inpt = [rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
    outpt = simu(rho, T, t_d0, n_r, r, v_lim, eta_target, noise_std_ratio = 5, noise=noise, learning=learning, trade_off=trade_off, alpha=alpha, a=a)
    ### Saving simulation ###

    res_sim = [inpt,outpt]

    name = "simulation"

    os.makedirs(name, exist_ok=True )

    with open(os.path.join(script_dir, name, f"{name}.pkl"), 'wb') as file_to_write:
        dill.dump(res_sim, file_to_write)
