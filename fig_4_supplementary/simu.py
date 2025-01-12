import scipy as sp
import pickle as pk
import os
import numpy as np
import dill
import tqdm
import matplotlib.pyplot as plt

def gamma_of_rho(rho,t_d0,rho_s):

    res = np.copy(rho)

    res[np.where(rho>=rho_s)] = rho_s/t_d0
    
    res[np.where(rho<rho_s)] = rho[np.where(rho<rho_s)]/t_d0

    return res

def local_depletion(rho_r, pars):

    t_d0 = pars[0]
    rho_s = pars[1]
    t = pars[2]

    ts = (rho_r-rho_s)/rho_s*t_d0

    if rho_r>rho_s:

        if t<ts:

            f = rho_s/t_d0*t

        else:

            f = rho_s/t_d0*ts + rho_s * (1 - np.exp(-(t-ts)/t_d0)) #### 

    else:

        f = rho_r * (1 - np.exp(-t/t_d0))

    return f


def depletion(rho_r, pars):

    t_d0 = pars[0]
    rho_s = pars[1]
    t = pars[2]

    f = np.zeros(len(rho_r))

    if np.isscalar(t):

        for i in range(len(rho_r)):

            f[i] = local_depletion(rho_r[i], pars)

    else:
        
        for i in range(len(rho_r)):

            f[i] = local_depletion(rho_r[i], [t_d0,rho_s,t[i]])

    return f


def simu(rho_init, T, t_d0, n_r, r, v_lim, eta_target, noise=0, learning=False, trade_off=False, alpha=0, a=1):

    rho = np.copy(rho_init)
    rho = np.concatenate((rho,rho[:2*n_r]))
    rho = np.concatenate((rho[-2*n_r:], rho))
    rho_0 = np.copy(rho)
    length = len(rho)
    rho_s = 5
    v = 2.5
    v_list = []

    n = 0
    t = 0

    n_stop = n_r
    jump_size = n_r

    while n + n_r < length:

        if n==n_stop:
            n_stop = n + jump_size
            #if np.sum(depletion(rho[n:n + n_r], [t_d0, r / n_r / v, ]))*v > eta_target:
                #v = min(v_lim,-r/n_r/t_d0 * 1/(np.log(eta_target*t_d0/(r*rho_0[n]))))

            #else:
            #    v = v_lim
            
            #if np.sum(depletion(rho[n:n + n_r], [t_d0, rho_s, r / n_r / v ]))*v > eta_target:
            
            time = r/n_r/v_lim
            dt = r/n_r/v_lim/10
            
            while (np.sum(depletion(rho[n:n + n_r], [t_d0, rho_s, time+dt ])) - np.sum(depletion(rho[n:n + n_r], [t_d0, rho_s, time ])) )/dt  > eta_target: #

                
                time = time + dt
                #tau = tau + r/n_r/v_lim /10 #
                #tau = t_d0 * np.log(np.sum(rho[n:n+n_r])/t_d0/eta_target)
                
            v = r/n_r/time
            
            #else:

            #    v = v_lim
                
        else:
            v = v_lim
            
        #print(n,"/",length)

        ###################### Feeding ###

        rho_depleted = depletion(rho[n:n + n_r], [t_d0, rho_s, r / n_r / v ])
        rho[n:n + n_r] = rho[n:n + n_r] - rho_depleted
        eta = np.sum(rho_depleted) * v 
        #if v!=v_lim:
        #    print(eta)
        #    print('t=', r / n_r / v)

        v_list.append(v)
        n = n + 1
        # t = t + r / n_r / v

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
    aux = np.load(os.path.join(script_dir_parent, 'fig_2_supplementary', "opt_0.npy"))
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


    ### Simulation ###
    inpt = [rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
    outpt = simu(rho, T, t_d0, n_r, r, v_lim, eta_target, noise=noise, learning=learning, trade_off=trade_off, alpha=alpha, a=a)
    ### Saving simulation ###

    res_sim = [inpt,outpt]

    name = "simulation"

    os.makedirs(name, exist_ok=True )

    with open(os.path.join(script_dir, name, f"{name}.pkl"), 'wb') as file_to_write:
        dill.dump(res_sim, file_to_write)
