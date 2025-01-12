import numpy as np
import matplotlib.pyplot as plt
import time
import dill
import simu
from tqdm.auto import tqdm
import os

script_dir = os.path.dirname(__file__)
script_dir_parent = os.path.abspath(os.path.join(script_dir, os.pardir))

eta_target = np.linspace(20,200,20)
    
t_d0 = 1 # depletion time scale
n_r = 20 # size in bins of the depletion section
r = 1*n_r # physical size of the depletion section
v_lim = 200 # maximum speed
t_min = r/n_r/v_lim

rho_s = 5

T = 100

for i_env in [0, 1, 2]:

    rho = np.load(os.path.join(script_dir_parent, f'rho_{i_env}.npy'))
    
    length = len(rho)
        
    ### Simulation ###
    
    tail = n_r
    
    #rho = np.concatenate((rho,np.ones(n_r*2)*rho[-1]))
    #rho = np.concatenate((np.ones(n_r * 2) * rho[0], rho))
    
    eta_bar = []
    
    
    for eta_star in tqdm(eta_target) :
    
        schedule = simu.simu(rho, T, t_d0, n_r, r, v_lim, eta_star)
    
        time_felt = []
        for i in range(len(schedule)):
            time_felt.append(np.sum(schedule[i-n_r+1:i+1]))
        time_felt = np.array(time_felt)[tail*2:-tail]
        schedule = schedule[tail*2:-tail]
    
        density_eaten = simu.actual_depletion(rho, [t_d0, rho_s, time_felt])[tail*2:-tail]
        
        total_food_eaten = np.sum(density_eaten)*r/n_r
    
        eta_bar.append(total_food_eaten/np.sum(schedule))
    
    
    print('optimal eta* = ', eta_target[eta_bar.index(max(eta_bar))])
    print('maximum eta_bar = ', max(eta_bar))
    
    print((eta_bar))
    
    np.save(os.path.join(script_dir, f'opt_{i_env}.npy'), [eta_target,eta_bar])
    
    # with open(f"opt_{i_env}.pkl", 'wb') as file_to_write:
    #     dill.dump([eta_target,eta_bar], file_to_write)
    
    #plt.plot(eta_target,eta_bar_1)
    #plt.plot(eta_target,eta_bar_2)
    #plt.plot(np.linspace(0,9,2),np.linspace(0,9,2),'--')
    #plt.xlabel('eta^*')
    #plt.ylabel('eta_bar')
    #plt.show()


    

    