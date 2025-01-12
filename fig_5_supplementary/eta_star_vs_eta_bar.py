import numpy as np
import matplotlib.pyplot as plt
import time
import dill
import simu_math
import simu_intermitent
from tqdm.auto import tqdm
import os

script_dir = os.path.dirname(__file__)
script_dir_parent = os.path.abspath(os.path.join(script_dir, os.pardir))

eta_target = 50
    
t_d0 = 1 # depletion time scale
n_r = 20 # size in bins of the depletion section
r = 1*n_r # physical size of the depletion section
v_lim = 200 # maximum speed
t_min = r/n_r/v_lim

T = 100

#with open("../environment_test_1.pkl", 'rb') as fileopen:
#    rho = dill.load(fileopen) * 10

for i_env in [0, 1, 2]:

    rho = np.load(os.path.join(script_dir_parent, f'rho_{i_env}.npy'))

    long_rho = np.copy(rho)

    for _ in range(500):

        long_rho = np.concatenate((long_rho,rho))

    length = len(rho)

    tail = n_r

    #long_rho = np.concatenate((long_rho,np.ones(n_r*2)*long_rho[-1]))
    #long_rho = np.concatenate((np.ones(n_r * 2) * long_rho[0], long_rho))

    eta_bar = []

    simu_res = simu_math.simu(long_rho, T, t_d0, n_r, r, v_lim, eta_target, learning=True, alpha = 5*1e-4)

    schedule = simu_res[0]
    eta_m_star_list = simu_res[1]

    time_felt = []
    for i in range(len(schedule)):
        time_felt.append(np.sum(schedule[i-n_r+1:i+1]))

    time_felt = np.array(time_felt)[tail*2:-tail-1]
    schedule = schedule[tail*2:-tail-1]
    density_eaten = simu_math.depletion(long_rho[:-1], [t_d0, time_felt])[tail*2:-tail-1]
    eta_m_star_list = eta_m_star_list[tail*2:-tail-1]

    total_food_eaten = np.sum(density_eaten[-len(rho):])*r/n_r

    eta_bar = total_food_eaten/np.sum(schedule[-len(rho):])
    eta_m_star = np.sum(eta_m_star_list[-len(rho):] * schedule[-len(rho):]/np.sum(schedule[-len(rho):]))

    #print('optimal eta* = ', eta_target[eta_bar.index(max(eta_bar))])
    #print('maximum eta_bar = ', max(eta_bar))

    #with open(f"opt_math_{n}.pkl", 'wb') as file_to_write:
    #    dill.dump([eta_m_star,eta_bar], file_to_write)    
    np.save(os.path.join(script_dir, f'opt_math_{i_env}.npy'), [eta_m_star,eta_bar])

for i_env in [0, 1, 2]:

    rho = np.load(os.path.join(script_dir_parent, f'rho_{i_env}.npy'))

    long_rho = np.copy(rho)

    for _ in range(500):

        long_rho = np.concatenate((long_rho,rho))

    length = len(rho)

    tail = n_r

    #long_rho = np.concatenate((long_rho,np.ones(n_r*2)*long_rho[-1]))
    #long_rho = np.concatenate((np.ones(n_r * 2) * long_rho[0], long_rho))

    eta_bar = []

    simu_res = simu_intermitent.simu(long_rho, T, t_d0, n_r, r, v_lim, eta_target, learning=True, alpha = 5*1e-4)

    schedule = simu_res[0]
    eta_m_star_list = simu_res[1]

    time_felt = []
    for i in range(len(schedule)):
        time_felt.append(np.sum(schedule[i-n_r+1:i+1]))

    time_felt = np.array(time_felt)[tail*2:-tail-1]
    schedule = schedule[tail*2:-tail-1]
    density_eaten = simu_intermitent.depletion(long_rho[:-1], [t_d0, time_felt])[tail*2:-tail-1]
    eta_m_star_list = eta_m_star_list[tail*2:-tail-1]

    total_food_eaten = np.sum(density_eaten[-len(rho):])*r/n_r

    eta_bar = total_food_eaten/np.sum(schedule[-len(rho):])
    eta_m_star = np.sum(eta_m_star_list[-len(rho):] * schedule[-len(rho):]/np.sum(schedule[-len(rho):]))

    #print('optimal eta* = ', eta_target[eta_bar.index(max(eta_bar))])
    #print('maximum eta_bar = ', max(eta_bar))

    #with open(f"opt_intermitent_{n}.pkl", 'wb') as file_to_write:
    #    dill.dump([eta_m_star,eta_bar], file_to_write)

    np.save(os.path.join(script_dir, f'opt_intermitent_{i_env}.npy'), [eta_m_star,eta_bar])

caca

#plt.plot(eta_target,eta_bar_1)
#plt.plot(eta_target,eta_bar_2)
#plt.plot(np.linspace(0,9,2),np.linspace(0,9,2),'--')
#plt.xlabel('eta^*')
#plt.ylabel('eta_bar')
#plt.show()


    

    