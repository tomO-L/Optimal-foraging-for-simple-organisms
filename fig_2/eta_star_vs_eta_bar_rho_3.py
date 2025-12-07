import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import time
import dill
import simu
import sys
from tqdm.auto import tqdm
import os

script_dir = os.path.dirname(__file__)
script_dir_parent = os.path.abspath(os.path.join(script_dir, os.pardir))

eta_target = np.linspace(20,200,20)
# eta_target_1 = np.linspace(20,94,20)
# eta_target_2 = np.linspace(94,101,20)
# eta_target_3 = np.linspace(101,201,20)

# eta_target = np.concatenate((eta_target_1,eta_target_2))
# eta_target = np.concatenate((eta_target,eta_target_3))


t_d0 = 1 # depletion time scale
n_r = 20 # size in bins of the depletion section
r = 1*n_r # physical size of the depletion section
v_lim = 200 # maximum speed
t_min = r/n_r/v_lim

T = 100

# patch_array = np.load(os.path.join(script_dir_parent, f'patch_array_rho_3.npy'))
# patch_length_array = np.load(os.path.join(script_dir_parent, f'patch_length_array_rho_3.npy'))

# patch_list = []

# for i in range(len(patch_length_array)):

#     patch_start = int(np.sum(patch_length_array[:i-1]))

#     patch_list.append(patch_array[patch_start:patch_start+int(patch_length_array[i])])


def flat_patch(height, width, smoothness):

    unsmoothed_patch = np.ones(width)*height
    unsmoothed_patch = np.concatenate((np.zeros(smoothness*3), unsmoothed_patch))
    unsmoothed_patch = np.concatenate((unsmoothed_patch, np.zeros(smoothness*3)))

    patch = gaussian_filter1d(unsmoothed_patch, smoothness)

    return patch

### Generate Random Environment ###

np.random.seed(13)

distance_range = np.arange(60, 200)
height_range = np.linspace(0, 8, 100)
width_range = np.arange(100, 200)
smoothness_range = np.arange(10, 20)

n_patches = 10
patch_list = []

for i in range(n_patches):

    height = np.random.choice(height_range)
    width = np.random.choice(width_range)
    smoothness = np.random.choice(smoothness_range)
    distance = np.random.choice(distance_range)

    patch = flat_patch(height, width, smoothness)
    patch = np.concatenate((patch,np.zeros(n_r)))
    patch = np.concatenate((np.zeros(n_r),patch))

    patch_list.append(patch)

patch_list.append(np.zeros(100))
patch_list.append(np.zeros(200))
patch_list.append(np.zeros(300))

rho_3 = np.array([])
rho_patch_indexes = []

for i in range(100):

    patch_index = np.random.choice(np.arange(len(patch_list)))
    rho_patch_indexes.append(patch_index)
    patch_to_add = patch_list[patch_index]
    
    rho_3 = np.concatenate((rho_3,patch_to_add))

folder_to_save = "" # To save it in the current folder
np.save(os.path.join(folder_to_save, "rho_3.npy"), rho_3)


### Simulation ###
    
tail = n_r
    
#rho = np.concatenate((rho,rho[:2*n_r]))
#rho = np.concatenate((rho[-2*n_r:], rho))
    
eta_bar = []
    

for eta_star in tqdm(eta_target) :

    patch_schedule_list = []
    patch_time_felt_list = []
    patch_total_food_eaten_list = []

    for patch in patch_list:

        patch_schedule = simu.simu(patch, T, t_d0, n_r, r, v_lim, eta_star)
    
        patch_time_felt = []
        for i in range(len(patch_schedule)):
            patch_time_felt.append(np.sum(patch_schedule[i-n_r+1:i+1]))
        patch_time_felt = np.array(patch_time_felt)[tail*2:-tail-1]
        patch_schedule = patch_schedule[tail*2:-tail-1]
        
        patch_schedule_list.append(patch_schedule)
        patch_time_felt_list.append(patch_time_felt)

        # patch_density_eaten = simu.depletion(patch[:-1], [t_d0, patch_time_felt])[tail*2:-tail-1]
        patch_density_eaten = simu.depletion(patch[:-1], [t_d0, patch_time_felt])[tail*2:-tail-1]
        patch_total_food_eaten_list.append(np.sum(patch_density_eaten)*r/n_r)
    
    
    total_food_eaten = 0
    total_time = 0

    for index in rho_patch_indexes:

        total_food_eaten += patch_total_food_eaten_list[index]
        total_time += np.sum(patch_schedule_list[index])

    eta_bar.append(total_food_eaten/total_time)


    
    
print('optimal eta* = ', eta_target[eta_bar.index(max(eta_bar))])
print('maximum eta_bar = ', max(eta_bar))
    
np.save(os.path.join(script_dir, f'opt_3.npy'), [eta_target,eta_bar])



#plt.plot(eta_target,eta_bar_1)
#plt.plot(eta_target,eta_bar_2)
#plt.plot(np.linspace(0,9,2),np.linspace(0,9,2),'--')
#plt.xlabel('eta^*')
#plt.ylabel('eta_bar')
#plt.show()


    

    
