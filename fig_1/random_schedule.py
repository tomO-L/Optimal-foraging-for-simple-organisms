import pickle as pk
import numpy as np
import dill
import os
import matplotlib.pyplot as plt

def depletion(rho_r, pars):
    
    t_d0 = pars[0]
    t = pars[1]
    
    f = rho_r * (1 - np.exp(-t/t_d0))

    return f

### Model parameters ###
t_d0 = 1 # depletion time scale
n_r = 20 # size in bins of the depletion section
r = 1*n_r # physical size of the depletion section
v_lim = 200 # maximum speed

eta_target = 6
#0.6 # (initial) rho_final
noise = 0.
learning = False
alpha = 1e-1 # learning coeficient. alpha = 0 when you care just about the past, and alpha = 1 when you care just about the present
trade_off = False
a = 250 # trade-off coeficient. The higher it is, the less trade-off you have
T = 15

def arbitrary_schedule(x):
    
    res1 = abs((x+50)*(x-2001)*(x-2500)*(x-3500)*9e-11 - 30)# -200)/2.1
    # res1 = abs((x+50)*(x-2001)*(x-2500)*(x-3500)*9e-11 - 30)# -200)/2.1
    res2 = 1/res1
    res3 = np.maximum(res2,1/v_lim)
    res = np.copy(res3)
    for n in range(len(res3)):
        if n>100:
            res[n] = np.mean(res3[n-100:n])
    res[:200] = 1/v_lim
    res = res + .2*np.exp((x-3500)/300) 
    return res


# with open("../environment_test.pkl", 'rb') as fileopen:
#     rho = pk.load(fileopen) * 10
    
name = "simulation_test"
folder_environment = ".." # Relative path to the files with the environment

rho = np.load(os.path.join(folder_environment, "rho.npy"))
environment_2D = np.load(os.path.join(folder_environment, "environment_2D.npy"))
traj_2D = np.load(os.path.join(folder_environment, "traj_2D.npy"))

print(len(rho))

tail = n_r
x = np.linspace(0,len(rho)-1,len(rho))
schedule = arbitrary_schedule(x)

plt.plot(x, schedule)
# plt.plot(x, schedule + .07*np.exp((x-3500)/500))
plt.show()

schedule = np.concatenate((schedule,np.zeros(tail)))
schedule = np.concatenate((schedule[0]*np.ones(2*tail),schedule))



### Simulation ###
inpt = [rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a, environment_2D, traj_2D]
outpt = schedule

### Saving simulation ###
res_sim = [inpt,outpt]

os.makedirs(name, exist_ok=True )
with open(f"{name}/depletion.pkl", 'wb') as file_to_write:
    dill.dump(depletion, file_to_write)
with open(f"{name}/{name}.pkl", 'wb') as file_to_write:
    dill.dump(res_sim, file_to_write)