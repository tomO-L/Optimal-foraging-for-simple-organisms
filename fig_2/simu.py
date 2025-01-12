import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import Bounds
import pickle as pk
import os
import numpy as np
import dill
import tqdm
import sys
import matplotlib.pyplot as plt

def gamma_of_rho(rho,t_d0):

    res = rho/t_d0

    return res

def depletion(rho_r, pars):
    
    t_d0 = pars[0]
    t = pars[1]
    
    f = rho_r * (1 - np.exp(-t/t_d0))

    return f

def simu_rough(rho_init, T, t_d0, n_r, r, v_lim, eta_target):

    rho_0 = np.copy(rho_init)
    rho = np.copy(rho_init)
    #rho = np.concatenate((rho,np.ones(n_r*2)*rho[-1]))
    #rho = np.concatenate((np.ones(n_r * 2) * rho[0], rho))
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
        
        rho_depleted = depletion(rho[n:n + n_r], [t_d0, r / n_r / v, ])
        rho[n:n + n_r] = rho[n:n + n_r] - rho_depleted
        eta = np.sum(rho_depleted) * v * np.random.uniform(1 - noise, 1 + noise)

		
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
        
        v_list.append(v)
        n += 1
    schedule = r / n_r / np.array(v_list)

    return schedule

def simu(rho_init, T, t_d0, n_r, r, v_lim, eta_target, noise=0, learning=False, trade_off=False, alpha=0, a=1):
        
    tail = n_r

    rho = np.copy(rho_init)
    rho = np.concatenate((rho,rho[:2*n_r]))
    rho = np.concatenate((rho[-2*n_r:], rho))
    length = len(rho)
    food_eaten = np.zeros(length)
    #v_list_t = np.array([])
    v = 20
    v_list = []
    
    if learning==True:
        eta_star_list = []
        

    n = 0
    n_stop = n_r
    t = 0
    t_min = r/n_r/v_lim
    #for t in np.linspace(0,T,int(T/dt)):        

    #schedule = np.ones(length)*t_min
    
    def compute_eta_m(rho):

        eta_m = []

        for n in range(length):
     
            eta_m.append(r/n_r*np.sum(gamma_of_rho(rho[n:n + n_r],t_d0)))
        
        return np.array(eta_m)       
    
    def compute_rho_f(schedule, rho):

        rho_final = np.copy(rho)

        for n in range(length):
            
            t = schedule[n]
            rho_depleted = depletion(rho_final[n:n + n_r], [t_d0, t])
            rho_final[n:n + n_r] = rho_final[n:n + n_r] - rho_depleted

        return np.array(rho_final)

    def smooth(x):
        
        x_out = np.convolve(x, np.ones(n_r),'same') / n_r
            
        return x_out

    schedule = np.concatenate((simu_rough(rho, T, t_d0, n_r, r, v_lim, eta_target),np.zeros(tail)))

    nb_of_corrections = 30 # For Alfonso to play with # Set it to 1 if you are running eta_star_vs_eta_bar.py
    CHI_before = np.inf

    dt = t_min*10

    for iteration in tqdm.tqdm(range(nb_of_corrections)):

        tolerance = 1 #0.1 for simulations that are not shown For Alfonso to play with
        n = 0
        chi_local = np.inf
        chi = np.array([0])

        while abs(chi_local) >= tolerance and chi.size!=0:
            # 1 
            
            remaining_rho = compute_rho_f(schedule,rho)
            eta_m_list = compute_eta_m(remaining_rho)
            
            # 2
            
            chi_search = eta_m_list-eta_target
            
            condition = np.logical_and(chi_search<0,schedule==t_min)
            points_of_interest = np.where(np.logical_not(condition))[0]
            #points_of_interest = np.where(np.logical_and(abs(chi_search)>tolerance,schedule>t_min))[0]
            
            chi = eta_m_list[points_of_interest]-eta_target
            CHI = np.sum(abs(eta_m_list-eta_target))

            if chi.size!=0:
                ind_list = np.where(max(abs(chi)) == abs(chi))[0]
                ind = points_of_interest[ind_list[0]]
                print("eta_m=", eta_m_list[ind],"position=", ind, "time=", schedule[ind], "chi=", chi[ind_list[0]], "dt=", dt, "Itération=", iteration)
                schedule[ind] = max(schedule[ind] + np.sign(chi[ind_list[0]])*dt,t_min)
                
                verif_remaining_rho = compute_rho_f(schedule,rho)
                verif_eta_m_list = compute_eta_m(verif_remaining_rho)
                verif_points_of_interest = np.where(np.logical_or(verif_eta_m_list>eta_target,schedule>t_min))[0]
                verif_chi = verif_eta_m_list[verif_points_of_interest]-eta_target
                
                new_CHI = np.sum(abs(verif_eta_m_list-eta_target))
                
                
                #verif_ind_list = np.where(max(abs(verif_chi)) == abs(verif_chi))[0]
                #verif_ind = verif_points_of_interest[verif_ind_list[0]]
                
            
                #print(abs(verif_chi[verif_ind_list[0]]),abs(chi_before))
                
                if new_CHI>CHI:
                    #schedule[ind] = schedule[ind] - np.sign(chi[ind_list[0]])*dt
                    dt = dt*0.99 # For Alfonso to play with
                    # 0.999 for rho_0
                    # for rho_1
                    # for rho_2
    
                chi_local = chi[ind_list[0]]
            n = n+1
        dt = t_min/10


        print("Smoothy?")
        if iteration<nb_of_corrections-1:
            schedule = smooth(schedule)
            print("Smoothyyyy :)")
        else:
            print("Not smoothy :(")
        schedule = np.maximum(schedule,t_min)


    condition_to_stop = np.any(
            np.logical_and(
                eta_m_list<eta_target-tolerance,schedule!=t_min
                ))

    if condition_to_stop:

        print("Fuck it + I'm not doing this shit + your program crashes")
        sys.exit()

    else:

        print("I will let it slide... for now")

        

    return schedule[:-tail]



if __name__=='__main__':

    script_dir = os.path.dirname(__file__)
    script_dir_parent = os.path.abspath(os.path.join(script_dir, os.pardir))

    ### Model parameters ###
    t_d0 = 1 # depletion time scale
    n_r = 20 # size in bins of the depletion section
    r = 1*n_r # physical size of the depletion section
    v_lim = 200 # maximum speed
    
    # with open(f"/home/tom/Ingénieur_Etude_2023-2024/python/simulations_celegans/Plots/fig_2/opt_0.pkl", 'rb') as fileopen:
    #     opt_eta_target_list, opt_eta_average_list = dill.load(fileopen)
    aux = np.load(os.path.join(script_dir, "opt_0.npy"))
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

    #with open("/home/tom/Ingénieur_Etude_2023-2024/python/simulations_celegans/Plots/environment_test.pkl", 'rb') as fileopen:
    #    rho = pk.load(fileopen)*10
    
    rho = np.load(os.path.join(script_dir_parent, 'rho_0.npy')) 


    ### Simulation ###
    inpt = [rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
    outpt = simu(rho, T, t_d0, n_r, r, v_lim, eta_target, noise=noise, learning=learning, trade_off=trade_off, alpha=alpha, a=a)

    schedule = np.copy(outpt)

    tail = n_r

    time_felt = []
    for i in range(len(schedule)):
        time_felt.append(np.sum(schedule[i-n_r+1:i+1]))
    time_felt = np.array(time_felt)[tail*2:-tail-1]
    schedule = schedule[tail*2:-tail-1]

    density_eaten = depletion(rho[:-1], [t_d0, time_felt])[tail*2:-tail-1]
    
    total_food_eaten = np.sum(density_eaten)*r/n_r

    print(total_food_eaten/np.sum(schedule))
    print(len(outpt))
    ### Saving simulation ###

    res_sim = [inpt,outpt]

    name = "simulation"

    os.makedirs(name, exist_ok=True )

    with open(os.path.join(script_dir, name, f"{name}.pkl"), 'wb') as file_to_write:
        dill.dump(res_sim, file_to_write)
