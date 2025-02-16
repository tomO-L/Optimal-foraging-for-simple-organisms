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

def simu(rho_init, T, t_d0, n_r, r, v_lim, eta_target, noise=0, learning=False, trade_off=False, alpha=0, a=1):

    rho = np.copy(rho_init)
    rho = np.concatenate((rho,rho[:2*n_r]))
    rho = np.concatenate((rho[-2*n_r:], rho))
    rho_0 = np.copy(rho)
    
    length = len(rho)
    v = v_lim
    v_list = []
    eta_target_list = []

    n = 0
    t = 0

    n_stop = n_r
    jump_size = n_r
		
    def J_minimize(v,pars):
		
        t_d = pars[0]
        r = pars[1]
        eta_star = pars[2]
        eta_m_star = pars[3]
		
        res = (eta_m_star * v*t_d/r * (np.exp(r/(t_d*v)) - 1) - eta_star)**2
		
        return res

    eta_list = []
    while n + n_r < length:
		
        v_list.append(v)
        ### Feeding ###

        rho_depleted = depletion(rho[n:n + n_r], [t_d0, r / n_r / v, ])
        rho[n:n + n_r] = rho[n:n + n_r] - rho_depleted
        eta = np.sum(rho_depleted) * v * np.random.uniform(1 - noise, 1 + noise)
        eta_list.append(eta)
        
        
        
        #print(n,"/",length)        
        
        
        ### Learning ### 

        if learning==True:
            alpha_now = alpha/v # Make learning rate proportional to time spent in the bin, so that we average over time and not over bins
            eta_target = eta_target*(1 - alpha_now) + eta*alpha_now
            # print(f"alpha = {alpha}")
            # print(f"v = {v}")
            # print(f"alpha_now = {alpha_now}")
            # print(f"eta = {eta}")
            # print(f"eta_target = {eta_target}")
            
            # eta_target = (eta_target-eta)*np.exp(-alpha*(r/v/t_d0)) + eta
            #eta_star = (1-alpha)*eta_star + alpha*eta 
            #eta_star = np.sum(food_eaten)/t  
            
            eta_target_list.append(eta_target)
        
        ### Speed Control ###
        
        """
        # Math rule #
		
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
        """
        
        # Intermitent rule #

        if n==n_stop:
            n_stop = n + jump_size
            if rho_0[n] > eta_target*t_d0/r:
                #v = min(v_lim,-r/n_r/t_d0 * 1/(np.log(eta_target*t_d0/(r*rho_0[n]))))
                v = min(v_lim,-r/n_r/t_d0 * 1/(np.log(eta_target*t_d0/(r*rho_0[n]))))

            else:
                v = v_lim
        else:
            v = v_lim
        
        # v_list.append(v)
        n += 1
    
    print(f"eta_target_final = {eta_target}")    
    schedule = r / n_r / np.array(v_list)
    
    # # START FIGURE
    # my_dpi = 96

    # axes_font_size = 30
    # title_font_size = 20
    # graduation_font_size = 25
    # legend_font_size = 25

    # factor_inset = 2

    # image_width = 1200
    # image_height = 400 #388

    # color_rho = (102/255,166/255,30/255)
    # color_v = (117/255,112/255,179/255)
    # color_tau = (231/255,41/255,138/255)
    # color_eta = (27/255,158/255,119/255)
    # color_eta_m = (217/255,95/255,2/255)
    # color_meta = (102/255,102/255,102/255)

    # alpha_red = 0.5

    # linewidth = 4
    # ticks_width = 1
    # ticks_length = 4

    # top=0.95
    # bottom=0.22
    # right=0.9
    # left=0.15

    # x_max = 40000
    # x_ticks = [0, 10000, 20000, 30000, 40000]
    # x = np.linspace(1, len(schedule), len(schedule))
    # fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)
    # axes.plot(x, np.ones(len(x))*opt_eta_target, color = 'black', linestyle='--', linewidth = linewidth)

    # axes.plot(x, eta_list, color=color_eta, linewidth = linewidth)
    # # axes.plot(x, eta_m_list, color=color_eta_m, linewidth = linewidth)
    # axes.plot(x, eta_target_list, color='black', linewidth = linewidth, linestyle=':')

    # axes.set_xlim(0, x_max)
    # axes.set_xticks(x_ticks)
    # axes.set_ylim(0, 150)
    # axes.set_yticks([0, 50, 100, 150])
    # axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
    # axes.set_ylabel(r'Feeding rate ($\eta$)',fontsize=axes_font_size)
    # #axes[2].set_title("Feeding rate profiles", fontsize = title_font_size)
    # for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    #     label.set_fontsize(graduation_font_size)
    # # END FIGURE
    
    eta_bar = np.sum(np.array(eta_list)*np.array(schedule))/np.sum(np.array(schedule))
    print(f"eta_bar = {eta_bar}")
    
    return schedule, np.array(eta_target_list), eta_list



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
    
    eta_target = 20 # Should be 20
    noise = 0.
    learning = False
    alpha = 1e-2 # learning coeficient. alpha = 0 when you care just about the past, and alpha = 1 when you care just about the present
    trade_off = False
    a = 250 # trade-off coeficient. The higher it is, the less trade-off you have

    T = 100

    #with open("../environment_test.pkl", 'rb') as fileopen:
    #    rho = pk.load(fileopen) * 10

    rho = np.load(os.path.join(script_dir_parent, 'rho_0.npy'))

    long_rho = np.copy(rho)

    for i in range(50):

        long_rho = np.concatenate((long_rho,rho))

    ### Simulation ###
    inpt = [long_rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
    outpt = simu(long_rho, T, t_d0, n_r, r, v_lim, eta_target, noise=noise, learning=True, trade_off=trade_off, alpha=alpha, a=a)
    ### Saving simulation ###

    res_sim = [inpt,outpt]

    
    name = "simulation"

    os.makedirs(name, exist_ok=True )

    with open(os.path.join(script_dir, name, f"{name}.pkl"), 'wb') as file_to_write:
        dill.dump(res_sim, file_to_write)
    
