import scipy as sp
import pickle as pk
import os
import numpy as np
import dill

def gamma_of_rho(rho,t_d0):

    res = rho/t_d0

    return res

def depletion(rho_r, pars):
    
    t_d0 = pars[0]
    t = pars[1]
    
    f = rho_r * (1 - np.exp(-t/t_d0))

    return f

def simu(rho_init, T, t_d0, n_r, r, v_lim, eta_target, noise=0, learning=False, trade_off=False, alpha=0, a=1):

    eta_threshold = eta_target#/(v_lim*t_d0/r*(1-np.exp(-r/(t_d0*v_lim))))
    
    #c = 1/(v_lim*t_d0/r*(1-np.exp(-r/(t_d0*v_lim))))
    
    rho_0 = np.copy(rho_init)
    rho = np.copy(rho_init)
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

    #for t in np.linspace(0,T,int(T/dt)):        



    while n<length and t<T:

        ###################### Speed control rule ###

        ### perfect rule ###

        if eta_target*t_d0/r <= rho_0[n+n_r] and eta_target*t_d0/r <= rho_0[n]:

            v = -r/n_r/t_d0 * 1/np.log(eta_target*t_d0/r/rho_0[n]) * n_r
            v = min(v,v_lim)
        
        else:

            v = v_lim

        ### math rule ###

        #def J(speed,pars):

        #    r = pars[0]
        #    td = pars[1]
        #    eta = pars[2]
        #    eta_target = pars[3]

        #    res = (speed * t_d0/r * ( eta / (1-np.exp(-r/speed/td)) - eta_target) - eta)**2

        #    return res

        #res_minimization = sp.optimize.minimize(J, v, [r, t_d0, eta, eta_target], bounds=[(0,v_lim)],options={'maxiter': 15000})
        #v = min(res_minimization.x[0],v_lim)
        

        ### dashes rule ###        

        #if n==n_stop:
        #    n_stop = n + n_r
        #    if rho_0[n] > eta_target*t_d0/r:
        #        v = min(v_lim,-r/n_r/t_d0 * 1/(np.log(eta_target*t_d0/(r*rho_0[n]))))
        #    else:
        #        v = v_lim
        #else:
        #    v = v_lim
            

        v_list.append(v)


        ###################### Feeding ###

        rho_depleted = depletion(rho[n:n + n_r], [t_d0, r/n_r/v,])
        rho[n:n + n_r] = rho[n:n + n_r] - rho_depleted
        eta = np.sum(rho_depleted)*v *np.random.uniform(1-noise,1+noise)
    

        ###################### Learning rule ###
        if learning==True:
            
            eta_star = (eta_star-eta)*np.exp(-alpha*(r/v/t_d0)) + eta
            #eta_star = (1-alpha)*eta_star + alpha*eta 
            #eta_star = np.sum(food_eaten)/t  
            
            eta_star_list.append(eta_star)

        #np.append(v_list_t)

        n = n + 1
        t = t + r/n_r/v

    schedule = r/n_r/np.array(v_list)


    return schedule



if __name__=='__main__':

    ### Model parameters ###
    t_d0 = 1 # depletion time scale
    n_r = 20 # size in bins of the depletion section
    r = 1*n_r # physical size of the depletion section
    v_lim = 200 # maximum speed
    
    eta_target = 6 * r/t_d0
    
    noise = 0.
    learning = False
    alpha = 1e-1 # learning coeficient. alpha = 0 when you care just about the past, and alpha = 1 when you care just about the present
    trade_off = False
    a = 250 # trade-off coeficient. The higher it is, the less trade-off you have

    T = 15

    ### Loading the environement (beware of the *10 factor at importation)
    with open("../environment_test.pkl", 'rb') as fileopen:
        rho = pk.load(fileopen) * 10

    ### Simulation ###
    inpt = [rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
    outpt = simu(rho, T, t_d0, n_r, r, v_lim, eta_target, noise=noise, learning=learning, trade_off=trade_off, alpha=alpha, a=a)

    ### Saving simulation ###

    res_sim = [inpt,outpt]

    name = "simulation_OLD"

    os.makedirs(name, exist_ok=True )

    with open(f"{name}/depletion.pkl", 'wb') as file_to_write:
        dill.dump(depletion, file_to_write)

    with open(f"{name}/{name}.pkl", 'wb') as file_to_write:
        dill.dump(res_sim, file_to_write)

   
