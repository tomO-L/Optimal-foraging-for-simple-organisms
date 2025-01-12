import scipy as sp
import pickle as pk
import os
import numpy as np
import dill
import tqdm
import matplotlib.pyplot as plt


def gamma_of_rho(rho, t_d0):
    res = rho / t_d0

    return res


def depletion(rho_r, pars):
    t_d0 = pars[0]
    t = pars[1]

    f = rho_r * (1 - np.exp(-t / t_d0))

    return f


def simu(rho_init, T, t_d0, n_r, r, v_lim, eta_target, noise=0, learning=False, trade_off=False, alpha=0, a=1):
    rho_0 = np.copy(rho_init)
    rho = np.copy(rho_init)
    length = len(rho)
    food_eaten = np.zeros(length)
    # v_list_t = np.array([])
    v = 20
    v_list = []

    if learning == True:
        eta_star_list = []

    n = 0
    n_stop = n_r
    t = 0
    t_min = r / n_r / v_lim
    # for t in np.linspace(0,T,int(T/dt)):

    schedule = np.ones(length) * t_min

    def compute_eta_m(rho):

        eta_m = []

        for n in range(length):
            eta_m.append(r / n_r * np.sum(gamma_of_rho(rho[n:n + n_r], t_d0)))

        return np.array(eta_m)

    def compute_rho_f(schedule, rho):

        rho_final = np.copy(rho)

        for n in range(length):
            t = schedule[n]
            rho_depleted = depletion(rho_final[n:n + n_r], [t_d0, t])
            rho_final[n:n + n_r] = rho_final[n:n + n_r] - rho_depleted

        return np.array(rho_final)

    def smooth(x):

        x_out = np.convolve(x, np.ones(n_r), 'same') / n_r

        return x_out

    nb_of_corrections = 2
    CHI_before = np.inf

    for iteration in tqdm.tqdm(range(nb_of_corrections)):
        # print("caca")
        dt = t_min * 100
        n = 0
        chi_local = np.inf
        chi_before = np.inf

        while abs(chi_local) > 1e-2:
            # 1
            remaining_rho = compute_rho_f(schedule, rho)  # rho - depletion(rho,[t_d0, tau])
            eta_m_list = compute_eta_m(remaining_rho)
            # print(remaining_rho)
            # 2
            """
            points_of_interest = np.where(np.logical_or(eta_m_list>eta_target,schedule>t_min))[0]
            print(points_of_interest)
            chi = eta_m_list[points_of_interest]-eta_target
            ind_list = np.where(abs(max(chi)) == abs(chi))[0]
            ind = points_of_interest[ind_list[0]]
            print(eta_m_list[ind],ind,n)
            schedule[ind] = schedule[ind] + np.sign(chi[ind])*dt
            schedule[ind] = max(schedule[ind],t_min)
            """

            points_of_interest = np.where(np.logical_or(eta_m_list > eta_target, schedule > t_min))[0]
            # print(points_of_interest)
            chi = eta_m_list[points_of_interest] - eta_target
            ind_list = np.where(max(abs(chi)) == abs(chi))[0]
            # print(ind_list[0])
            ind = points_of_interest[ind_list[0]]
            # print("eta_m=", eta_m_list[ind], "position=", ind, "time=", schedule[ind], "chi=", chi[ind_list[0]], "dt=",dt)
            schedule[ind] = schedule[ind] + np.sign(chi[ind_list[0]]) * dt
            schedule[ind] = max(schedule[ind], t_min)

            # CHI = np.sum(abs(chi))

            if abs(chi[ind_list[0]]) > abs(chi_before):
                schedule[ind] = schedule[ind] - np.sign(chi[ind_list[0]]) * dt
                # print("CHI=",CHI, "previous CHI=", CHI_before, "Number of time variations=", n)
                dt = dt * 0.99
                # print(ind)

            # if n==0:
            #    plt.plot(ind,eta_m_list[ind],'o',color='red', alpha=0.1)

            # print(abs(chi_local))
            chi_before = chi[ind_list[0]]
            chi_local = chi[ind_list[0]]
            n = n + 1

        # plt.xlabel('X')
        # plt.ylabel('eta_m')
        # plt.xlim([0,length])
        # plt.plot(np.linspace(0,length-1,length),eta_m_list)
        # plt.show()
        """
        print("Smoothy?")
        if iteration < nb_of_corrections - 1:
            schedule = smooth(schedule)
            print("Smoothyyyy :)")
        else:
            # print(ind_list)
            print("Not smoothy :(")
        """
        schedule = np.maximum(schedule, t_min)

    return schedule


if __name__ == '__main__':
    ### Model parameters ###
    t_d0 = 1  # depletion time scale
    n_r = 10  # size in bins of the depletion section
    r = 1 * n_r  # physical size of the depletion section
    v_lim = 200  # maximum speed

    eta_target = 6.
    # 0.6 # (initial) rho_final
    noise = 0.
    learning = False
    alpha = 1e-1  # learning coeficient. alpha = 0 when you care just about the past, and alpha = 1 when you care just about the present
    trade_off = False
    a = 250  # trade-off coeficient. The higher it is, the less trade-off you have

    T = 100

    with open("../environment_test.pkl", 'rb') as fileopen:
        rho = pk.load(fileopen)
    # print(len(rho))

    ### Simulation ###
    inpt = [rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
    outpt = simu(rho, T, t_d0, n_r, r, v_lim, eta_target, noise=noise, learning=learning, trade_off=trade_off,
                 alpha=alpha, a=a)
    # print(outpt[:400])
    ### Saving simulation ###

    res_sim = [inpt, outpt]

    name = "simulation"

    os.makedirs(name, exist_ok=True)

    with open(f"{name}/{name}.pkl", 'wb') as file_to_write:
        dill.dump(res_sim, file_to_write)
