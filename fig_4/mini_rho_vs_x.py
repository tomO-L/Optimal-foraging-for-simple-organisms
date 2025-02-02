import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scipy as sp
import dill

name = "simulation"

with open(f"{name}/{name}.pkl", 'rb') as fileopen:
        inpt, outpt = dill.load(fileopen)

from simu import depletion, gamma_of_rho

rho, T, t_d, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a = inpt
schedule = outpt

tail = n_r

window = 650
schedule_local = schedule[window-n_r:window+5*n_r]

x = np.linspace(window,window + 20*n_r-1,20*n_r) ### tom

rho_local = np.ones(20*n_r) * 10 ### tom
#rho_local = np.concatenate((rho_local,np.ones(n_r*2)*rho_local[-1]))
#rho_local = np.concatenate((np.ones(n_r * 2) * rho_local[0], rho_local))
l = len(schedule_local)

rho_list = []
# sample_loc = np.array([25, 30, 30, 45])
# sample_loc = np.array([25, 30, 30, 45]) + 40
sample_loc = np.array([20, 30, 30, 40]) + 40

switch = True

for n in range(l):
    
    t = schedule_local[n]

    if n==sample_loc[1] and switch:
        
        rho_depleted = depletion(rho_local[n:n + n_r], [t_d, t/2])
        rho_to_append = np.copy(rho_local)
        rho_to_append[n:n+n_r] = rho_to_append[n:n+n_r] - rho_depleted
        rho_list.append(rho_to_append)

        switch = not(switch)
    
    rho_depleted = depletion(rho_local[n:n + n_r], [t_d, t])
    
    rho_local[n:n + n_r] = rho_local[n:n + n_r] - rho_depleted
    
    if np.any(n==sample_loc) and n!=sample_loc[1]:
        
        rho_list.append(np.copy(rho_local))
    
    elif np.any(n==sample_loc) and n==sample_loc[1] and switch==False:
    
        rho_list.append(np.copy(rho_local))
    
    

### PLOTS ###

my_dpi = 96

axes_font_size = 30
title_font_size = 20
graduation_font_size = 25
legend_font_size = 25

factor_inset = 2

image_width = 1200
image_height = 400 #388

color_rho = (102/255,166/255,30/255)
color_v = (117/255,112/255,179/255)
color_tau = (231/255,41/255,138/255)
color_eta = (27/255,158/255,119/255)
color_eta_m = (217/255,95/255,2/255)
color_meta = (102/255,102/255,102/255)

alpha_red = 0.5

linewidth = 4
ticks_width = 1
ticks_length = 4

top=0.95
bottom=0.22
right=0.9
left=0.15

for i in range(4):

    fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi*.6, image_height/my_dpi), dpi=my_dpi, sharex=True)

    axes.fill_between(x, np.ones(20*n_r) * 10, color = color_rho, step='mid')#axes.set_xlim(T[599+n_r],T[599+3*n_r+1]) ### tom
    axes.fill_between(x, rho_list[i], color = 'black', alpha = 0.4, step='mid')#axes.set_xlim(T[599+n_r],T[599+3*n_r+1])
    
    axes.scatter(sample_loc[i], 0, color = 'black')#axes.set_xlim(T[599+n_r],T[599+3*n_r+1])
    axes.plot(np.ones(50)*(window+sample_loc[i]-0.5), np.linspace(0,20), '--', linewidth = linewidth, color = color_meta)
    axes.plot(np.ones(50)*(window+sample_loc[i]+n_r-0.5), np.linspace(0,20), '--', linewidth = linewidth, color = color_meta)
    
    axes.set_xlim(window+20,window+150)
    axes.set_xticks([])
    if i == 3:
        axes.set_xlabel('Position ($X$)', fontsize=axes_font_size)
    axes.set_ylim(0, 12)
    axes.set_yticks([0, 5, 10])
    axes.set_ylabel(r'Density ($\rho$)',fontsize=axes_font_size)

    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(graduation_font_size)
    
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    
    plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
    
    #plt.savefig(f'images/minirho_{i}.png', dpi=my_dpi)

    #legend_elements = [Line2D([0], [0], color='blue', label=r'Speed of the animal ($v$)'),
    #                   Line2D([0], [0], color='orange', label=r'Feeding rate ($\eta$)')]

    #axes[i].legend(handles=legend_elements, loc='center', fontsize = legend_font_size)
#axes[1].set_aspect('equal')

plt.show()