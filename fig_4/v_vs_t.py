import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scipy as sp
import dill
import os

script_dir = os.path.dirname(__file__)
name = "simulation"

with open(os.path.join(script_dir,name,f"{name}.pkl"), 'rb') as fileopen:
        inpt, outpt = dill.load(fileopen)

from simu import depletion, gamma_of_rho

rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a = inpt
schedule = outpt

l = len(schedule)

tail = n_r

window = 650 + 2*tail

rho = np.concatenate((rho,np.ones(n_r*2)*rho[-1]))
rho = np.concatenate((np.ones(n_r * 2) * rho[0], rho))

rho_final = np.copy(rho)

eta_list = []
eta_list_interp = []

for n in range(l):
    
    t = schedule[n]
    
    if t != r/n_r/v_lim:

        eta_list_interp.append([])    
        time_step = t/50
        
        for i in range(50):
    
            rho_depleted = depletion(rho_final[n:n + n_r], [t_d0, time_step])
            rho_final[n:n + n_r] = rho_final[n:n + n_r] - rho_depleted
            eta = np.sum(rho_depleted)*r/n_r/time_step
            eta_list_interp[-1].append(eta)

        eta_list.append(eta)

    else:
            
        rho_depleted = depletion(rho_final[n:n + n_r], [t_d0, t])
        rho_final[n:n + n_r] = rho_final[n:n + n_r] - rho_depleted
        eta = np.sum(rho_depleted)*r/n_r/t
        eta_list.append(eta)

schedule = schedule[2*tail:-tail]
eta_list = eta_list[2*tail:-tail]
window = window - tail*2

schedule = schedule[:3500]
eta_list = eta_list[:3500]

v = r/n_r/schedule[:3500]

k = 0

T = np.array([])
eta_time = np.array([])

for m in range(len(schedule)):

    if schedule[m]!=r/n_r/v_lim:
        
        #print(T[-1])
        #print(np.cumsum(np.ones(50)*schedule[m]/50)[-1], T[-1])
        
        T = np.concatenate( ( T, ( np.cumsum(np.ones(50)*schedule[m]/50) + T[-1] ) ) )
        eta_time = np.concatenate((eta_time,eta_list_interp[k]))
        #print(T[-1])

        k = k+1

    else:

        T = np.append(T,np.sum(schedule[:m]))
        eta_time = np.append(eta_time,eta_list[m])


T_to_display = T[np.where(np.logical_and(np.cumsum(schedule)[window+5*n_r]>T, np.cumsum(schedule)[window+n_r]<T))[0]]
eta_to_display = eta_time[np.where(np.logical_and(np.cumsum(schedule)[window+5*n_r]>T, np.cumsum(schedule)[window+n_r]<T))[0]]


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

fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)

# axes.plot(v[window+n_r:window+4*n_r+1], linewidth=linewidth)
# caca
t_show = np.cumsum(schedule)[window+n_r:window+5*n_r+1]
axes.step(t_show, v[window+n_r:window+5*n_r+1], linewidth=linewidth, where='pre', color=color_v)

sample_loc = np.array([20, 30, 30, 40]) + 40
# sample_loc = np.array([25, 30, 30, 45])
axes.scatter(np.cumsum(schedule)[window+sample_loc[0]],0, color='black', s=80, zorder=5)
axes.scatter((np.cumsum(schedule)[window+sample_loc[1]-1] + np.cumsum(schedule)[window+sample_loc[1]])/2,0, color='black', s=80, zorder=5)
axes.scatter(np.cumsum(schedule)[window+sample_loc[2]],0, color='black', s=80, zorder=5)
axes.scatter(np.cumsum(schedule)[window+sample_loc[3]],0, color='black', s=80, zorder=5)

time_ticks = [np.cumsum(schedule)[window+sample_loc[0]],
              (np.cumsum(schedule)[window+sample_loc[1]-1] + np.cumsum(schedule)[window+sample_loc[1]])/2,
              np.cumsum(schedule)[window+sample_loc[2]],
              np.cumsum(schedule)[window+sample_loc[3]]]

axes.set_xlim(min(t_show), 14.34)
axes.set_xticks(time_ticks)
# axes.set_xticklabels(["$t_B$", "$t_C$", "$t_D$", "$t_E$"])
axes.set_xticklabels(["$t_B$", "$t_C$", "$t_D$", "$t_E$"], color="white")
axes.set_yscale("log")
axes.set_ylim(1,1.5*v_lim)
axes.set_xlabel(r'Time ($t$)',fontsize=axes_font_size)
axes.set_ylabel(r'Speed ($v$)',fontsize=axes_font_size)

for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

ax_bis = axes.twinx()
#ax_bis.plot(T[window+n_r:window+3*n_r+1], eta_time[window+n_r:window+3*n_r+1], color='orange', marker = '+')
ax_bis.plot(T_to_display, eta_to_display, linewidth=linewidth, color=color_eta)
ax_bis.plot(T_to_display, np.ones(len(T_to_display))*eta_target, color = 'black', linestyle='--', linewidth = linewidth)
#ax_bis.plot(T, eta_time, color='orange', marker = '+')

ax_bis.set_ylabel(r'Feeding rate ($\eta$)',fontsize=axes_font_size)
ax_bis.set_ylim([0, 200])
ax_bis.set_yticks([0, 100, 200])
ax_bis.set_xlabel(r'Time ($t$)',fontsize=axes_font_size)
for label in (ax_bis.get_xticklabels() + ax_bis.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

#legend_elements = [Line2D([0], [0], color='blue', label=r'Speed of the animal ($v$)'),
#                   Line2D([0], [0], color='orange', label=r'Feeding rate ($\eta$)')]

#axes.legend(handles=legend_elements, loc='upper left', fontsize = legend_font_size)
#axes[1].set_aspect('equal')
plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

#plt.savefig(f'images/v_vs_t.png', dpi=my_dpi)

plt.show()

