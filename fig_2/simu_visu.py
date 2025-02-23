import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scipy as sp
import dill
import os

# inpt = [rho, length, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
# outpt = [v_list_x[0:n],rho_0[0:n] - np.array(food_eaten[0:n]), np.array(eta_list), [np.sum(food_eaten)/T,eta_star_list[-1]], eta_star_list]

script_dir = os.path.dirname(__file__)
script_dir_parent = os.path.abspath(os.path.join(script_dir, os.pardir))

name = "simulation"

with open(os.path.join(script_dir,name,f"{name}.pkl"), 'rb') as fileopen:
        inpt, outpt = dill.load(fileopen)

from simu import depletion, gamma_of_rho

rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a = inpt
schedule = outpt

l = len(schedule)

tail = n_r

print(l)

rho = np.concatenate((rho,rho[:2*n_r]))
rho = np.concatenate((rho[-2*n_r:], rho))

rho_final = np.copy(rho)
eta_list = []
eta_m_list = []
tau = []

for m in range(len(schedule)):
    tau.append(np.sum(schedule[m-n_r+1:m+1]))

for n in range(l):
    
    t = schedule[n]
    rho_depleted = depletion(rho_final[n:n + n_r], [t_d0, t])
    rho_final[n:n + n_r] = rho_final[n:n + n_r] - rho_depleted
    eta = np.sum(rho_depleted)*r/n_r/t
    eta_list.append(eta)

for i in range(l):
    eta_m_list.append(r/n_r*np.sum(gamma_of_rho(rho_final[i:i + n_r],t_d0)))

rho = rho[2*tail:-2*tail-1]
rho_final = rho_final[2*tail:-2*tail-1]
tau = tau[2*tail:-tail-1]
schedule = schedule[2*tail:-tail-1]
eta_list = eta_list[2*tail:-tail-1]
eta_m_list = eta_m_list[2*tail:-tail-1]

eta_bar = np.sum(rho - rho_final)*l*r/n_r/T
v = r/n_r/schedule

x = np.linspace(1, len(schedule), len(schedule))

### Plots ###

print("eta_bar = ", eta_bar)

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

xticks_environment = [0, 1000, 2000, 3000]


# 1
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 2')

axes.fill_between(x, rho, color = color_rho, step="mid")
axes.fill_between(x, rho_final, color = 'black', step="mid", alpha=0.7)
axes.set_xlim(min(x),max(x))
axes.set_ylim(0,1.1*max(rho))
#axes[0].set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Food density ($\rho$)',fontsize=axes_font_size)
#axes[0].set_title("Density profiles", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

legend_elements = [Patch(facecolor='olivedrab',
                         label=r'Initial food density ($\rho_0$)'),
                   Patch(facecolor='black', alpha=0.7,
                         label=r'Final food density ($\rho_f$)')]

axes.set_xlim(min(x),max(x))
axes.set_xticks(xticks_environment)
axes.set_xticklabels([],color='w')
axes.set_yticks([0, 5, 10])

#axes.legend(handles=legend_elements, loc='upper center', fontsize = legend_font_size, frameon=False)
#axes[0].set_aspect('equal')

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

#plt.savefig('images/a.png', dpi=my_dpi)

# 2
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)

axes.plot(x, v, color=color_v, linewidth = linewidth, zorder=1)
axes.set_xlim(min(x),max(x))
#axes.set_ylim(0,1.1*v_lim)
axes.set_yscale('log')
#axes[1].set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Speed ($v$)',fontsize=axes_font_size)
#axes[1].set_title("Strategy", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

ax_bis = axes.twinx()
ax_bis.plot(x, tau, color=color_tau, linewidth = linewidth, alpha=0.6, zorder=2)
ax_bis.set_ylabel(r'Contact time ($\tau_f$)',fontsize=axes_font_size)
for label in (ax_bis.get_xticklabels() + ax_bis.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

legend_elements = [Line2D([0], [0], color='blue', label=r'Speed ($v$)'),
                   Line2D([0], [0], color='grey', label=r'Contact time ($\tau$)')]

axes.set_xlim(min(x),max(x))
axes.set_xticks(xticks_environment)
axes.set_xticklabels([],color='w')
axes.set_ylim(10,1.1*v_lim)
ax_bis.set_ylim([0, 1])
ax_bis.set_yticks([0, .5, 1])

#axes.legend(handles=legend_elements, loc='center', fontsize = legend_font_size, frameon=False)
#axes[1].set_aspect('equal')

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

#plt.savefig('images/b.png', dpi=my_dpi)

# 3

fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)

axes.plot(x, eta_list, color=color_eta, linewidth = linewidth)
axes.plot(x, eta_m_list, color=color_eta_m, linewidth = linewidth)
axes.plot(x, np.ones(len(x))*eta_target, color = 'black', linestyle='--', linewidth = linewidth)
axes.set_xlim(min(x),max(x))
# axes.set_ylim(0,1.1*max(eta_list))
axes.set_ylim(0,150)
axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Feeding rate ($\eta$)',fontsize=axes_font_size)
#axes[2].set_title("Feeding rate profiles", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xlim(min(x),max(x))
axes.set_xticks(xticks_environment)
axes.set_yticks([0, 50, 100, 150])

legend_elements = [Line2D([0], [0], color='blue', label=r'Feeding rate ($\eta$)'),
                   Line2D([0], [0], color='orange', label=r'Marginal feeding rate ($\eta_m$)')]

#axes.legend(handles=legend_elements, loc='upper center', fontsize = legend_font_size, frameon=False)
#axes[2].set_aspect('equal')

#fig.tight_layout()

#plt.gca().set_axis_off()
#plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#            hspace = 0, wspace = 0)
#plt.margins(0,0)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

#plt.savefig('images/c.png', dpi=my_dpi)

plt.show()


