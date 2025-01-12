import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import scipy as sp
import dill
import os

### Plots ###

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

cmap = LinearSegmentedColormap.from_list("", [(1,1,1), color_rho])

alpha_red = 0.5

linewidth = 4
ticks_width = 1
ticks_length = 4

top=0.95
bottom=0.22
right=0.9
left=0.15

xticks_environment = [0, 1000, 2000, 3000]

with open(os.path.join("..", "fig_3","simulation", "simulation.pkl"), 'rb') as fileopen:
        inpt, outpt = dill.load(fileopen)

from simu_fig2 import depletion, gamma_of_rho

rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a = inpt
schedule = outpt

# 1
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)

eta_all = np.array([0])
t_all = np.array([0])
step_time = .001
# Travel period
t_travel = .3
t = np.arange(0, t_travel, step_time)
eta = np.zeros(len(t))
t_all = np.concatenate((t_all, t_all[-1] + step_time + t))
eta_all = np.concatenate((eta_all, eta))

# First patch
eta_bar = eta_target
rho_0 = 140
t_leave = np.log(rho_0/eta_bar)
t = np.arange(0, t_leave, step_time)
eta = rho_0*np.exp(-t)
t_all = np.concatenate((t_all, t_all[-1] + step_time + t))
eta_all = np.concatenate((eta_all, eta))

# Travel period
t_travel = .4
t = np.arange(0, t_travel, step_time)
eta = np.zeros(len(t))
t_all = np.concatenate((t_all, t_all[-1] + step_time + t))
eta_all = np.concatenate((eta_all, eta))

# Second patch
rho_0 = 70
t_leave = .001 #• np.log(rho_0/eta_bar)
t = np.arange(0, t_leave, step_time)
eta = rho_0*np.exp(-t)
t_all = np.concatenate((t_all, t_all[-1] + step_time + t))
eta_all = np.concatenate((eta_all, eta))

# Travel period
t_travel = .3
t = np.arange(0, t_travel, step_time)
eta = np.zeros(len(t))
t_all = np.concatenate((t_all, t_all[-1] + step_time + t))
eta_all = np.concatenate((eta_all, eta))

# Third patch
rho_0 = 110
t_leave = np.log(rho_0/eta_bar)
t = np.arange(0, t_leave, step_time)
eta = rho_0*np.exp(-t)
t_all = np.concatenate((t_all, t_all[-1] + step_time + t))
eta_all = np.concatenate((eta_all, eta))

# Travel period
t_travel = .2
t = np.arange(0, t_travel, step_time)
eta = np.zeros(len(t))
t_all = np.concatenate((t_all, t_all[-1] + step_time + t))
eta_all = np.concatenate((eta_all, eta))

axes.plot(t_all, eta_bar*np.ones(len(t_all)), color='black', linestyle='--', linewidth=linewidth)
axes.plot(t_all, eta_all, color=color_eta, linewidth=linewidth)
axes.set_xlim(0, max(t_all))
axes.set_ylim(0,150)
axes.set_xticks([])
axes.set_yticks([])
axes.set_xlabel('Time', fontsize=axes_font_size)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)


plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

plt.savefig('images/a.png', dpi=my_dpi)


# 2
# rho = np.load(os.path.join("..", "rho_0.npy"))


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

fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)
background = np.tile(rho, [100, 1])
eta_max = 150
plt.imshow(background, cmap=cmap, extent=[0, len(rho), 0, eta_max], aspect='auto')

axes.plot(x, eta_list, color=color_eta, linewidth = linewidth)
axes.plot(x, eta_m_list, color=color_eta_m, linewidth = linewidth)
axes.plot(x, np.ones(len(x))*eta_target, color = 'black', linestyle='--', linewidth = linewidth)
axes.set_xlim(0,l)
# axes.set_ylim(0,1.1*max(eta_list))
axes.set_ylim(0,150)
axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
# axes.set_ylabel(r'Feeding rate',fontsize=axes_font_size)
#axes[2].set_title("Feeding rate profiles", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xlim(min(x),max(x))
axes.set_xticks([])
# axes.set_xticks(xticks_environment)
# axes.set_yticks([0, 50, 100, 150])
axes.set_yticks([])
# axes.spines['top'].set_visible(False)

ax_bis = axes.twinx()
ax_bis.plot(x, v, color=color_v, linewidth = linewidth, zorder=1)
ax_bis.set_xlim(0,l)
#axes.set_ylim(0,1.1*v_lim)
ax_bis.set_yscale('log')
#axes[1].set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
ax_bis.set_ylabel(r'Speed ($v$)',fontsize=axes_font_size)
#axes[1].set_title("Strategy", fontsize = title_font_size)
for label in (ax_bis.get_xticklabels() + ax_bis.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

ax_bis.set_xlim(min(x),max(x))
ax_bis.set_xticks([])
# ax_bis.set_xticks(xticks_environment)
ax_bis.set_xticklabels([],color='w')
ax_bis.set_ylim(10,1.1*v_lim)
# ax_bis.spines['top'].set_visible(False)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

plt.savefig('images/b.png', dpi=my_dpi)
caca

ax_bis.plot(x, tau, color=color_tau, linewidth = linewidth, alpha=0.6, zorder=2)
ax_bis.set_ylabel(r'Contact time ($\tau$)',fontsize=axes_font_size)
for label in (ax_bis.get_xticklabels() + ax_bis.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

legend_elements = [Line2D([0], [0], color='blue', label=r'Speed ($v$)'),
                   Line2D([0], [0], color='grey', label=r'Contact time ($\tau$)')]


ax_bis.set_ylim([0, 1])
ax_bis.set_yticks([0, .5, 1])




caca























# 2
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)
zoom_region = [2450, 2800, 1.5, 250]

axes.plot(x, v, color=color_v, linewidth = linewidth, zorder=1)
axes.plot([zoom_region[_] for _ in [0, 1, 1, 0, 0]], [zoom_region[_] for _ in [2, 2, 3, 3, 2]], color="black", linewidth=1.5)

axes.set_xlim(min(x),max(x))
axes.set_ylim(1, 1.5*v_lim)
axes.set_yscale('log')
#axes[1].set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Speed ($v$)',fontsize=axes_font_size)
#axes[1].set_title("Strategy", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

ax_bis = axes.twinx()
ax_bis.plot(x, tau, color=color_tau, linewidth = linewidth, alpha=0.6, zorder=2)
ax_bis.set_ylabel(r'Contact time ($\tau$)',fontsize=axes_font_size)
ax_bis.set_ylim(0, .8)
ax_bis.set_yticks([0, .2, .4, .6, .8])
for label in (ax_bis.get_xticklabels() + ax_bis.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

legend_elements = [Line2D([0], [0], color='blue', label=r'Speed ($v$)'),
                   Line2D([0], [0], color='grey', label=r'Contact time ($\tau$)')]

axes.set_xticklabels([],color='w')

#axes.legend(handles=legend_elements, loc='center', fontsize = legend_font_size, frameon=False)
#axes[1].set_aspect('equal')

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

plt.savefig('images/b.png', dpi=my_dpi)

# 2bis: Inset
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi*.3, image_height/my_dpi*.4), dpi=my_dpi, sharex=True)
zoom_region = [2450, 2800, 1.5, 250]

axes.plot(x, v, color=color_v, linewidth = linewidth, zorder=1)
ax_bis = axes.twinx()
ax_bis.plot(x, tau, color=color_tau, linewidth = linewidth, alpha=0.6, zorder=2)

axes.set_xlim(zoom_region[0], zoom_region[1])
ax_bis.set_xlim(zoom_region[0], zoom_region[1])
axes.set_ylim(zoom_region[2], zoom_region[3])
ax_bis.set_ylim(.1, .78)
axes.set_yscale('log')
#axes[1].set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
# axes.set_xlabel('$X$', fontsize=axes_font_size)
axes.set_xticks([])
# axes.set_ylabel(r'$v$', fontsize=axes_font_size)
axes.set_yticks([])
ax_bis.set_yticks([])
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(1.5)

# #axes[1].set_title("Strategy", fontsize = title_font_size)
# for label in (axes.get_xticklabels() + axes.get_yticklabels()):
#     label.set_fontsize(graduation_font_size)

# ax_bis = axes.twinx()
# ax_bis.plot(x, tau, color=color_tau, linewidth = linewidth, alpha=0.6, zorder=2)
# ax_bis.set_ylabel(r'Contact time ($\tau$)',fontsize=axes_font_size)
# ax_bis.set_ylim(0, .8)
# ax_bis.set_yticks([0, .2, .4, .6, .8])
# for label in (ax_bis.get_xticklabels() + ax_bis.get_yticklabels()):
#     label.set_fontsize(graduation_font_size)

# legend_elements = [Line2D([0], [0], color='blue', label=r'Speed ($v$)'),
#                    Line2D([0], [0], color='grey', label=r'Contact time ($\tau$)')]

# axes.set_xticklabels([],color='w')

#axes.legend(handles=legend_elements, loc='center', fontsize = legend_font_size, frameon=False)
#axes[1].set_aspect('equal')

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

plt.savefig('images/b_inset.png', dpi=my_dpi)


# 3

fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)
zoom_region = [2450, 2800, 80, 150]

axes.plot(x, eta_list, color=color_eta, linewidth = linewidth)
axes.plot(x, eta_m_list, color=color_eta_m, linewidth = linewidth)
axes.plot(x, np.ones(len(x))*eta_target, color = 'black', linestyle='--', linewidth = linewidth)
axes.plot([zoom_region[_] for _ in [0, 1, 1, 0, 0]], [zoom_region[_] for _ in [2, 2, 3, 3, 2]], color="black", linewidth=1.5)
axes.set_xlim(min(x),max(x))
axes.set_ylim(0,1.1*max(eta_list))
axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Feeding rate',fontsize=axes_font_size)
axes.set_yticks([0, 50, 100, 150])
#axes[2].set_title("Feeding rate profiles", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

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

plt.savefig('images/c.png', dpi=my_dpi)

fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi*.3, image_height/my_dpi*.4), dpi=my_dpi, sharex=True)

axes.plot(x, eta_list, color=color_eta, linewidth = linewidth)
axes.plot(x, eta_m_list, color=color_eta_m, linewidth = linewidth)
axes.plot(x, np.ones(len(x))*eta_target, color = 'black', linestyle='--', linewidth = linewidth)
axes.set_xlim(zoom_region[0], zoom_region[1])
axes.set_ylim(zoom_region[2], zoom_region[3])
# axes.set_xlabel(r'$X$',fontsize=axes_font_size)
# axes.set_ylabel(r'$\eta$',fontsize=axes_font_size)
axes.set_xticks([])
axes.set_yticks([])
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(1.5)
#axes[2].set_title("Feeding rate profiles", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

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

plt.savefig('images/c_inset.png', dpi=my_dpi)


plt.show()


