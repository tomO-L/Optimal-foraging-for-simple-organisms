import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy as sp
import dill
import os

# inpt = [rho, length, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a]
# outpt = [v_list_x[0:n],rho_0[0:n] - np.array(food_eaten[0:n]), np.array(eta_list), [np.sum(food_eaten)/T,eta_star_list[-1]], eta_star_list]

script_dir = os.path.dirname(__file__)
script_dir_parent = os.path.abspath(os.path.join(script_dir, os.pardir))

name = "simulation_test"

with open(os.path.join(script_dir,name,f"{name}.pkl"), 'rb') as fileopen:
        inpt, outpt = dill.load(fileopen)

from simu import depletion, gamma_of_rho

rho, T, t_d0, n_r, r, v_lim, eta_target, noise, learning, trade_off, alpha, a, environment_2D, traj_2D = inpt
schedule = outpt

print('n_r=',n_r)

l = len(schedule)

tail = n_r

rho = np.concatenate((rho,rho[:2*n_r]))
rho = np.concatenate((rho[-2*n_r:], rho))

rho_final = np.copy(rho)
eta_list = []
eta_m_list = []
tau = []
mid = 1750 + 2*tail


for m in range(len(schedule)):

    tau.append(np.sum(schedule[m-n_r+1:m+1]))


for n in range(l):
    
    t = schedule[n]
    rho_depleted = depletion(rho_final[n:n + n_r], [t_d0, t])
    rho_final[n:n + n_r] = rho_final[n:n + n_r] - rho_depleted
    eta = np.sum(rho_depleted)*r/n_r/t
    eta_list.append(eta)

    if n == mid:

        rho_mid = np.copy(rho_final)
        local_gamma = rho_depleted/t

for i in range(l):
    eta_m_list.append(r/n_r*np.sum(gamma_of_rho(rho_final[i:i + n_r],t_d0)))
    if i == mid:
        marginal_local_gamma = r/n_r*gamma_of_rho(rho_final[i:i + n_r],t_d0)

rho = rho[2*tail:-2*tail-1]
rho_final = rho_final[2*tail:-2*tail-1]
rho_mid = rho_mid[2*tail:-2*tail-1]
tau = tau[2*tail:-tail-1]
schedule = schedule[2*tail:-tail-1]
eta_list = eta_list[2*tail:-tail-1]
eta_m_list = eta_m_list[2*tail:-tail-1]

mid = mid - 2*tail

gamma = np.zeros(len(schedule))
gamma[mid:mid+n_r] = local_gamma
marginal_gamma = np.zeros(len(schedule))
marginal_gamma[mid:mid+n_r] = marginal_local_gamma

eta_bar = np.sum(rho - rho_final)*l*r/n_r/T
v = r/n_r/schedule

res = 1000

x = np.linspace(1, len(schedule), len(schedule))

local_time_ex = np.linspace(0,tau[mid],res)
local_time_pos = int(mid+n_r/2)
local_rho_ex = rho[mid] - depletion(rho[mid],[t_d0, local_time_ex])
local_rho_ex = np.concatenate((np.ones(res)*rho[mid],local_rho_ex))
local_gamma_ex = t_d0*(rho[mid] - depletion(rho[mid],[t_d0, local_time_ex]))
local_gamma_ex = np.concatenate((np.zeros(res)*rho[mid]/t_d0,local_gamma_ex))
mid_mid_time = np.sum(schedule[mid:mid+int(n_r/2)])
mid_mid_index = (np.sum(schedule[:mid])+mid_mid_time<=np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)).argmax()
#np.sum(schedule[:mid])+mid_mid_time

#print(mid_mid_index)
#print(np.sum(schedule[:mid]) + mid_mid_time)
#print(np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],100))

### Plots ###

#print("eta_bar = ", eta_bar)


#########################################################
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
#########################################################

# 0 2D environment and trajectory
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 A')


plt.imshow(environment_2D, cmap=cmap)

axes.plot(traj_2D[0], traj_2D[1], linewidth=linewidth, color=color_meta)
cbar = plt.colorbar(ax=axes)#, label='Food density')#(fraction=0.02, pad=0.04)
cbar.ax.tick_params(labelsize=graduation_font_size)
cbar.set_label(label='Food density', fontsize=axes_font_size)
cbar.ax.set_yticks([0,5,10])
axes.set_axis_off()


plt.subplots_adjust(top=top, bottom=bottom, right=right, left=left)
#plt.savefig('images/zero.png', dpi=my_dpi)

# 1
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 B')

axes.fill_between(x, rho[:l], color = color_rho)
axes.set_xlim(min(x),max(x))
axes.set_ylim(0,1.1*max(rho))
#axes.set_xlabel(r'$X$',fontsize=axes_font_size)
axes.set_ylabel('Initial\nfood density ($\\rho_0$)',fontsize=axes_font_size)
#axes.set_title("Initial density", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xticks([0,1000,2000,3000])
axes.set_xticklabels([0,1000,2000,3000],color='w')
axes.set_yticks([0,5,10])
axes.set_yticklabels([0,5,10])

axes.tick_params(length = ticks_length, width = ticks_width)

plt.subplots_adjust(top=top, bottom=bottom, right=right, left=left)
#plt.savefig('images/a.png', dpi=my_dpi)



# 2
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 E')

axes.fill_between(x, rho[:l], color = color_rho)
axes.plot(np.ones(50)*mid, np.linspace(0,20), '--', color = color_meta)
axes.plot(np.ones(50)*(mid+n_r), np.linspace(0,20), '--', color = color_meta)
axes.fill_between(x, rho_mid[:l], color = 'black', alpha=0.4)
axes.set_xlim(min(x),max(x))
axes.set_ylim(0,1.1*max(rho))
#axes.set_xlabel(r'$X$',fontsize=axes_font_size)
axes.set_ylabel(r'Food density ($\rho$)',fontsize=axes_font_size)
#axes.set_title("During foraging", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xticks([0,1000,2000,3000])
axes.set_xticklabels([0,1000,2000,3000],color='w')
axes.set_yticks([0,5,10])
axes.set_yticklabels([0,5,10])
axes.tick_params(length = ticks_length, width = ticks_width)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/b.png', dpi=my_dpi)

# 3
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 H')

axes.fill_between(x, rho[:l], color = color_rho)
axes.fill_between(x, rho_final[:l], color = 'black', alpha=0.7)
axes.plot(np.ones(50)*mid, np.linspace(0,20), '--', color = color_meta)
axes.plot(np.ones(50)*(mid+n_r), np.linspace(0,20), '--', color = color_meta)
axes.set_xlim(min(x),max(x))
axes.set_ylim(0,1.1*max(rho))
#axes.set_xlabel(r'$X$',fontsize=axes_font_size)
axes.set_ylabel(r'Food density ($\rho$)',fontsize=axes_font_size)
#axes.set_title("After foraging", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xticks([0,1000,2000,3000])
axes.set_xticklabels([0,1000,2000,3000],color='w')
axes.set_yticks([0,5,10])
axes.set_yticklabels([0,5,10])
axes.tick_params(length = ticks_length, width = ticks_width)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/c.png', dpi=my_dpi)

# 4
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 C')

axes.plot(x, v_lim*np.ones(len(x)), color='black', linestyle='--')
axes.plot(x, v, color = color_v, linewidth = linewidth)
axes.set_yscale('log')
axes.set_xlim(min(x),max(x))
axes.set_ylim(1,8*v_lim)
#axes.set_xlabel(r'$X$',fontsize=axes_font_size)
axes.set_ylabel(r'Speed ($v$)',fontsize=axes_font_size)
#axes.set_title("Strategy", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xticks([0,1000,2000,3000])
axes.set_xticklabels([0,1000,2000,3000],color='w')
axes.set_yticks([1, 10, 100])
# axes.set_yticklabels(["$10^0$", "$10^1$", "$10^2$", "$v_{max}$"])
#axes.set_yticks([0,50,100,150,200])
#axes.set_yticklabels([0,50,100,150,200])
axes.tick_params(length = ticks_length, width = ticks_width)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/d.png', dpi=my_dpi)

# 5
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 F')

axes.plot(x, gamma, color = 'black', linewidth = linewidth, zorder=4)
axes.scatter(local_time_pos,gamma[local_time_pos], color='black', s=80, zorder=5)
axes.fill_between(x, gamma, color = color_eta, linewidth = linewidth, alpha=0.5, zorder=2)
axes.plot(np.ones(50)*mid, np.linspace(0,20), '--', color = color_meta, zorder=6)
axes.plot(np.ones(50)*(mid+n_r), np.linspace(0,20), '--', color = color_meta, zorder=7)
axes.set_xlim(1600,1830)
axes.set_ylim(0,2.5)

#axes.set_xlabel(r'$X$',fontsize=axes_font_size)
axes.set_ylabel('Depletion\nrate ($\\gamma$)',fontsize=axes_font_size)
#axes.set_title("Depletion rate", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xticks([])
axes.set_xticklabels([],color='w')
axes.set_yticks([0,1,2])
# axes.set_yticklabels([0,5,10])
axes.tick_params(length = ticks_length, width = ticks_width)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/e.png', dpi=my_dpi)

# 6
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 I')

axes.plot(x, marginal_gamma, color = 'black', linewidth = linewidth)
axes.fill_between(x, marginal_gamma, color = color_eta_m, linewidth = linewidth, alpha=0.5)
axes.plot(np.ones(50)*mid, np.linspace(0,20), '--', color = color_meta)
axes.plot(np.ones(50)*(mid+n_r), np.linspace(0,20), '--', color = color_meta)
axes.set_xlim(1600,1830)
axes.set_ylim(0,2.5)
#axes.set_xlabel(r'$X$',fontsize=axes_font_size)
axes.set_ylabel('Marginal\ndeplet. rate ($\\gamma_m$)',fontsize=axes_font_size)
#axes.set_title("Marginal depletion rate", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xticks([])
axes.set_xticklabels([],color='w')
axes.set_yticks([0,1,2])
# axes.set_yticklabels([0,5,10])
axes.tick_params(length = ticks_length, width = ticks_width)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/f.png', dpi=my_dpi)

# 7
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 D')

axes.plot(x, tau, color = color_tau, linewidth = linewidth)
axes.set_xlim(min(x),max(x))
axes.set_ylim(0,4)
axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel('Contact time\n($\\tau$)',fontsize=axes_font_size)
#axes.set_title("Schedule", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xticks([0,1000,2000,3000])
axes.set_yticks([0,2,4])
axes.tick_params(length = ticks_length, width = ticks_width)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/g.png', dpi=my_dpi)

# 8
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 G')

axes.plot(x[:mid], eta_list[:mid], color = color_eta, linewidth = linewidth)
axes.plot(x[mid:], eta_list[mid:], linestyle = ':', color = color_eta, linewidth = linewidth)
axes.plot(np.ones(50)*mid, np.linspace(0,400), '--', color = color_meta)
axes.plot(np.ones(50)*(mid+n_r), np.linspace(0,400), '--', color = color_meta)
axes.scatter(mid,eta_list[mid], color='black', s=80, zorder=5)
axes.set_xlim(min(x),max(x))
axes.set_ylim(0,1.1*max(eta_list))
axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Feeding rate ($\eta$)',fontsize=axes_font_size)
#axes.set_title("Feeding rate", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xticks([0,1000,2000,3000])
axes.set_yticks([0, 100, 200])
axes.tick_params(length = ticks_length, width = ticks_width)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/h.png', dpi=my_dpi)

# 9
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 J')

axes.plot(x, eta_m_list, color = color_eta_m, linewidth = linewidth)
axes.set_xlim(min(x),max(x))
axes.set_ylim(0,1.1*max(eta_list))
axes.plot(np.ones(50)*mid, np.linspace(0,300), '--', color = color_meta)
axes.plot(np.ones(50)*(mid+n_r), np.linspace(0,300), '--', color = color_meta)
axes.scatter(mid,eta_m_list[mid], color='black', s=80, zorder=5)
axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel('Marginal\nfeed. rate ($\\eta_m$)',fontsize=axes_font_size)
#axes.set_title("Marginal feedign rate", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xticks([0,1000,2000,3000])
axes.set_yticks([0,100,200])
axes.tick_params(length = ticks_length, width = ticks_width)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/i.png', dpi=my_dpi)

#plt.legend()
plt.show()

# 1 bis: Density as a function of time for one point (inset)
fig,axes_bis = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 F inset 1')

axes_bis.plot(np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[:mid_mid_index], local_rho_ex[:mid_mid_index], color = color_rho, linewidth = linewidth*factor_inset)
axes_bis.plot(np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[mid_mid_index-1:], local_rho_ex[mid_mid_index-1:], linestyle = ':', color = color_rho, linewidth = linewidth*factor_inset)
axes_bis.plot(np.ones(n_r)*np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[mid_mid_index],np.linspace(0,3,n_r), linestyle = '--', color = color_meta, linewidth = linewidth*factor_inset)
axes_bis.plot(np.linspace(np.sum(schedule[:mid])+tau[mid],np.sum(schedule[:mid])+2*tau[mid],2*res), np.ones(2*res)*local_rho_ex[-1], linestyle = ':', color = color_rho, linewidth = linewidth*factor_inset)

"""
axes_bis.plot(np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res), local_rho_ex, color = color_rho, linewidth = linewidth*factor_inset)
axes_bis.plot(np.ones(n_r)*np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[mid_mid_index],np.linspace(0,3,n_r), linestyle = '--', color = color_meta, linewidth = linewidth*factor_inset)
axes_bis.plot(np.linspace(np.sum(schedule[:mid])+tau[mid],np.sum(schedule[:mid])+2*tau[mid],2*res), np.ones(2*res)*local_rho_ex[-1], linestyle = ':', color = color_rho, linewidth = linewidth*factor_inset)
"""
axes_bis.set_xlim(9.15,9.7)
axes_bis.set_ylim(0,1.1*rho[mid])
#axes_bis[0].set_xlabel(r'$t$',fontsize=axes_font_size)
axes_bis.set_ylabel(r'$\rho$',fontsize=axes_font_size*factor_inset)
# axes_bis.set_ylabel(r'Food density ($\rho)$',fontsize=axes_font_size)
#axes_bis[0].set_title("Density", fontsize = title_font_size)
for label in (axes_bis.get_xticklabels() + axes_bis.get_yticklabels()):
    label.set_fontsize(graduation_font_size*factor_inset)

#axes_bis.set_xticks([0,1,2,3])
axes_bis.set_xticks([9.15,9.7])
axes_bis.set_xticklabels([9.15,9.7], color = 'w')
axes_bis.set_yticks([1,2])

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/j.png', dpi=my_dpi)


# 2 bis: Depletion rate as a function of time for one point (inset)
fig,axes_bis = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 1 F inset 2')

axes_bis.plot(np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[:res], local_gamma_ex[:res], color = 'black', linewidth = linewidth*factor_inset)
axes_bis.plot(np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[res:mid_mid_index], local_gamma_ex[res:mid_mid_index], color = 'black', linewidth = linewidth*factor_inset)
axes_bis.plot(np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[mid_mid_index-1:], local_gamma_ex[mid_mid_index-1:], linestyle = ':', color = 'black', linewidth = linewidth*factor_inset)
axes_bis.plot(np.ones(n_r)*np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[mid_mid_index],np.linspace(0,3,n_r), linestyle = '--', color = color_meta, linewidth = linewidth*factor_inset)
axes_bis.plot(np.linspace(np.sum(schedule[:mid])+tau[mid],np.sum(schedule[:mid])+2*tau[mid],2*res), np.zeros(2*res), linestyle = ':', color = 'black', linewidth = linewidth*factor_inset)

a = np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[-1]
b = np.linspace(np.sum(schedule[:mid])+tau[mid],np.sum(schedule[:mid])+2*tau[mid],2*res)[0]
c = np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[:res][-1]
d = np.linspace(np.sum(schedule[:mid])-tau[mid],np.sum(schedule[:mid])+tau[mid],2*res)[res:mid_mid_index][0]

axes_bis.plot([a,b], [local_gamma_ex[-1],0], linestyle = ':', color = 'black', linewidth = linewidth*factor_inset)
axes_bis.plot([c,d], [0,local_gamma_ex[res]], color = 'black', linewidth = linewidth*factor_inset)

axes_bis.set_xlim(9.15,9.7)
axes_bis.set_ylim(0,1.1*rho[mid])
axes_bis.set_xlabel(r'Time ($t$)',fontsize=axes_font_size*factor_inset)
axes_bis.set_ylabel(r'$\gamma$',fontsize=axes_font_size*factor_inset)
#axes_bis[1].set_title("Depletion rate", fontsize = title_font_size)
for label in (axes_bis.get_xticklabels() + axes_bis.get_yticklabels()):
    label.set_fontsize(graduation_font_size*factor_inset)

#axes_bis.set_xticks([0,1,2,3])
#axes_bis.set_xticklabels([0,1,2,3], color = 'w')
axes_bis.set_xticks([])
axes_bis.set_yticks([0,1,2])

#fig.tight_layout()

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/k.png', dpi=my_dpi)

plt.show()
