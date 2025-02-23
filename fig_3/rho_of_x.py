import numpy as np
import matplotlib.pyplot as plt


def depletion(rho_0, pars):
    t_d0 = pars[0]
    t = pars[1]
    r = pars[2]
    n_r = pars[3]

    res = rho_0 * (1 - np.exp(-t / t_d0))

    return res

rho = np.ones(150) * 10
rho_init = np.copy(rho)

r = 20
n_r = 20
t_d = 1
v = 10
dt = r/n_r/v

t = np.cumsum(np.ones(n_r)*dt)
t = np.flip(t)

pars = [t_d,t,r,n_r]

rho[50:50+n_r] = rho[50:50+n_r] - depletion(rho[50:50+n_r], pars)
rho_f = min(rho[50:50+n_r])

rho[:50] = rho_f


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

fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 3')

pos_txt_x = 1.025
pos_txt_y = 1

axes.fill_between(np.linspace(0,149,150), rho_init, color = color_rho, step="mid")
# axes.fill_between(np.linspace(0,149,150), rho, color = 'black', step="mid", alpha=0.7)
axes.fill_between(np.linspace(0,149,150), rho, color = 'black', alpha=0.7)
y_max = 1.2*max(rho)
axes.plot([50, 50], [0, y_max], color=color_meta, linewidth=linewidth, linestyle="--")
axes.plot([70, 70], [0, y_max], color=color_meta, linewidth=linewidth, linestyle="--")
axes.set_ylim(0, y_max)
axes.set_xlim(20,80)

axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Food density ($\rho$)',fontsize=axes_font_size)
axes.set_xticks([])
axes.set_yticks([0, 5, 10])
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)

for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/density.png', dpi=my_dpi)

plt.show()
