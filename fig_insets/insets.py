import numpy as np
import matplotlib.pyplot as plt
import dill


rho = np.load("../rho.npy")
rho_1 = np.load("../rho_1.npy")
rho_2 = np.load("../rho_2.npy")

# with open("../environment_test_1.pkl", 'rb') as fileopen:
#     rho_1 = dill.load(fileopen) * 10

# with open("../environment_test_2.pkl", 'rb') as fileopen:
#     rho_2 = dill.load(fileopen) * 10

for i in range(3):

    rho = np.concatenate((rho,rho))
    rho_1 = np.concatenate((rho_1[:-200],rho_1))
    rho_2 = np.concatenate((rho_2[:-400],rho_2))


### PLOTS ###


my_dpi = 96

axes_font_size = 30
title_font_size = 20
graduation_font_size = 25
legend_font_size = 25

factor_inset = 2

image_width = 1200
image_height = 100 #388

color_rho = [(102/255,166/255,30/255), (230/255,171/255,2/255), (166/255,118/255,29/255) ]
color_v = (117/255,112/255,179/255)
color_tau = (231/255,41/255,138/255)
color_eta = (27/255,158/255,119/255)
color_eta_m = (217/255,95/255,2/255)
color_meta = (102/255,102/255,102/255)

linewidth = 4
ticks_width = 1
ticks_length = 4

top=1
bottom=0
right=1
left=0

# 1
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)

axes.fill_between(np.linspace(0,len(rho)-1,len(rho)), rho, color = color_rho[0], step="mid")

#axes.set_ylim(0,1.1 * max(rho))
#axes.set_xlim(40,80)

axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Density ($\rho$)',fontsize=axes_font_size)

for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_axis_off()

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

plt.savefig('images/a.png', dpi=my_dpi)

# 2
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)

axes.fill_between(np.linspace(0,len(rho_1)-1,len(rho_1)), rho_1, color = color_rho[1], step="mid")

#axes.set_ylim(0,1.1 * max(rho))
#axes.set_xlim(40,80)

axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Density ($\rho$)',fontsize=axes_font_size)

for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_axis_off()

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

plt.savefig('images/b.png', dpi=my_dpi)

# 3
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)

axes.fill_between(np.linspace(0,len(rho_2)-1,len(rho_2)), rho_2, color = color_rho[2], step="mid")

#axes.set_ylim(0,1.1 * max(rho))
#axes.set_xlim(40,80)

axes.set_xlabel(r'Position ($X$)',fontsize=axes_font_size)
axes.set_ylabel(r'Density ($\rho$)',fontsize=axes_font_size)

for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_axis_off()

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

plt.savefig('images/c.png', dpi=my_dpi)

plt.show()
