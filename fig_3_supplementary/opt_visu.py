import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import dill
import os

script_dir = os.path.dirname(__file__)
script_dir_parent = os.path.abspath(os.path.join(script_dir, os.pardir))

## Plots ###

my_dpi = 96

axes_font_size = 30
title_font_size = 20
graduation_font_size = 25
legend_font_size = 25

factor_inset = 2

image_width = 1200
image_height = 1200 #388

color_rho = [(102/255,166/255,30/255), (230/255,171/255,2/255), (166/255,118/255,29/255) ]
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


# 1
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)

for i in range(3):

    # with open(f"opt_{i}.pkl", 'rb') as fileopen:
    #     x,y = dill.load(fileopen)
    # with open(f"../fig_2/opt_{i}.pkl", 'rb') as fileopen:
    #     x_opt,y_opt = dill.load(fileopen)
    
    aux = np.load(os.path.join(script_dir, f"opt_{i}.npy"))
    x = aux[0]
    y = aux[1]    
    aux = np.load(os.path.join(script_dir_parent, "fig_2_supplementary", f"opt_{i}.npy"))
    x_opt = aux[0]
    y_opt = aux[1]

    axes.plot(x, y, color = color_rho[i], zorder=2, linewidth=linewidth, linestyle='--')
    axes.plot(x_opt, y_opt, color = color_rho[i], zorder=2, linewidth=linewidth)

axes.plot(np.linspace(0,200,2),np.linspace(0,200,2), linestyle=':', linewidth=linewidth, color='black', zorder=1)
axes.set_xlim(25,200)
axes.set_ylim(25,200)
axes.set_xticks([50, 100, 150, 200])
axes.set_yticks([50, 100, 150, 200])
axes.set_xlabel(r'Target marginal feeding rate ($\eta_m^*$)',fontsize=axes_font_size)
axes.set_ylabel(r'Average feeding rate ($\bar{\eta}$)',fontsize=axes_font_size)
#axes[0].set_title("Density profiles", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

legend_elements = [Line2D([0], [0], color='blue', label=r'Average feeding rate for environment 1 ($\bar{\eta}_1$)'),
                   Line2D([0], [0], color='red', label=r'$1:1$')]

#axes.legend(handles=legend_elements, loc='upper center', fontsize = legend_font_size, frameon=False)
#axes.set_aspect('equal')

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)

#plt.savefig(os.path.join(script_dir, "images", "d.png"), dpi=my_dpi)

plt.show()

