# -*- coding: utf-8 -*-
"""
This script generates the 2D environment and trajectory shown in Figure 1A, and the density profile along that trajectory, which will be used everywhere else.
It saves all this information in npy files in the simulation_test folder

Created on Thu Aug 29 20:39:27 2024

@author: APELab
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splev, splrep
import os

x_im = np.arange(0, 3, .001)
y_im = np.arange(0, 1, .001)
(X, Y) = np.meshgrid(x_im, y_im)

# High density uniform patch
xy_drop1 = np.array([.4, .6])
xy_drop2 = xy_drop1 + np.array([.35, .1])
sigma_drop1 = .1
sigma_drop2 = .02
potential = np.exp(-((X - xy_drop1[0])**2 + (Y - xy_drop1[1])**2)/sigma_drop1) + np.exp(-((X - xy_drop2[0])**2 + (Y - xy_drop2[1])**2)/sigma_drop2)
patch = potential > .8
patch = patch.astype(np.float32)
patch1 = gaussian_filter(patch, 20)

# Low density uniform patch
xy_drop1 = np.array([1.3, .1])
xy_drop2 = xy_drop1 + np.array([.1, .3])
xy_drop3 = xy_drop2 + np.array([-.15, .3])
xy_drop4 = xy_drop3 + np.array([0, .45])
sigma_drop1 = .1
sigma_drop2 = .02
sigma_drop3 = .1
sigma_drop4 = .05
potential = (np.exp(-((X - xy_drop1[0])**2 + (Y - xy_drop1[1])**2)/sigma_drop1) 
            + np.exp(-((X - xy_drop2[0])**2 + (Y - xy_drop2[1])**2)/sigma_drop2)
            + np.exp(-((X - xy_drop3[0])**2 + (Y - xy_drop3[1])**2)/sigma_drop3)
            + np.exp(-((X - xy_drop4[0])**2 + (Y - xy_drop4[1])**2)/sigma_drop4))
patch = potential > .8
patch = patch.astype(np.float32)
patch2 = gaussian_filter(patch, 20)

xy_drop1 = np.array([2.15, .35])
xy_drop2 = xy_drop1 + np.array([.3, -.2])
sigma_drop1 = .05
sigma_drop2 = .02
hetero = np.exp(-((X - xy_drop1[0])**2 + (Y - xy_drop1[1])**2)/sigma_drop1)# + .8*np.exp(-((X - xy_drop2[0])**2 + (Y - xy_drop2[1])**2)/sigma_drop2)

environment = 10*(patch1 + patch2*.2 + hetero)

xy_traj = np.array([
 # (-.1, 0.2),
 (0.045362903225806384, 0.14649067540322575),
 (0.251008064516129, 0.15102696572580637),
 (0.331149193548387, 0.20546244959677412),
 (0.36895161290322576, 0.30072454637096774),
 (0.471774193548387, 0.33550277217741936),
 (0.609375, 0.3385269657258064),
 (0.7121975806451613, 0.3717930947580645),
 (0.8815524193548387, 0.4186680947580645),
 (0.9692540322580645, 0.3460874495967741),
 (1.061491935483871, 0.262922127016129),
 (1.141633064516129, 0.262922127016129),
 (1.21875, 0.279555191532258),
 (1.3215725806451613, 0.34911164314516124),
 (1.4606854838709677, 0.6439705141129033),
 # (1.4213709677419355, 0.7634261592741935),
 # (1.4501008064516128, 0.8178616431451613),
 # (1.5151209677419355, 0.7634261592741935),
 # (1.569556451612903, 0.7256237399193548),
 # (1.6391129032258065, 0.718063256048387),
 # (1.7222782258064515, 0.7150390625),
 # (1.7888104838709675, 0.718063256048387),
 # (1.8296370967741935, 0.7044543850806453),
 # (1.867439516129032, 0.6757245463709678),
 # (1.905241935483871, 0.6424584173387098),
 # (1.938508064516129, 0.604655997983871),
 # (1.982358870967742, 0.541147933467742),
 # (1.4788306451612903, 0.6137285786290323),
 # (1.4758064516129032, 0.6742124495967743),
 (1.4380040322580645, 0.7740108366935485),
 (1.489415322580645, 0.8133253528225808),
 (1.577116935483871, 0.8284463205645163),
 (1.649697580645161, 0.8208858366935485),
 (1.717741935483871, 0.7830834173387098),
 (1.7827620967741935, 0.7679624495967743),
 (1.864415322580645, 0.7392326108870968),
 (1.9264112903225805, 0.7210874495967743),
 (1.988407258064516, 0.7074785786290323),
 (2.0473790322580645, 0.7105027721774193),
 (2.101814516129032, 0.7815713205645163),
 (2.139616935483871, 0.8133253528225808),
 (2.1713709677419355, 0.8103011592741935),
 (2.200100806451613, 0.764938256048387),
 (2.2182459677419355, 0.7105027721774193),
 (2.239415322580645, 0.6409463205645163),
 (2.263608870967742, 0.5698777721774193),
 (2.3119959677419355, 0.5078818044354838),
 (2.366431451612903, 0.464030997983871),
 (2.4359879032258065, 0.42774067540322575),
 (2.5433467741935485, 0.41564390120967737),
 (2.641633064516129, 0.4232043850806451),
 (2.745967741935484, 0.45798261088709674),
 (2.8200604838709675, 0.510905997983871),
 (2.8805443548387095, 0.5517326108870968),
 (2.936491935483871, 0.5819745463709678),])


x_indiv = xy_traj[:, 0]
y_indiv = xy_traj[:, 1]
x_sp = splrep(np.arange(len(x_indiv)), x_indiv)
y_sp = splrep(np.arange(len(x_indiv)), y_indiv)
x_traj = splev(np.arange(0, len(x_indiv), .001), x_sp)
y_traj = splev(np.arange(0, len(x_indiv), .001), y_sp)



color_food = (102/255,166/255,30/255)
# colormap = [np.linspace(1, color_food[0], 64), np.linspace(1, color_food[1], 64), np.linspace(1, color_food[2], 64)]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1,1,1), color_food])

x_pixel = x_traj*len(x_im)/x_im[-1] # Trajectory in pixels
y_pixel = (1 - y_traj)*len(y_im)/y_im[-1] # Trajectory in pixels



# Now get the density along the trajectory

# First, get a new trajectory with steps of 1 pixel
v = np.sqrt((x_pixel[1:] - x_pixel[0:-1])**2 + (y_pixel[1:] - y_pixel[0:-1])**2)
dist = np.concatenate(([0], np.cumsum(v))) # Distance traveled
x_new = np.interp(np.arange(0, dist[-1], 1), dist, x_pixel)
y_new = np.interp(np.arange(0, dist[-1], 1), dist, y_pixel)

# Now get the density in each pixel
density = np.zeros(len(x_new)) # To preallocate
for i_step in range(len(density)):
    density[i_step] = environment[int(np.round(y_new[i_step])), int(np.round(x_new[i_step]))]

fig, ax = plt.subplots()  
plt.imshow(environment, extent=[0, 3, 0, 1], cmap=cmap)
plt.plot(x_traj, y_traj, color=(102/255, 102/255, 102/255))
plt.colorbar(fraction=0.02, pad=0.04, label='Food density')
ax.set_axis_off()
plt.show()

ax = plt.imshow(environment, cmap=cmap)
plt.plot(x_new, y_new)
# 
# plt.plot(x_pixel, y_pixel)
plt.show()

plt.plot(density)
plt.show()

rho = density[0:3500]

folder_to_save = "" # To save it in the current folder
np.save(os.path.join(folder_to_save, "rho.npy"), rho)
np.save(os.path.join(folder_to_save, "environment_2D.npy"), environment)
traj_2D = [x_new, y_new]
np.save(os.path.join(folder_to_save, "traj_2D.npy"), traj_2D)