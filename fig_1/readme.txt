Steps to generate the panels in Figure 1:

- Run Plots\env_generator_al to generate the 2D environment, the 2D trajectory, and the density profile along the trajectory
- Run Plots\fig_1\random_schedule to generate the arbitrary speed profile used in Figure 1. This script also takes the results of the previous one, and a lot of other stuff, and saves them all together in a file that will be opened later.
- Run Plots\fig_1\simu_visu

Changes made by Alfonso:
- New panel with 2D environment and trajectory
- New density profile
- Change arbitrary schedule so that it does 