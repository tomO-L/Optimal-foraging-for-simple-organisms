import subprocess
import os
import threading

# Decide what figure to show. Set to True the variables corresponding to the figures you want to show;
plot_fig_1 = True
plot_fig_2 = False
plot_fig_3 = False
plot_fig_4 = False
plot_fig_5 = False

plot_fig_2_supplementary = False
plot_fig_3_supplementary = False
plot_fig_4_supplementary = False
plot_fig_5_supplementary = False

# Function to run a Python script
def run_script(script_name):
    subprocess.run(["python", script_name])

# Set the path of this script
main_script_dir = os.path.dirname(__file__)

# Initialize the list that will contain the threads of scripts generating the figure to run in parallel
threads = []

### Main Figures ###

# Will add Figure 1 to the list of figures to plot if plot_fig_1 is True

if plot_fig_1:
    
    fig_1_path = os.path.join(main_script_dir, "fig_1")

    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_1_path, "simu_visu.py"),)) )

# Will add Figure 2 to the list of figures to plot if plot_fig_2 is True
if plot_fig_2:
    
    fig_2_path = os.path.join(main_script_dir, "fig_2")

    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_2_path, "simu_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_2_path, "opt_visu.py"),)) )

# Will add Figure 3 to the list of figures to plot if plot_fig_3 is True
if plot_fig_3:
    
    fig_3_path = os.path.join(main_script_dir, "fig_3")

    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_3_path, "simu_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_3_path, "opt_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_3_path, "eta_vs_v.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_3_path, "rho_of_x.py"),)) )

# Will add Figure 4 to the list of figures to plot if plot_fig_4 is True
if plot_fig_4:
    
    fig_4_path = os.path.join(main_script_dir, "fig_4")

    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_4_path, "simu_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_4_path, "opt_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_4_path, "mini_rho_vs_x.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_4_path, "v_vs_t.py"),)) )

# Will add Figure 5 to the list of figures to plot if plot_fig_5 is True
if plot_fig_5:
    
    fig_5_path = os.path.join(main_script_dir, "fig_5")

    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_5_path, "simu_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_5_path, "opt_visu.py"),)) )


### Supplementary Figures ###

# Will add Figure 2 supplementary to the list of figures to plot if plot_fig_2 is True
if plot_fig_2_supplementary:
    
    fig_2_path_supplementary = os.path.join(main_script_dir, "fig_2_supplementary")

    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_2_path_supplementary, "simu_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_2_path_supplementary, "opt_visu.py"),)) )

# Will add Figure 3 supplementary to the list of figures to plot if plot_fig_3 is True
if plot_fig_3_supplementary:
    
    fig_3_path_supplementary = os.path.join(main_script_dir, "fig_3_supplementary")

    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_3_path_supplementary, "simu_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_3_path_supplementary, "opt_visu.py"),)) )

# Will add Figure 4 supplementary to the list of figures to plot if plot_fig_4 is True
if plot_fig_4_supplementary:
    
    fig_4_path_supplementary = os.path.join(main_script_dir, "fig_4_supplementary")

    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_4_path_supplementary, "simu_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_4_path_supplementary, "opt_visu.py"),)) )

# Will add Figure 5 supplementary to the list of figures to plot if plot_fig_5 is True
if plot_fig_5_supplementary:
    
    fig_5_path_supplementary = os.path.join(main_script_dir, "fig_5_supplementary")

    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_5_path_supplementary, "simu_visu.py"),)) )
    threads.append( threading.Thread(target=run_script, args=(os.path.join(fig_5_path_supplementary, "opt_visu.py"),)) )



### Execute all the selected scripts ###
for thread in threads:
    thread.start()
    
for thread in threads:
    thread.join()

