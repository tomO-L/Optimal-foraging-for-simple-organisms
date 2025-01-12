# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 07:37:08 2024

@author: apadmin
"""

schedule = outpt[0]
eta_target_list = outpt[1]
eta_list = outpt[2]

eta_target = 103
eta_target_new = []
alpha = 5*10**-3
for i_step in range(len(schedule)):    
    v = 1/schedule[i_step]
    alpha_now = alpha/v # Make learning rate proportional to time spent in the bin, so that we average over time and not over bins
    eta_target = eta_target*(1 - alpha_now) + eta_list[i_step]*alpha_now    
    print(f"alpha = {alpha}")
    print(f"v = {v}")
    print(f"alpha_now = {alpha_now}")
    print(f"eta = {eta}")
    print(f"eta_target = {eta_target}")
    caca
    eta_target_new.append(eta_target)
    
    