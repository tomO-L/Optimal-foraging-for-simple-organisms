import numpy as np
import matplotlib.pyplot as plt

def feeding_rate(rho_0, pars):
    
    t_d0 = pars[0]
    t = pars[1]
    
    res = r/n_r/t*rho_0*(1 - np.exp(-t/t_d0))

    return res

r = 20
n_r = 1*r
t_d = 1
eta_f = 105#10 * r/t_d
v_lim = 200

l = 1000

v = np.linspace(1,300,l)

# densities = np.array([5, 10, 15, 20])
densities = np.array([4, 8, 16, 32])
# densities = np.array([3, 6, 12, 24])

eta_1 = n_r*feeding_rate(densities[0], [t_d, r/v])
eta_2 = n_r*feeding_rate(densities[1], [t_d, r/v])
eta_3 = n_r*feeding_rate(densities[2], [t_d, r/v])
eta_4 = n_r*feeding_rate(densities[3], [t_d, r/v])

eta_target = eta_f * v/(r/t_d) * (np.exp(r/(v*t_d))-1)
#eta_target = eta_f * v * (np.exp(r/(v*t_d))-1)/(r/t_d)

eta_m_1 = densities[0] * r * np.exp(-r/(t_d*v))
eta_m_2 = densities[1] * r * np.exp(-r/(t_d*v))
eta_m_3 = densities[2] * r * np.exp(-r/(t_d*v))
eta_m_4 = densities[3] * r * np.exp(-r/(t_d*v))

# dots

v_pt_1 = v_lim #- r/t_d * 1/np.log(eta_f/(5*r/t_d))
v_pt_2 = - r/t_d * 1/np.log(eta_f/(densities[1]*r/t_d))
v_pt_3 = - r/t_d * 1/np.log(eta_f/(densities[2]*r/t_d))
v_pt_4 = - r/t_d * 1/np.log(eta_f/(densities[3]*r/t_d))

eta_pt_1 = n_r*feeding_rate(densities[0], [t_d, r/v_pt_1])
eta_pt_2 = n_r*feeding_rate(densities[1], [t_d, r/v_pt_2])
eta_pt_3 = n_r*feeding_rate(densities[2], [t_d, r/v_pt_3])
eta_pt_4 = n_r*feeding_rate(densities[3], [t_d, r/v_pt_4])

eta_m_pt_1 = densities[0] * r/t_d * np.exp(-r/(t_d*v_pt_1))
eta_m_pt_2 = densities[1] * r/t_d * np.exp(-r/(t_d*v_pt_2))
eta_m_pt_3 = densities[2] * r/t_d * np.exp(-r/(t_d*v_pt_3))
eta_m_pt_4 = densities[3] * r/t_d * np.exp(-r/(t_d*v_pt_4))

### PLOTS ###


my_dpi = 96

axes_font_size = 30
title_font_size = 20
graduation_font_size = 25
legend_font_size = 25
text_font_size = 25

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


pos_txt_x = 1.025
pos_txt_y = 1

#print(n_r*feeding_rate(1,[t_d, r/n_r/v[500]]))
#print(v[500])
# 1
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True, num='Figure 3')

axes.plot(np.ones(50)*200, np.linspace(0,400), color = 'black', linestyle = '--', linewidth=linewidth, label = r'$\eta_m^*$')

axes.plot(v, eta_1, color = color_eta, label = r'$\eta(v)$', alpha = 0.25, linewidth=linewidth)
axes.plot(v, eta_2, color = color_eta, label = r'$\eta(v)$', alpha = 0.5, linewidth=linewidth)
axes.plot(v, eta_3, color = color_eta, label = r'$\eta(v)$', alpha = 0.75, linewidth=linewidth)
axes.plot(v, eta_4, color = color_eta, label = r'$\eta(v)$', alpha = 1, linewidth=linewidth)
axes.plot(v, eta_target, linestyle = ':', color = color_meta, linewidth=linewidth, label = r'$\eta(v)$')#, alpha=0.5)
#axes.scatter(v[500], n_r*feeding_rate(1,[t_d, r/n_r/v[500]]), marker = '+', s = 150, linewidths = 3, c = 'darkgoldenrod', label = r'$\eta(v)$')
axes.scatter(v_pt_1, eta_pt_1, s = 100, color = color_eta, label = r'$\eta(v)$', alpha = 0.25, zorder=10)
axes.scatter(v_pt_2, eta_pt_2, s = 100, color = color_eta, label = r'$\eta(v)$', alpha = 0.5, zorder=10)
axes.scatter(v_pt_3, eta_pt_3, s = 100, color = color_eta, label = r'$\eta(v)$', alpha = 0.75, zorder=10)
axes.scatter(v_pt_4, eta_pt_4, s = 100, color = color_eta, label = r'$\eta(v)$', alpha = 1, zorder=10)


#axes.text(v[-1]*pos_txt_x,eta_1[-1]*pos_txt_y, r'$\rho_0 = 0.5$', fontsize = text_font_size)
#axes.text(v[-1]*pos_txt_x,eta_2[-1]*pos_txt_y, r'$\rho_0 = 1.0$', fontsize = text_font_size)
#axes.text(v[-1]*pos_txt_x,eta_3[-1]*pos_txt_y, r'$\rho_0 = 1.5$', fontsize = text_font_size)
#axes.text(v[-1]*pos_txt_x,eta_4[-1]*pos_txt_y, r'$\rho_0 = 2.0$', fontsize = text_font_size)


axes.set_xlim(4,300)
axes.set_ylim(0,350)
axes.set_yticks([0, 100, 200, 300])
axes.set_xlabel(r'Speed ($v$)',fontsize=axes_font_size)
axes.set_ylabel(r'Feed. rate ($\eta$)',fontsize=axes_font_size)
#axes.set_title("Schedule", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)

axes.set_xscale('log')
plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/eta.png', dpi=my_dpi)

#plt.legend()


# 2
fig,axes = plt.subplots(1,1, figsize=(image_width/my_dpi, image_height/my_dpi), dpi=my_dpi, sharex=True)

axes.plot(np.ones(50)*200, np.linspace(0,350), color = "black", linestyle = '--', linewidth=linewidth, label = r'$\eta_m^*$')

axes.plot(v, eta_m_1, color = color_eta_m, label = r'$\eta(v)$', alpha = 0.25, linewidth=linewidth)
axes.plot(v, eta_m_2, color = color_eta_m, label = r'$\eta(v)$', alpha = 0.5, linewidth=linewidth)
axes.plot(v, eta_m_3, color = color_eta_m, label = r'$\eta(v)$', alpha = 0.75, linewidth=linewidth)
axes.plot(v, eta_m_4, color = color_eta_m, label = r'$\eta(v)$', alpha = 1, linewidth=linewidth)
axes.plot(v, np.ones(l)*eta_f, color = color_meta, linestyle = ':', linewidth=linewidth, label = r'$\eta_m^*$')#, alpha=0.5)

axes.scatter(v_pt_1, eta_m_pt_1, s = 100, color = color_eta_m, label = r'$\eta(v)$', alpha = 0.25, zorder=10)
axes.scatter(v_pt_2, eta_m_pt_2, s = 100, color = color_eta_m, label = r'$\eta(v)$', alpha = 0.5, zorder=10)
axes.scatter(v_pt_3, eta_m_pt_3, s = 100, color = color_eta_m, label = r'$\eta(v)$', alpha = 0.75, zorder=10)
axes.scatter(v_pt_4, eta_m_pt_4, s = 100, color = color_eta_m, label = r'$\eta(v)$', alpha = 1, zorder=10)


#axes.text(v[-1]*pos_txt_x,eta_m_1[-1]*pos_txt_y, r'$\rho_0 = 0.5$', fontsize = text_font_size)
#axes.text(v[-1]*pos_txt_x,eta_m_2[-1]*pos_txt_y, r'$\rho_0 = 1.0$', fontsize = text_font_size)
#axes.text(v[-1]*pos_txt_x,eta_m_3[-1]*pos_txt_y, r'$\rho_0 = 1.5$', fontsize = text_font_size)
#axes.text(v[-1]*pos_txt_x,eta_m_4[-1]*pos_txt_y, r'$\rho_0 = 2.0$', fontsize = text_font_size)

axes.set_xlim(4,300)
axes.set_ylim(0,350)
axes.set_yticks([0, 100, 200, 300])
# axes.set_xlabel(r'Speed ($v$)',fontsize=axes_font_size)
axes.set_ylabel('Marginal\nfeed. rate ($\eta_m$)',fontsize=axes_font_size)
#axes.set_title("Schedule", fontsize = title_font_size)
for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontsize(graduation_font_size)
# axes.set_xticks([])
# axes.set_xticklabels([], color="white")
axes.set_xscale('log')
axes.set_xticklabels([])

plt.subplots_adjust(top=top, bottom=bottom, right=right , left=left)
#plt.savefig('images/eta_m.png', dpi=my_dpi)

#plt.legend()
plt.show()

