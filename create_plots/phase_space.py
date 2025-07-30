import numpy as np
import matplotlib.pyplot as plt
import pickle
import xpart as xp
from matplotlib import colors
import mplhep as hep 

p0c = 6800e9
nTurn = 1000
n_particles = 15
n_lumigrid_cells = 300
bunch_intensity = 0.7825E11 #3e11 #// og = 0.7825E11 // try 1e5 

vdm = True
flat = False
local = False

if vdm == True:
    if flat == True :
        physemit_x = (4.142E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
        physemit_y = (2.071E-6*xp.PROTON_MASS_EV)/p0c  #3.75 for LHC 
    else :
        physemit_x = (2.946E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
        physemit_y = (2.946E-6*xp.PROTON_MASS_EV)/p0c  #3.75 for LHC
    beta_x = 19.17 #in meters 0.3 m for LHC, 19.17 for vdM, 0.15m for HL-LHC
    beta_y = 19.17 ## round BEAM (/4 for flat)
    beta_xl1 = 108.022378 #4746.325091 # for LHC // 108.022378 for vdM
    beta_yl1 = 84.702917 #4703.922409 # for LHC // 84.702917 for vdM
    beta_xr1 = 85.523581 #4703.922409 # for LHC // 85.523581 for vdM
    beta_yr1 = 108.033490 #4746.325090 # for LHC // 108.033490 for vdM
    fo = 'vdM_global_nobb'
else : 
    if flat == True : 
        ## FLAT beams
        physemit_x = (4.142E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
        physemit_y = (2.071E-6*xp.PROTON_MASS_EV)/(p0c)  #3.75 for LHC
    else :
        physemit_x = (3.75E-6*xp.PROTON_MASS_EV)/p0c
        physemit_y = (3.75E-6*xp.PROTON_MASS_EV)/p0c
    beta_x = 0.3 #in meters 0.3 m for LHC, 19.17 for vdM, 0.15m for HL-LHC
    beta_y = 0.3  ## round BEAM (/4 for flat)
    beta_xl1 = 4746.325091 #108.022378
    beta_yl1 = 4703.922409
    beta_xr1 = 4703.922409
    beta_yr1 = 4746.325090
    fo='LHC_global_nobb'
    
    
sigmax_star = np.sqrt(physemit_x*beta_x)
sigmay_star = np.sqrt(physemit_x*beta_y)
sigmax_l = np.sqrt(physemit_x*beta_xl1)
sigmay_l = np.sqrt(physemit_x*beta_yl1)
sigmax_r = np.sqrt(physemit_x*beta_xr1)
sigmay_r = np.sqrt(physemit_x*beta_yr1)
# Delta_x = (12*sigma/n_lumigrid_cells)# to make in um (1e2 to make in cm)
# Delta_y = (12*sigma_y/n_lumigrid_cells) # to make in um (1e2 to make in cm)

x1_1 = []; y1_1 = []; px1_1 = []; py1_1 = []; x1_2 = []; y1_2 = []; px1_2 = []; py1_2 = []
rms_1 = []; rms_2 = []
if local == True : 
    folder = ['0coup','1e-4','5e-4','1e-3'] 
    x1_3 = []; y1_3 = []; px1_3 = []
    py1_3 = []; x1_4 = []; y1_4 = []; px1_4 = []; py1_4 = []
    rms_3 = []; rms_4 = []
else : 
    folder = ['0coup','5coup','8coup','16coup'] 

for f in folder :
    file = f'./outputs/phase_space_2/{fo}/{f}.pkl'
    with open(file,"rb") as file:
        data = pickle.load(file)
        x1_1.append(data['x1_1'])
        y1_1.append(data['y1_1'])
        px1_1.append(data['px1_1'])
        py1_1.append(data['py1_1'])
        
        x1_2.append(data['x1_2'])
        y1_2.append(data['y1_2'])
        px1_2.append(data['px1_2'])
        py1_2.append(data['py1_2'])
        
        if local == True :
            x1_3.append(data['x1_3'])
            y1_3.append(data['y1_3'])
            px1_3.append(data['px1_3'])
            py1_3.append(data['py1_3'])
            
            x1_4.append(data['x1_4'])
            y1_4.append(data['y1_4'])
            px1_4.append(data['px1_4'])
            py1_4.append(data['py1_4'])
        
colours = plt.cm.viridis(np.linspace(0,1,n_particles))

if local == True :
    x1 = [x1_1, x1_2, x1_3, x1_4]
    y1 = [y1_1, y1_2, y1_3, y1_4]
    px1 = [px1_1, px1_2, px1_3, px1_4]
    py1 = [py1_1, py1_2, py1_3, py1_4]
else :
    x1 = [x1_1, x1_2]
    y1 = [y1_1, y1_2]
    px1 = [px1_1, px1_2]
    py1 = [py1_1, py1_2]

sigma_x = [sigmax_star, sigmax_star, sigmax_star, sigmax_star]
sigma_y = [sigmay_star, sigmay_star, sigmay_star, sigmay_star]


# for i in range(n_particles):
#     rms_1.append(np.std(x1_4[0][0][i])) #coup0
#     rms_2.append(np.std(x1_4[1][0][i])) #1e-4
#     if local == True :
#         rms_3.append(np.std(x1_4[2][0][i])) #5e-4
#         rms_4.append(np.std(x1_4[3][0][i])) #1e-3
# rms = [np.mean(rms_1), np.mean(rms_2)]
# if local == True : rms = [np.mean(rms_1), np.mean(rms_2), np.mean(rms_3), np.mean(rms_4)]

# print(np.array(x_1[0]).shape)
# print(np.array(x_1[1])[:,2].shape)

for j in range(len(x1)):

    fig_y,axes_y = plt.subplots(1,len(folder),figsize=(28,5))
    fig_y.suptitle("phase space in $y$, global coupling, no BB",fontsize=20) #scaled by $\sqrt{\\beta}$, $\pi/2$ phase advance, BB, 1 beam half-corrected",fontsize=17)
    plt.subplots_adjust(top=0.85)
    
    fig_x,axes_x = plt.subplots(1,len(folder),figsize=(28,5))
    fig_x.suptitle("phase space in $x$, global coupling, no BB",fontsize=20) # scaled by $\sqrt{\\beta}$, $\pi/2$ phase advance, BB, 1 beam half-corrected",fontsize=17)
    plt.subplots_adjust(top=0.85)
    
    fig_xy,axes_xy = plt.subplots(1,len(folder),figsize=(28,5))
    fig_xy.suptitle("$x$-$y$ coordinates, global coupling, no BB",fontsize=20) # scaled by $\sqrt{\\beta}$, $\pi/2$ phase advance, BB, 1 beam half-corrected",fontsize=17)
    plt.subplots_adjust(top=0.85)
    
    for i in range(len(folder)):
        for p in range(n_particles):
            axes_x[i].plot(np.array(x1[j][i])[:,p]/sigma_x[j],np.array(px1[j][i])[:,p]/sigma_x[j], '.', color=colours[p],markersize=1.5)
            axes_x[i].grid(True)
            axes_x[i].set_xlabel('$x [\sigma]$')
            axes_x[0].set_ylabel('$p_x [\sigma]$')
            # if j == 1 or j == 2 :  
            #     axes_x[i].set_xlim(-50,50)
            #     axes_x[i].set_ylim(-30,30)
                #axes_x[i].set_ylim(-1e-5,1e-5)
            #else : #axes_x[i].set_xlim(-4,4)
            #axes_x[i].set_ylim(-2e-3,2e-3)
            axes_x[i].set_title(f'{folder[i]}')
            
            axes_y[i].plot(np.array(y1[j][i])[:,p]/sigma_y[j],np.array(py1[j][i])[:,p]/sigma_y[j], '.', color=colours[p],markersize=1.5)
            axes_y[i].grid(True)
            axes_y[i].set_xlabel('$y [\sigma]$')
            axes_y[0].set_ylabel('$p_y [\sigma]$')
            #axes_y[i].set_xlim(-4,4)
            # if j == 1 or j == 2 : 
            #     axes_y[i].set_xlim(-50,50)
            #     axes_y[i].set_ylim(-30,30)
            #     #axes_y[i].set_ylim(-1e-5,1e-5)
            #     axes_y[3].set_xlim(-15,15)
                #axes_y[3].set_ylim(-2.5e-4,2.5e-4)
            #else : 
            #axes_y[i].set_ylim(-1e-5,1e-5)
            axes_y[i].set_title(f'{folder[i]}')
            
            axes_xy[i].plot(np.array(x1[j][i])[:,p]/sigma_x[j],np.array(y1[j][i])[:,p]/sigma_y[j], '.', color=colours[p],markersize=1.5)
            axes_xy[i].grid(True)
            axes_xy[i].set_xlabel('$x [\sigma]$')
            axes_xy[0].set_ylabel('$y [\sigma]$')
            # if j == 1 or j == 2 : 
            #     axes_xy[i].set_xlim(-50,50)
            #     axes_xy[i].set_ylim(-50,50)
            # else :
            #     axes_xy[i].set_xlim(-4,4)
            #     axes_xy[i].set_ylim(-4,4)
            axes_xy[i].set_title(f'{folder[i]}')
            
    fig_x.savefig(f'./lumiGrid_local/phase_space_final/{fo}/M{j+1}_x_px.png')
    fig_y.savefig(f'./lumiGrid_local/phase_space_final/{fo}/M{j+1}_y_py.png')
    fig_xy.savefig(f'./lumiGrid_local/phase_space_final/{fo}/M{j+1}_x_y.png')

# ar = np.array(np.squeeze(np.array(x1[1])[1]))
# print(ar.shape)
# rms1 = []; rms2 = []; rms3 = []; rms4 = []
# for i,f in enumerate(folder) :
#     rms1.append((np.std(np.squeeze(np.array(x1[i])[0])/sigma_x[0],axis=0)))
#     rms2.append((np.std(np.squeeze(np.array(x1[i])[1])/sigma_x[1],axis=0)))
#     rms3.append((np.std(np.squeeze(np.array(x1[i])[2])/sigma_x[2],axis=0)))
#     rms4.append((np.std(np.squeeze(np.array(x1[i])[3])/sigma_x[3],axis=0)))

# m2 = [100., 111.19447194, 248.49604546, 459.72799949]
# m1 = [100.,100.,99.99999998,99.99999996]
# m3 = [100.,111.19447194, 248.49604546, 459.72799949]
# m4 = [100.,100.,99.99999998, 99.99999996]
    
# plt.figure(0)
# # plt.plot(folder, m1, '+-',label='Monitor 1')
# plt.plot(folder, m2, '+-',label='Monitor 2')
# # plt.plot(folder, m3, '+-',label='Monitor 3')
# # plt.plot(folder, m4, '+-',label='Monitor 4')
# plt.xlabel('Skew quadrupole strength')
# plt.ylabel('sigma/sigma0 [%]')
# plt.legend()
# plt.savefig(f'./lumiGrid_local/{fol}/std_x.png')

# everything for beam 2
# fig_x2,axes_x2 = plt.subplots(1,len(folder),figsize=(16,5))
# fig_x2.suptitle("phase space in $x$ of particles over turns for beam 2")

# fig_y2,axes_y2 = plt.subplots(1,len(folder),figsize=(16,5))
# fig_y2.suptitle("phase space in $y$ of particles over turns for beam 2")

# fig_xy2,axes_xy2 = plt.subplots(1,len(folder),figsize=(16,5))
# fig_xy2.suptitle("$x$-$y$ coordinates of particles over turns for beam 2")

# for i in range(len(folder)):
#     for p in range(n_particles):
#         axes_x2[i].plot(np.array(x_2[i])[:,p],np.array(px_2[i])[:,p], '.', color=colours[p],markersize=1.5)
#         axes_x2[i].grid(True)
#         axes_x2[i].set_xlabel('$x [\sigma]$')
#         axes_x2[0].set_ylabel('$p_x$')
#         axes_x2[i].set_title(f'{folder[i]}')
        
#         axes_y2[i].plot(np.array(y_2[i])[:,p],np.array(py_2[i])[:,p], '.', color=colours[p],markersize=1.5)
#         axes_y2[i].grid(True)
#         axes_y2[i].set_xlabel('$y [\sigma]$')
#         axes_y2[0].set_ylabel('$p_y$')
#         axes_y2[i].set_title(f'{folder[i]}')
        
#         axes_xy2[i].plot(np.array(x_2[i])[:,p],np.array(y_2[i])[:,p], '.', color=colours[p],markersize=1.5)
#         axes_xy2[i].grid(True)
#         axes_xy2[i].set_xlabel('$x [\sigma]$')
#         axes_xy2[0].set_ylabel('$y [\sigma]$')
#         #axes_xy2[i].set_ylim(-0.0003,0.0003)
#         axes_xy2[i].set_title(f'{folder[i]}')
# fig_x2.savefig(f'./lumiGrid_local/tests/phase_space/{fol}/x_px2.png')
# fig_y2.savefig(f'./lumiGrid_local/tests/phase_space/{fol}/y_py2.png')
# fig_xy2.savefig(f'./lumiGrid_local/tests/phase_space/{fol}/x_y2.png')
