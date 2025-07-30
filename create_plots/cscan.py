import numpy as np
import pickle
import matplotlib.pyplot as plt
import xpart as xp
import matplotlib.ticker as ticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams['axes.linewidth'] = 0.7
plt.rcParams.update({'font.size': 10})
cmap = plt.get_cmap('Set2')

p0c = 6800e9
bunch_intensity = 0.7825E11

vdm = True

if vdm == True :
    physemit_x = (2.946E-6 * xp.PROTON_MASS_EV) / p0c
    physemit_y = (2.946E-6 * xp.PROTON_MASS_EV) / (p0c)
    beta_x = 19.17
    beta_y = 19.17 
    # beta_x = 108.022378 #4746.325091 # for LHC // 108.022378 for vdM
    # beta_y = 84.702917 
else : 
    physemit_x = (3.75E-6 * xp.PROTON_MASS_EV) / p0c
    physemit_y = (3.75E-6 * xp.PROTON_MASS_EV) / (p0c)
    beta_x = 0.3
    beta_y = 0.3
    
sig_x=np.sqrt(physemit_x*beta_x)
# print(sig_x)
sig_y=np.sqrt(physemit_y*beta_y)
# print(sig_y)
# n_lumigrid_cells = 1200
# rang = 12
# sigma_range = np.arange(-rang/2, rang/2, rang/n_lumigrid_cells)
# Delta_x = (12*sig_x/n_lumigrid_cells)*1e4 # to make in um (1e2 to make in cm)
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 62.31
Qy = 60.32
Qs = 2.1E-3
frev = 11245.5 
nTurn = 1000 #700 #1000
phase_adv = 0.25
cells = 300

foll = ['0.0']#['0.0','0.0001','0.0002','0.0003','0.0004','0.0005','0.0006','0.0007','0.0008','0.0009','0.001']#np.insert(np.linspace(1e-5, 1e-3, 20), 0, 0.0) 
skew = [0,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3]
# separations = [0]
separations = np.arange(0,5.5,0.5)
#print(folder.shape)
# n_part = [1e5,1e6,1e7]
# n_cells = [100,300,600,900,1200]
fol = 'vdM_flat'
# fol = 'lhc_sigma_few'
sep = 0 #3

## ANALYTICAL LUMI ###################################
# def Lumi_analytical(Nb, N1, N2, frev, Delta_i, sig_i, sig_x, sig_y):
#     W = np.exp(-Delta_i**2/(4*sig_i**2))
#     return ((Nb * N1 * N2 * frev * W)/(4 * np.pi * (sig_x) * (sig_y))) # x100 to make sure that we are in cm^-2
# lumi_ana = (Lumi_analytical(1, bunch_intensity, bunch_intensity, frev, sep*np.sqrt(physemit_x*beta_x),np.sqrt(physemit_x*beta_x),sig_x, sig_y))
# # lumi_ana = np.ones([len(n_cells),len(n_part)])*lumi_ana
# print('analytical = '+ str(lumi_ana))

def Lumi_formula(Nb, N1, N2, frev, Delta_i, sig_i, sig_x, sig_y):
    W = np.exp(-Delta_i**2/(4*sig_i**2))
    return ((Nb * N1 * N2 * frev)/(2 * np.pi * np.sqrt(2*(sig_x**2)) * np.sqrt(2*(sig_y**2)))) # x100 to make sure that we are in cm^-2

# lumin = np.zeros([len(n_cells),len(n_part)])
# lumin_grid = np.zeros([len(n_cells),len(n_part)])
# table = [['' for _ in range(len(n_part))] for _ in range(len(n_cells))]
# print(lumin.shape)
# # lumis=[7.04891722e+32,7.06018073e+32,7.06548080e+32,7.05056440e+32,7.06920604e+32,7.05619588e+32,7.05922926e+32,7.07713864e+32,7.05084051e+32,7.06654766e+32,7.07204039e+32,7.04541910e+32,
# #        7.07728286e+32,7.06609775e+32,7.04185604e+32,7.07227015e+32,7.05916230e+32,7.05794702e+32,7.06678191e+32,7.04306167e+32,
# #        7.06396342e+32,7.06664818e+32,7.04475371e+32,7.06930767e+32,7.05616407e+32,7.04599579e+32,7.05800587e+32,7.05006913e+32,
# #        7.06161180e+32,7.05763103e+32,7.05159605e+32,7.06366213e+32,7.06016836e+32,7.05081658e+32,7.05822260e+32,7.06088549e+32,
# #        7.04861597e+32,7.05670027e+32,7.06803918e+32,7.04921641e+32,7.05234903e+32,7.06713747e+32,7.05556568e+32,7.05635172e+32,
# #        7.05465446e+32,7.04988303e+32,7.07013909e+32,7.04546754e+32,7.06175963e+32,7.07975121e+32,7.04891722e+32,7.06018073e+32,
# #        7.06548080e+32,7.05056440e+32,7.06920604e+32,7.05619588e+32,7.05922926e+32,7.07713864e+32,7.05084051e+32,7.06654766e+32,
# #        7.07204039e+32,7.04541910e+32,7.07728286e+32,7.06609775e+32,7.04185604e+32,7.07227015e+32,7.05916230e+32,7.05794702e+32,
# #        7.06678191e+32,7.04306167e+32,7.06396342e+32,7.06664818e+32,7.04475371e+32,7.06930767e+32,7.05616407e+32,7.04599579e+32,
# #        7.05800587e+32,7.05006913e+32,7.06161180e+32,7.05763103e+32,7.05159605e+32,7.06366213e+32,7.06016836e+32,7.05081658e+32,
# #        7.05822260e+32,7.06088549e+32,7.04861597e+32,7.05670027e+32,7.06803918e+32,7.04921641e+32,7.05234903e+32,7.06713747e+32,
# #        7.05556568e+32,7.05635172e+32,7.05465446e+32,7.04988303e+32,7.07013909e+32,7.04546754e+32,7.06175963e+32,7.07975121e+32]

# for i,p in enumerate(n_part) :
#     p = round(p)
#     for j,c in enumerate(n_cells) :
#         on_x = []
#         file = f'./outputs/IP1/{fol}/{p}p_{c}c.pkl'
#         with open(file,"rb") as file:
#             data = pickle.load(file)
#             #print(data['int_luminosity_values'])
#             lumin[j][i]=(np.mean(data['int_luminosity_values']))
#             table[j][i]=(str("{:.1e}".format(n_part[i]))+', '+str(n_cells[j]))
            # lumi = data['int_luminosity_values']
            # for k in range(nTurn):
            #     #step = int(len(lumi)/nTurn)
            #     lumis.append(np.mean(lumi[:k]))
        #     on_x_grid_b1 = data['lumigrid_b1']
        #     on_x_grid_b2 = data['lumigrid_b2']
        #     on_x_grid_b1 = on_x_grid_b1.reshape(c, c)
        #     on_x_grid_b2 = on_x_grid_b2.reshape(c, c)
        #     on_x_grid_b1 = on_x_grid_b1.T
        #     on_x_grid_b2 = np.flip(on_x_grid_b2.T)
        #     on_x_grid_multiplied = on_x_grid_b1 * on_x_grid_b2      # luminous region
        #     on_x.append(on_x_grid_multiplied)
        # lumin_grid[j][i] = np.mean(on_x)

# print(lumin)
# print(lumin_grid)

# print('table : ')
# print(np.array(table))
# lumin = np.abs((lumin-lumi_ana)/lumi_ana) *100

# NO SEPARATION ###################################

# colours = plt.cm.viridis(np.linspace(0,1,len(skew)))

# round, no bb : lhc_sigma_loc
# with bb : lhc_sigma_bb
# flat : lhc_flat

# fol = 'vdm_sigma_bb'

# # lumi_noc = []
# lumi_c = []
# # sigma_noc = []
# sigma_c = []
# # # for fol in folder :
# # # tun = np.linspace(0,99,100)
# # for s,sep in enumerate(separations) :
#     # print(s)
# file_0 = f'./outputs/IP1/{fol}/0.0.pkl'
# lum0 = []
# with open(file_0,"rb") as file:
#     data_0 = pickle.load(file)
#     # print(data_0['int_luminosity_values'])
#     # lum0.append(np.mean(data_0['int_luminosity_values']))
#     lumi_noc.append(data_0['int_luminosity_values'])
#     sigma_noc.append(data_0['rms_x_1'])

# for i,f in enumerate(foll) :
#     # f = round(f,5)
#     # f = str(f)
#     file = f'./outputs/IP1/{fol}/{f}.pkl'
#     # lum = []
#     with open(file,"rb") as file:
#         data = pickle.load(file)
#         # print(sep,f)
#         # lum.append(np.mean(data['int_luminosity_values']))
#         lumi_c.append(data['int_luminosity_values'])
#         sigma_c.append(data['rms_x_1'])

# print(len(lumi_c[0]))

# print(lumi0)

# print((lumi_ana/lum0-1) * 100)
# print(np.mean(luminosity))

# print(tun)
# print(luminosity[0])

# print(sig)

# plt.figure()
# plt.plot(tun, np.array(sig)[0])
# plt.show()
    
# SEPARATION ###################################
l = len(foll)

sigma_0x = np.zeros((6, 1))
sigma_0y = np.zeros((6, 1))
sigma_x = np.zeros((6, l))
sigma_y = np.zeros((6, l))
sigma_x_2 = np.zeros((6, l))
sigma_y_2 = np.zeros((6, l))
lumi=[]#np.zeros((11, l))
lumi_2=np.zeros((6, l))
lumi0=np.zeros((6, 1))
# # fol = ['lhc_sigma_few','vdm_sigma_few']
# # fol_ = ['lhc_sigma','vdm_sigma']

colours = plt.cm.viridis(np.linspace(0,1,l))

fol = 'vdm_half'
fol2 = 'LHC_flat'
luminosity = []

# for fol in folder :
for s,sep in enumerate(separations) :
    # file_0 = f'./outputs/IP1/{fol}/0.0.pkl'
    # lum0 = []
    # with open(file_0,"rb") as file:
    #     data_0 = pickle.load(file)
    #     # rms_0x = np.mean(data_0['rms_x_1'])
    #     # rms_0y = np.mean(data_0['rms_y_1'])
    #     lum0.append(data_0['int_luminosity_values'])
    #     luminosity.append(data_0['int_luminosity_values'])

    # # sigma_0x[s][0] = np.mean(rms_0x)
    # # sigma_0y[s][0] = np.mean(rms_0y)
    # lumi0[s][0] = np.mean(lum0)
    # # sigma_0x.append(rms_0x)
    # # sigma_0y.append(rms_0y)
    # # lumi0.append(lum0)
    

    for i,f in enumerate(foll) :
        #f = round(f,5)
        #f = str(f)
        file = f'./outputs/IP1/{fol}/{f}_{sep}_nbb.pkl'
        lum = []
        on_x = []
        with open(file,"rb") as file:
            data = pickle.load(file)
            # rms_x = np.mean(data['rms_x_1'])
            # rms_y = np.mean(data['rms_y_1'])
            lum.append(data['int_luminosity_values']/1e4)
            # on_x_grid_b1 = data['lumigrid_b1']
            # on_x_grid_b2 = data['lumigrid_b2']
            # on_x_grid_b1 = on_x_grid_b1.reshape(cells, cells)
            # on_x_grid_b2 = on_x_grid_b2.reshape(cells, cells)
            # on_x_grid_b1 = on_x_grid_b1.T
            # on_x_grid_b2 = np.flip(on_x_grid_b2.T)
            # on_x_grid_multiplied = on_x_grid_b1 * on_x_grid_b2      # luminous region
            # on_x.append(on_x_grid_multiplied)
        # sigma_x[s][i] = rms_x
        # sigma_y[s][i] = rms_y
        # lumi[s][i] = np.mean(lum)
        # lumi_2[s][i] = np.mean(on_x)
        # sigma_x.append(rms_x)
        # sigma_y.append(rms_y)
        lumi.append(np.mean(lum))
    # for i, f in enumerate(foll):
    #     file = f'./outputs/IP1/{fol2}/{f}_bb.pkl'
    #     lum_2 = []
    #     with open(file, "rb") as file:
    #         data_2 = pickle.load(file)
    #         # rms_x_2 = np.mean(data_2['rms_x_1'])
    #         # rms_y_2 = np.mean(data_2['rms_y_1'])
    #         lum_2.append(data_2['int_luminosity_values']/1e4)
 
    #     # sigma_x_2[s][i] = rms_x_2
    #     # sigma_y_2[s][i] = rms_y_2
    #     lumi_2[s][i] = np.mean(lum_2)
        
        
# print('numerical : ')
# print(lumi)
# print('grid : ')
# print(lumi_2)

# sigma_lhc_norm = ((sigma_x[:11]-sigma_0x[0])/sigma_0x[0]) *100
# sigma_vdm = sigma_x[0]
# lumi_bb = lumi[0]
# sigma_vdm_norm = ((sigma_x[0]-sigma_0x[0])/sigma_0x[0]) *100
# sigma_nbb = sigma_x_2[0]
# lumi_nbb = lumi_2[0]
# lumi_lhc_norm = ((lumi[:11]-lumi0[0])/lumi0[0]) *100
# lumi_vdm_norm = ((lumi[-11:]-lumi0[1])/lumi0[1]) *100
# lumi_bb = 100*(lumi[0]-lumi0[0])/lumi0[0]
# lumi_nobb = 100*(lumi_2[0]-lumi0[0])/lumi0[0]
# lumi_vdm = lumi[0]
# lumi_lhc = lumi_2[0]

# lumi_form=[]
# for i in range(len(sigma_vdm)) :
#     lumi_form.append((Lumi_formula(1, bunch_intensity, bunch_intensity, frev, 0,np.sqrt(physemit_x*beta_x),sigma_vdm[i], sigma_vdm[i]))/1e4)

# turns = np.linspace(0,99,100)
# print(turns)

plt.figure(0)
for i,f in enumerate(foll) :
    plt.plot(separations, lumi, '-',label=str(f)+' coupling', color = colours[i])
plt.grid()
plt.ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')
plt.xlabel('Beam separation [$\sigma$]')
plt.title('Luminosity as a function of beam separation')
plt.savefig(f'./lumiGrid_local/lumi/half_lumi.png')
        
# plt.figure(0)
# for i,f in enumerate(foll) :
#     plt.plot(turns, (sigma_c[i] - np.mean(sigma_c[i])), '-',label='coup  = '+ str(f), color = colours[i])
# # plt.plot(skew, sigma_lhc, '+-',label='sigma LHC, no BB, round beam')
# # plt.plot(skew, sigma_vdm, '+-',label='sigma LHC, BB, round beam')
# # plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
# #plt.ylim(-10, 10)
# plt.xlabel('Number of turns')
# plt.ylabel('$\sigma - \sigma_\mathrm{avg}$')
# plt.title('Beam size for different values of coupling \n as a function of number of turns') # as a function of beam separation \n for different skew quadrupole strengths')
# plt.legend()
# plt.savefig(f'./lumiGrid_local/ksl_scan/sigma_vdm_turns.png')  # Save as PNG with high resolution
# plt.clf()

# plt.figure(0)
# # for i,f in enumerate(foll) :
# #     plt.plot(separations, (lumi[:,i]/lumi_2[:,i]-1)*100, '+-',label='coup  = '+ str(f), color = colours[i])
# plt.plot(skew, lumi_vdm, '+-',label='Numerical \n luminosity', color=cmap(0),linewidth=2)
# plt.plot(skew, lumi_form, '+-',label='Luminosity calculated \n from beam size', color=cmap(1),linewidth=2)
# # plt.plot(skew, lumi_form, '+-',label='luminosity calculated using beam size')
# plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
# plt.xlabel('Skew quadrupole strength',fontsize=10)
# # plt.xlabel('Nominal Beam Separation [$\sigma$]')
# # plt.ylabel('$(L_{bb+c}-L_{0})/L_{0}$ [%]',fontsize=14)
# plt.ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')
# plt.title('Luminosity as a function of different linear coupling strengths',fontsize=13,pad=-5)
# plt.legend()
# plt.grid()
# plt.savefig(f'./lumiGrid_local/ksl_scan2/lumi_form_lhc_nobb.png')  # Save as PNG with high resolution
# plt.clf()

# plt.figure(1)
# # for i,f in enumerate(foll) :
# #     plt.plot(separations, (lumi[:,i]/lumii_2[:,i]-1)*100, '+-',label='coup  = '+ str(f), color = colours[i])
# plt.plot(skew, lumi_bb, '+-',label='BB', color=cmap(0))
# plt.plot(skew, lumi_nobb, '+-',label='no BB', color=cmap(1))
# # plt.plot(skew, lumi_form, '+-',label='luminosity calculated using beam size')
# plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
# plt.xlabel('Skew quadrupole strength')
# # plt.xlabel('Nominal Beam Separation [$\sigma$]')
# # plt.ylabel('(Lbb+c-Lc)/Lc [%]')
# plt.ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')
# plt.title('vdM Luminosity for different skew quadrupole strengths')
# plt.legend()
# plt.grid()
# plt.savefig(f'./lumiGrid_local/ksl_scan2/luminosity.png')  # Save as PNG with high resolution
# plt.clf()

# fig, (ax1,ax2) = plt.subplots(2,1,height_ratios=[5, 3])
# # ax1.plot(skew, lumi_lhc_norm, '+-',label='vdM, no BB, round beam')
# ax1.plot(skew, lumi_vdm, '+-',label='Numerical \n luminosity', color=cmap(0))
# ax1.plot(skew, lumi_form, '+-',label='Luminosity calculated \n from beam size', color=cmap(1))
# ax2.plot(skew, (np.divide(lumi_vdm,lumi_form)-1)*100, '+-', color=cmap(2))
# ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
# ax2.set_xlabel('Skew quadrupole strength')
# # ax1.set_ylabel('$( \sigma - \sigma0 ) / \sigma0 $ [%]')
# # ax2.set_ylabel('$\sigma / \sigma0 $')
# ax1.set_ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')   
# ax2.set_ylabel('$L_\\text{num}/L_\\text{form} - 1$ [%]')
# # plt.ylabel('L')
# fig.tight_layout(pad=1.0)
# fig.subplots_adjust(top=0.9)
# fig.suptitle('Luminosity for different skew quadrupole strengths')
# ax1.legend()
# ax1.grid(True)
# ax2.grid(True)
# fig.savefig(f'./lumiGrid_local/ksl_scan2/lumi_form_lhc_nobb_ratio.png')  # Save as PNG with high resolution
# plt.clf()

# plt.figure(2)
# ratio = []
# for i,f in enumerate(separations) :
#     ratio.append(np.mean(sigma_x[i]/sigma_x[0]))
# plt.plot(separations, ratio, '+-')
# # plt.plot(skew, lhc, '+-',label='vdM, BB, round beam')
# # plt.plot(skew, vdm, '+-',label='vdM, BB, flat beam')
# plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
# # plt.xlabel('Skew quadrupole strength')
# plt.xlabel('Nominal Beam Separation [$\sigma$]')
# plt.ylabel('sigma/sigma0 [%]')
# plt.title('Ratio of beam widths as a function of beam separation \n for different skew quadrupole strengths')
# # plt.legend()
# plt.savefig(f'./lumiGrid_local/ksl_scan/sigma_ratio.png')  # Save as PNG with high resolution
# plt.clf()


## 2D convergence 

# n_par = ['1e5','1e6','1e7']

# plt.figure('2D convergence')
# plt.imshow(lumin,cmap='GnBu',origin='lower',interpolation='none',vmin=0.03,vmax=0.5)
# plt.xticks(np.arange(len(n_par)),(n_par))
# plt.yticks(np.arange(len(n_cells)),n_cells)
# plt.colorbar()
# for j,c in enumerate(n_cells):
#     for i,p in enumerate(n_part):
#         plt.text(i, j, round(lumin[j][i],3), ha='center', va='center',  color='black')  # Contrast text color
# #ax = plt.gca()
# #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
# #plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
# plt.xlabel('Number of macroparticles')
# plt.ylabel('Number of cells')
# plt.title('Convergence to the analytical luminosity [%]', pad=20) #\n for 3 $\sigma$ beam seaparation')
# plt.savefig(f'./lumiGrid_local/ksl_scan/convergence/vdm_conv_2.png')  # Save as PNG with high resolution