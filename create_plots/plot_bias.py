import lumigrid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams['axes.linewidth'] = 1
plt.rcParams.update({'font.size': 18})
cmap = plt.get_cmap('Set2')

# coup = '0.0'
# skew = '0'
couplings = ['0.0', '0.0001','0.0005','0.001'] #['0coup','5coup','8coup','16coup'] #['0.0', '0.0001','0.0005','0.001']
skew = ['0','1e-4','5e-4','1e-3'] #['0','5','8','16']#['0','1e-4','5e-4','1e-3']
#coup,folder,Delta_x,ratio,b1, b2,on_x,off_x,on_x_mult, off_x_mult, rms_x_1, rms_x_2, rms_y_1, rms_y_2, rms_x_1_, rms_x_2_, rms_y_1_, rms_y_2_, on_pos, off_pos = lumigrid.bias()
#coup,folder,directory,Delta_x,ratio,b1, b2,on_x,off_x,on_x_mult, off_x_mult, on_pos, off_pos,rms_x_1, rms_x_2, rms_y_1, rms_y_2, rms_x_1_, rms_x_2_, rms_y_1_, rms_y_2_, rms_x, rms_y = lumigrid.bias()


# lumi_x_ratio[lumi_x_ratio == np.inf] = np.nan
# lumi_x_ratio[lumi_x_ratio == -np.inf] = np.nan
# lumi_x_ratio[lumi_x_ratio > 100] = np.nan
# lumi_x_ratio[lumi_x_ratio < -100] = np.nan


separation =  np.arange(0,5.5,0.5)

plt.figure(0)
plt.figure(1)
plt.figure(2)
# ax_inset = inset_axes(plt.gca(), width="40%", height="40%", loc='upper right')
l_0_bb = []
l_0_nbb = []


for i,coup in enumerate(couplings):
    on_x, on_x_mult,off_x, off_x_mult,folder, directory, sigma, lumi, lumi0 = lumigrid.bias(coup)
    lumi_x_ratio = ((on_x_mult-off_x_mult)/off_x_mult)*100
    lumi_ratio = np.divide(lumi,lumi0)*100
    if i == 0 : 
        l_0_bb = lumi
        l_0_nbb = lumi0
    
## PLOT BIAS
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Setup subplots
    # fig, axs = plt.subplots()

    # axs.set_position([0.1, 0.2, 0.8, 1.6])
    # #titles = ['Beam 1', 'Beam 2', 'Luminous Region']
    # #ratios = [beam_1_ratio, beam_2_ratio, lumi_x_ratio]
    # titles = ['Luminous Region']
    # ratios = [lumi_x_ratio]

    # for sepX in separation:

    #     # Loop to create each subplot
    #     origin_x, origin_y = 150, 150
    #     change = 12/300
    #     #extent = [-origin_x*change, (ratios[0].shape[1] - origin_x)*change, -origin_y*change, (ratios[0].shape[0] - origin_y)*change]
    #     im = axs.imshow(ratios[0], extent = (-6,6,-6,6), cmap='bwr',interpolation='none',vmin=-5, vmax=5)
    #     #im = axs.imshow(ratios[0], extent = (-6,6,-6,6), cmap='bwr',interpolation='nearest',vmin=-10, vmax=10)  # Control the color range for better visual distinction
    #     axs.set_title(f'{titles[0]} bias for local {skew[i]} coupling, flat beam',fontsize=20,pad=20)
    #     # axs.set_title(f'beam-beam+coupling bias for {skew[i]} coupling',fontsize=20,pad=20)
    #     # 200 turns, 2.1 phase adv
    #     axs.set_xlabel('X ($\sigma_x$)', fontsize=18)
    #     axs.set_ylabel('Y ($\sigma_y$)', fontsize=18)
    #     #axs.set_box_aspect(0.8)
    # plt.subplots_adjust(top=0.85)
    # cbar = plt.colorbar(im, ax=axs)  # Add a colorbar to each subplot
    # cbar.set_label('Bias (%)', fontsize=18)
    # #plt.suptitle(f'Bias from beam-beam and {coup} coupling ',fontsize=16)
    # #fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle
    # fig.savefig(f'{directory}/bias/flat_{skew[i]}.png')  # Save as PNG with high resolution
    # plt.clf()

## PLOT LUMI RATIO
    plt.figure(0)
    plt.plot(separation, lumi_ratio, '-',color=cmap(i),linewidth=2,label=skew[i]+' coupling')

    plt.figure(1)
    plt.plot(separation, np.divide(lumi,1e4), '-',color=cmap(i),linewidth=2,label=skew[i]+' coupling')
    
    plt.figure(2)
    plt.plot(separation, np.divide(lumi,l_0_bb)*100, '-',color=cmap(i),linewidth=2,label=skew[i]+' coupling')
    
    
    
# plt.figure(0)
# plt.xlabel('Nominal beam separation ($\sigma$)')
# plt.ylabel('$L_{c} / L_{0} $ (%)')
# plt.grid()
# plt.legend()
# plt.title(f'Luminosity coupling bias for local coupling', pad=20)
# plt.savefig(f'{directory}/lumi/vdm_coup_ratio.png')
# plt.clf()

# plt.figure(1)
# plt.xlabel('Nominal beam separation ($\sigma$)')
# plt.grid()
# plt.legend(loc='lower left')
# plt.ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')
# # plt.ylabel('$L_\mathrm{c+bb} / L_\mathrm{bb} $ (%)')
# plt.title(f'Luminosity for local coupling, BB', pad=20)
# ax_inset = inset_axes(plt.gca(), width="40%", height="40%",  loc='upper right')
# for i, coup in enumerate(couplings):
#     on_x, on_x_mult,off_x, off_x_mult,folder, directory, sigma, lumi, lumi0 = lumigrid.bias(coup)
#     ax_inset.plot(separation, np.divide(lumi,1e4), '-', color=cmap(i)) #, linewidth=2, label=skew[i]+' coupling')
# ax_inset.set_xlim(-0.02, 0.3)
# ax_inset.set_ylim(6.7e28, 7.12e28)  # Adjust the y-axis limits as needed
# ax_inset.grid()
# # ax_inset.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# mark_inset(plt.gca(), ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.savefig(f'{directory}/lumi/vdm.png')
# plt.clf()

plt.figure(2)
plt.xlabel('Nominal beam separation ($\sigma$)')
plt.grid()
plt.legend(loc='upper left')
# plt.ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')
plt.ylabel('$L_\mathrm{c+bb} / L_\mathrm{bb} $ (%)')
plt.title(f'Luminosity ratio for local coupling, BB', pad=20)
# ax_inset = inset_axes(plt.gca(), width="40%", height="40%",  loc='upper right')
# for i, coup in enumerate(couplings):
#     on_x, on_x_mult,off_x, off_x_mult,folder, directory, sigma, lumi, lumi0 = lumigrid.bias(coup)
#     ax_inset.plot(separation, np.divide(lumi,1e4), '-', color=cmap(i)) #, linewidth=2, label=skew[i]+' coupling')
# ax_inset.set_xlim(-0.01, 0.25)
# ax_inset.set_ylim(6.8e28, 7.1e28)  # Adjust the y-axis limits as needed
# ax_inset.grid()
# ax_inset.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# mark_inset(plt.gca(), ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
plt.savefig(f'{directory}/lumi/vdm_ratio_bb.png')
plt.clf()



