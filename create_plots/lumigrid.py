import numpy as np
import matplotlib.pyplot as plt
import pickle
import xpart as xp
from matplotlib import colors
import mplhep as hep 
hep.style.use("ROOT")
plt.rcParams["figure.figsize"] = [10, 8]
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns



p0c = 6800e9
bunch_intensity = 0.7825E11
flat = True
if flat == True :
        physemit_x = (4.142E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
        physemit_y = (2.071E-6*xp.PROTON_MASS_EV)/p0c  #3.75 for LHC 
else :
    physemit_x = (2.946E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
    physemit_y = (2.946E-6*xp.PROTON_MASS_EV)/p0c  #3.75 for LHC
beta_x = 19.17
beta_y = 19.17 
n_lumigrid_cells = 300  #binsize
sigma=np.sqrt(physemit_x*beta_x)
#print(sigma)
sigma_y = np.sqrt(physemit_y*beta_y)
#print(sigma_y)
rang = 12
sigma_range = np.arange(-rang/2, rang/2, rang/n_lumigrid_cells)
Delta_x = (12*sigma/n_lumigrid_cells)*1e4 # to make in um (1e2 to make in cm)
#print(Delta_x)

#linspace_array = np.linspace(1e-4, 1e-3, 10)
#separation = np.concatenate(([0], linspace_array))
separation = [0.0]#np.arange(0,5.5,0.5)
#print(separation)
#sepX = '5.0'
directory='./lumiGrid_local/'
# coup2 = '0.0'
folder='vdm_sepscan_full'
# rms_x_1 = []; rms_x_2 = []; rms_y_1 = []; rms_y_2 = []
# rms_x_1_ = []; rms_x_2_ = []; rms_y_1_ = []; rms_y_2_ = []
# x_1 = []; x_1_ = []; y_1 = []; y_1_ = []; z_1 = []; z_1_ = []
# x_2 = []; x_2_ = []; y_2 = []; y_2_ = []; z_2 = []; z_2_ = []
# px_1 = []; px_1_ = []; py_1 = []; py_1_ = []
# px_2 = [] py_2 = []
# cx_1 = []; cx_2 = []; cx_1_ = []; cx_2_ = []
# cy_1 = []; cy_2 = []; cy_1_ = []; cy_2_ = []
# rms_x = []; rms_y = []
# x1_noc = []; y1_noc = []; px1_noc = []; py1_noc = []
# x1_coup2 = []; y1_coup2 = []; px1_coup2 = []; py1_coup2 = []
# x1_coup3 = []; y1_coup3 = []; px1_coup3 = []; py1_coup3 = []
# x1_coup4 = []; y1_coup4 = []; px1_coup4 = []; py1_coup4 = []


def bias(coup) :
    
    on_x_mult = np.zeros([n_lumigrid_cells,n_lumigrid_cells])
    on_x = []
    off_x_mult = np.zeros([n_lumigrid_cells,n_lumigrid_cells])
    off_x = []
    on_y_mult = np.zeros([n_lumigrid_cells,n_lumigrid_cells])
    on_y = []
    off_y_mult = np.zeros([n_lumigrid_cells,n_lumigrid_cells])
    off_y = []
    ratio = []
    b1 = []
    b2 = []
    lumi = [[] for _ in range(len(separation))]
    lumi0 = [[] for _ in range(len(separation))]

    for i,shift in enumerate(separation):
        shift = round(shift,4)
        shift_str = str(shift)
        # i = separation.tolist().index(shift)
        
        # Paths setup
        # on_xshift_file_name 
        # nocoup = f'./outputs/IP1/{folder}/{coup}_{shift}.pkl' #2.1
        on_xshift_file_name = f'./outputs/IP1/{folder}/{coup}_{shift}_bb.pkl'
        off_xshift_file_name = f'./outputs/IP1/{folder}/0.0_{shift}_nbb.pkl'
        # # off_xshift_file_name
        # coup2 = f'./outputs/IP1/{folder2}/{c2}_{shift}.pkl'
        # coup3 = f'./outputs/IP1/{folder2}/{c3}_{shift}.pkl'
        # coup4 = f'./outputs/IP1/{folder2}/{c4}_{shift}.pkl'
        #on_yshift_file_name = f'./outputs/IP1/1E6_new_y/{folder}/IP1_lc_bb_{coup}_sepX0_sepY{shift_str}.pkl'
        #off_yshift_file_name = f'./outputs/IP1/1E6_new_y/{folder2}/IP1_lc_nbb_{coup2}_sepX0_sepY{shift_str}.pkl'
        
        # Data loading
        on_x_data_loaded = pickle.load(open(on_xshift_file_name, 'rb'))
        off_x_data_loaded = pickle.load(open(off_xshift_file_name, 'rb'))
        #on_y_data_loaded = pickle.load(open(on_yshift_file_name, 'rb'))
        #off_y_data_loaded = pickle.load(open(off_yshift_file_name, 'rb'))
        # nocoup_data = pickle.load(open(nocoup, 'rb'))
        # coup2_data = pickle.load(open(coup2, 'rb'))
        # coup3_data = pickle.load(open(coup3, 'rb'))
        # coup4_data = pickle.load(open(coup4, 'rb'))

        lumi[i] = np.mean(on_x_data_loaded['int_luminosity_values'])
        lumi0[i] = np.mean(off_x_data_loaded['int_luminosity_values'])
        
        # Lumi Grid    
        on_x_grid_b1 = on_x_data_loaded['lumigrid_b1']
        on_x_grid_b2 = on_x_data_loaded['lumigrid_b2']
        on_x_grid_b1 = on_x_grid_b1.reshape(n_lumigrid_cells, n_lumigrid_cells)
        on_x_grid_b2 = on_x_grid_b2.reshape(n_lumigrid_cells, n_lumigrid_cells)
        on_x_grid_b1 = on_x_grid_b1.T
        on_x_grid_b2 = np.flip(on_x_grid_b2.T)
        on_x_grid_multiplied = on_x_grid_b1 * on_x_grid_b2      # luminous region
        on_x.append(on_x_grid_multiplied)
        on_x_mult = on_x_mult + on_x_grid_multiplied        # sum of luminous regions
        #print(on_x_mult)

        # lumi = on_x_data_loaded['int_luminosity_values']    
        #print(lumi)    
        # Z = on_x #on_x[i]
        # x = np.linspace(-6/Delta_x,6/Delta_x,120)
        # y = np.linspace(-6/Delta_x,6/Delta_x,120)
        # sum_x = np.sum(Z, axis=0)
        # sum_x_n = sum_x/np.sum(sum_x)
        # mean_x = np.sum(x * sum_x_n)
        # rms_x.append(np.sqrt(np.sum(sum_x_n*((x - mean_x) ** 2))))
        # sum_y = np.sum(Z, axis=1)
        # sum_y_n = sum_y/np.sum(sum_y)
        # mean_y = np.sum(y * sum_y_n)
        # rms_y.append(np.sqrt(np.sum(sum_y_n*((y - mean_y) ** 2))))
        
        off_x_grid_b1 = off_x_data_loaded['lumigrid_b1']
        off_x_grid_b2 = off_x_data_loaded['lumigrid_b2']
        off_x_grid_b1 = off_x_grid_b1.reshape(n_lumigrid_cells, n_lumigrid_cells)
        off_x_grid_b2 = off_x_grid_b2.reshape(n_lumigrid_cells, n_lumigrid_cells)
        off_x_grid_b1 = off_x_grid_b1.T
        off_x_grid_b2 = np.flip(off_x_grid_b2.T)
        off_x_grid_multiplied = off_x_grid_b1 * off_x_grid_b2
        off_x.append(off_x_grid_multiplied)
        off_x_mult = off_x_mult + off_x_grid_multiplied
        
        # division = on_x_grid_multiplied/off_x_grid_multiplied
        # ratio.append(division)
        
        # beam_1_ratio = on_x_grid_b1/off_x_grid_b1 
        # b1.append(beam_1_ratio)
        # beam_2_ratio = on_x_grid_b2/off_x_grid_b2 
        # b2.append(beam_2_ratio)

        # on_y_grid_b1 = on_y_data_loaded['lumigrid_b1']
        # on_y_grid_b2 = on_y_data_loaded['lumigrid_b2']
        # on_y_grid_b1 = on_y_grid_b1.reshape(1200, 1200)*Delta_x
        # on_y_grid_b2 = on_y_grid_b2.reshape(1200, 1200)*Delta_x
        # on_y_grid_b1 = np.flip(on_y_grid_b1.T)
        # on_y_grid_b2 = np.flip(on_y_grid_b2.T)
        # on_y_grid_multiplied = on_y_grid_b1 * on_y_grid_b2
        # on_y.append(on_y_grid_multiplied)
        # on_y_grid_multiplied = on_y_grid_b1 * on_y_grid_b2
        # on_y_mult = on_y_mult + on_y_grid_multiplied
        
        # off_y_grid_b1 = off_y_data_loaded['lumigrid_b1']
        # off_y_grid_b2 = off_y_data_loaded['lumigrid_b2']
        # off_y_grid_b1 = off_y_grid_b1.reshape(1200, 1200)*Delta_x
        # off_y_grid_b2 = off_y_grid_b2.reshape(1200, 1200)*Delta_x
        # off_y_grid_b1 = np.flip(off_y_grid_b1.T)
        # off_y_grid_b2 = np.flip(off_y_grid_b2.T)
        # off_y_grid_multiplied = off_y_grid_b1 * off_y_grid_b2
        # off_y.append(off_y_grid_multiplied)
        # off_y_grid_multiplied = off_y_grid_b1 * off_y_grid_b2
        # off_y_mult = off_y_mult + off_y_grid_multiplied
        
        # division = on_y_grid_multiplied/off_y_grid_multiplied
        # ratio.append(division)
        
        # beam_1_ratio = on_y_grid_b1/off_y_grid_b1 
        # b1.append(beam_1_ratio)
        # beam_2_ratio = on_y_grid_b2/off_y_grid_b2 
        # b2.append(beam_2_ratio)
        
        # x1_noc.append(nocoup_data['x_1'][0])
        # y1_noc.append(nocoup_data['y_1'][0])
        # px1_noc.append(nocoup_data['p_x1'][0])
        # py1_noc.append(nocoup_data['p_y1'][0])
        # x1_coup2.append(coup2_data['x_1'][0])
        # y1_coup2.append(coup2_data['y_1'][0])
        # px1_coup2.append(coup2_data['p_x1'][0])
        # py1_coup2.append(coup2_data['p_y1'][0])
        # x1_coup3.append(coup3_data['x_1'][0])
        # y1_coup3.append(coup3_data['y_1'][0])
        # px1_coup3.append(coup3_data['p_x1'][0])
        # py1_coup3.append(coup3_data['p_y1'][0])
        # x1_coup4.append(coup4_data['x_1'][0])
        # y1_coup4.append(coup4_data['y_1'][0])
        # px1_coup4.append(coup4_data['p_x1'][0])
        # py1_coup4.append(coup4_data['p_y1'][0])
     
    # pos_1_noc = [x1_noc, y1_noc, px1_noc, py1_noc]
    # pos_1_coup2 = [x1_coup2, y1_coup2, px1_coup2, py1_coup2]
    # pos_1_coup3 = [x1_coup3, y1_coup3, px1_coup3, py1_coup3]
    # pos_1_coup4 = [x1_coup4, y1_coup4, px1_coup4, py1_coup4]
    #, z_1, z_2, cx_1, cx_2,cy_1, cy_2] 
    #off_pos = [x_1_, x_2_, y_1_, y_2_]#, z_1_, z_2_, cx_1_, cx_2_,cy_1_, cy_2_]
    
    #print(rms_x_1)
        
    return on_x, on_x_mult,off_x, off_x_mult,folder, directory, sigma, lumi, lumi0 #pos_1_noc, pos_1_coup2, pos_1_coup3, pos_1_coup4, sigma
#coup,folder,directory,Delta_x,ratio,b1, b2,on_x,off_x,on_x_mult, off_x_mult, on_pos, off_pos,rms_x_1, rms_x_2, rms_y_1, rms_y_2, rms_x_1_, rms_x_2_, rms_y_1_, rms_y_2_, rms_x, rms_y
#coup,folder,Delta_x,ratio,b1, b2,on_x,off_x,on_x_mult, off_x_mult, rms_x_1, rms_x_2, rms_y_1, rms_y_2, rms_x_1_, rms_x_2_, rms_y_1_, rms_y_2_, on_pos, off_pos
#coup,folder,Delta_x,ratio,b1, b2,on_y,off_y,on_y_mult, off_y_mult, rms_x_1, rms_x_2, rms_y_1, rms_y_2, rms_x_1_, rms_x_2_, rms_y_1_, rms_y_2_, on_pos, off_pos
