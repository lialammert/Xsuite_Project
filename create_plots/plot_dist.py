import lumigrid
import numpy as np
import matplotlib.pyplot as plt

#coup,folder,Delta_x,ratio,b1, b2,on_x,off_x,on_x_mult, off_x_mult, rms_x_1, rms_x_2, rms_y_1, rms_y_2, rms_x_1_, rms_x_2_, rms_y_1_, rms_y_2_, on_pos, off_pos = lumigrid.bias()
on_x_mult, off_x_mult, coup, folder, directory, pos_1_noc, pos_1_coup2, pos_1_coup3, pos_1_coup4, sigma = lumigrid.bias()
#separation = [0]#np.arange(0, 5.5, 0.5)
# coup = ['0coup', '1e-4', '1e']

pos = [ pos_1_coup2, pos_1_coup3, pos_1_coup4]
coup = [ '1e-2', '1e-3', '1e-4']
colours = plt.cm.viridis(np.linspace(0,1,len(coup)))

for i,p in enumerate(pos) :
    x1 = np.array(p)[:,0][0]/sigma
    y1 = np.array(p)[:,0][1]/sigma
    px1 = np.array(p)[:,0][2]/sigma
    py1 = np.array(p)[:,0][3]/sigma

    #print(coup[pos.index(p)])

    slope, intercept = np.polyfit(x1[:,0], py1[:,0], 1)


    plt.figure(0)
    plt.plot(x1,px1, '.', label='Beam 1, coup  = '+str(coup[i]), color= colours[i])
    plt.plot(x1, slope*x1 + intercept, 'r', label='Fit : y = '+str(round(slope,4))+'x + '+str(round(intercept,4)), color=colours[i])
plt.ylabel('$\Delta p_x$')
plt.xlabel('$x$')
plt.grid(True)  # Show grid
plt.legend()
plt.show()

# for sepX in separation:
#     i = list(separation).index(sepX)
    
    # Particle dsitribution
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(on_pos[0],on_pos[2],on_pos[4],'.',s=0.5)  
    # ax.set_title('Beam 1 distribution, BB')  
    # fig.savefig(f'./distribution/particles/flat/BB1_{coup}coup_{sepX}.png')
    
    # plt.figure(1)
    # plt.plot(on_pos[0],on_pos[2], 'k.')
    # plt.ylim(-0.04,0.04)
    # plt.savefig(f'./lumiGrid_tests/{folder}/BB1_{coup}coup_{sepX}_2D.png')
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(projection='3d')
    # ax1.scatter(off_pos[0],off_pos[2],off_pos[4],'.',s=0.5)
    # ax1.set_title('Beam 1 distribution, no BB')
    # fig.savefig(f'./distribution/particles/nBB1_{coup}coup_{sepX}.png')
    # plt.clf()
    
    # plt.figure(2)
    # plt.plot(off_pos[1],off_pos[3], 'k.')
    # plt.ylim(-0.04,0.04)
    # plt.savefig(f'./lumiGrid_tests/{folder}/nBB1_{coup}coup_{sepX}_2D.png')
    
    # fig = plt.figure()
    # ax2 = fig.add_subplot(projection='3d')
    # ax2.scatter(on_pos[1],on_pos[3],on_pos[5],'.',s=0.5)  
    # ax2.set_title('Beam 2 distribution, BB')  
    # fig.savefig(f'./distribution/particles/flat/BB2_{coup}coup_{sepX}.png')
    
    # fig = plt.figure()
    # ax3 = fig.add_subplot(projection='3d')
    # ax3.scatter(off_pos[1],off_pos[3],off_pos[5],'.',s=0.5)
    # ax3.set_title('Beam 2 distribution, no BB')
    # fig.savefig(f'./distribution/particles/flat/nBB2_{coup}coup_{sepX}.png')
    
    # Beam size
    # print('with coupling, Beam 1 size in x = '+ str(rms_x_1) +' and in y = ' + str(rms_y_1))
    # print('without coupling, Beam 1 size in x = '+ str(rms_x_1_) +' and in y = ' + str(rms_y_1_))
    
    # Centroids
    # plt.figure()
    # plt.plot(on_pos[6],label = 'Beam 1, BB')
    # plt.plot(on_pos[7],label = 'Beam 2, BB')
    # plt.plot(off_pos[6],label = 'Beam 1, no BB')
    # plt.plot(off_pos[7],label = 'Beam 2, no BB')
    # plt.legend()
    # plt.xlabel('Turns')
    # plt.ylabel('x centroid position')
    # plt.title('Centroid position in x')
    # plt.savefig(f'./distribution/centroid/flat/BB_{coup}coup_{sepX}.png')
    
    # plt.figure()
    # plt.plot(on_pos[8],label = 'Beam 1, BB')
    # plt.plot(on_pos[9],label = 'Beam 2, BB')
    # plt.plot(off_pos[8],label = 'Beam 1, no BB')
    # plt.plot(off_pos[9],label = 'Beam 2, no BB')
    # plt.legend()
    # plt.xlabel('Turns')
    # plt.ylabel('y centroid position')
    # plt.title('Centroid position in y')
    # plt.savefig(f'./distribution/centroid/flat/nBB_{coup}coup_{sepX}.png') 
    
