import lumigrid
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

# cmap = plt.get_cmap('tab10')
cmap = plt.cm.rainbow(np.linspace(0,1,11))
# plt.rcParams['image.cmap'] = 'tab10'

cells = 300

separation = np.arange(0, 5.5, 0.5)

#lim = [1e20,1e20,1e22,1e23,1e23,1e24,1e25,1e26,1e26,1e26,1e27]
#coup = '1'

# print(np.array(rms_x_1).shape)
# print(separation.shape)

plt.figure(0)
plt.figure(1)
plt.figure(2)
plt.figure(3)
plt.figure(4)
plt.figure(5)
plt.figure(14)
plt.figure(15)


couplings = ['0.001'] #['0.0', '0.0001','0.0005','0.001']#['0coup','5coup','8coup','16coup'] #['0.0', '0.0001','0.0005','0.001']
skew = ['1e-3']#['0','1e-4','5e-4','1e-3']#['0','5','8','16']#['0','1e-4','5e-4','1e-3']
# classmethod = ['cornflowerblue', 'blueviolet']
for c,coup in enumerate(couplings):
    on_x, on_x_mult,off_x, off_x_mult,folder, directory, sigma, lumi, lumi0 = lumigrid.bias(coup)
    rms_x=[]
    rms_y=[]
    rms_x_=[]
    rms_y_=[]
    bb_fit_x=[]
    bb_fit_y=[]
    nbb_fit_x=[]
    nbb_fit_y=[]
    sigma_x=[]
    sigma_y=[]
    sigma_x_=[]
    sigma_y_=[]
    sum_x = np.zeros([len(separation),cells])
    sum_x_n = np.zeros([len(separation),cells])
    sum_x_ = np.zeros([len(separation),cells])
    sum_x_n_ = np.zeros([len(separation),cells])
    sum_y = np.zeros([len(separation),cells])
    sum_y_n = np.zeros([len(separation),cells])
    sum_y_ = np.zeros([len(separation),cells])
    sum_y_n_ = np.zeros([len(separation),cells])
    for sepX in separation:
        i = list(separation).index(sepX)
        Delta_x = (12*sigma/cells)*1e4
        # Z = on_x_mult # Z = on_x[i]
        Z = on_x[i]
        x = np.linspace(-6,6,cells)
        y = np.linspace(-6,6,cells)
        # sum_x = np.sum(Z, axis=0) # sum_x[i] = np.sum(Z, axis=0)
        sum_x[i] = np.sum(Z, axis=0)/1e4
        # sum_x_n = sum_x/np.sum(sum_x) #sum_x_n[i] = sum_x[i]/np.sum(sum_x[i])
        sum_x_n[i] = sum_x[i]/np.sum(sum_x[i])
        mean_x = np.sum(x * sum_x_n[i])
        # rms_x = (np.sqrt(np.sum(sum_x_n*((x - mean_x) ** 2)))) # rms_x.append(np.sqrt(np.sum(sum_x_n[i]*((x - mean_x) ** 2))))
        rms_x.append(np.sqrt(np.sum(sum_x_n[i]*((x - mean_x) ** 2))))
        # sum_y = np.sum(Z, axis=1)
        sum_y[i] = np.sum(Z, axis=1)/1e4
        # sum_y_n = sum_y/np.sum(sum_y)
        sum_y_n[i] = sum_y[i]/np.sum(sum_y[i])
        mean_y = np.sum(y * sum_y_n[i])
        rms_y.append(np.sqrt(np.sum(sum_y_n[i]*((y - mean_y) ** 2))))
        # print(rms_x,rms_y)
        x1 = np.interp(rms_x,x,sum_x[i])
        y1 = np.interp(rms_y,y,sum_y[i])
    
        # Z_ = off_x_mult #off_x[i]
        Z_ = off_x[i]
        sum_x_[i] = np.sum(Z_, axis=0)/1e4
        sum_x_n_[i] = sum_x_[i]/np.sum(sum_x_[i])
        mean_x_ = np.sum(x * sum_x_n_[i])
        rms_x_.append(np.sqrt(np.sum(sum_x_n_[i]*((x - mean_x_) ** 2))))
        sum_y_[i] = np.sum(Z_, axis=1)/1e4
        sum_y_n_[i] = sum_y_[i]/np.sum(sum_y_[i])
        mean_y_ = np.sum(y * sum_y_n_[i])
        rms_y_.append(np.sqrt(np.sum(sum_y_n_[i]*((y - mean_y_) ** 2))))
        # print(rms_x_,rms_y_)
        x2 = np.interp(rms_x_,x,sum_x_[i])
        y2 = np.interp(rms_y_,y,sum_y_[i])
        
        plt.figure(0)
        plt.plot(x,sum_x[i],label='sep = '+str(sepX),color=cmap[i])
        # plt.plot(x,sum_x_[i],label='no BB',color=cmap(i+1))
        # plt.hlines(x1, -rms_x, rms_x, 'b',label='$\Sigma$ BB = '+str(np.round(rms_x,2))+ '\n $\mu_x$ = '+str(np.round(mean_x,2)),color=cmap(0))
        # plt.hlines(x2, -rms_x_, rms_x_, 'g',label='$\Sigma$ no BB = '+str(np.round(rms_x_,2))+ '\n $\mu_x$ = '+str(np.round(mean_x_,2)),color=cmap(1))
        

        plt.figure(2)
        plt.plot(y,sum_y[i],label='sep = '+str(sepX),color=cmap[i])
        # plt.plot(y,sum_y_[i],label='no BB',color=cmap(1))
        # plt.hlines(y1, -rms_y, rms_y, 'b',label='$\Sigma$ BB = '+str(np.round(rms_y,2))+'\n $\mu_y$ = '+str(np.round(mean_y,2)),color=cmap(0))
        # plt.hlines(y2, -rms_y_, rms_y_, 'g',label='$\Sigma$ no BB = '+str(np.round(rms_y_,2))+'\n $\mu_y$ = '+str(np.round(mean_y_,2)),color=cmap(1))

        
        
        ## fit the data to a gaussian 
        def func_bb(x, a, x0, sigma): 
            return a*np.exp(-(x-x0)**2/(2*sigma**2)) 
        def func_nbb(x, a, x0, sigma): 
            return a*np.exp(-(x-x0)**2/(2*sigma**2)) 
        px, cx = curve_fit(func_bb, x, sum_x[i],  p0=[2e42, 0, 2.5]) #sum_x[i]
        px_, cx_ = curve_fit(func_nbb, x, sum_x_[i],  p0=[3e39, 0, 2.5])
        py, cy = curve_fit(func_bb, y, sum_y[i],  p0=[2e42, 0, 2.5]) # sum_y[i]
        py_, cy_ = curve_fit(func_nbb, y, sum_y_[i],  p0=[3e39, 0, 2.5])
        bb_fit_x.append(px)
        nbb_fit_x.append(px_)
        bb_fit_y.append(py)
        nbb_fit_y.append(py_)

        fit_x = func_bb(x, px[0], px[1], px[2]) 
        fit_y = func_bb(y, py[0], py[1], py[2]) 
        fit_x_ = func_nbb(x, px_[0], px_[1], px_[2]) 
        fit_y_ = func_nbb(y, py_[0], py_[1], py_[2]) 

    # pull_x = (sum_x-fit_x) / rms_x
    # pull_x = pull_x / np.max(pull_x)
    # pull_error_x = rms_x / sum_x
    # pull_error_x = pull_error_x / np.max(pull_error_x)
    # print(pull_error_x)
    # pull_x_ = (sum_x_-fit_x_) / rms_x_
    # pull_error_x_ = rms_x_ / sum_x_
    # pull_y = (sum_y-fit_y) / rms_y
    # pull_error_y = rms_y / sum_y
    # pull_y_ = (sum_y_-fit_y_) / rms_y_
    # pull_error_y_ = rms_y_ / sum_y_
        
        sigma_x.append(np.abs(px[2]))
        sigma_y.append(np.abs(py[2]))
        sigma_x_.append(np.abs(px_[2]))
        sigma_y_.append(np.abs(py_[2]))
    
    ## gaussian distribution ratio plots to see what BB does
#     g_ratio_x = (np.divide(np.array(fit_x),np.array(fit_x_))-1)*100
#     plt.figure(4)
#     plt.plot(x, g_ratio_x, label = f'{skew[i]} coup') #, label = 'sep = '+str(sepX),color=cmap(i / 11))
    
#     g_ratio_y = (np.divide(np.array(fit_y),np.array(fit_y_))-1)*100
#     plt.figure(5)
#     plt.plot(y, g_ratio_y, label = f'{skew[i]} coup') #, label = 'sep = '+str(sepX),color=cmap(i / 11))

## RMS // width plots
# plt.figure(11)
# plt.plot(separation,rms_x_1,'.-',label='beam 1 width BB')
# plt.plot(separation,rms_x_2,'.-',label='beam 2 width BB')
# # plt.plot(separation,rms_x_1_,'.-',label='beam 1 width, no BB')
# # plt.plot(separation,rms_x_2_,'.-',label='beam 2 width, no BB')
# plt.xlabel('Nominal beam separation [$\sigma$]')
# plt.ylabel('RMS width')
# plt.legend()
# plt.title('RMS of luminosity in x')
# plt.savefig(f'{directory}/gauss/beam_width_x.png')
# plt.clf()

# plt.figure(11)
# plt.plot(separation,rms_y_1,'.-',label='beam 1 width')
# plt.plot(separation,rms_y_2,'.-',label='beam 2 width')
# # plt.plot(separation,rms_y_1_,'.-',label='beam 1 width, no BB')
# # plt.plot(separation,rms_y_2_,'.-',label='beam 2 width, no BB')
# plt.xlabel('Nominal beam separation [$\sigma$]')
# plt.ylabel('RMS width')
# plt.legend()
# plt.title('RMS of luminosity in y')
# plt.savefig(f'{directory}/gauss/beam_width_y.png')
# plt.clf()

    d = 1/np.sqrt(2)
    
    if c == 0:
        print(np.array(rms_x))
        print(np.array(rms_x_))

    # plt.figure(10)
    # plt.plot(separation,rms_x/d,'.-',label='rms x with BB',color=cmap(0))
    # plt.plot(separation,rms_x_/d,'.-',label='rms x without BB',color=cmap(1))
    # plt.plot(separation,rms_y/d,'.-',label='rms y with BB',color=cmap(2))
    # plt.plot(separation,rms_y_/d,'.-',label='rms y without BB',color=cmap(3))
    # plt.hlines(1, 0, 5, linestyle='--',color = 'black')
    # # plt.text(4.5,d,'$\\frac{1}{\\sqrt{2}}$',color='black',va='top',fontsize=20)
    # plt.xlabel('Nominal beam separation ($\\sigma$)',fontsize=20)
    # plt.ylabel('Width ($\\sigma$)',fontsize=20)
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # plt.legend()
    # plt.title('RMS of luminosity in x and y for coup = '+str(skew[c]),fontsize=30)
    # plt.savefig(f'{directory}/gauss_global/rms_{skew[c]}.png')
    # plt.clf()

    # plt.figure(11)
    # plt.plot(separation,sigma_x/d,'.-',label='sigma x with BB',color=cmap(0))
    # plt.plot(separation,sigma_x_/d,'.-',label='sigma x without BB',color=cmap(1))
    # plt.plot(separation,sigma_y/d,'.-',label='sigma y with BB',color=cmap(2))
    # plt.plot(separation,sigma_y_/d,'.-',label='sigma y without BB',color=cmap(3))
    # plt.hlines(1, 0, 5, linestyle='--',color = 'black')
    # # plt.text(4.5,d,'$\\frac{1}{\\sqrt{2}}$',color='black',va='top',fontsize=20)
    # plt.xlabel('Nominal beam separation ($\\sigma$)',fontsize=20)
    # plt.ylabel('Width ($\\sigma$)',fontsize=20)
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # plt.legend()
    # plt.title('Width of luminous region in x and y for coup = '+str(skew[c]),fontsize=30)
    # plt.savefig(f'{directory}/gauss_global/sigma_{skew[c]}.png')
    # plt.clf()

    # plt.figure(12)
    # plt.plot(separation,sigma_x/d,'.-',label='sigma x with BB',color=cmap(0))
    # plt.plot(separation,sigma_x_/d,'.-',label='sigma x without BB',color=cmap(1))
    # plt.plot(separation,rms_x/d,'.-',label='rms x with BB',color=cmap(2))
    # plt.plot(separation,rms_x_/d,'.-',label='rms x without BB',color=cmap(3))
    # plt.hlines(1, 0, 5, linestyle='--',color = 'black')
    # # plt.text(4.5,d,'$\\frac{1}{\\sqrt{2}}$',color='black',va='top',fontsize=20)
    # plt.xlabel('Nominal beam separation ($\\sigma$)',fontsize=20)
    # plt.ylabel('Width ($\\sigma$)',fontsize=20)
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # plt.legend()
    # plt.title('RMS and luminous region width in x for coup = '+str(skew[c]),fontsize=30)
    # plt.savefig(f'{directory}/gauss/width_x_{skew[c]}.png')
    # plt.clf()

    # plt.figure(13)
    # plt.plot(separation,sigma_y/d,'.-',label='sigma y with BB',color=cmap(0))
    # plt.plot(separation,sigma_y_/d,'.-',label='sigma y without BB',color=cmap(1))
    # plt.plot(separation,rms_y/d,'.-',label='rms y with BB',color=cmap(2))
    # plt.plot(separation,rms_y_/d,'.-',label='rms y without BB',color=cmap(3))
    # plt.hlines(1, 0, 5, linestyle='--',color = 'black')
    # # plt.text(4.5,d,'$\\frac{1}{\\sqrt{2}}$',color='black',va='top',fontsize=20)
    # plt.xlabel('Nominal beam separation ($\\sigma$)',fontsize=20)
    # plt.ylabel('Width ($\\sigma$)',fontsize=20)
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # plt.legend(loc='upper left')
    # plt.title('RMS and luminous region width in y for coup = '+str(skew[c]),fontsize=30)
    # plt.savefig(f'{directory}/gauss/width_y_{skew[c]}.png')
    # plt.clf()

    # ratio_x = (np.divide(np.array(rms_x),np.array(rms_x_))-1)*100
    # ratio_y = (np.divide(np.array(rms_y),np.array(rms_y_))-1)*100

    # plt.figure(14)
    # plt.plot(separation, ratio_x, '.-', label='coup = '+str(skew[c]),color=cmap(c))
    

    # plt.figure(15)
    # plt.plot(separation, ratio_y, '.-', label='coup = '+str(skew[c]),color=cmap(c))
    

# plt.figure(14)
# plt.legend()
# plt.xlabel('Nominal beam separation ($\\sigma$)',fontsize=20)
# plt.ylabel('Width ratio in x [%]',fontsize=20)
# plt.title('Ratio of width in x')
# plt.grid()
# plt.savefig(f'{directory}/gauss/ratio_x.png')
# plt.clf()

# plt.figure(15)
# plt.legend()
# plt.xlabel('Nominal beam separation ($\\sigma$)',fontsize=20)
# plt.ylabel('Width ratio in y [%]',fontsize=20)
# plt.title('Ratio of width in y')
# plt.grid()
# plt.savefig(f'{directory}/gauss/ratio_y.png')
# plt.clf()

plt.figure(0)
plt.xlabel('Grid in x')
plt.ylabel('Luminosity projected in x [$\\text{cm}^{-2} \\text{s}^{-1}$]')
plt.title(f'X profile of luminosity with BB, coup = '+skew[0],pad=20)
plt.legend()
plt.savefig(f'{directory}/gauss/X_sep_{skew[0]}.png')
plt.clf()

plt.figure(2)
plt.xlabel('Grid in y')
plt.ylabel('Luminosity projected in y [$\\text{cm}^{-2} \\text{s}^{-1}$]')
plt.title(f'Y profile of luminosity with BB, coup = '+skew[0],pad=20)
plt.legend()
plt.savefig(f'{directory}/gauss/Y_sep_{skew[0]}.png')
plt.clf()