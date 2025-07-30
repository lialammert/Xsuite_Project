import lumigrid
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import seaborn as sns

cmap = plt.get_cmap('Set2')
plt.rcParams['image.cmap'] = 'Set2'
plt.rcParams.update({'font.size': 20})

cells = 300

separation = [0.0]#np.arange(0, 5.5, 0.5)

#lim = [1e20,1e20,1e22,1e23,1e23,1e24,1e25,1e26,1e26,1e26,1e27]
#coup = '1'

# print(np.array(rms_x_1).shape)
# print(separation.shape)

plt.figure(1)
plt.figure(3)
plt.figure(4)
plt.figure(5)

couplings = ['0.0','0.0001','0.0005','0.001']#,'0.001']
skew = ['0','1e-4','5e-4','1e-3']#,'1e-3']
# c = ['cornflowerblue', 'blueviolet']
for i,coup in enumerate(couplings):
    on_x, on_x_mult,off_x, off_x_mult,folder, directory, sigma, lumi, lumi0 = lumigrid.bias(coup)
    # lumii = np.sum(lumi, axis=1)
    # print(np.sum(lumii))
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
    #for sepX in separation:
    #    i = list(separation).index(sepX)
    Delta_x = (12*sigma/cells)*1e4
    Z = on_x_mult # Z = on_x[i]
    x = np.linspace(-6,6,cells)
    y = np.linspace(-6,6,cells)
    sum_x = np.sum(Z, axis=0)/1e4 # sum_x[i] = np.sum(Z, axis=0)
    sum_x_n = sum_x/np.sum(sum_x) #sum_x_n[i] = sum_x[i]/np.sum(sum_x[i])
    mean_x = np.sum(x * sum_x_n)
    rms_x = (np.sqrt(np.sum(sum_x_n*((x - mean_x) ** 2)))) # rms_x.append(np.sqrt(np.sum(sum_x_n[i]*((x - mean_x) ** 2))))
    sum_y = np.sum(Z, axis=1)/1e4 
    sum_y_n = sum_y/np.sum(sum_y)
    mean_y = np.sum(y * sum_y_n)
    rms_y = np.sqrt(np.sum(sum_y_n*((y - mean_y) ** 2)))
    print(rms_x,rms_y)
    x1 = np.interp(rms_x,x,sum_x)
    y1 = np.interp(rms_y,y,sum_y)

    Z_ = off_x_mult #off_x[i]
    sum_x_ = np.sum(Z_, axis=0)/1e4 
    sum_x_n_ = sum_x_/np.sum(sum_x_)
    mean_x_ = np.sum(x * sum_x_n_)
    rms_x_ = (np.sqrt(np.sum(sum_x_n_*((x - mean_x_) ** 2))))
    sum_y_ = np.sum(Z_, axis=1)/1e4 
    sum_y_n_ = sum_y_/np.sum(sum_y_)
    mean_y_ = np.sum(y * sum_y_n_)
    rms_y_ = (np.sqrt(np.sum(sum_y_n_*((y - mean_y_) ** 2))))
    print(rms_x_,rms_y_)
    x2 = np.interp(rms_x_,x,sum_x_)
    y2 = np.interp(rms_y_,y,sum_y_)
        
        
    # plot the luminosity profiles
    # plt.figure(0)
    # plt.plot(x,sum_x,label='BB',color=cmap(0))
    # plt.plot(x,sum_x_,label='no BB',color=cmap(1))
    # plt.hlines(x1, -rms_x, rms_x, 'b',label='$\Sigma$ BB = '+str(np.round(rms_x,2))+ '\n $\mu_x$ = '+str(np.round(mean_x,2)),color=cmap(0))
    # plt.hlines(x2, -rms_x_, rms_x_, 'g',label='$\Sigma$ no BB = '+str(np.round(rms_x_,2))+ '\n $\mu_x$ = '+str(np.round(mean_x_,2)),color=cmap(1))
    # plt.xlabel('Grid in x')
    # plt.ylabel('Luminosity projected in x')
    # plt.title(f'X profile of luminosity with BB for coup = {skew[i]}',pad=20)
    # plt.legend()
    # plt.savefig(f'{directory}/gauss/X_{skew[i]}.png')
    # plt.clf()

    plt.figure(1)
    plt.plot(x,((sum_x/sum_x_)-1), label = f'{skew[i]} coup',color=cmap(i))#,linewidth=1)

    # plt.figure(2)
    # plt.plot(y,sum_y,label='BB',color=cmap(0))
    # plt.plot(y,sum_y_,label='no BB',color=cmap(1))
    # plt.hlines(y1, -rms_y, rms_y, 'b',label='$\Sigma$ BB = '+str(np.round(rms_y,2))+'\n $\mu_y$ = '+str(np.round(mean_y,2)),color=cmap(0))
    # plt.hlines(y2, -rms_y_, rms_y_, 'g',label='$\Sigma$ no BB = '+str(np.round(rms_y_,2))+'\n $\mu_y$ = '+str(np.round(mean_y_,2)),color=cmap(1))
    # plt.xlabel('Grid in y')
    # plt.ylabel('Luminosity projected in y')
    # plt.title(f'Y profile of luminosity with BB for coup = {skew[i]}',pad=20)
    # plt.legend()
    # plt.savefig(f'{directory}/gauss/Y_{skew[i]}.png')
    # plt.clf()

    plt.figure(3)
    plt.plot(y,((sum_y/sum_y_)-1), label = f'{skew[i]} coup',color=cmap(i))#,linewidth=1)



    #folder = 'zero_1'
        
        ## fit the data to a gaussian 
    def func_bb(x, a, x0, sigma): 
        return a*np.exp(-(x-x0)**2/(2*sigma**2)) 
    def func_nbb(x, a, x0, sigma): 
        return a*np.exp(-(x-x0)**2/(2*sigma**2)) 
    def fit_uncertainty(x, popt, pcov):
        # Partial derivatives of the Gaussian with respect to parameters
        A, mu, sigma = popt
        partials = np.array([
            np.exp(-0.5 * ((x - mu) / sigma) ** 2),  # ∂f/∂A
            A * ((x - mu) / sigma**2) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),  # ∂f/∂mu
            A * ((x - mu)**2 / sigma**3) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),  # ∂f/∂sigma
        ])
        # Propagate the covariance matrix
        return np.sqrt(np.einsum('ij,ji->i', partials.T @ pcov, partials))

    
    px, cx = curve_fit(func_bb, x, sum_x,  p0=[2.4e38, 0, 1]) #sum_x[i]
    px_, cx_ = curve_fit(func_nbb, x, sum_x_,  p0=[2.4e38, 0, 1])
    py, cy = curve_fit(func_bb, y, sum_y,  p0=[2.4e38, 0, 1]) # sum_y[i]
    py_, cy_ = curve_fit(func_nbb, y, sum_y_,  p0=[2.4e38, 0, 1])
    bb_fit_x.append(px)
    nbb_fit_x.append(px_)
    bb_fit_y.append(py)
    nbb_fit_y.append(py_)

    fit_x = func_bb(x, px[0], px[1], px[2]) 
    fit_y = func_bb(y, py[0], py[1], py[2]) 
    fit_x_ = func_nbb(x, px_[0], px_[1], px_[2]) 
    fit_y_ = func_nbb(y, py_[0], py_[1], py_[2]) 
    
    # Compute fit uncertainties
    fit_x_err = fit_uncertainty(x, px, cx)
    fit_y_err = fit_uncertainty(y, py, cy)
    fit_x_err_ = fit_uncertainty(x, px_, cx_)
    fit_y_err_ = fit_uncertainty(y, py_, cy_)

    pull_x = (sum_x-fit_x)  / fit_x 
    pull_x_err = np.sqrt((np.std(sum_x) / fit_x)**2 + ((-sum_x / fit_x**2 + 1 / fit_x) * fit_x_err)**2) 
    pull_y = (sum_y-fit_y)  / fit_y
    pull_y_err = np.sqrt((np.std(sum_y) / fit_y)**2 + ((-sum_y / fit_y**2 + 1 / fit_y) * fit_y_err)**2)
    pull_x_ = (sum_x_-fit_x_)  / fit_x_
    pull_x_err_ = np.sqrt((np.std(sum_x_) / fit_x_)**2 + ((-sum_x_ / fit_x_**2 + 1 / fit_x_) * fit_x_err_)**2)
    pull_y_ = (sum_y_-fit_y_)  / fit_y_
    pull_y_err_ = np.sqrt((np.std(sum_y_) / fit_y_)**2 + ((-sum_y_ / fit_y_**2 + 1 / fit_y_) * fit_y_err_)**2)
    
    err=50

    # fig_x,ax_x = plt.subplots(2,1,figsize=(14,9), gridspec_kw={'height_ratios': [2, 1]})
    # ax_x[0].plot(x, sum_x, 'kx', label='data',markersize = 10) 
    # ax_x[0].plot(x, fit_x, '-', label='fit : a = %.2e \n x0 = %.2e \n sigma = %.2e' % tuple(px),color=cmap(0),linewidth=2) 
    # ax_x[0].legend(fontsize=20,loc='upper right')
    # ax_x[0].set_ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')
    # #ax_x[0].set_xlim(-665,665)
    # ax_x[0].set_ylim(1e24,np.max(sum_x)*10)
    # ax_x[0].set_yscale('log')
    # fig_x.suptitle('Fit of luminosity profile in x with BB and coup = '+str(skew[i]), fontsize=28)
    # ax_x[1].set_xlabel('Grid in x [$\\sigma$]', fontsize=24)
    # ax_x[1].plot(x,(sum_x-fit_x)/sum_x * 100,'k+',markersize = 10)
    # #ax_x[1].plot(x,pull_x,'+')
    # # ax_x[1].errorbar(x, pull_x, yerr=pull_x_err)#np.abs((sum_x-fit_x)/sum_x))
    # ax_x[1].set_ylabel('$\\frac{\\text{ data }}{\\text{ fit }}$ - 1 [%]')
    # ax_x[1].set_yscale('symlog', linthresh=1)
    # ax_x[1].set_ylim(-err,err)
    # ax_x[1].axhline(y=0, color=cmap(1), linestyle='--', linewidth=1)
    # ax_x[0].grid(True)
    # ax_x[1].grid(True)
    # plt.subplots_adjust(top=0.9)
    # #ax_x[1].set_xlim(-665,665)
    # fig_x.savefig(f'{directory}/gauss_0sep/fit_X_BB_{skew[i]}.png')
    # fig_x.clf()
        
    # fig_y,ax_y = plt.subplots(2,1,figsize=(14,9), gridspec_kw={'height_ratios': [2, 1]})
    # ax_y[0].plot(y, sum_y, 'kx', label='data',markersize = 10) 
    # ax_y[0].plot(y, fit_y, '-', label='fit : a = %.2e \n x0 = %.2e \n sigma = %.2e' % tuple(py),color=cmap(0),linewidth=2) 
    # ax_y[0].legend(fontsize=20,loc='upper right')
    # ax_y[0].set_ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')
    # #ax_y[0].set_xlim(-665,665)
    # ax_y[0].set_ylim(1e24,np.max(sum_y)*10)
    # ax_y[0].set_yscale('log')
    # fig_y.suptitle('Fit of luminosity profile in y with BB and coup = ' +str(skew[i]), fontsize=28)
    # ax_y[1].set_xlabel('Grid in y [$\\sigma$]', fontsize=24)
    # ax_y[1].plot(y,(sum_y-fit_y)/sum_y * 100,'k+',markersize = 10)
    # ax_y[1].set_ylabel('$\\frac{\\text{ data }}{\\text{ fit }}$ - 1 [%]')
    # ax_y[1].set_yscale('symlog', linthresh=1)
    # ax_y[1].set_ylim(-err,err)
    # ax_y[1].axhline(y=0, color=cmap(1), linestyle='--', linewidth=1)
    # ax_y[0].grid(True)
    # ax_y[1].grid(True)
    # plt.subplots_adjust(top=0.9)
    # #ax_y[1].set_xlim(-665,665)
    # fig_y.savefig(f'{directory}/gauss_0sep/fit_Y_BB_{skew[i]}.png')
    # fig_y.clf()

    # fig_x_,ax_x_ = plt.subplots(2,1,figsize=(14,9), gridspec_kw={'height_ratios': [2, 1]})
    # ax_x_[0].plot(x, sum_x_, 'kx', label='data',markersize = 10) 
    # ax_x_[0].plot(x, fit_x_, '-', label='fit : a = %.2e \n x0 = %.2e \n sigma = %.2e' % tuple(px),color=cmap(0),linewidth=2) 
    # ax_x_[0].legend(fontsize=20,loc='upper right')
    # ax_x_[0].set_ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')
    # #ax_x_[0].set_xlim(-665,665)
    # ax_x_[0].set_ylim(1e24,np.max(sum_x_)*10)
    # ax_x_[0].set_yscale('log')
    # fig_x_.suptitle('Fit of luminosity profile in x without BB and coup = ' +str(skew[i]), fontsize=28)
    # ax_x_[1].set_xlabel('Grid in x [$\\sigma$]', fontsize=24)
    # ax_x_[1].plot(x,(sum_x_-fit_x_)/sum_x_ * 100,'k+',markersize = 10)
    # ax_x_[1].set_ylabel('$\\frac{\\text{ data }}{\\text{ fit }}$ - 1 [%]')
    # ax_x_[1].set_yscale('symlog', linthresh=1)
    # ax_x_[1].set_ylim(-err,err)
    # ax_x_[1].axhline(y=0, color=cmap(1), linestyle='--', linewidth=1)
    # ax_x_[0].grid(True)
    # ax_x_[1].grid(True)
    # plt.subplots_adjust(top=0.9)
    # #ax_x_[1].set_xlim(-665,665)
    # fig_x_.savefig(f'{directory}/gauss_0sep/fit_X_noBB_{skew[i]}.png')
    # fig_x_.clf()

    # fig_y_,ax_y_ = plt.subplots(2,1,figsize=(14,9), gridspec_kw={'height_ratios': [2, 1]})
    # ax_y_[0].plot(y, sum_y_, 'kx', label='data',markersize = 10) 
    # ax_y_[0].plot(y, fit_y_, '-', label='fit : a = %.2e \n x0 = %.2e \n sigma = %.2e' % tuple(px),color=cmap(0),linewidth=2) 
    # ax_y_[0].legend(fontsize=20,loc='upper right')
    # ax_y_[0].set_ylabel('Luminosity [$\\text{cm}^{-2} \\text{s}^{-1}$]')
    # #ax_y_[0].set_xlim(-665,665)
    # ax_y_[0].set_ylim(1e24,np.max(sum_y_)*10)
    # ax_y_[0].set_yscale('log')
    # fig_y_.suptitle('Fit of luminosity profile in y without BB and coup = ' +str(skew[i]), fontsize=28)
    # ax_y_[1].set_xlabel('Grid in y [$\\sigma$]', fontsize=24)
    # ax_y_[1].plot(y,(sum_y_-fit_y_)/sum_y_ *100,'k+',markersize = 10)
    # ax_y_[1].set_yscale('symlog', linthresh=1)
    # ax_y_[1].set_ylim(-err,err)
    # ax_y_[1].axhline(y=0, color=cmap(1), linestyle='--', linewidth=1)
    # ax_y_[0].grid(True)
    # ax_y_[1].grid(True)
    # ax_y_[1].set_ylabel('$\\frac{\\text{ data }}{\\text{ fit }}$ - 1 [%]')
    # plt.subplots_adjust(top=0.9)
    # #ax_y_[1].set_xlim(-665,665)
    # fig_y_.savefig(f'{directory}/gauss_0sep/fit_Y_noBB_{skew[i]}.png')
    # fig_y_.clf()
        
    # sigma_x.append(np.abs(px[2]))
    # sigma_y.append(np.abs(py[2]))
    # sigma_x_.append(np.abs(px_[2]))
    # sigma_y_.append(np.abs(py_[2]))
    
    ## gaussian distribution ratio plots to see what BB does
    # g_ratio_x = (np.divide(np.array(fit_x),np.array(fit_x_))-1)*100
    # plt.figure(4)
    # plt.plot(x, g_ratio_x, label = f'{skew[i]} coup',color=cmap(i)) #, label = 'sep = '+str(sepX),color=cmap(i / 11))
    
    # g_ratio_y = (np.divide(np.array(fit_y),np.array(fit_y_))-1)*100
    # plt.figure(5)
    # plt.plot(y, g_ratio_y, label = f'{skew[i]} coup',color=cmap(i)) #, label = 'sep = '+str(sepX),color=cmap(i / 11))

plt.figure(1)
# plt.figsize=(18,10)
plt.xlabel('Grid in x')
plt.ylabel('($L_{bb+c}$/ $L_0$) - 1, $\it{symlog}$',fontsize=24)
plt.title('Effect of beam-beam and linear coupling in x \n for 0$\sigma$ beam separation',pad=20)
plt.yscale('symlog')
plt.ylim(-1e-1,1e-1)
# plt.yticks(np.arange(-1e-1, 1e-1, 0.05))
plt.legend()
plt.grid()
plt.savefig(f'{directory}/gauss/bb_X_0sep.png')

plt.figure(3)
# plt.figsize=(18,10)
plt.xlabel('Grid in y')
plt.ylabel('($L_{bb+c}$ / $L_0$) - 1, $\it{symlog}$',fontsize=24)
plt.title('Effect of beam-beam and linear coupling in y \n for 0$\sigma$ beam separation',pad=20)
plt.yscale('symlog')
plt.legend()
plt.ylim(-1e-1,1e-1)
# plt.yticks(np.arange(-1e-1, 1e-1, 0.05))
plt.grid()
plt.savefig(f'{directory}/gauss/bb_Y_0sep.png')

# plt.figure(4)
# plt.legend()
# plt.xlabel('Nominal beam separation in x')
# plt.ylabel('Gaussian ditribution ratio in x [%]')
# plt.title('BB/noBB ratio of Gaussian distribution in x',pad=20)
# plt.grid()
# plt.savefig(f'{directory}/gauss/g_ratio_x.png')

# plt.figure(5)
# plt.legend()
# plt.xlabel('Nominal beam separation in x')
# plt.ylabel('Gaussian ditribution ratio in y [%]')
# plt.title('BB/noBB ratio of Gaussian distribution in y',pad=20)
# plt.grid()
# plt.savefig(f'{directory}/gauss/g_ratio_y.png')
