import numpy as np
from matplotlib import pyplot as plt
import xobjects as xo
import xtrack as xt
import xfieldsdevlumi as xf
import xpart as xp
import pickle
from datetime import datetime
start_time = datetime.now()


print(xo.__version__)
print(xt.__version__)
print(xf.__version__)
print(xp.__version__)

context = xo.ContextCpu()

p0c = 6800e9 #6800e9
bunch_intensity = 0.7825E11 #3e11 #// og = 0.7825E11 // try 1e5 

vdm = False
lhc = True
flat = True
#hl_lhc = True

if vdm == True:
    if flat == True :
        physemit_x = (5.892E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
        physemit_y = (1.473E-6*xp.PROTON_MASS_EV)/p0c  #3.75 for LHC 
    else :
        physemit_x = (2.946E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
        physemit_y = (2.946E-6*xp.PROTON_MASS_EV)/p0c  #3.75 for LHC
    beta_x = 19.17 #in meters 0.3 m for LHC, 19.17 for vdM, 0.15m for HL-LHC
    beta_y = 19.17 ## round BEAM (/4 for flat)
    folder = 'vdM_flat'
    print('vdm')
if lhc == True :
    if flat == True : 
        ## FLAT beams
        physemit_x = (5.892E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
        physemit_y = (1.473E-6*xp.PROTON_MASS_EV)/(p0c)  #3.75 for LHC
    else :
        physemit_x = (3.75E-6*xp.PROTON_MASS_EV)/p0c
        physemit_y = (3.75E-6*xp.PROTON_MASS_EV)/p0c
    beta_x = 0.3 #in meters 0.3 m for LHC, 19.17 for vdM, 0.15m for HL-LHC
    beta_y = 0.3  ## round BEAM (/4 for flat)
    folder = 'LHC_flat'
    print('lhc')
    
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 62.31
Qy = 60.32
Qs = 2.1E-3
frev = 11245.5 
nTurn = 1000 #700 #1000
phase_adv = 0.25 #0.25  # [1 = 2pi] so [0.5 = pi] and [0.25 = pi/2]

n_macroparticles = int(1e6) # try 1e7 (at least 21)
lumi_cells = 300

totalshift = 0.0
xshift = 0.0
# xshifts = np.arange(0,6,1)
yshift = 0
coupling = 0.0
qxchange = phase_adv
qychange = phase_adv
# ksl = 0.0
# name_coupling = str(coupling)

shift_label = "xshift"

## need a different random seed for each of the 4 distributions ???
np.random.seed(0) 
dist1_x = np.random.randn(n_macroparticles)
np.random.seed(1)
dist1_y = np.random.randn(n_macroparticles)
np.random.seed(2)
dist2_x = np.random.randn(n_macroparticles)
np.random.seed(3)
dist2_y = np.random.randn(n_macroparticles)
np.random.seed(4)
dist1_px = np.random.randn(n_macroparticles)
np.random.seed(5)
dist1_py = np.random.randn(n_macroparticles)
np.random.seed(6)
dist2_px = np.random.randn(n_macroparticles)
np.random.seed(7)
dist2_py = np.random.randn(n_macroparticles)

print('Initialising particles')
bb_particles_b1 = xp.Particles(_context=context,
    p0c=p0c,
    x=np.sqrt(physemit_x*beta_x)*(dist1_x), #(np.random.randn(n_macroparticles)), #(dist0_x),
    px=np.sqrt(physemit_x/beta_x)*dist1_px,  #(np.random.randn(n_macroparticles)), #dist0_x,
    y=np.sqrt(physemit_y*beta_y)*(dist1_y),   #(np.random.randn(n_macroparticles)), #(dist0_y),
    py=np.sqrt(physemit_y/beta_y)*dist1_py,  #(np.random.randn(n_macroparticles)), #dist0_y,
    zeta=sigma_z*(np.random.randn(n_macroparticles)), #dist0,
    delta=sigma_delta*(np.random.randn(n_macroparticles)), #dist0,
    weight=bunch_intensity/n_macroparticles
)

bb_particles_b2 = xp.Particles(_context=context,
    p0c=p0c,
    x=np.sqrt(physemit_x*beta_x)*(dist2_x),       #(np.random.randn(n_macroparticles)), #(dist1_x),
    px=np.sqrt(physemit_x/beta_x)*dist2_px,    #(np.random.randn(n_macroparticles)), #dist1_x,
    y=np.sqrt(physemit_y*beta_y)*(dist2_y),  #(np.random.randn(n_macroparticles)), #(dist1_y),
    py=np.sqrt(physemit_y/beta_y)*dist2_py,  #(np.random.randn(n_macroparticles)), #dist1_y,
    zeta=sigma_z*(np.random.randn(n_macroparticles)), #dist1,
    delta=sigma_delta*(np.random.randn(n_macroparticles)), #dist1,
    weight=bunch_intensity/n_macroparticles
)

# bb_particles_b1 = xp.Particles(_context=context,
#     p0c=p0c,
#     x=np.sqrt(physemit_x*beta_x)*(np.random.randn(n_macroparticles)),
#     px=np.sqrt(physemit_x/beta_x)*np.random.randn(n_macroparticles),
#     y=np.sqrt(physemit_y*beta_y)*(np.random.randn(n_macroparticles)),
#     py=np.sqrt(physemit_y/beta_y)*np.random.randn(n_macroparticles),
#     zeta=sigma_z*np.random.randn(n_macroparticles),
#     delta=sigma_delta*np.random.randn(n_macroparticles),
#     weight=bunch_intensity/n_macroparticles
# )
# # print(np.sqrt(physemit_x*beta_x)*np.random.uniform(-1,1,n_macroparticles))
# # print(np.sqrt(physemit_x/beta_x)*np.linspace(-1,1,n_macroparticles))
# bb_particles_b2 = xp.Particles(_context=context,
#     p0c=p0c,
#     x=np.sqrt(physemit_x*beta_x)*(np.random.randn(n_macroparticles)),
#     px=np.sqrt(physemit_x/beta_x)*np.random.randn(n_macroparticles),
#     y=np.sqrt(physemit_y*beta_y)*(np.random.randn(n_macroparticles)),
#     py=np.sqrt(physemit_y/beta_y)*np.random.randn(n_macroparticles),
#     zeta=sigma_z*np.random.randn(n_macroparticles),
#     delta=sigma_delta*np.random.randn(n_macroparticles),
#     weight=bunch_intensity/n_macroparticles
# )

# plt.figure(1)
# plt.plot(bb_particles_b1.x,bb_particles_b1.y,'.')
# plt.show()
#print(bb_particles_b1.x)
nbb_particles_b1 = bb_particles_b1.copy()
nbb_particles_b2 = bb_particles_b2.copy()
# beambeamstates = [True] #noBB
bbstate = True
# for xshift in xshifts:
name_sep = str(xshift)
#     for coupling in couplings:
# coupling = round(coupling,4)
ksl = coupling
name_coupling = str(coupling)
# if coupling == 0:
#     qxchange = 0
#     qychange = 0
#     ksl = 0.0 
#     #l = 0.01
#     name_coupling = '0coup'
# elif coupling ==5:
#     qxchange = phase_adv #0.000657
#     qychange = phase_adv #0.000757
#     ksl = 0.16839
#     name_coupling = '5coup'
# elif coupling == 8:
#     qxchange = phase_adv #0.0019406
#     qychange = phase_adv #0.0020571
#     ksl = 0.26220
#     name_coupling = '8coup'
# elif coupling == 16:
#     qxchange = phase_adv #0.006931
#     qychange = phase_adv #0.00666
#     ksl = 0.5244
#     name_coupling = '16coup'
# elif coupling == -1.5:
#     qxchange = phase_adv
#     qychange = phase_adv
#     ksl = 0.15 #1e-3
#     name_coupling = '1.5e-1'
# elif coupling == -1:
#     qxchange = phase_adv
#     qychange = phase_adv
#     ksl = 0.1 #1e-3
#     name_coupling = '1e-1'
# elif coupling == -2:
#     qxchange = phase_adv
#     qychange = phase_adv
#     ksl = 0.01 #1e-3
#     name_coupling = '1e-2'
#     #l = ll
# elif coupling == -3:
#     qxchange = phase_adv
#     qychange = phase_adv
#     ksl = 0.001 #1e-3
#     name_coupling = '1e-3'
# elif coupling == -4:
#     qxchange = phase_adv
#     qychange = phase_adv
#     ksl = 0.0001 #1e-4
#     name_coupling = '1e-4'
# elif coupling == -4.5:
#     qxchange = phase_adv
#     qychange = phase_adv
#     ksl = 0.0005
#     name_coupling = '5e-4'
# elif coupling == -5:
#     qxchange = phase_adv
#     qychange = phase_adv
#     ksl = 0.00001 #1e-5
#     name_coupling = '1e-5'     
# elif coupling == -5.5:
#     qxchange = phase_adv
#     qychange = phase_adv
#     ksl = 0.00005 #5e-5
#     name_coupling = '5e-5'     
# else:
#     raise ValueError('coupling not defined correctly - please choose 0, 5 or 8')
print(f'Prepare BB for beam stats: {bbstate}, coup = {coupling}, sep = {xshift}')
if bbstate == True:
    particles_b1 = bb_particles_b1
    particles_b2 = bb_particles_b2
    charge_b1 = particles_b1.q0
    charge_b2 = particles_b2.q0
    name_bb = 'bb'
if bbstate == False:
    particles_b1 = nbb_particles_b1
    particles_b2 = nbb_particles_b2
    charge_b1 = 0
    charge_b2 = 0
    name_bb = 'nbb'
    
print('coupling = ' + name_coupling)
print(qxchange, qychange, ksl)

pipeline_manager = xt.PipelineManager()
pipeline_manager.add_particles('b1',0)
pipeline_manager.add_particles('b2',0)
pipeline_manager.add_element('IP1')
particles_b1.init_pipeline('b1')
particles_b2.init_pipeline('b2')

#############
#############
# Beam-beam #
#############
slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=1, mode = 'shatilov')
config_for_update_b1 = xf.ConfigForUpdateBeamBeamBiGaussian3D(
pipeline_manager=pipeline_manager,
element_name='IP1',
partner_particles_name = 'b2',
slicer=slicer,
update_every=1,
)
config_for_update_b2 = xf.ConfigForUpdateBeamBeamBiGaussian3D(
pipeline_manager=pipeline_manager,
element_name='IP1',
partner_particles_name = 'b1',
slicer=slicer,
update_every=1,
)

print('build bb elements...')
bbeam_b1 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = charge_b2,
            phi = 0,alpha=0,
            config_for_update = config_for_update_b1,
            flag_numerical_luminosity=1,
            flag_kick=1,
            n_lumigrid_cells = lumi_cells, #1200,  ######### BINSIZE
            sig_lumigrid_cells=np.sqrt(physemit_x*beta_x),
            range_lumigrid_cells = 12, #15
            n_macroparticles=n_macroparticles,
            nTurn=nTurn,
            update_lumigrid_sum = 1,
            ref_shift_x = xshift*np.sqrt(physemit_x*beta_x)/2,
            ref_shift_y = yshift*np.sqrt(physemit_x*beta_x)/2)
bbeam_b2 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = charge_b1,
            phi = 0,alpha=0,
            config_for_update = config_for_update_b2,
            flag_numerical_luminosity=1,
            flag_kick=1,
            n_lumigrid_cells = lumi_cells, #1200, 
            sig_lumigrid_cells=np.sqrt(physemit_x*beta_x),
            range_lumigrid_cells = 12, #15
            n_macroparticles=n_macroparticles,
            nTurn = nTurn,
            update_lumigrid_sum = 1,
            ref_shift_x = xshift*np.sqrt(physemit_x*beta_x)/2,
            ref_shift_y = -yshift*np.sqrt(physemit_x*beta_x)/2)

print(bbeam_b1.flag_kick)

#################################################################
# arcs (here they are all the same with half the phase advance) #
#################################################################

print('Prepare arcs and tracker')
if vdm == True :
    beta_xl1 = 108.022378 #4746.325091 # for LHC // 108.022378 for vdM
    beta_yl1 = 84.702917 #4703.922409 # for LHC // 84.702917 for vdM
    beta_xr1 = 85.523581 #4703.922409 # for LHC // 85.523581 for vdM
    beta_yr1 = 108.033490 #4746.325090 # for LHC // 108.033490 for vdM
else :
    beta_xl1 = 4746.325091 #108.022378
    beta_yl1 = 4703.922409
    beta_xr1 = 4703.922409
    beta_yr1 = 4746.325090
arc_b1 = xt.LineSegmentMap(betx = beta_x,bety = beta_y,
        qx = Qx-2*qxchange, qy = Qy-2*qxchange,bets = beta_s, qs=Qs) # note that this is only half // -2*qxchange, -2*qychange
arc = xt.LineSegmentMap(betx = beta_x,bety = beta_y,
        qx = qxchange, qy = qychange, bets = beta_s, qs=Qs)
quad = xt.Multipole(order=1, ksl =[0,ksl*np.sqrt(beta_xl1*beta_yl1)/np.sqrt(beta_x*beta_y)], length  = 1)
quad2 = xt.Multipole(order=1, ksl = [0,-ksl*np.sqrt(beta_xr1*beta_yr1)/np.sqrt(beta_x*beta_y)], length  = 1)
arc_b2 = xt.LineSegmentMap(betx = beta_x,bety = beta_y,
        qx = Qx-2*qxchange, qy = Qy-2*qychange,bets = beta_s, qs=Qs) # note that this is only half // same 

# arc_b1 = xt.LineSegmentMap(betx = beta_x,bety = beta_y,
#         qx = (Qx+qxchange), qy =(Qy-qychange),bets = beta_s, qs=Qs) # note that this is only half
# arc_b2 = xt.LineSegmentMap(betx = beta_x, bety = beta_y, 
#         qx = (Qx+qxchange), qy = (Qy-qychange), bets = beta_s, qs=Qs)
# quad = xt.Quadrupole(k1=0, k1s = ksl, length  = 0.01)


#################################################################
# Tracker                                                       #
#################################################################
print(bbeam_b1.other_beam_q0, bbeam_b2.other_beam_q0)
# monitor_1 = xt.BeamSizeMonitor(start_at_turn=0, stop_at_turn=nTurn)
# monitor_2 = xt.BeamSizeMonitor(start_at_turn=0, stop_at_turn=nTurn)
#monitor_1 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=nTurn,num_particles=n_macroparticles)
#monitor_2 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=nTurn,num_particles=n_macroparticles)
elements_b1 = [quad,arc,bbeam_b1, arc, quad2, arc_b1] #[bbeam_b1,arc,quad,arc_b1, quad2, arc]
elements_b2 = [quad, arc_b2, quad2, arc, bbeam_b2, arc]  #[bbeam_b2, arc_b2]    #
line_b1 = xt.Line(elements=elements_b1)
# line_b1.insert_element(element=monitor_1, name='m1',index=2)
print(line_b1.element_names)
line_b2 = xt.Line(elements=elements_b2)
# line_b2.insert_element(element=monitor_2, name='m2', index=5)
print(line_b2.element_names)
line_b1.build_tracker()
line_b2.build_tracker()
branch_b1 = xt.PipelineBranch(line_b1,particles_b1)
branch_b2 = xt.PipelineBranch(line_b2,particles_b2)
multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])

print('Tracking...')
#x1 = []; y1 = [] #; x2 = []; y2 = []; px1 = []; py1 = []; px2 = []; py2 = []
# rms_x1 = []; rms_y1 = [] #; rms_x2 = []; rms_y2 = [] #; c_x1 = []; c_y1 = []; c_x2 = []; c_y2 = []
# for i in range(1,nTurn+1) :
#     print(i)
#     multitracker.track(num_turns=1,turn_by_turn_monitor=True)
#     # rms
#     rms_x1.append(np.sqrt(np.mean(particles_b1.x**2)))
#     rms_y1.append(np.sqrt(np.mean(particles_b1.y**2)))
#     rms_x2.append(np.sqrt(np.mean(particles_b2.x**2)))
#     rms_y2.append(np.sqrt(np.mean(particles_b2.y**2)))
#     #centroid
#     c_x1.append(np.mean(particles_b1.x**2))
#     c_y1.append(np.mean(particles_b1.y**2))
#     c_x2.append(np.mean(particles_b2.x**2))
#     c_y2.append(np.mean(particles_b2.y**2))

multitracker.track(num_turns=nTurn,turn_by_turn_monitor=False)
# x1.append(monitor_1.x)
# # print(np.std(x1))
# y1.append(monitor_1.y)
# x2.append(particles_b2.x)
# y2.append(particles_b2.y)
# px1.append(particles_b1.px)
# py1.append(particles_b1.py)
# px2.append(particles_b2.px)
# py2.append(particles_b2.py)

# print(x1)
# print(px1)

# beam width
# rms_x1.append(monitor_1.x_std)
# rms_y1.append(monitor_1.y_std)
# rms_x2.append(monitor_2.x_std)
# rms_y2.append(monitor_2.y_std)
# print(rms_x1)

# c_x1.append(np.mean(monitor_ip.x))
# c_y1.append(np.mean(monitor_ip.y))
# c_x2.append(np.mean(monitor_ip_2.x))
# c_y2.append(np.mean(monitor_ip_2.y))
    
#multitracker.track(num_turns=nTurn,turn_by_turn_monitor=True)
line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

print('Get lumi')
num_lumi_b1_beambeam = [] 
num_lumi_b1_beambeam = bbeam_b1.numlumitable.numerical_luminosity
data = {
    'shift': (xshift, yshift),
    'int_luminosity_values': num_lumi_b1_beambeam,
    'lumigrid_b1': bbeam_b1.lumigrid_sum,
    'lumigrid_b2': bbeam_b2.lumigrid_sum
    # # 'centroid_x_1': c_x1,
    # 'centroid_y_1': c_y1,
    # 'rms_x_1': rms_x1,
    # 'rms_y_1': rms_y1
    # # # # 'centroid_x_2': c_x2,
    # # # # 'centroid_y_2': c_y2,
    # 'rms_x_2': rms_x2,
    # 'rms_y_2': rms_y2
    # 'x_1': x1,
    # 'y_1': y1
    # # #'z_1' : particles_b1.zeta,
    # 'x_2': x2,
    # 'y_2': y2,
    # #'z_2' : particles_b2.zeta
    # 'p_x1': px1,
    # 'p_y1': py1,
    # 'p_x2': px2,
    # 'p_y2': py2
}
# print(rms_x1)
# print(num_lumi_b1_beambeam)
# print(np.array(num_lumi_b1_beambeam).shape)
# print(np.array(bbeam_b1.lumigrid_sum).shape)
#print(data['int_luminosity_values'][:10])
#print(data['centroid_x_1'][:10])

file_name = f'/afs/cern.ch/work/l/llammert/public/xsuite_project/outputs/IP1/{folder}/{name_coupling}_{name_bb}.pkl' #changed path
#filename = f'/afs/cern.ch/work/l/llammert/public/xsuite_project/outputs/scripts2/IP1_lc_{name_bb}_{name_coupling}_sepX{
# }_sepY{yshift}.pkl
print('Save data')
with open(file_name, 'wb') as file:
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

line_b1.discard_tracker()
line_b2.discard_tracker()
#multitracker.stop_tracking()
    
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


