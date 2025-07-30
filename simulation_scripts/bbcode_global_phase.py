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

vdm = True
flat = False

if vdm == True:
    if flat == True :
        physemit_x = (4.142E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
        physemit_y = (2.071E-6*xp.PROTON_MASS_EV)/p0c  #3.75 for LHC 
    else :
        physemit_x = (2.946E-6*xp.PROTON_MASS_EV)/p0c #2.946 for vdM
        physemit_y = (2.946E-6*xp.PROTON_MASS_EV)/p0c  #3.75 for LHC
    beta_x = 19.17 #in meters 0.3 m for LHC, 19.17 for vdM, 0.15m for HL-LHC
    beta_y = 19.17 ## round BEAM (/4 for flat)
    folder = 'vdM_global_nobb'
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
    folder = 'LHC_global_nobb'
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 62.31
Qy = 60.32
Qs = 2.1E-3
frev = 11245.5 
nTurn = 1000 #700 #1000
phase_adv = 0.25 #0.25  # [1 = 2pi] so [0.5 = pi] and [0.25 = pi/2]
#ll = 0.1
print(folder)

n_macroparticles = int(15) #1e7

random = False

totalshift = 0
xshifts = [0]
yshift = 0
couplings = [0,5,8,16] #[0,-2] # try 1e-2

shift_label = "xshift"
#folder = ['noC_noBB', 'noC_BB', 'C_noBB', 'C_BB']


print('Initialising particles for vdm = '+str(vdm))
if random == True :
    bb_ptcles_b1 = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x)*(np.random.randn(n_macroparticles)), #np.linspace(-6,6,n_macroparticles)
        px=np.sqrt(physemit_x/beta_x)*np.random.randn(n_macroparticles),
        y=np.sqrt(physemit_y*beta_y)*(np.random.randn(n_macroparticles)),
        py=np.sqrt(physemit_y/beta_y)*np.random.randn(n_macroparticles),
        zeta=sigma_z*np.random.randn(n_macroparticles),
        delta=sigma_delta*np.random.randn(n_macroparticles),
        weight=bunch_intensity/n_macroparticles
    )

    bb_ptcles_b2 = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x)*(np.random.randn(n_macroparticles)),
        px=np.sqrt(physemit_x/beta_x)*np.random.randn(n_macroparticles),
        y=np.sqrt(physemit_y*beta_y)*(np.random.randn(n_macroparticles)),
        py=np.sqrt(physemit_y/beta_y)*np.random.randn(n_macroparticles),
        zeta=sigma_z*np.random.randn(n_macroparticles),
        delta=sigma_delta*np.random.randn(n_macroparticles),
        weight=bunch_intensity/n_macroparticles
    )
else : 
    bb_ptcles_b1 = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x)*(np.linspace(-6,6,n_macroparticles)), #np.linspace(-6,6,n_macroparticles)
        px=np.sqrt(physemit_x/beta_x)*np.linspace(-6,6,n_macroparticles),
        y=np.sqrt(physemit_y*beta_y)*(np.linspace(-6,6,n_macroparticles)),
        py=np.sqrt(physemit_y/beta_y)*np.linspace(-6,6,n_macroparticles),
        zeta=sigma_z*np.random.randn(n_macroparticles),
        delta=sigma_delta*np.random.randn(n_macroparticles),
        weight=bunch_intensity/n_macroparticles
    )

    bb_ptcles_b2 = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x)*(np.linspace(-6,6,n_macroparticles)),
        px=np.sqrt(physemit_x/beta_x)*np.linspace(-6,6,n_macroparticles),
        y=np.sqrt(physemit_y*beta_y)*(np.linspace(-6,6,n_macroparticles)),
        py=np.sqrt(physemit_y/beta_y)*np.linspace(-6,6,n_macroparticles),
        zeta=sigma_z*np.random.randn(n_macroparticles),
        delta=sigma_delta*np.random.randn(n_macroparticles),
        weight=bunch_intensity/n_macroparticles
    )
#print(bb_particles_b1.x)
# plt.figure(1)
# plt.plot(bb_ptcles_b1.x,bb_ptcles_b1.y,'.')
# plt.show()
   
for xshift in xshifts:     
    for coupling in couplings :
        if coupling == 0:
            qxchange = 0
            qychange = 0
            ksl = 0.0 
            #l = 0.01
            name_coupling = '0coup'
        elif coupling ==5:
            qxchange = 0.000657
            qychange = 0.000757
            ksl = 0.16839
            name_coupling = '5coup'
        elif coupling == 8:
            qxchange = 0.0019406
            qychange = 0.0020571
            ksl = 0.26220
            name_coupling = '8coup'
        elif coupling == 16:
            qxchange = 0.006931
            qychange = 0.00666
            ksl = 0.5244
            name_coupling = '16coup' 
        else:
            raise ValueError('coupling not defined correctly - please choose 0, 5 or 8')

        #variables to change - xshift, yshift, or other, 
        #coupling no coupling, medium coupling
        #run for each single separation beam beam no beam beam for same seed - copy the particles and run again - joanna sent you the code
        #move this to outside the beam beam no bb loop to keep the same distribution for each single separation case, make sure the particles reset back to o>

        bb_particles_b1 = bb_ptcles_b1.copy()
        bb_particles_b2 = bb_ptcles_b2.copy()
        nbb_particles_b1 = bb_ptcles_b1.copy()
        nbb_particles_b2 = bb_ptcles_b2.copy()
        beambeamstates = [False] #noBB
        for bbstate in beambeamstates:
            print(f'Prepare BB for beam stats: {bbstate}')
            if bbstate == True:
                particles_b1 = bb_particles_b1
                particles_b2 = bb_particles_b2
                charge_b1 = particles_b1.q0
                charge_b2 = particles_b2.q0
                name_bb = 'BB'
            if bbstate == False:
                particles_b1 = nbb_particles_b1
                particles_b2 = nbb_particles_b2
                charge_b1 = 0
                charge_b2 = 0
                name_bb = 'noBB'
                #ksl = 0
                #qxchange = 0
                #qychange = 0
                
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
                        flag_kick=0,
                        n_lumigrid_cells = 120,  ######### BINSIZE
                        sig_lumigrid_cells=np.sqrt(physemit_x*beta_x),
                        range_lumigrid_cells = 12, #15
                        n_macroparticles=n_macroparticles,
                        nTurn=nTurn,
                        update_lumigrid_sum = 0,
                        ref_shift_x = xshift*np.sqrt(physemit_x*beta_x)/2,
                        ref_shift_y = yshift*np.sqrt(physemit_x*beta_x)/2)
            bbeam_b2 = xf.BeamBeamBiGaussian3D(
                        _context=context,
                        other_beam_q0 = charge_b1,
                        phi = 0,alpha=0,
                        config_for_update = config_for_update_b2,
                        flag_numerical_luminosity=1,
                        flag_kick=0,
                        n_lumigrid_cells = 120, 
                        sig_lumigrid_cells=np.sqrt(physemit_x*beta_x),
                        range_lumigrid_cells = 12, #15
                        n_macroparticles=n_macroparticles,
                        nTurn = nTurn,
                        update_lumigrid_sum = 0,
                        ref_shift_x = xshift*np.sqrt(physemit_x*beta_x)/2,
                        ref_shift_y = -yshift*np.sqrt(physemit_x*beta_x)/2)
            # n_lumigrid_cells = 1200
            # sigma=np.sqrt(physemit_x*beta_x)
            # Delta_x = (12*sigma/n_lumigrid_cells)*1e2 # to make in um (1e2 to make in cm)

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
                    qx = (Qx+qxchange), qy =(Qy-qychange),bets = beta_s, qs=Qs) # note that this is only half
            arc_b2 = xt.LineSegmentMap(betx = beta_x, bety = beta_y, 
                    qx = (Qx+qxchange), qy = (Qy-qychange), bets = beta_s, qs=Qs)
            quad = xt.Quadrupole(k1=0, k1s = ksl, length  = 0.01)
            
            # arc_b1 = xt.LineSegmentMap(betx = beta_x,bety = beta_y,
            #         qx = (Qx+qxchange), qy =(Qy-qychange),bets = beta_s, qs=Qs) # note that this is only half
            # arc_b2 = xt.LineSegmentMap(betx = beta_x, bety = beta_y, 
            #         qx = (Qx+qxchange), qy = (Qy-qychange), bets = beta_s, qs=Qs)
            # quad = xt.Quadrupole(k1=0, k1s = ksl, length  = 0.01)
            
            
            #################################################################
            # Tracker                                                       #
            #################################################################
            print(bbeam_b1.other_beam_q0, bbeam_b2.other_beam_q0)
            monitor_1 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=nTurn,num_particles=n_macroparticles)
            monitor_2 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=nTurn,num_particles=n_macroparticles)
            elements_b1 = [bbeam_b1, quad, arc_b1] #[bbeam_b1,arc,quad,arc_b1, quad2, arc]
            elements_b2 = [bbeam_b2, quad, arc_b2]  #[bbeam_b2, arc_b2]    
            line_b1 = xt.Line(elements=elements_b1)
            line_b1.insert_element(element=monitor_1, name='m1',index=0)
            line_b1.insert_element(element=monitor_2, name='m2', index=2)
            line_b2 = xt.Line(elements=elements_b2)
            line_b1.build_tracker()
            line_b2.build_tracker()
            branch_b1 = xt.PipelineBranch(line_b1,particles_b1)
            branch_b2 = xt.PipelineBranch(line_b2,particles_b2)
            multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])
            
            #print(particles_b1.x)

            print('Tracking...')
            #rms_x1 = []; rms_y1 = []; rms_x2 = []; rms_y2 = []; c_x1 = []; c_y1 = []; c_x2 = []; c_y2 = []
            #x1 = []; y1 = []; x2 = []; y2 = []; px1 = []; py1 = []; px2 = []; py2 = []
            x1_1 = []; y1_1 = []; px1_1 = []; py1_1 = []; x1_2 = []; y1_2 = []; px1_2 = []; py1_2 = [] #; x1_3 = []; y1_3 = []; px1_3 = []; py1_3 = []; x1_4 = []; y1_4 = []; px1_4 = []; py1_4 = []
            #num_lumi_b1_beambeam = []
            # for i in range(1,nTurn+1) :
            # #     print(i)
            #     multitracker.track(num_turns=1,turn_by_turn_monitor=True)
            # #     #positions
            #     print(particles_b1.x)
            #     x1.append(particles_b1.x)
            # #     y1.append(particles_b1.y/Delta_x)
            # #     x2.append(particles_b2.x/Delta_x)
            # #     y2.append(particles_b2.y/Delta_x)
                
            # #     #momenta
            # #     px1.append(particles_b1.px/Delta_x)
            # #     py1.append(particles_b1.py/Delta_x)
            # #     px2.append(particles_b2.px/Delta_x)
            # #     py2.append(particles_b2.py/Delta_x)  
            #     print(x1)
                
                
            #multitracker.track(num_turns=nTurn,turn_by_turn_monitor=False)
            #line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
            multitracker.track(num_turns=nTurn,turn_by_turn_monitor=True)
            x1_1.append(monitor_1.x)
            y1_1.append(monitor_1.y)
            px1_1.append(monitor_1.px)
            py1_1.append(monitor_1.py)
            x1_2.append(monitor_2.x)
            y1_2.append(monitor_2.y)
            px1_2.append(monitor_2.px)
            py1_2.append(monitor_2.py)
 
            line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
            
            print('Get lumi')
            num_lumi_b1_beambeam = bbeam_b1.numlumitable.numerical_luminosity
            data = {
                'shift': (xshift, yshift),
                'int_luminosity_values': num_lumi_b1_beambeam,
                #'lumigrid_b1': bbeam_b1.lumigrid_sum,
                #'lumigrid_b2': bbeam_b2.lumigrid_sum,
                'x1_1': x1_1,
                'y1_1': y1_1,
                #'z_1' : particles_b1.zeta,
                #'z_2' : particles_b2.zeta
                'px1_1': px1_1,
                'py1_1': py1_1,
                'x1_2': x1_2,
                'y1_2': y1_2,
                'px1_2': px1_2,
                'py1_2': py1_2
            }

            file_name = f'/afs/cern.ch/work/l/llammert/public/xsuite_project/outputs/phase_space_2/{folder}/{name_coupling}.pkl' #changed path
            #filename = f'/afs/cern.ch/work/l/llammert/public/xsuite_project/outputs/scripts2/IP1_lc_{name_bb}_{name_coupling}_sepX{xshift}_sepY{yshift}.pkl
            print('Save data')
            with open(file_name, 'wb') as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            
            line_b1.discard_tracker()
            line_b2.discard_tracker()
    
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

