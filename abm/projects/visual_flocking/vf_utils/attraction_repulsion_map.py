import numpy as np

from abm.projects.visual_flocking.vf_simulation.vf_sims import VFSimulation

# VF_GAM=0.1
# VF_V0=0
# VF_ALP0=1
# VF_ALP1=0.09
# VF_ALP2=0
# VF_BET0=1
# VF_BET1=0.09
# VF_BET2=0

width = 800
height = 800
undersample = 2
radius = 5

foc_ag_pos = (int(width/2)-radius, int(height/2)-radius)
dv_matrix = np.zeros((int(width/undersample), int(height/undersample)))
dpsi_matrix = np.zeros((int(width/undersample), int(height/undersample)))
ablob_matrix = np.zeros((int(width/undersample), int(height/undersample)))
aedge_matrix = np.zeros((int(width/undersample), int(height/undersample)))
bblob_matrix = np.zeros((int(width/undersample), int(height/undersample)))
bedge_matrix = np.zeros((int(width/undersample), int(height/undersample)))

sim = VFSimulation(N=2,
                   T=width*height+1,
                   v_field_res=2000,
                   agent_fov=1,
                   framerate=2000,
                   with_visualization=0,
                   width=width,
                   height=height,
                   show_vis_field=True,
                   show_vis_field_return=False,
                   vision_range=2000,
                   visual_exclusion=False,
                   show_vision_range=True,
                   use_ifdb_logging=False,
                   use_ram_logging=True,
                   save_csv_files=False,
                   use_zarr=True,
                   parallel=False,
                   window_pad=0,
                   agent_behave_param_list=None,
                   collide_agents=False)

sim.prepare_start()
for w in range(0, width, undersample):
    print(f"progress: {w}/{width}")
    for h in range(0, height, undersample):
        agent0 = list(sim.agents)[0]
        agent1 = list(sim.agents)[1]
        agent0.position = np.array(foc_ag_pos, dtype=np.float64)
        agent0.orientation = np.pi / 2
        agent0.velocity = 0
        agent0.verbose_supcalc = True
        agent1.position = np.array((w, h), dtype=np.float64)
        sim.step_sim()
        dv_matrix[int(w/undersample), int(h/undersample)] = agent0.dv
        dpsi_matrix[int(w/undersample), int(h/undersample)] = agent0.dphi
        ablob_matrix[int(w/undersample), int(h/undersample)] = agent0.ablob
        aedge_matrix[int(w / undersample), int(h / undersample)] = agent0.aedge
        bblob_matrix[int(w/undersample), int(h/undersample)] = agent0.bblob
        bedge_matrix[int(w / undersample), int(h / undersample)] = agent0.bedge

dv_matrix[dv_matrix>0.2] = 0.2
dv_matrix[dv_matrix<-0.2] = -0.2
ablob_matrix[ablob_matrix>0.2] = 0.2
ablob_matrix[ablob_matrix<-0.2] = -0.2
aedge_matrix[aedge_matrix>0.2] = 0.2
aedge_matrix[aedge_matrix<-0.2] = -0.2

dpsi_matrix[dpsi_matrix>0.2] = 0.2
dpsi_matrix[dpsi_matrix<-0.2] = -0.2
bblob_matrix[bblob_matrix>0.2] = 0.2
bblob_matrix[bblob_matrix<-0.2] = -0.2
bedge_matrix[bedge_matrix>0.2] = 0.2
bedge_matrix[bedge_matrix<-0.2] = -0.2

import matplotlib.pyplot as plt
import matplotlib

cmap_vel = matplotlib.colors.LinearSegmentedColormap.from_list(
    'mycmap', ["red", "white", "green"])
cmap_ori = matplotlib.colors.LinearSegmentedColormap.from_list(
    'mycmap', ["orange", "white", "blue"])
plt.figure()
plt.imshow(dv_matrix.T, cmap=cmap_vel)
plt.show()
plt.figure()
plt.imshow(dpsi_matrix.T, cmap=cmap_ori)
plt.show()
plt.imshow(ablob_matrix.T, cmap=cmap_vel)
plt.show()
plt.imshow(aedge_matrix.T, cmap=cmap_vel)
plt.show()
plt.imshow(bblob_matrix.T, cmap=cmap_ori)
plt.show()
plt.imshow(bedge_matrix.T, cmap=cmap_ori)
plt.show()
