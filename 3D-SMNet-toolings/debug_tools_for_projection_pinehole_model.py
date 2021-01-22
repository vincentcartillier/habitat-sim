import os
import sys
import cv2
import json
import math

import numpy as np

import habitat_sim

import habitat_utils

import matplotlib.pyplot as plt

# TODO: 
# get projection parameters from cfg (hov, widht, height)

"""
    Settings / Paths
"""
data_dir = './data/replicaCAD_dataset_v1_4/'
stage = 'frl_apartment_stage.glb'
layout = 'stage_0.scene_apt_0.id_0.scene_dataset_config.json'

scene = os.path.join(data_dir, "stages", stage )   # Scene path
dataset_file = os.path.join(data_dir, "3D-SMNet-dataset", layout)

output_dir = './data/point_clouds'

env = '.'.join(layout.split('.')[:-2])


path = json.load(open(
                    os.path.join("./data/paths/",
                                "path_"+env+'.json'),
                 'r')
                 )


"""
    Load habitat
"""

sim_settings = habitat_utils.make_default_settings(scene, dataset_file)

sim, sim_cfg, cfg, obj_attr_mgr, prim_attr_mgr, stage_attr_mgr = habitat_utils.make_simulator_from_settings(sim_settings)

sim.reset()
sim.load_scene_instances(sim_cfg)





"""
    Init Projector
"""
# assumption: ALL sensors have the same parameters
hfov = float(cfg.agents[0].sensor_specifications[0].parameters['hfov'])
hfov = hfov * np.pi / 180.0
near = float(cfg.agents[0].sensor_specifications[0].parameters['near'])
far  = float(cfg.agents[0].sensor_specifications[0].parameters['hfov'])

height, width = cfg.agents[0].sensor_specifications[0].resolution

vfov = hfov * height / width


f_x = width / (2.0*math.tan(hfov/2.0))
f_y = height / (2.0*math.tan(vfov/2.0))
cy = height / 2.0
cx = width / 2.0
K = np.array([[f_x, 0, cx],
              [0, f_y, cy],
              [0, 0, 1.0]])


x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
x_grid = x_grid.astype(np.float) 
y_grid = y_grid.astype(np.float)

# 0.5 is so points are projected through the center of pixels
x_scale = (x_grid - cx - 0.5) / f_x#; print(x_scale[0,0,:])
y_scale = (y_grid - cy - 0.5) / f_y#; print(y_scale[0,:,0]); stop
 


"""
    BUILD POINT CLOUD
"""
positions = path['positions']
orientations = path['orientations']

point_cloud = np.empty((0,7))

print(' -- creating point cloud')
for i, data in tqdm(enumerate(zip(positions, orientations))):
    
    # -- set agent pos/rot
    pos, ori = data
    
    agent_state = sim.agents[0].get_state()

    agent_state.position[0] = pos[0]
    agent_state.position[1] = pos[1]
    agent_state.position[2] = pos[2]

    agent_state.rotation.x = ori[0]
    agent_state.rotation.y = ori[1]
    agent_state.rotation.z = ori[2]
    agent_state.rotation.w = ori[3]
    
    sim.agents[0].set_state(agent_state)
    
    observations = sim.get_sensor_observations()
    
    # -- get agent sensor position
    agent_state = sim.agents[0].get_state()
    sensor_state = agent_state.sensor_states['depth_sensor_1st_person']

    position = sensor_state.position
    rotation = sensor_state.rotation

    s = rotation.norm()
    qi = rotation.x
    qj = rotation.y
    qk = rotation.z
    qr = rotation.w
    
    R = np.array([
        [1 - 2*s*(qj**2 + qk**2), 2*s*(qi*qj-qk*qr), 2*s*(qi*qk+qj*qr)],
        [2*s*(qi*qj+qk*qr), 1 - 2*s*(qi**2 + qk**2), 2*s*(qj*qk-qi*qr)],
        [2*s*(qi*qk-qj*qr), 2*s*(qj*qk+qi*qr), 1 - 2*s*(qi**2 + qj**2)],
    ])
    
    #rotate 180 around x to put y up
    # convert CV to robotics conventions coords
    R_yup = np.array([
        [1, 0, 0],
        [0, np.cos(np.pi), -np.sin(np.pi)],
        [0, np.sin(np.pi), np.cos(np.pi)]
    ])
    
    
    R = np.matmul(R, R_yup)
    
 
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3,3] = np.array(position)
    T[3,3] =  1




    depth = observations["depth_sensor_1st_person"]
    depth = depth.astype(np.float32)
    
    mask_outliers = depth == 0

    z = depth 
    x = z * x_scale
    y = z * y_scale
    ones = np.ones(z.shape)
    
    pc = np.concatenate((x[:,:,np.newaxis],
                         y[:,:,np.newaxis],
                         z[:,:,np.newaxis],
                         ones[:,:,np.newaxis],
                        ), axis=2)
    
    pc = pc[~mask_outliers]

    # TRANSFORMATION 
    pc = np.matmul(T, pc.T)
    pc = pc.T
    pc = pc[:,:3]

    rgb = observations["color_sensor_1st_person"]
    rgb = rgb[~mask_outliers]

    pc = np.concatenate((pc, rgb), axis=1)

    point_cloud = np.concatenate((point_cloud, pc), axis=0)
   
    if i < 13:
        file = open(
                    os.path.join(output_dir, 'pc_{}--debug-{}.txt'.format(env, i)),
                    'w')
         
        tmp_point_cloud = point_cloud[0:-1:10, :]
        for v in tqdm(tmp_point_cloud):
            line = str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2])
            line = line + ' ' + str(v[3]) + ' ' + str(v[4])  + ' ' + str(v[5]) +  '\n'
            file.write(line)
        
        file.close()

    if i==13: break


