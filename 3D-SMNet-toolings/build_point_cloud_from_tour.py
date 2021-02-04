import os
import sys
import cv2
import json
from tqdm import tqdm
import open3d as o3d

import numpy as np

import habitat_sim

import habitat_utils
from projector import PointCloud
from projector.core import _transform3D


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
import torch
# assumption: ALL sensors have the same parameters
hfov = float(cfg.agents[0].sensor_specifications[0].parameters['hfov'])
hfov = hfov * np.pi / 180.0
near = float(cfg.agents[0].sensor_specifications[0].parameters['near'])
far  = float(cfg.agents[0].sensor_specifications[0].parameters['hfov'])

image_height, image_width = cfg.agents[0].sensor_specifications[0].resolution

vfov = hfov * image_height / image_width

# -- set to default
z_clip = 3.0 #m
world_shift = torch.FloatTensor([0,0,0])

projector = PointCloud(vfov,
                       1,
                       image_height,
                       image_width,
                       world_shift,
                       z_clip,
                       device = torch.device("cpu")
                      )


"""
    BUILD POINT CLOUD
"""

positions = path['positions']
orientations = path['orientations']

point_cloud = o3d.geometry.PointCloud()

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

    sensor_pos = sensor_state.position
    sensor_rot = sensor_state.rotation

    T = _transform3D(sensor_pos,
                     sensor_rot)


    observations = sim.get_sensor_observations()
    depth = observations["depth_sensor_1st_person"]
    depth = depth.astype(np.float32)
    depth_var = torch.FloatTensor(depth.copy()).unsqueeze(0).unsqueeze(0)

    pc, mask_outliers = projector.forward(depth_var, T)

    pc = pc[~mask_outliers]
    pc = pc.numpy()


    rgba = observations["color_sensor_1st_person"]
    mask_inliers = ~mask_outliers[0].numpy()
    rgba = rgba[mask_inliers]
    
    rgb = rgba[:,:3]
    rgb = rgb.astype(np.float)
    rgb = rgb / 255.0

    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(pc)
    tmp_pcd.colors = o3d.utility.Vector3dVector(rgb)

    point_cloud += tmp_pcd



# convert coordinates to match VoteNet -> Done in VoteNet preprocess point cloud
# -- point_cloud[...,[0,1,2]] = point_cloud[...,[0,2,1]]
# -- point_cloud[...,1] *= 1

o3d.io.write_point_cloud(os.path.join(output_dir, 
                                      'pc_{}.ply'.format(env)), 
                         point_cloud)


"""


VIEWING DEBUGGING TOOLS


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

pc = point_cloud[0:-1:100, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = pc[:,0]
y = pc[:,1]
z = pc[:,2]

ax.scatter(x, y, z)

plt.show()
"""



