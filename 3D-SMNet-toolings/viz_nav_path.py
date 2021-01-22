import os
import sys
import cv2
import json

import numpy as np

import habitat_sim

import habitat_utils


"""
    Settings / Paths
"""
data_dir = './data/replicaCAD_dataset_v1_4/'
stage = 'frl_apartment_stage.glb'
layout = 'stage_0.scene_apt_0.id_0.scene_dataset_config.json'

scene = os.path.join(data_dir, "stages", stage )   # Scene path
dataset_file = os.path.join(data_dir, "3D-SMNet-dataset", layout)


env = '.'.join(layout.split('.')[:-2])


path = json.load(open(
                    os.path.join("./data/paths/",
                                "path_"+env+'.json'),
                 'r')
                 )




"""
    VIZ PATHS
"""

sim_settings = habitat_utils.make_default_settings(scene, dataset_file)

sim, sim_cfg, cfg, obj_attr_mgr, prim_attr_mgr, stage_attr_mgr = habitat_utils.make_simulator_from_settings(sim_settings)

sim.reset()
sim.load_scene_instances(sim_cfg)


positions = path['positions']
orientations = path['orientations']

for i, data in enumerate(zip(positions, orientations)):

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
    rgb = observations["color_sensor_1st_person"]

    bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

    cv2.imshow("window", bgr)
    key = cv2.waitKey(100)
    if key == 27 or key == ord('q'): #esc
        cv2.destroyAllWindows()
        break


