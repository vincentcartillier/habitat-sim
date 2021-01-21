import os
import sys
import cv2
import json
import math
import random

import numpy as np

import habitat_sim

import habitat_utils

dir_path = './'
data_path = os.path.join(dir_path, "data")


# %%
# @title Initialize Simulator and Load Scene { display-mode: "form" }

# convienience functions defined in Utility cell manage global variables
scene = "./data/replicaCAD_dataset_v1_4/stages/frl_apartment_stage.glb",  # Scene path
dataset_file = "./data/replicaCAD_dataset_v1_4/replicaCAD.scene_dataset_config.json",

sim_settings = habitat_utils.make_default_settings(scene, dataset_file)
# set globals: sim,
 
sim, obj_attr_mgr, prim_attr_mgr, stage_attr_mgr = habitat_utils.make_simulator_from_settings(sim_settings)

sim.reset()
sim.load_scene_instances(sim_cfg)


state = sim.agents[0].state

position = state.position
rotation = state.rotation

data = {}
positions = [[float(x) for x in position]]
orientations = [[float(rotation.x), float(rotation.y),
                 float(rotation.z), float(rotation.w)]]
actions = []


while True:
    observations = sim.get_sensor_observations()
    rgb = observations["color_sensor_1st_person"]

    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("window", rgb)
    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'): #esc
        data[env] = {'positions': positions,
                     'orientations': orientations,
                     'actions': actions}
        json.dump(data, open('test-replicaCAD/paths/path_{}.json'.format(env), 'w'))
        break
    elif key == ord('w'):
        sim.step('move_forward')
        action = 0
    elif key == ord('a'):
        sim.step('turn_left')
        action = 1
    elif key == ord('d'):
        sim.step('turn_right')
        action = 2
    else:
        print("Unknown key:", key)
        continue

    agent_state = sim.agents[0].state
    position = agent_state.position
    rotation = agent_state.rotation

    positions.append([float(x) for x in position])
    orientations.append([float(rotation.x), float(rotation.y),
                         float(rotation.z), float(rotation.w)])
    actions.append(action)

