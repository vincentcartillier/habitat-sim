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
layout = 'stage_0.scene_apt_3.id_1.scene_dataset_config.json'

scene = os.path.join(data_dir, "stages", stage )   # Scene path
dataset_file = os.path.join(data_dir, "3D-SMNet-dataset", layout)

output_dir = "./data/paths/"


env = '.'.join(layout.split('.')[:-2])


"""
    CREATING PATHS MANUALLY
"""
sim_settings = habitat_utils.make_default_settings(scene, dataset_file)

sim, sim_cfg, cfg, obj_attr_mgr, prim_attr_mgr, stage_attr_mgr = habitat_utils.make_simulator_from_settings(sim_settings)

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
        data = {'positions': positions,
                'orientations': orientations,
                'actions': actions}
        json.dump(data,
                  open(
                      os.path.join(output_dir, 'path_{}.json'.format(env)),
                      'w'))
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

