import os
import sys
import json
import math
import random

import magnum as mn
import numpy as np

import habitat_sim


dir_path = './'
data_path = os.path.join(dir_path, "data")

def make_cfg(settings):
    global sim_cfg
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene.id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.scene_dataset_config_file = settings["dataset-file"]
    sim_cfg.frustum_culling = True
    sim_cfg.requires_textures = True

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor_1st_person": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "orientation": [settings["sensor_pitch"], 0.0, 0.0],
        },
        "depth_sensor_1st_person": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "orientation": [settings["sensor_pitch"], 0.0, 0.0],
        },
        "semantic_sensor_1st_person": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "orientation": [settings["sensor_pitch"], 0.0, 0.0],
        },
        # configure the 3rd person cam specifically:
        "color_sensor_3rd_person": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"] + 0.2, 0.2],
            "orientation": np.array([-math.pi / 4, 0, 0]),
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_default_settings(scene, dataset_file):
    settings = {
        "width": 640,  # Spatial resolution of the observations
        "height": 480,
        #"scene": "./data/replicaCAD_dataset_v1_4/stages/frl_apartment_stage.glb",  # Scene path
        #"dataset-file":"./data/replicaCAD_dataset_v1_4/replicaCAD.scene_dataset_config.json",
        "scene": scene,
        "dataset-file": dataset_file, 
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": -math.pi / 8.0,  # sensor pitch (x rotation in rads)
        "color_sensor_1st_person": True,  # RGB sensor
        "color_sensor_3rd_person": True,  # RGB sensor 3rd person
        "depth_sensor_1st_person": False,  # Depth sensor
        "semantic_sensor_1st_person": False,  # Semantic sensor
        "seed": 1,
        "enable_physics": False,  # enable dynamics simulation
    }
    return settings


def make_simulator_from_settings(sim_settings):
    cfg = make_cfg(sim_settings)
    # initialize the simulator
    sim = habitat_sim.Simulator(cfg)
    # Managers of various Attributes templates
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(str(os.path.join(data_path, "test_assets/objects")))
    prim_attr_mgr = sim.get_asset_template_manager()
    stage_attr_mgr = sim.get_stage_template_manager()

    return sim, obj_attr_mgr, prim_attr_mgr, stage_attr_mgr



def remove_all_objects(sim):
    for obj_id in sim.get_existing_object_ids():
        sim.remove_object(obj_id)


# Set an object transform relative to the agent state
def set_object_state_from_agent(
    sim,
    ob_id,
    offset=np.array([0, 2.0, -1.5]),
    orientation=mn.Quaternion(((0, 0, 0), 1)),
    ):
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    ob_translation = agent_transform.transform_point(offset)
    sim.set_translation(ob_translation, ob_id)
    sim.set_rotation(orientation, ob_id)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    scene = "./data/replicaCAD_dataset_v1_4/stages/frl_apartment_stage.glb",  # Scene path
    dataset_file = "./data/replicaCAD_dataset_v1_4/replicaCAD.scene_dataset_config.json",


    sim_settings = make_default_settings(scene, dataset_file)
    sim, obj_attr_mgr, prim_attr_mgr, stage_attr_mgr = make_simulator_from_settings(sim_settings)
    
    sim.reset()
    sim.load_scene_instances(sim_cfg)

    observations = sim.get_sensor_observations()
    rgb = observations["color_sensor_1st_person"]

    plt.imshow(rgb)



