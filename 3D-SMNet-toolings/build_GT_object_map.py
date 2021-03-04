"""
    This script builds GT object based map given a scene ID.
    It grabs all the objects instantiated in the scene within Habitat
    It outputs a json file of a list of objects (ID, center, sizes)
    It outputs an OFF file of object bboxes to be displayed in MeshLab
"""
import os
import sys
import json
import argparse
import numpy as np

import habitat_sim

import habitat_utils


#TODO:
"""
    Set colors for each object categories.
    Define object categories.
    Probably going to wait to receive full data before choosing a set of
    categories.
    Default color: blue
"""

"""
    Set Global Paths
"""
output_dir = './data/GT_object_maps'
output_dir_off = './data/GT_object_maps_off'

# example of stage/layout
#stage = 'frl_apartment_stage.glb'
#layout = 'stage_0.scene_apt_3.id_0.scene_dataset_config.json'

def get_GT_object_map(input_dir, stage, layout):
    """
        Settings / Paths
    """
    scene = os.path.join(input_dir, "stages", stage )   # Scene path
    dataset_file = os.path.join(input_dir, "3D-SMNet-dataset", layout)

    env = '.'.join(layout.split('.')[:-2])

    """
        Load habitat
    """

    sim_settings = habitat_utils.make_default_settings(scene, dataset_file)

    sim, sim_cfg, cfg, obj_attr_mgr, prim_attr_mgr, stage_attr_mgr = habitat_utils.make_simulator_from_settings(sim_settings)

    sim.reset()
    sim.load_scene_instances(sim_cfg)


    """
        GET OBJECT MAP
    """
    object_ids = sim.get_existing_object_ids()

    object_map = []

    semantic_ids = []
    vertices = []
    faces = []
    colors = []
    vid=0
    for oid in object_ids:
        node = sim.get_object_visual_scene_nodes(oid)[0]
        semantic_id = node.semantic_id
        semantic_ids.append(semantic_id)
        aabb = node.cumulative_bb

        center = np.array([aabb.center()[0],
                          aabb.center()[1],
                          aabb.center()[2],
                          1.0,]
                         )
        sizes  = np.array([aabb.size()[0],
                          aabb.size()[1],
                          aabb.size()[2],]
                         )


        T = np.array(node.transformation)


        object_map.append({'id': semantic_id,
                           'center': center.tolist(),
                           'sizes': sizes.tolist(),
                           'transformation': T.tolist(),
                          })




        v0 = center.copy(); v0[:3] -= sizes/2
        v1 = center.copy(); v1[:2] -= sizes[:2]/2; v1[2]+=sizes[2]/2
        v2 = center.copy(); v2[0]  -= sizes[0]/2;  v2[1]+=sizes[1]/2; v2[2] -= sizes[2]/2
        v3 = center.copy(); v3[0]  -= sizes[0]/2;  v3[1]+=sizes[1]/2; v3[2] += sizes[2]/2
        v4 = center.copy(); v4[:2] += sizes[:2]/2; v4[2]+=sizes[2]/2
        v5 = center.copy(); v5[0]  += sizes[0]/2;  v5[1]+=sizes[1]/2; v5[2] -= sizes[2]/2
        v6 = center.copy(); v6[0]  += sizes[0]/2;  v6[1]-=sizes[1]/2; v6[2] += sizes[2]/2
        v7 = center.copy(); v7[0]  += sizes[0]/2;  v7[1]-=sizes[1]/2; v7[2] -= sizes[2]/2

        v0 = np.matmul(T, v0)
        v1 = np.matmul(T, v1)
        v2 = np.matmul(T, v2)
        v3 = np.matmul(T, v3)
        v4 = np.matmul(T, v4)
        v5 = np.matmul(T, v5)
        v6 = np.matmul(T, v6)
        v7 = np.matmul(T, v7)

        v0 = [v0[0], v0[2], v0[1]]
        v1 = [v1[0], v1[2], v1[1]]
        v2 = [v2[0], v2[2], v2[1]]
        v3 = [v3[0], v3[2], v3[1]]
        v4 = [v4[0], v4[2], v4[1]]
        v5 = [v5[0], v5[2], v5[1]]
        v6 = [v6[0], v6[2], v6[1]]
        v7 = [v7[0], v7[2], v7[1]]

        vertices.append(str(v0[0])+' '+str(v0[1])+' '+str(v0[2]))
        vertices.append(str(v1[0])+' '+str(v1[1])+' '+str(v1[2]))
        vertices.append(str(v2[0])+' '+str(v2[1])+' '+str(v2[2]))
        vertices.append(str(v3[0])+' '+str(v3[1])+' '+str(v3[2]))
        vertices.append(str(v4[0])+' '+str(v4[1])+' '+str(v4[2]))
        vertices.append(str(v5[0])+' '+str(v5[1])+' '+str(v5[2]))
        vertices.append(str(v6[0])+' '+str(v6[1])+' '+str(v6[2]))
        vertices.append(str(v7[0])+' '+str(v7[1])+' '+str(v7[2]))

        faces.append('4 '+ str(vid+0)+ ' '+ str(vid+1)+ ' '+ str(vid+3)+ ' '+ str(vid+2))
        faces.append('4 '+ str(vid+0)+ ' '+ str(vid+1)+ ' '+ str(vid+6)+ ' '+ str(vid+7))
        faces.append('4 '+ str(vid+6)+ ' '+ str(vid+7)+ ' '+ str(vid+5)+ ' '+ str(vid+4))
        faces.append('4 '+ str(vid+5)+ ' '+ str(vid+4)+ ' '+ str(vid+3)+ ' '+ str(vid+2))
        faces.append('4 '+ str(vid+1)+ ' '+ str(vid+3)+ ' '+ str(vid+4)+ ' '+ str(vid+6))
        faces.append('4 '+ str(vid+0)+ ' '+ str(vid+2)+ ' '+ str(vid+5)+ ' '+ str(vid+7))

        vid += 8

        color = [0,0,255]
        for _ in range(6):
            colors.append(color)

    print("\n Saving json file for stage {} and layouts {}".format(stage, layout))

    json.dump(object_map, open(os.path.join(output_dir,
                                            'object_map_{}.json'.format(env)),
                                            'w'
                              )
             )

    print(" \n Saving OFF file for stage {} and layouts {} \n ".format(stage, layout))
    file = open(
                os.path.join(output_dir_off,
                             'object_map_{}.off'.format(env)
                            ),
                'w')
    file.write('OFF\n')
    file.write('{} {} 0\n'.format(str(len(vertices)), str(len(faces))))
    for v in vertices:
        file.write(v+'\n')
    for f, c in zip(faces[:-1], colors[:-1]):
        line = f + ' ' + str(c[0]) + ' ' + str(c[1]) + ' ' + str(c[2]) + ' 125\n'
        file.write(line)
    f, c = faces[-1], colors[-1]
    line = f + ' ' + str(c[0]) + ' ' + str(c[1]) + ' ' + str(c[2]) + ' 125\n'
    file.write(line)
    file.close()

    sim.close()

    del sim, sim_cfg, cfg, obj_attr_mgr, prim_attr_mgr, stage_attr_mgr


def get_GT_object_map_all_envs(input_dir):
    stage = 'frl_apartment_stage.glb' # -- TODO: only one stage at the moment

    layout_files = os.listdir(os.path.join(input_dir,
                                           "3D-SMNet-dataset")
                             )
    layout_files = [x for x in layout_files if x[:5]=='stage']

    for layout in layout_files:
        print('\n \n \n ', layout, '\n \n \n ')
        get_GT_object_map(input_dir, stage, layout)




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default='./data/replicaCAD_dataset_v1_4/',
        help="Dir of input files",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default='frl_apartment_stage.glb',
        help="file name of stage.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default='stage_0.scene_apt_3.id_0.scene_dataset_config.json',
        help="file name of layout.",
    )
    parser.add_argument(
        "--parse_all",
        dest="parse_all",
        action="store_true",
        help="Parse all input files in input_dir.",
    )
    args = parser.parse_args()


    if args.parse_all:
        get_GT_object_map_all_envs(args.input_dir)
    else:
        get_GT_object_map(args.input_dir,
                         args.stage,
                         args.layout)

