import os
import sys
import time
import argparse
import torch
import pickle
import mujoco_py
import numpy as np
from uhc.utils.config_utils.copycat_config import Config
from uhc.smpllib.smpl_robot import Robot
from mujoco_py import load_model_from_path, load_model_from_xml, MjSim, MjViewer


if __name__ == "__main__":
    #unrelated, would like to see if there are other poses I can test with

    check = input("would you like the simulation to run (y/n): ")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="copycat_40")
    args = parser.parse_args()

    # retrieve filename
    fn = input("please enter filename for the smpl body pose: ")
    filepath = "/home/hrl5/data/uhc_models/" + fn
    # load in model from pkl file
    my_model = pickle.load(open(filepath, 'rb'))

    # betas are not currently loading properly -- will work on this
    my_betas = my_model["betas"]

    # retrieve body pose and convert to torch
    my_body_pose = np.asarray(my_model["body_pose"]) 
    temp = torch.from_numpy(my_body_pose)
    temp = torch.reshape(temp, (-1, 72))

    # build robot configuration
    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.robot_cfg["model"] = "smpl"
    cfg.robot_cfg["mesh"] = True
    smpl_robot = Robot(cfg.robot_cfg, data_dir="/home/hrl5/data/smpl", masterfoot=False)
    params_names = smpl_robot.get_params(get_name=True)


    # load the body pose from the smpl_robot class
    smpl_robot.load_from_skeleton(gender=[1], pose=temp)
    # smpl_robot.load_from_skeleton(gender=[1])
    smpl_robot.write_xml(f"humanoid_smpl.xml")
    model = load_model_from_path(f"humanoid_smpl.xml")

    # amass_data = smpl_to_qpose(pose = full_pose, mj_model = model, trans = trans)

    print(f"mass {mujoco_py.functions.mj_getTotalmass(model)}")
    sim = MjSim(model)
    t1 = time.time()

    viewer = MjViewer(sim)
    print(sim.data.qpos.shape, sim.data.ctrl.shape)

    if check == "y":
        stop = False
        paused = False
        while not stop:
            sim.step()
            viewer.render()
    elif check == "s":
        stop = False
        paused = False
        while not stop:
            sim.step()
            viewer.render()
            input("press enter to step")
    else:
        stop = False
        paused = False
        while not stop:
            viewer.render()