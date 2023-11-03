import yaml
from infer_vis_grasp import pointcloud_to_grasp, vis_grasp, MyGraspNet
# from sapien_grasp_env import GraspEnv
from sapien_gym import GraspEnv
from utils.config_parser import parse_yaml
import os, glob, cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from utils.visu import save_imgs_to_video
from datetime import datetime
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--demo_save_path', type=str, default=None)
parser.add_argument('--obj_start', type=int, default=None)
# parser.add_argument('--save_video', type=bool, default=None)
cfgs_args = parser.parse_args()

cfgs = parse_yaml("config.yaml")
if cfgs_args.demo_save_path is not None:
    cfgs['demo_save_path'] = cfgs_args.demo_save_path
if cfgs_args.obj_start is not None:
    cfgs['obj_start'] = cfgs_args.obj_start
# if cfgs_args.save_video is not None:
#     cfgs['save_video'] = cfgs_args.save_video
# import pdb; pdb.set_trace()

graspnet = MyGraspNet(cfgs["graspnet"])

mesh_paths =sorted(glob.glob(f"assets/ycb_all/*toma*/poisson/textured.obj"))

for mesh_path in mesh_paths[cfgs['obj_start']:]:
    name = mesh_path.split("/")[-3]
    cfgs["sapien_env"]["objs"][0]["name"] = name
    cfgs["sapien_env"]["objs"][0]["mesh_path"] = mesh_path

    env = GraspEnv(cfgs["sapien_env"])

    for episode in range(cfgs["try_random_objpose_num"]):
        obs = env.reset(random = True)
        pcs = obs["pc_xyz"]
        obj_pcs = pcs[(1-obs['pc_seg']['robot']).astype(np.bool_)]
        # cv2.imwrite(f"test_{name}.png",obs["img"][0])
        # continue
        # import pdb; pdb.set_trace()
        gg = graspnet.inference(obj_pcs)
        gg = gg.nms()
        gg = gg.sort_by_score()
        
        steps_per_action = 10
        if cfgs["graspnet"]["vis"]:
            gg_top1 = gg[0]
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            grippers = gg_top1.to_open3d_geometry()
            tmp_length = gg_top1.rotation_matrix[:,0]*0.03
            gg_top1.translation -= tmp_length
            grippers2 = gg_top1.to_open3d_geometry()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
            o3d.visualization.draw_geometries([cloud, grippers, grippers2, frame])      

        for try_i in range(min(cfgs["try_grasppose_num"], len(gg))):

            total_step = 0
            obs = env.reset()

            

            gg_top1 = gg[try_i]
            delta_m = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
            R.from_quat(env.end_effector.get_pose().q).as_matrix()

            grasp_quat_R = R.from_matrix(np.dot(gg_top1.rotation_matrix, delta_m)).as_quat()
            grasp_quat_SAPIEN = [grasp_quat_R[3],grasp_quat_R[0],grasp_quat_R[1],grasp_quat_R[2]]
            
            grasp_xyz = gg_top1.translation
            pose = np.concatenate([grasp_xyz, grasp_quat_SAPIEN])
            pose_robot_frame = pose.copy()
            pose_robot_frame[:3] -= env.robot_base_xyz
            
            pose_in_robot_coord_grasping = pose_robot_frame.copy()
            pose_in_robot_coord_reaching = pose_in_robot_coord_grasping.copy()

            rotation_unit_vect = gg_top1.rotation_matrix[:,0]
            pose_in_robot_coord_reaching[:3] -= rotation_unit_vect*0.2
            pose_in_robot_coord_grasping[:3] -= rotation_unit_vect*0.05
            result = env.planner.plan_screw(pose_in_robot_coord_reaching, env.robot.get_qpos(), time_step=cfgs["sapien_env"]["time_step"])
            
            if result['status'] != "Success":
                print("Fail")
                continue
            
            current_time = datetime.now()
            formatted_time = current_time.strftime("%m-%d-%H-%M-%S")
            save_root = cfgs['demo_save_path'] + f"/{name}-{episode}-{try_i}-{formatted_time}"
            data_root = f"{save_root}/data"
            img_root = f"{save_root}/imgs"
            os.makedirs(data_root, exist_ok=True)
            os.makedirs(img_root, exist_ok=True)
            n_step = result['position'].shape[0] // steps_per_action
            for step in range(n_step):
                print("name", name, "episode", episode, "try_i", try_i, "total_step", total_step)
                if cfgs["sapien_env"]["use_viewer"]:
                    env.render()
                action = {
                    'position': result['position'][step*steps_per_action], 
                    'velocity': result['velocity'][step*steps_per_action], 
                    'gripper': [10.0, 10.0]}
                data_to_save = {
                    "obs": obs,
                    "action": action,
                }
                img_now = Image.fromarray((obs["img"][0]).astype(np.uint8))
                img_now.save(f"{img_root}/{total_step:04d}.png")
                np.save(f"{data_root}/{total_step:04d}.npy", data_to_save, allow_pickle=True)
                obs, reward, done, info = env.step(action)
                total_step += 1


            print("finish reaching tmp position")
            
            result_2 = env.planner.plan_screw(pose_in_robot_coord_grasping, env.robot.get_qpos(), time_step=cfgs["sapien_env"]["time_step"])
            if result_2['status'] != "Success":
                print("Fail")
                continue
            n_step_2 = result_2['position'].shape[0] // steps_per_action
            for step in range(n_step_2):
                print("name", name, "episode", episode, "try_i", try_i, "total_step", total_step)
                if cfgs["sapien_env"]["use_viewer"]:
                    env.render()
                action = {
                    'position': result_2['position'][step*steps_per_action], 
                    'velocity': result_2['velocity'][step*steps_per_action], 
                    'gripper': [10.0, 10.0]}
                data_to_save = {
                    "obs": obs,
                    "action": action,
                }
                img_now = Image.fromarray((obs["img"][0]).astype(np.uint8))
                img_now.save(f"{img_root}/{total_step:04d}.png")
                np.save(f"{data_root}/{total_step:04d}.npy", data_to_save, allow_pickle=True)
                obs, reward, done, info = env.step(action)
                total_step += 1

            print("finish reaching pre-grasp position")
            
            for step in range(n_step+n_step_2, n_step+n_step_2+10):
                print("name", name, "episode", episode, "try_i", try_i, "total_step", total_step)
                if cfgs["sapien_env"]["use_viewer"]:
                    env.render()
                action = {
                    'gripper': [-10.0, -10.0]}
                data_to_save = {
                    "obs": obs,
                    "action": action,
                }
                img_now = Image.fromarray((obs["img"][0]).astype(np.uint8))
                img_now.save(f"{img_root}/{total_step:04d}.png")
                np.save(f"{data_root}/{total_step:04d}.npy", data_to_save, allow_pickle=True)
                obs, reward, done, info = env.step(action)
                total_step += 1

            print("finish grasping")

            pose_in_robot_coord_lift = pose_in_robot_coord_grasping.copy()
            pose_in_robot_coord_lift[2] += 0.3
            result_3 = env.planner.plan_screw(pose_in_robot_coord_lift, env.robot.get_qpos(), time_step=cfgs["sapien_env"]["time_step"])
            if result_3['status'] != "Success":
                print("Fail")
                continue
            n_step_3 = result_3['position'].shape[0] // steps_per_action
            for step in range(n_step_3):
                print("name", name, "episode", episode, "try_i", try_i, "total_step", total_step)
                if cfgs["sapien_env"]["use_viewer"]:
                    env.render()

                action = {
                    'position': result_3['position'][step*steps_per_action], 
                    'velocity': result_3['velocity'][step*steps_per_action], 
                    'gripper': [-10.0, -10.0]}
                data_to_save = {
                    "obs": obs,
                    "action": action,
                }
                img_now = Image.fromarray((obs["img"][0]).astype(np.uint8))
                img_now.save(f"{img_root}/{total_step:04d}.png")
                np.save(f"{data_root}/{total_step:04d}.npy", data_to_save, allow_pickle=True)
                obs, reward, done, info = env.step(action)
                total_step += 1

            print("finish reaching pre-grasp position")
            success_flag = env.objs[0].get_pose().p[2] >= 0.25 and total_step<=500
            if success_flag :
                print("Success")
            else:
                print("Fail")
            success_name = "success" if success_flag else "fail"

            if cfgs['save_video']:
                save_imgs_to_video(output_path = img_root, video_name = success_name)

            meta_data = {
                "success": success_flag,
                "cfgs": cfgs,
                "obj_name": name,
                "obj_pose": env.obj_pose_now,
            }
            np.save(f"{save_root}/{success_name}-meta.npy", meta_data, allow_pickle=True)

    env.close()