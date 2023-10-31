import yaml
from infer_vis_grasp import pointcloud_to_grasp, vis_grasp, MyGraspNet
# from sapien_grasp_env import GraspEnv
from sapien_gym import GraspEnv
from utils.config_parser import parse_yaml
import os, glob
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from utils.visu import save_imgs_to_video

cfgs = parse_yaml("config.yaml")
def debug(env, qpos):
    env.robot.set_qpos(qpos)
    env.render()

    for step in range(3000000000000):
        if cfgs["sapien_env"]["use_viewer"]: env.render()
        obs, reward, done, info = env.step()

graspnet = MyGraspNet(cfgs["graspnet"])

mesh_paths = glob.glob(f"/home/haoran/Projects/GraspingPolicy/assets/ycb_all/*/poisson/textured.obj")
for mesh_path in mesh_paths[13:]:
    name = mesh_path.split("/")[-3]
    cfgs["sapien_env"]["objs"][0]["name"] = name
    cfgs["sapien_env"]["objs"][0]["mesh_path"] = mesh_path

    env = GraspEnv(cfgs["sapien_env"])

    for episode in range(cfgs["try_random_objpose_num"]):
        obs = env.reset(random = True)
        pcs = obs["pc_xyz"]
        obj_pcs = pcs[(1-obs['pc_seg']['robot']).astype(np.bool_)]
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
            obs_traj = []
            total_step = 0
            obs = env.reset()
            obs_traj.append(obs)
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
            n_step = result['position'].shape[0] // steps_per_action
            for step in range(n_step):
                print("name", name, "episode", episode, "try_i", try_i, "total_step", total_step)
                if cfgs["sapien_env"]["use_viewer"]:
                    env.render()
                action = {
                    'position': result['position'][step*steps_per_action], 
                    'velocity': result['velocity'][step*steps_per_action], 
                    'gripper': [100.0, 100.0]}
                obs, reward, done, info = env.step(action)
                total_step += 1
                obs_traj.append(obs)

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
                    'gripper': [100.0, 100.0]}
                obs, reward, done, info = env.step(action)
                total_step += 1
                obs_traj.append(obs)
                if done:
                    print(f'Done at step {step}')
                    break
            print("finish reaching pre-grasp position")
            
            for step in range(n_step+n_step_2, n_step+n_step_2+10):
                print("name", name, "episode", episode, "try_i", try_i, "total_step", total_step)
                if cfgs["sapien_env"]["use_viewer"]:
                    env.render()
                action = {
                    'gripper': [-1.0, -1.0]}
                obs, reward, done, info = env.step(action)
                total_step += 1
                obs_traj.append(obs)
                if done:
                    print(f'Done at step {step}')
                    break
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
                    'gripper': [-1.0, -1.0]}
                obs, reward, done, info = env.step(action)
                total_step += 1
                obs_traj.append(obs)
                if done:
                    print(f'Done at step {step}')
                    break
            print("finish reaching pre-grasp position")
            if  env.objs[0].get_pose().p[2] >= 0.25 and total_step<=500:
                print("Success")
                imgs = [obs["img"][0] for obs in obs_traj]
                obj_name = env.objs[0].name
                save_path = cfgs["demo_save_path"]
                os.makedirs(f"{save_path}/{obj_name}_{episode}", exist_ok=True)
                save_imgs_to_video(imgs, f"{save_path}/{obj_name}_{episode}")

                os.makedirs(f"{save_path}", exist_ok=True)
                traj_data = {
                    "obs_traj": obs_traj,
                    "cfgs": cfgs,
                    "obj_name": obj_name,
                    "obj_pose": env.obj_pose_now,
                }
                np.save(f"{save_path}/{obj_name}_{episode}.npy", obs_traj, allow_pickle=True)
            else:
                print("Fail")
                continue
        
env.close()