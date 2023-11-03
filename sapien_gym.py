"""Lift environment."""

import numpy as np
from gym import spaces

import sapien.core as sapien
from sapien.core import Pose
from sapien.utils.viewer import Viewer
from utils.sapien_env import SapienEnv
import mplib
from scipy.spatial.transform import Rotation as R
import time
from PIL import Image
import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utilss

class GraspEnv(SapienEnv):
    def __init__(self, cfgs):
        self.cfgs = cfgs

        self.init_qpos = [-2.3467421e-04, -2.2423947e-01,  5.5315247e-04, -2.1298823e+00,
        2.7728686e-04,  1.9058064e+00,  7.8549516e-01,  1.0883382e-08, 9.3260359e-09]
        # [0, 0.19634954084936207, 0.0, -2.617993877991494,
        #  0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
    
        self.table_height = self.cfgs["table_height"]
        super().__init__(control_freq=self.cfgs["control_freq"], timestep=self.cfgs["time_step"])

        # The arm is controlled by the internal velocity drive
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=100, damping=20)
        # for joint in self.active_joints[:5]:
        #     joint.set_drive_property(stiffness=0, damping=4.8)
        # for joint in self.active_joints[5:7]:
        #     joint.set_drive_property(stiffness=0, damping=0.72)
        # The gripper will be controlled directly by the torque

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[self.dof * 2 + 13], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=[self.dof], dtype=np.float32)

        # light
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self._scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self._scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self._scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        
        self._setup_camera()
        self._setup_mplib_planner()

        self.seg_offset = 14

    def _build_world(self):
        physical_material = self._scene.create_physical_material(1.0, 1.0, 0.0)
        self._scene.default_physical_material = physical_material
        self._scene.add_ground(-0.05)
        # robot
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(self.cfgs["robot"]["robot_urdf_path"])
        self.robot.set_name('panda')
        self.robot_base_xyz = np.array([-0.4, 0, self.table_height])
        self.robot.set_root_pose(sapien.Pose([-0.4, 0, self.table_height], [1, 0, 0, 0]))
        self.robot.set_qpos(self.init_qpos)
        self.robot = self.get_articulation('panda')
        self.end_effector = self.robot.get_links()[8]
        self.dof = self.robot.dof
        assert self.dof == 9, 'Panda should have 9 DoF'
        self.active_joints = self.robot.get_active_joints()

        # table top
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.4, 0.4, 0.025])
        builder.add_box_visual(half_size=[0.4, 0.4, 0.025], color=[0.70, 0.70, 0.70])
        self.table = builder.build_kinematic(name='table')
        self.table.set_pose(Pose([0, 0, self.table_height - 0.05]))


        self.objs = []
        # obj
        # builder = self._scene.create_actor_builder()
        # builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        # builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1, 0, 0])
        # self.cube = builder.build(name='cube')
        # self.cube.set_pose(Pose([0, 0, self.table_height + 0.02]))

        builder = self._scene.create_actor_builder()
        self.obj_pose_now = []
        for obj_id, obj_cfg in enumerate(self.cfgs["objs"]):
            builder = self._scene.create_actor_builder()
            builder.add_collision_from_file(filename=obj_cfg["mesh_path"], scale = obj_cfg["scale"])
            builder.add_visual_from_file(filename=obj_cfg["mesh_path"], scale = obj_cfg["scale"])
            obj_ins = builder.build(name=obj_cfg["name"])
            self.obj_pose_now.append(Pose(p=obj_cfg["pose"]))
            obj_ins.set_pose(self.obj_pose_now[obj_id])
            self.objs.append(obj_ins)
            
        # builder.add_collision_from_file(filename='/home/haoran/Projects/sapien/SAPIEN/assets/aligned/steel_ball/visual_mesh.obj')
        # builder.add_visual_from_file(filename='/home/haoran/Projects/sapien/SAPIEN/assets/aligned/steel_ball/visual_mesh.obj')
        # self.steel_ball = builder.build(name='mesh')
        # self.steel_ball.set_pose(sapien.Pose(p=[-0.15, 0, self.table_height]))
        # self.objs.append(self.steel_ball)

        # builder = self._scene.create_actor_builder()
        # builder.add_collision_from_file(filename='/home/haoran/Projects/sapien/SAPIEN/assets/aligned/champagne/visual_mesh.obj')
        # builder.add_visual_from_file(filename='/home/haoran/Projects/sapien/SAPIEN/assets/aligned/champagne/visual_mesh.obj')
        # self.champagne = builder.build(name='mesh')
        # self.champagne.set_pose(sapien.Pose(p=[0.15, 0, self.table_height]))
        # self.objs.append(self.champagne)

    def _setup_ik_solver(self):
        self.ik_chain = ikpy.chain.Chain.from_urdf_file(self.cfgs["robot"]["robot_urdf_path"])

    def step(self, action = None):
        if action is not None:
            if "position" in action:
                # Use internal velocity drive
                for idx in range(7):
                    self.active_joints[idx].set_drive_target(action['position'][idx])
                    self.active_joints[idx].set_drive_velocity_target(action['velocity'][idx])
                    # self.active_joints[idx].set_drive_velocity_target(action[idx])

            # Control the gripper directly by torque
        qf = self.robot.compute_passive_force(True, True, False)

        if action is not None:
            if "gripper" in action:
                qf[-2:] += action["gripper"]
        
        self.robot.set_qf(qf)

        for i in range(self.control_freq):
            self._scene.step()

        obs = self._get_obs(options = self.cfgs["obs"]["runtime_options"])
        reward = self._get_reward()
        done = 0 #self.cube.get_pose().p[2] > self.table_height + 0.04 # TODO
        if done:
            reward += 100.0

        return obs, reward, done, {}

    def reset(self, random = False):
        self.robot.set_qpos(self.init_qpos)
        for obj_id, obj_ins in enumerate(self.objs):
            # obj_ins.set_pose(Pose(self.cfgs["objs"][obj_id]["pose"]))
            if random:
                self.obj_pose_now[obj_id] = Pose([np.random.randn() * 0.2, np.random.randn() * 0.2, 0.02])
                obj_ins.set_pose(self.obj_pose_now[obj_id])
            else:
                obj_ins.set_pose(self.obj_pose_now[obj_id])
        # for i in range(5):
        self._scene.step()
        return self._get_obs(options = self.cfgs["obs"]["reset_options"])

    def _get_obs(self, options = []):
        obs = {}
        
        # state
        obs_state = []
        assert "robot_state" in options
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()
        ee_pose = self.end_effector.get_pose()
        ee_p = ee_pose.p
        ee_q = ee_pose.q
        obs_state = np.hstack([qpos, qvel, ee_p, ee_q])

        if "obj_state" in options:
            for obj_ins in self.objs:
                obj_pose = obj_ins.get_pose()
                obj_to_ee = ee_pose.p - obj_pose.p
                obj_p = obj_pose.p
                obj_q = obj_pose.q
                obs_state = np.hstack([obs_state, obj_p, obj_q, obj_to_ee])
        obs["state"] = obs_state
        obs["meta"] = options
        obs["state_dim"] = obs_state.shape[0]

        # pc
        if "pc_xyz" in options or \
            "pc_rgb" in options or \
            "img" in options:

            use_timer = self.cfgs["obs"]["timer"]
            if use_timer:
                s = time.time()
            self._scene.update_render()
            [cam.take_picture() for cam in self.cameras]
            if use_timer:
                t = time.time()
                print("pic", t - s)

              # [H, W, 4]
            
            if "pc_rgb" in options or "img" in options:
                if use_timer:
                    s = time.time()
                rgbas = [cam.get_float_texture('Color') for cam in self.cameras]
                if use_timer:
                    t = time.time()
                    print("rgba", t - s)
            if "pc_xyz" in options:
                if use_timer:
                    s = time.time()
                positions = [cam.get_float_texture('Position') for cam in self.cameras]
                if use_timer:
                    t = time.time()
                    print("position", t - s)
                model_matrixs = [cam.get_model_matrix() for cam in self.cameras]
                points_worlds = []
                pc_save = []
                for cam_id in range(len(self.cameras)):
                    position = positions[cam_id]
                    model_matrix = model_matrixs[cam_id]
                    points_opengl = position.reshape(-1, position.shape[-1])[..., :3]
                    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
                    points_worlds.append(points_world)
                    if cam_id in self.cfgs["cams"]["pc_save_ids"]:
                        pc_save.append(points_world)
                points_worlds = np.concatenate(points_worlds, axis=0)
                if "pc_save" in options:
                    obs["pc_save"] = pc_save
                # valid mask
                pc_valid_mask = (points_worlds[..., 0] < 0.38) & (points_worlds[..., 0] > -0.38) \
                    & (points_worlds[..., 1] < 0.38) & (points_worlds[..., 1] > -0.38)
                
                # register xyz
                obs["pc_xyz"] = points_worlds[pc_valid_mask]
                if "pc_rgb" in options:
                    points_colors = [rgba.reshape(-1, rgba.shape[-1])[..., :3] for rgba in rgbas]
                    points_colors = np.concatenate(points_colors, axis=0)
                    # register rgb
                    obs["pc_rgb"] = points_colors[pc_valid_mask]
                if "pc_seg" in options:
                    seg_labels = [cam.get_uint32_texture('Segmentation') for cam in self.cameras]
                    label0_image = np.concatenate([seg_label.reshape(-1, seg_label.shape[-1])[..., 0].astype(np.uint8) for seg_label in seg_labels], axis=0)
                    obj_mask_img = label0_image == self.seg_offset
                    robot_mask_img = (label0_image >= 2)&(label0_image <= 12)
                    # register seg
                    obs["pc_seg"] = {
                        "obj":obj_mask_img[pc_valid_mask].astype(np.uint8),
                        "robot": robot_mask_img[pc_valid_mask].astype(np.uint8),
                        }
                    if "img_seg" in options:
                        obs["img_seg"] = {
                            "obj":obj_mask_img.astype(np.uint8),
                            "robot": robot_mask_img.astype(np.uint8),
                        }
                
            if "img" in options:
                rgba_imgs = [(rgba * 255).clip(0, 255).astype("uint8") for rgba in rgbas]
                obs["img"] = rgba_imgs
                if self.cfgs["obs"]["save_img_for_debug"]:
                    for id, rgba_img in enumerate(rgba_imgs):
                        rgba_pil = Image.fromarray(rgba_img)
                        rgba_pil.save(f'color_{id}.png')
        
        return obs

    def _get_reward(self):
        # reaching reward
        # cube_pose = self.cube.get_pose()
        # ee_pose = self.end_effector.get_pose()
        # distance = np.linalg.norm(ee_pose.p - cube_pose.p)
        # reaching_reward = 1 - np.tanh(10.0 * distance)

        # # lifting reward
        # lifting_reward = max(
        #     0, self.cube.pose.p[2] - self.table_height - 0.02) / 0.02

        return 0 #reaching_reward + lifting_reward

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):

        self._scene.set_ambient_light([.4, .4, .4])
        self._scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_viewer(self):
        self._setup_lighting()
        if self.cfgs["use_viewer"]:
            self.viewer = Viewer(self._renderer)
            self.viewer.set_scene(self._scene)
            self.viewer.set_camera_xyz(x=1.5, y=0.0, z=2.0)
            self.viewer.set_camera_rpy(y=3.14, p=-0.5, r=0)

    def _setup_mplib_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf=self.cfgs["robot"]["robot_urdf_path"],
            srdf="assets/robot/panda/panda.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))

    def _setup_camera(self):
        self.cameras = [
            self._scene.add_camera(
                name="camera",
                width=self.cfgs["cams"]["width"],
                height=self.cfgs["cams"]["height"],
                fovy=np.deg2rad(35),
                near=self.cfgs["cams"]["near"],
                far=self.cfgs["cams"]["far"],
            ) for cam in self.cfgs["cams"]["options"]
        ]

        for cam_id, cam_pose in enumerate(self.cfgs["cams"]["options"]):
            q = R.from_euler('xyz', cam_pose[3:], degrees=True).as_quat()
            self.cameras[cam_id].set_pose(sapien.Pose(p=cam_pose[:3], q=q))    

def main():
    env = GraspEnv()
    env.reset()

    for episode in range(100):
        for step in range(100):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.step(action)
            if done:
                print(f'Done at step {step}')
                break
        obs = env.reset()
    env.close()


if __name__ == '__main__':
    main()
